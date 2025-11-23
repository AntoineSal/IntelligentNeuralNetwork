import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import math
import json
import requests
from collections import Counter
from src.model import HybridINN

# ==============================================================================
# CONFIGURATION INNv3
# ==============================================================================
CONFIG = {
    'batch_size': 20,
    'seq_len': 64,
    'learning_rate': 5e-4,
    'epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Hybrid Params
    'd_model': 256,        # Taille du Stem (Standard Small Transformer)
    'stem_layers': 2,      # Profondeur du Stem
    'inn_d_model': 64,     # Taille par neurone INN
    'inn_neurons': 16,     # Nombre de neurones INN
    'inn_layers': 2        # Profondeur INN
}

# ... (Code de Data Loading identique à INNv2 - je le compacte ici pour la lisibilité) ...
class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]
    def __len__(self): return len(self.idx2word)

class Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'), build_vocab=True)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
    def tokenize(self, path, build_vocab=False):
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                if build_vocab:
                    for word in words: self.dictionary.add_word(word)
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if word in self.dictionary.word2idx: ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids

def batchify(data, bsz, device):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def main():
    print("=== TRAINING INNv3 (HYBRID) on WikiText-2 ===")
    
    # 1. Data
    if not os.path.exists("data/wikitext-2"):
        print("Downloading WikiText-2...")
        os.makedirs("data/wikitext-2", exist_ok=True)
        base_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/"
        for f in ['train.txt', 'valid.txt', 'test.txt']:
            r = requests.get(base_url + f)
            with open(f"data/wikitext-2/{f}", 'w') as out: out.write(r.text)
            
    corpus = Corpus("data/wikitext-2")
    vocab_size = len(corpus.dictionary)
    print(f"Vocab Size: {vocab_size}")
    
    device = torch.device(CONFIG['device'])
    train_data = batchify(corpus.train, CONFIG['batch_size'], device)
    val_data = batchify(corpus.valid, 10, device)
    
    # 2. Model
    model = HybridINN(
        vocab_size=vocab_size,
        d_model=CONFIG['d_model'],
        stem_layers=CONFIG['stem_layers'],
        inn_d_model=CONFIG['inn_d_model'],
        inn_neurons=CONFIG['inn_neurons'],
        inn_layers=CONFIG['inn_layers']
    ).to(device)
    
    print(f"Model Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Optimize
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=CONFIG['learning_rate'],
        total_steps=(train_data.size(0) // CONFIG['seq_len']) * CONFIG['epochs'],
        pct_start=0.1
    )
    
    # 4. Train
    for epoch in range(1, CONFIG['epochs']+1):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch, i in enumerate(range(0, train_data.size(0)-1, CONFIG['seq_len'])):
            data, targets = get_batch(train_data, i, CONFIG['seq_len'])
            data = data.t() # (B, L)
            
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits.reshape(-1, vocab_size), targets)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if batch % 100 == 0 and batch > 0:
                cur_loss = total_loss / 100
                print(f"| epoch {epoch} | batch {batch} | loss {cur_loss:.2f} | ppl {math.exp(cur_loss):.2f}")
                total_loss = 0
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, val_data.size(0)-1, CONFIG['seq_len']):
                data, targets = get_batch(val_data, i, CONFIG['seq_len'])
                logits = model(data.t())
                val_loss += len(data) * criterion(logits.reshape(-1, vocab_size), targets).item()
        val_loss /= (len(val_data)-1)
        
        print(f"=== Epoch {epoch} | Val Loss {val_loss:.2f} | Val PPL {math.exp(val_loss):.2f} ===")
        
        # Checkpoint
        torch.save(model.state_dict(), f"checkpoints/innv3_epoch{epoch}.pt")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()

