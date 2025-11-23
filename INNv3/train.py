import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import math
import json
import requests
from collections import Counter
from src.model import NeuronTokensINN

# ==============================================================================
# CONFIGURATION FINALE - "GOLDEN RUN"
# ==============================================================================
CONFIG = {
    'batch_size': 20,
    'seq_len': 128,
    'learning_rate': 5e-4, # LR robuste pour cette taille
    'epochs': 20,          # WikiText a besoin de temps
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Architecture (Optimisée pour WikiText ~10M params)
    'd_model': 256,
    'n_neurons': 16,
    'n_layers': 6,         # Plus profond = Meilleure abstraction
    'n_head': 4,
    'dropout': 0.1
}

# --- DATA LOADING (Standard & Robuste) ---
class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    def __len__(self): return len(self.idx2word)

class Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        # Train sert à construire le vocabulaire
        self.train = self.tokenize(os.path.join(path, 'train.txt'), build_vocab=True)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
    
    def tokenize(self, path, build_vocab=False):
        if not os.path.exists(path): return torch.LongTensor([])
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
                    # Gestion <unk> implicite : on ignore si pas dans vocab (ou on map au dernier)
                    # Pour WikiText standard, le vocab train couvre tout.
                    if word in self.dictionary.word2idx:
                        ids[token] = self.dictionary.word2idx[word]
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
    print("=== TRAINING INNv3 (NEURON TOKENS) ===")
    
    # 1. Data Check
    data_path = "data/wikitext-2"
    if not os.path.exists(f"{data_path}/train.txt"):
        print("Downloading WikiText-2...")
        os.makedirs(data_path, exist_ok=True)
        base_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/"
        for f in ['train.txt', 'valid.txt', 'test.txt']:
            r = requests.get(base_url + f)
            with open(f"{data_path}/{f}", 'w') as out: out.write(r.text)
            
    corpus = Corpus(data_path)
    vocab_size = len(corpus.dictionary)
    print(f"Vocab Size: {vocab_size}")
    
    device = torch.device(CONFIG['device'])
    train_data = batchify(corpus.train, CONFIG['batch_size'], device)
    val_data = batchify(corpus.valid, 10, device)
    
    # 2. Model
    model = NeuronTokensINN(
        vocab_size=vocab_size,
        d_model=CONFIG['d_model'],
        n_neurons=CONFIG['n_neurons'],
        n_layers=CONFIG['n_layers'],
        n_head=CONFIG['n_head'],
        dropout=CONFIG['dropout'],
        max_seq_len=CONFIG['seq_len']
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Params: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # 3. Optimize
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Scheduler OneCycle pour convergence rapide
    steps_per_epoch = train_data.size(0) // CONFIG['seq_len']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=CONFIG['learning_rate'],
        steps_per_epoch=steps_per_epoch,
        epochs=CONFIG['epochs'],
        pct_start=0.1
    )
    
    # 4. Train
    best_val_loss = float('inf')
    
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
                ppl = math.exp(cur_loss)
                print(f"| epoch {epoch} | batch {batch} | loss {cur_loss:.2f} | ppl {ppl:.2f} | lr {scheduler.get_last_lr()[0]:.2e}")
                total_loss = 0
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, val_data.size(0)-1, CONFIG['seq_len']):
                data, targets = get_batch(val_data, i, CONFIG['seq_len'])
                logits = model(data.t())
                val_loss += len(data) * criterion(logits.reshape(-1, vocab_size), targets).item()
        val_loss /= (len(val_data) // CONFIG['seq_len'])
        val_ppl = math.exp(val_loss)
        
        print(f"=== Epoch {epoch} | Val Loss {val_loss:.2f} | Val PPL {val_ppl:.2f} ===")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/innv3_best.pt")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()

