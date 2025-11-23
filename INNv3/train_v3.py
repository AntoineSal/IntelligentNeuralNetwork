import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import math
import json
import requests
from collections import Counter
from src.sparse_model import OptimizedINN

# ==============================================================================
# CONFIGURATION INNv3 - "LEAN & MEAN"
# ==============================================================================
CONFIG = {
    'batch_size': 20,
    'seq_len': 128,        # Augmenté un peu pour le contexte word-level
    'learning_rate': 5e-4,
    'epochs': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Optimized Params (Target ~9.5M)
    'd_embed': 128,        # Petit embedding
    'd_model': 256,        # Core dimension
    'num_neurons': 12,     # 12 Neurones
    'num_layers': 4,
    'n_head': 4,
    'top_k': 4,
    'num_static': 4
}

# --- DATA LOADING (Identique) ---
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
    print("=== TRAINING INNv3 (OPTIMIZED) on WikiText-2 ===")
    
    # 1. Data Check
    if not os.path.exists("data/wikitext-2/train.txt"):
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
    model = OptimizedINN(
        vocab_size=vocab_size,
        d_embed=CONFIG['d_embed'],
        d_model=CONFIG['d_model'],
        num_neurons=CONFIG['num_neurons'],
        num_layers=CONFIG['num_layers'],
        n_head=CONFIG['n_head'],
        top_k=CONFIG['top_k'],
        num_static=CONFIG['num_static']
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Params: {num_params:,} ({num_params/1e6:.2f}M)")
    
    if num_params > 12000000:
        print("⚠️ WARNING: Params > 12M. Check configuration.")
    
    # 3. Optimize
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # Warmup Scheduler
    total_steps = (train_data.size(0) // CONFIG['seq_len']) * CONFIG['epochs']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=CONFIG['learning_rate'],
        total_steps=total_steps,
        pct_start=0.05 # 5% Warmup (Rapide)
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
                print(f"| epoch {epoch} | batch {batch} | loss {cur_loss:.2f} | ppl {math.exp(cur_loss):.2f} | lr {scheduler.get_last_lr()[0]:.2e}")
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/innv3_best.pt")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()
