import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import math
import requests
from src.model import INNv4
from torch.amp import autocast, GradScaler

# ==============================================================================
# CONFIGURATION INNv4 "MODULAR BRAIN"
# ==============================================================================
CONFIG = {
    'batch_size': 8,        
    'accum_steps': 4,       
    'seq_len': 64,          
    'epochs': 10,           
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'd_model': 256,         
    'n_colonies': 4,        # 4 Départements
    'neurons_per_colony': 32,
    'n_layers': 4,          # Profondeur du graphe
    'learning_rate': 5e-4,
    'warmup_steps': 1000
}

# --- DATA LOADING (Standard) ---
class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    def __len__(self): return len(self.idx2word)

class Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'), build_vocab=True)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
    
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

def evaluate(model, data_source, criterion, vocab_size, seq_len):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_len):
            data, targets = get_batch(data_source, i, seq_len)
            data = data.t()
            
            with autocast('cuda'):
                logits = model(data)
                loss = criterion(logits.reshape(-1, vocab_size), targets)
            
            n_tokens = data.size(0) * data.size(1)
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            
    return total_loss / total_tokens

def main():
    print("=== TRAINING INNv4: THE MODULAR BRAIN ===")
    
    # 1. Data
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
    model = INNv4(
        vocab_size=vocab_size,
        d_model=CONFIG['d_model'],
        n_layers=CONFIG['n_layers'],
        n_colonies=CONFIG['n_colonies'],
        neurons_per_colony=CONFIG['neurons_per_colony']
    ).to(device)
    
    print(f"Model Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    
    steps_per_epoch = train_data.size(0) // CONFIG['seq_len'] // CONFIG['accum_steps']
    total_steps = steps_per_epoch * CONFIG['epochs']
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG['learning_rate'],
        total_steps=total_steps,
        pct_start=0.1
    )
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    
    # 4. Loop
    optimizer.zero_grad()
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        total_loss = 0
        total_tokens = 0
        start_time = time.time()
        
        for batch, i in enumerate(range(0, train_data.size(0)-1, CONFIG['seq_len'])):
            data, targets = get_batch(train_data, i, CONFIG['seq_len'])
            data = data.t()
            
            with autocast('cuda'):
                logits = model(data)
                loss = criterion(logits.reshape(-1, vocab_size), targets)
                loss = loss / CONFIG['accum_steps']
            
            scaler.scale(loss).backward()
            
            if (batch + 1) % CONFIG['accum_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            n_tokens = data.size(0) * data.size(1)
            total_loss += (loss.item() * CONFIG['accum_steps']) * n_tokens
            total_tokens += n_tokens
            
            if (batch + 1) % (100 * CONFIG['accum_steps']) == 0:
                cur_loss = total_loss / total_tokens
                ppl = math.exp(cur_loss)
                elapsed = time.time() - start_time
                print(f"| epoch {epoch} | {batch} batches | loss {cur_loss:.2f} | ppl {ppl:.2f}")
                total_loss = 0
                total_tokens = 0
                start_time = time.time()
                
        # Val
        val_loss = evaluate(model, val_data, criterion, vocab_size, CONFIG['seq_len'])
        val_ppl = math.exp(val_loss)
        print(f"=== Epoch {epoch} | Val Loss {val_loss:.2f} | Val PPL {val_ppl:.2f} ===")
        torch.save(model.state_dict(), "checkpoints/innv4_modular.pt")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()

