import torch
import torch.nn as nn
import math
import time
import os
import requests
import zipfile
import numpy as np

try:
    from mamba_ssm import Mamba
    print("✅ Mamba-SSM CUDA Kernel detected!")
except ImportError:
    print("❌ Mamba-SSM not found. Please install via `pip install mamba-ssm`")
    exit()

# === CONFIGURATION (MATCHING INN EXACTLY) ===
CONFIG = {
    'dataset': 'text8',
    'vocab_size': 27,
    'd_model': 256,        # Same width as INN
    'n_layers': 6,         # Same depth as INN
    'dropout': 0.1,
    'lr': 4e-4,            # Same LR
    'batch_size': 16,      # Same Batch Size
    'seq_len': 256,        # Same Context Window
    'epochs': 1,           # Same Duration (1 epoch on 100M)
    'subset_size': 100000000, # FULL DATASET
    'grad_clip': 1.0,
    'save_every': 5000
}

device = torch.device("cuda")
print(f"=== LAUNCHING MAMBA BASELINE (FULL TEXT8) ===")
print(f"Config: {CONFIG}")

# === DATA LOADING (Shared Logic) ===
class Text8Corpus:
    def __init__(self, path, subset_size=None):
        if not os.path.exists(path):
            self.download(path)
        print("Loading text8...")
        with open(path, 'r') as f: data = f.read()
        if subset_size: data = data[:subset_size]
        
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.train_data = torch.tensor([self.char_to_idx[ch] for ch in data], dtype=torch.long)
        
        n = len(self.train_data)
        train_end = int(n * 0.90)
        self.valid = self.train_data[train_end:]
        self.train = self.train_data[:train_end]
        print(f"Train: {len(self.train):,} | Valid: {len(self.valid):,}")

    def download(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = "http://mattmahoney.net/dc/text8.zip"
        r = requests.get(url)
        with open("data/text8.zip", "wb") as f: f.write(r.content)
        with zipfile.ZipFile("data/text8.zip", "r") as z: z.extractall("data")

def get_batch(data, i, seq_len):
    seq_len = min(seq_len, len(data) - 1 - i)
    x = data[i:i+seq_len]
    y = data[i+1:i+1+seq_len]
    return x.t().to(device), y.t().to(device)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data

# === MAMBA PURE MODEL ===
class MambaBaseline(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        
        for _ in range(n_layers):
            # Standard Mamba Block structure (Pre-Norm usually handled inside or outside)
            # Here we mimic INN structure: Norm -> Mamba -> Add
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(d_model),
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            ]))
            
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.head.weight = self.embedding.weight # Tying weights like INN

    def forward(self, x):
        # x: (L, B) -> (B, L) for Mamba (batch_first=True usually preferred by Mamba)
        x = x.transpose(0, 1) 
        x = self.embedding(x)
        
        for norm, mamba in self.layers:
            x_norm = norm(x)
            x = x + mamba(x_norm) # Residual connection
            
        x = self.norm_f(x)
        logits = self.head(x)
        # (B, L, V) -> (L, B, V) to match target shape
        return logits.transpose(0, 1)

# === TRAINING LOOP ===
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    corpus = Text8Corpus("data/text8", subset_size=CONFIG['subset_size'])
    
    model = MambaBaseline(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_layers']).to(device)
    print(f"Mamba Params: {sum(p.numel() for p in model.parameters()):,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    train_data = batchify(corpus.train, CONFIG['batch_size'])
    valid_data = batchify(corpus.valid, CONFIG['batch_size'])
    
    # Exact same scheduler as INN
    batches_per_epoch = len(range(0, train_data.size(0)-1, CONFIG['seq_len']))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CONFIG['lr'], total_steps=batches_per_epoch)
    crit = nn.CrossEntropyLoss()
    
    print(f"Starting training for {batches_per_epoch} steps...")
    model.train()
    total_loss = 0
    start = time.time()
    tokens = 0
    
    try:
        for batch, i in enumerate(range(0, train_data.size(0)-1, CONFIG['seq_len'])):
            x, y = get_batch(train_data, i, CONFIG['seq_len'])
            
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits.reshape(-1, CONFIG['vocab_size']), y.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            opt.step()
            sched.step()
            
            total_loss += loss.item() * len(x) * x.size(1)
            tokens += len(x) * x.size(1)
            
            if (batch+1) % 100 == 0:
                elapsed = time.time() - start
                print(f"Batch {batch+1} | Loss: {loss.item():.4f} | BPC: {loss.item()/math.log(2):.3f} | Speed: {tokens/elapsed:.0f} tok/s")
                start = time.time()
                tokens = 0
                
            if (batch+1) % CONFIG['save_every'] == 0:
                torch.save(model.state_dict(), f"models/mamba_text8_step{batch+1}.pth")
                
    except KeyboardInterrupt:
        print("Stop.")
        
    # Final Eval
    print("Evaluating...")
    model.eval()
    val_loss = 0
    val_tokens = 0
    with torch.no_grad():
        for i in range(0, valid_data.size(0)-1, CONFIG['seq_len']):
            x, y = get_batch(valid_data, i, CONFIG['seq_len'])
            loss = crit(model(x).reshape(-1, CONFIG['vocab_size']), y.reshape(-1))
            val_loss += loss.item() * x.numel()
            val_tokens += x.numel()
            
    final_bpc = (val_loss/val_tokens)/math.log(2)
    print(f"Valid BPC: {final_bpc:.4f}")

