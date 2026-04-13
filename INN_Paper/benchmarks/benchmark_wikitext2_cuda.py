import torch
import torch.nn as nn
import math
import time
import os
import requests
import zipfile
import numpy as np
from collections import Counter

try:
    from mamba_ssm import Mamba
    print("✅ Mamba-SSM CUDA Kernel detected!")
except ImportError:
    print("❌ Mamba-SSM not found. Please install via `pip install mamba-ssm`")
    exit()

# === CONFIGURATION (WIKITEXT-2 REVENGE) ===
CONFIG = {
    'dataset': 'wikitext-2',
    'd_model': 256,        # Standard width
    'n_neurons': 32,       # High capacity
    'n_layers': 6,         # Deep for reasoning
    'dropout': 0.1,
    'lr': 4e-4,            # Same proven LR
    'batch_size': 16,      
    'seq_len': 256,        
    'epochs': 10,          # WikiText is small, needs more epochs
    'grad_clip': 1.0,      
    'save_every': 1000     
}

device = torch.device("cuda")
print(f"=== LAUNCHING INNv2 CUDA (WIKITEXT-2) ===")
print(f"Config: {CONFIG}")

# === DATA LOADING ===
class WikiText2Corpus:
    def __init__(self, path):
        if not os.path.exists(path):
            self.download(path)
            
        print("Loading WikiText-2...")
        self.train = self.tokenize(os.path.join(path, 'wiki.train.tokens'))
        self.valid = self.tokenize(os.path.join(path, 'wiki.valid.tokens'))
        self.test = self.tokenize(os.path.join(path, 'wiki.test.tokens'))
        
        # Build Vocab
        print("Building vocabulary...")
        tokens = self.train + self.valid + self.test
        counter = Counter(tokens)
        self.vocab = sorted(counter.keys())
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        print(f"Vocab size: {self.vocab_size}")
        
        # Numericalize
        print("Numericalizing...")
        self.train_data = self.numericalize(self.train)
        self.valid_data = self.numericalize(self.valid)
        self.test_data = self.numericalize(self.test)
        
        print(f"Train tokens: {len(self.train_data):,} | Valid: {len(self.valid_data):,}")

    def download(self, path):
        print("Downloading WikiText-2...")
        os.makedirs(path, exist_ok=True)
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
        r = requests.get(url)
        with open("data/wikitext-2.zip", "wb") as f: f.write(r.content)
        with zipfile.ZipFile("data/wikitext-2.zip", "r") as z: z.extractall("data")
        # Move files if needed, but extraction usually creates 'wikitext-2' folder

    def tokenize(self, path):
        assert os.path.exists(path), f"File not found: {path}"
        with open(path, 'r', encoding='utf-8') as f:
            # Basic tokenization: split by space, preserving some structure
            # The dataset is already tokenized, so .split() is mostly sufficient
            tokens = []
            for line in f:
                tokens.extend(line.strip().split() + ['<eos>'])
        return tokens

    def numericalize(self, tokens):
        ids = [self.word2idx[t] for t in tokens]
        return torch.tensor(ids, dtype=torch.long)

def get_batch(data, i, seq_len):
    seq_len = min(seq_len, len(data) - 1 - i)
    x = data[i:i+seq_len]
    y = data[i+1:i+1+seq_len]
    return x.t().to(device), y.t().to(device) # (BSZ, SeqLen) moved to GPU here

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data # Keep on CPU

# === CUDA MODEL (PRE-NORM FIXED) ===
class INNv2CUDA(nn.Module):
    def __init__(self, vocab_size, num_neurons, d_model, num_layers, dropout=0.1):
        super().__init__()
        self.num_neurons = num_neurons
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # 1. Internal Intelligence (Parallel Mamba Neurons)
            neuron_pop = Mamba(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand=2
            )
            
            # 2. Communication (Attention)
            attn = nn.MultiheadAttention(d_model, 4, dropout=dropout, batch_first=True)
            
            # 3. Normalization
            norm1 = nn.LayerNorm(d_model)
            norm2 = nn.LayerNorm(d_model)
            
            self.layers.append(nn.ModuleList([neuron_pop, attn, norm1, norm2]))
            
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.head.weight = self.embedding.weight

    def forward(self, x):
        B, L = x.shape
        # 1. Embed & Replicate
        x = self.embedding(x) # (B, L, D)
        x = x.unsqueeze(1).expand(-1, self.num_neurons, -1, -1).reshape(B*self.num_neurons, L, -1)
        
        for mamba, attn, norm1, norm2 in self.layers:
            # A. Internal Dynamics (Pre-Norm)
            x_norm = norm1(x)
            x_mem = mamba(x_norm) 
            x = x + x_mem    # Residual
            
            # B. Communication (Pre-Norm)
            x_norm2 = norm2(x)
            x_comm = x_norm2.view(B, self.num_neurons, L, -1).permute(0, 2, 1, 3).reshape(B*L, self.num_neurons, -1)
            
            comm_out, _ = attn(x_comm, x_comm, x_comm) 
            
            # Restore shape
            x = x + comm_out.view(B, L, self.num_neurons, -1).permute(0, 2, 1, 3).reshape(B*self.num_neurons, L, -1)
            
        # Aggregation
        x = x.view(B, self.num_neurons, L, -1).mean(dim=1) # (B, L, D)
        return self.head(self.norm_f(x))

# === TRAINING ===
if __name__ == "__main__":
    # Force clean memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    os.makedirs("models", exist_ok=True)
    # Adjust path for extraction
    corpus = WikiText2Corpus("data/wikitext-2/wikitext-2") 
    
    CONFIG['vocab_size'] = corpus.vocab_size
    
    model = INNv2CUDA(CONFIG['vocab_size'], CONFIG['n_neurons'], CONFIG['d_model'], CONFIG['n_layers']).to(device)
    print(f"INNv2 CUDA Params: {sum(p.numel() for p in model.parameters()):,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    train_data = batchify(corpus.train_data, CONFIG['batch_size'])
    valid_data = batchify(corpus.valid_data, CONFIG['batch_size'])
    
    total_steps = (train_data.size(0) // CONFIG['seq_len']) * CONFIG['epochs']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CONFIG['lr'], total_steps=total_steps)
    crit = nn.CrossEntropyLoss()
    
    print(f"Starting training for {total_steps} steps ({CONFIG['epochs']} epochs)...")
    model.train()
    total_loss = 0
    start = time.time()
    tokens = 0
    
    try:
        step = 0
        for epoch in range(CONFIG['epochs']):
            print(f"--- Epoch {epoch+1} ---")
            for i in range(0, train_data.size(0)-1, CONFIG['seq_len']):
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
                step += 1
                
                if step % 100 == 0:
                    elapsed = time.time() - start
                    cur_loss = loss.item()
                    print(f"Step {step} | Loss: {cur_loss:.4f} | PPL: {math.exp(cur_loss):.2f} | Speed: {tokens/elapsed:.0f} tok/s")
                    
                if step % CONFIG['save_every'] == 0:
                    torch.save(model.state_dict(), f"models/inn_wikitext2_cuda_step{step}.pth")
                    
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
            
    final_loss = val_loss / val_tokens
    print(f"Valid Loss: {final_loss:.4f} | Valid PPL: {math.exp(final_loss):.2f}")

