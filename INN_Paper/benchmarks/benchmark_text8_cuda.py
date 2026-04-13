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

# === CONFIGURATION (TEXT8 VICTORY LAP) ===
CONFIG = {
    'dataset': 'text8',
    'vocab_size': 27,      # a-z + space
    'd_model': 256,        # Standard width
    'n_neurons': 32,       # High capacity
    'n_layers': 6,         # Deeper for reasoning
    'dropout': 0.1,
    'lr': 4e-4,            # Proven LR for Text8
    'batch_size': 16,      # Can increase batch size with smaller vocab
    'seq_len': 256,        
    'epochs': 1,           # 1 Epoch on 100M tokens is plenty
    'subset_size': 100000000, # FULL DATASET
    'grad_clip': 1.0,      # Standard clip
    'save_every': 5000     
}

device = torch.device("cuda")
print(f"=== LAUNCHING INNv2 CUDA (TEXT8 FULL) ===")
print(f"Config: {CONFIG}")

# === DATA LOADING ===
class Text8Corpus:
    def __init__(self, path, subset_size=None):
        if not os.path.exists(path):
            self.download(path)
            
        print("Loading text8...")
        with open(path, 'r') as f:
            data = f.read()
        
        if subset_size: 
            data = data[:subset_size]
            
        print(f"Data loaded. Length: {len(data):,}")
        
        # Create vocab
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        print(f"Vocab size found: {self.vocab_size} (expected 27)")
        
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Encode
        print("Encoding data...")
        self.train_data = torch.tensor([self.char_to_idx[ch] for ch in data], dtype=torch.long)
        
        # Split 90/10
        n = len(self.train_data)
        train_end = int(n * 0.90)
        self.valid = self.train_data[train_end:]
        self.train = self.train_data[:train_end]
        
        print(f"Train: {len(self.train):,} | Valid: {len(self.valid):,}")

    def download(self, path):
        print("Downloading text8...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = "http://mattmahoney.net/dc/text8.zip"
        r = requests.get(url)
        with open("data/text8.zip", "wb") as f: f.write(r.content)
        with zipfile.ZipFile("data/text8.zip", "r") as z: z.extractall("data")

def get_batch(data, i, seq_len):
    seq_len = min(seq_len, len(data) - 1 - i)
    x = data[i:i+seq_len]
    y = data[i+1:i+1+seq_len]
    return x.t().to(device), y.t().to(device) # (BSZ, SeqLen) moved to GPU here

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data # Keep on CPU until needed

# === CUDA MODEL ===
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
        # x is now (B*N, L, D)
        
        for mamba, attn, norm1, norm2 in self.layers:
            # A. Internal Dynamics (Pre-Norm)
            x_norm = norm1(x)
            x_mem = mamba(x_norm) # (B*N, L, D)
            x = x + x_mem    # Residual
            
            # B. Communication (Pre-Norm)
            x_norm2 = norm2(x)
            # Reshape to (B*L, N, D) for attention
            x_comm = x_norm2.view(B, self.num_neurons, L, -1).permute(0, 2, 1, 3).reshape(B*L, self.num_neurons, -1)
            
            comm_out, _ = attn(x_comm, x_comm, x_comm) # (B*L, N, D)
            
            # Restore shape
            x = x + comm_out.view(B, L, self.num_neurons, -1).permute(0, 2, 1, 3).reshape(B*self.num_neurons, L, -1)
            
        # Aggregation (Mean Pooling - Democracy)
        x = x.view(B, self.num_neurons, L, -1).mean(dim=1) # (B, L, D)
        return self.head(self.norm_f(x))

# === TRAINING ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args, _ = parser.parse_known_args()

    # Force clean memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Print memory status
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print(f"GPU Memory: {a/1e9:.2f}GB allocated, {r/1e9:.2f}GB reserved, {t/1e9:.2f}GB total")
    
    os.makedirs("models", exist_ok=True)
    corpus = Text8Corpus("data/text8", subset_size=CONFIG['subset_size'])
    
    model = INNv2CUDA(CONFIG['vocab_size'], CONFIG['n_neurons'], CONFIG['d_model'], CONFIG['n_layers']).to(device)
    print(f"INNv2 CUDA Params: {sum(p.numel() for p in model.parameters()):,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    train_data = batchify(corpus.train, CONFIG['batch_size'])
    valid_data = batchify(corpus.valid, CONFIG['batch_size'])
    
    # Fix: Calculate exact number of batches to avoid OneCycleLR off-by-one error
    batches_per_epoch = len(range(0, train_data.size(0)-1, CONFIG['seq_len']))
    total_steps = batches_per_epoch * CONFIG['epochs']
    
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CONFIG['lr'], total_steps=total_steps)
    crit = nn.CrossEntropyLoss()
    
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint) # We only saved state_dict, not full checkpoint
        # Estimate step from filename if possible, or manual
        try:
            start_step = int(args.resume.split('step')[-1].split('.pth')[0])
            print(f"Resuming at step {start_step}")
            # Fast-forward scheduler
            for _ in range(start_step):
                sched.step()
        except:
            print("Could not determine step from filename. Starting scheduler from 0 (suboptimal).")
    
    print(f"Starting training for {total_steps} steps...")
    model.train()
    total_loss = 0
    start = time.time()
    tokens = 0
    
    try:
        for batch, i in enumerate(range(0, train_data.size(0)-1, CONFIG['seq_len'])):
            # Skip batches if resuming
            if batch < start_step:
                continue
                
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
                # Reset counters for cleaner speed estimation
                start = time.time()
                tokens = 0
                
            if (batch+1) % CONFIG['save_every'] == 0:
                torch.save(model.state_dict(), f"models/inn_text8_cuda_step{batch+1}.pth")
                
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
            loss = crit(model(x).view(-1, CONFIG['vocab_size']), y.view(-1))
            val_loss += loss.item() * x.numel()
            val_tokens += x.numel()
            
    print(f"Valid BPC: {(val_loss/val_tokens)/math.log(2):.4f}")

