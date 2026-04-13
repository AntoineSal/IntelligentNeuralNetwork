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

# === CONFIGURATION (SOTA ATTEMPT) ===
CONFIG = {
    'dataset': 'enwik8',
    'vocab_size': 256,     # Byte-level
    'd_model': 256,        
    'n_neurons': 32,       
    'n_layers': 4,         # SIMPLIFY to 4 layers for stability
    'dropout': 0.1,
    'lr': 1e-4,            # LOWER LR (4x smaller)
    'batch_size': 8,       
    'seq_len': 256,        
    'epochs': 1,           
    'subset_size': 100000000, 
    'grad_clip': 0.5,      # TIGHTER CLIP
    'save_every': 5000     
}

device = torch.device("cuda")
print(f"=== LAUNCHING INNv2 CUDA (ENWIK8 FULL) ===")
print(f"Config: {CONFIG}")

# === DATA LOADING ===
class ByteCorpus:
    def __init__(self, path, subset_size=None):
        if not os.path.exists(path):
            self.download(path)
            
        with open(path, 'rb') as f:
            data = f.read()
        if subset_size: data = data[:subset_size]
        
        n = len(data)
        train_end = int(n * 0.90)
        val_end = int(n * 0.95)
        
        self.train = torch.from_numpy(np.frombuffer(data[:train_end], dtype=np.uint8)).long()
        self.valid = torch.from_numpy(np.frombuffer(data[train_end:val_end], dtype=np.uint8)).long()
        print(f"Train: {len(self.train):,} | Valid: {len(self.valid):,}")

    def download(self, path):
        print("Downloading enwik8...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = "http://mattmahoney.net/dc/enwik8.zip"
        r = requests.get(url)
        with open("data/enwik8.zip", "wb") as f: f.write(r.content)
        with zipfile.ZipFile("data/enwik8.zip", "r") as z: z.extractall("data")

def get_batch(data, i, seq_len):
    seq_len = min(seq_len, len(data) - 1 - i)
    x = data[i:i+seq_len].view(-1).to(device)
    y = data[i+1:i+1+seq_len].view(-1).to(device)
    return x.view(1, -1), y.view(1, -1) # Simple batching for stream

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

# === CUDA MODEL ===
class IntelligentNeuronCUDA(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # We use the Official Mamba Block as the neuron's internal brain
        self.mamba = Mamba(
            d_model=d_model, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
        )
        
    def forward(self, x):
        # x: (B, L, D) -> Mamba -> (B, L, D)
        return self.mamba(x)

class INNv2CUDA(nn.Module):
    def __init__(self, vocab_size, num_neurons, d_model, num_layers, dropout=0.1):
        super().__init__()
        self.num_neurons = num_neurons
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # 1. Internal Intelligence (Parallel Mamba Neurons)
            # Trick: Instead of N Mamba blocks (slow loop), we can use ONE big Mamba
            # if we reshape the input to (B*N, L, D). Mamba treats batches independently.
            # This gives us massive parallelism for free.
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
        # x is now (B*N, L, D) - treated as a super-batch by Mamba
        
        for mamba, attn, norm1, norm2 in self.layers:
            # A. Internal Dynamics (All neurons think in parallel via CUDA kernel)
            x_norm = norm1(x)
            x_mem = mamba(x_norm) # (B*N, L, D)
            x = x + x_mem    # Residual
            
            # B. Communication
            x_norm2 = norm2(x)
            # Reshape to (B*L, N, D) for attention
            x_comm = x_norm2.view(B, self.num_neurons, L, -1).permute(0, 2, 1, 3).reshape(B*L, self.num_neurons, -1)
            
            comm_out, _ = attn(x_comm, x_comm, x_comm) # (B*L, N, D)
            
            # Restore shape
            x = x + comm_out.view(B, L, self.num_neurons, -1).permute(0, 2, 1, 3).reshape(B*self.num_neurons, L, -1)
            
        # Aggregation
        x = x.view(B, self.num_neurons, L, -1).mean(dim=1) # (B, L, D)
        return self.head(self.norm_f(x))

# === TRAINING ===
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    corpus = ByteCorpus("data/enwik8", subset_size=CONFIG['subset_size'])
    
    model = INNv2CUDA(CONFIG['vocab_size'], CONFIG['n_neurons'], CONFIG['d_model'], CONFIG['n_layers']).to(device)
    print(f"INNv2 CUDA Params: {sum(p.numel() for p in model.parameters()):,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    train_data = batchify(corpus.train, CONFIG['batch_size'])
    valid_data = batchify(corpus.valid, CONFIG['batch_size'])
    
    total_steps = (train_data.size(0) // CONFIG['seq_len']) * CONFIG['epochs']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CONFIG['lr'], total_steps=total_steps)
    crit = nn.CrossEntropyLoss()
    
    print(f"Starting training for {total_steps} steps...")
    model.train()
    total_loss = 0
    start = time.time()
    tokens = 0
    
    try:
        for batch, i in enumerate(range(0, train_data.size(0)-1, CONFIG['seq_len'])):
            x, y = get_batch(train_data, i, CONFIG['seq_len'])
            
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits.view(-1, CONFIG['vocab_size']), y.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            opt.step()
            sched.step()
            
            total_loss += loss.item() * len(x) * x.size(1)
            tokens += len(x) * x.size(1)
            
            if (batch+1) % 100 == 0:
                elapsed = time.time() - start
                print(f"Batch {batch+1} | Loss: {loss.item():.4f} | BPC: {loss.item()/math.log(2):.3f} | Speed: {tokens/elapsed:.0f} tok/s")
                
            if (batch+1) % CONFIG['save_every'] == 0:
                torch.save(model.state_dict(), f"models/inn_enwik8_cuda_step{batch+1}.pth")
                
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
            loss = crit(model(x).view(-1, 256), y.view(-1))
            val_loss += loss.item() * x.numel()
            val_tokens += x.numel()
            
    print(f"Valid BPC: {(val_loss/val_tokens)/math.log(2):.4f}")

