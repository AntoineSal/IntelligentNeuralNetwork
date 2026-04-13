import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import requests
import zipfile

# === CONFIGURATION ===
CONFIG = {
    'dataset': 'text8',
    'vocab_size': 27,
    'd_model': 256,
    'n_layers': 4,    
    'dropout': 0.1,
    'lr': 1e-3,            # Fast LR for quick results
    'batch_size': 16,       
    'seq_len': 128,        
    'epochs': 1,           
    'subset_size': 1000000 # 1M subset for speed
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== ABLATION STUDY ON {device} ===")

# === DATA LOADING ===
class CharCorpus:
    def __init__(self, path, subset_size=None):
        self.char2idx = {' ': 0}
        for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
            self.char2idx[char] = i + 1
        with open(path, 'r') as f: text = f.read()
        if subset_size: text = text[:subset_size]
        n = len(text)
        self.train = self.tokenize(text[:int(n*0.9)])
        self.valid = self.tokenize(text[int(n*0.9):])
    def tokenize(self, text):
        return torch.tensor([self.char2idx.get(c, 0) for c in text], dtype=torch.int64)

def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data.to(device), target.to(device)

# === MODEL VARIANTS ===

# 1. FULL INN (Standard)
class INN_Full(nn.Module):
    # ... (Same as INNv2JIT but simplified for readability here) ...
    # We will use the JIT components defined below
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(27, 256)
        self.layers = nn.ModuleList([nn.ModuleList([
            MultiMambaBlockJIT(16, 256),
            nn.MultiheadAttention(256, 4, batch_first=True)
        ]) for _ in range(4)])
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, 27)
    def forward(self, x):
        B, L = x.shape
        x = self.emb(x).unsqueeze(1).expand(-1, 16, -1, -1).contiguous()
        for mamba, attn in self.layers:
            x = x + mamba(x) # Memory
            # Communication
            x_flat = x.permute(0, 2, 1, 3).reshape(B*L, 16, -1)
            x_flat = x_flat + attn(x_flat, x_flat, x_flat)[0]
            x = x_flat.view(B, L, 16, -1).permute(0, 2, 1, 3)
        return self.head(self.norm(x.mean(dim=1)))

# 2. NO ATTENTION (Isolated Neurons)
class INN_NoAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(27, 256)
        self.layers = nn.ModuleList([
            MultiMambaBlockJIT(16, 256) # ONLY MAMBA, NO ATTN
            for _ in range(4)])
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, 27)
    def forward(self, x):
        B, L = x.shape
        x = self.emb(x).unsqueeze(1).expand(-1, 16, -1, -1).contiguous()
        for mamba in self.layers:
            x = x + mamba(x) # Memory Only, No Comm
        return self.head(self.norm(x.mean(dim=1)))

# 3. NO MAMBA (Attention Only - "Recurrent Transformer")
class INN_NoMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(27, 256)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(256, 4, batch_first=True) # ONLY ATTN
            for _ in range(4)])
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, 27)
    def forward(self, x):
        B, L = x.shape
        x = self.emb(x).unsqueeze(1).expand(-1, 16, -1, -1).contiguous()
        for attn in self.layers:
            # No Memory Block
            x_flat = x.permute(0, 2, 1, 3).reshape(B*L, 16, -1)
            x_flat = x_flat + attn(x_flat, x_flat, x_flat)[0]
            x = x_flat.view(B, L, 16, -1).permute(0, 2, 1, 3)
        return self.head(self.norm(x.mean(dim=1)))

# === SHARED COMPONENTS (JIT) ===
@torch.jit.script
def ssm_jit(x, dt, A, B, C, D):
    dt = torch.clamp(dt, max=2.5) 
    dA = torch.exp(torch.einsum('bld,ds->blds', dt, A))
    dB = dt.unsqueeze(-1) * B.unsqueeze(2)
    h = torch.zeros(x.size(0), x.size(2), A.size(1), device=x.device, dtype=x.dtype)
    ys = []
    for t in range(x.size(1)):
        h = dA[:, t, :, :] * h + dB[:, t, :, :] * x[:, t, :].unsqueeze(-1)
        y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)
    return y + x * D

class MultiMambaBlockJIT(nn.Module):
    def __init__(self, num_neurons, d_model):
        super().__init__()
        d_state, d_conv, expand = 16, 4, 2
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(num_neurons * self.d_inner, num_neurons * self.d_inner, bias=True, kernel_size=d_conv, groups=num_neurons * self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    def forward(self, x):
        B, N, L, D = x.shape
        x_and_res = self.in_proj(x)
        (x_in, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        x_conv = x_in.permute(0, 1, 3, 2).reshape(B, N*self.d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L].reshape(B, N, self.d_inner, L).permute(0, 1, 3, 2)
        x_conv = F.silu(x_conv)
        x_flat = x_conv.reshape(B*N, L, self.d_inner)
        dt_rank_state = self.x_proj(x_flat)
        dt, B_ssm, C_ssm = torch.split(dt_rank_state, [self.dt_rank, 16, 16], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        y = ssm_jit(x_flat, dt, A, B_ssm, C_ssm, self.D)
        y = y.reshape(B, N, L, self.d_inner)
        return self.out_proj(y * F.silu(res))

# === TRAINING LOOP ===
def train_variant(model_class, name, corpus):
    print(f"\n>>> Training Variant: {name}")
    model = model_class().to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    train_data = corpus.train.view(CONFIG['batch_size'], -1).t().contiguous().to(device)
    valid_data = corpus.valid.view(CONFIG['batch_size'], -1).t().contiguous().to(device)
    
    total_steps = (train_data.size(0) // CONFIG['seq_len'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CONFIG['lr'], total_steps=total_steps)
    crit = nn.CrossEntropyLoss()
    
    model.train()
    start = time.time()
    
    for batch, i in enumerate(range(0, train_data.size(0)-1, CONFIG['seq_len'])):
        x, y = get_batch(train_data, i, CONFIG['seq_len'])
        opt.zero_grad()
        loss = crit(model(x).view(-1, 27), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        
        if (batch+1) % 200 == 0:
            print(f"Batch {batch+1} | Loss: {loss.item():.3f}")
            
    model.eval()
    val_loss = 0
    tokens = 0
    with torch.no_grad():
        for i in range(0, valid_data.size(0)-1, CONFIG['seq_len']):
            x, y = get_batch(valid_data, i, CONFIG['seq_len'])
            val_loss += len(x) * x.size(1) * crit(model(x).view(-1, 27), y).item()
            tokens += len(x) * x.size(1)
            
    bpc = (val_loss / tokens) / math.log(2)
    print(f"-> Final Valid BPC: {bpc:.3f}")
    return bpc

if __name__ == "__main__":
    if not os.path.exists("data/text8"): # Download if needed
        print("Downloading text8...")
        # ... (assume download code similar to before) ...
    
    corpus = CharCorpus("data/text8", subset_size=CONFIG['subset_size'])
    
    results = {}
    results['INN (Full)'] = train_variant(INN_Full, "INN Full", corpus)
    results['No Attention'] = train_variant(INN_NoAttention, "No Attention (Isolated)", corpus)
    results['No Mamba'] = train_variant(INN_NoMamba, "No Mamba (Comm Only)", corpus)
    
    print("\n=== ABLATION RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v:.3f} BPC")

