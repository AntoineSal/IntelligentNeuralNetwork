import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import requests
import zipfile

# === CONFIGURATION FOR LONG RUN (20M) ===
CONFIG = {
    'dataset': 'text8',
    'vocab_size': 27,
    'd_model': 256,
    'n_layers': 4,    
    'dropout': 0.1,
    'lr': 3e-4,            # STABLE LR
    'batch_size': 8,       
    'seq_len': 128,        
    'epochs': 1,           
    'subset_size': 20000000, # 20 MILLION CHARS
    'grad_clip': 0.5,      # STABLE CLIP
    'save_every': 5000     # Save every 5000 batches
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== LAUNCHING LONG RUN ON {device} ===")
print(f"Config: {CONFIG}")

# === DATA LOADING ===
class CharCorpus:
    def __init__(self, path, subset_size=None):
        self.char2idx = {' ': 0}
        for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
            self.char2idx[char] = i + 1
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        
        with open(path, 'r') as f:
            text = f.read()
        
        if subset_size:
            text = text[:subset_size]
            print(f"Using subset: {len(text):,} chars")
        
        n = len(text)
        train_end = int(n * 0.95) # 95% train for more data
        val_end = int(n * 0.98)
        
        self.train = self.tokenize(text[:train_end])
        self.valid = self.tokenize(text[train_end:val_end])
        
        print(f"Train tokens: {len(self.train):,}")
        print(f"Valid tokens: {len(self.valid):,}")

    def tokenize(self, text):
        ids = [self.char2idx.get(c, 0) for c in text]
        return torch.tensor(ids, dtype=torch.int64)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

# === MODEL ARCHITECTURE (JIT Safe) ===
@torch.jit.script
def ssm_jit(x, dt, A, B, C, D):
    dt = torch.clamp(dt, max=2.5) # Aggressive stability clamp
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
    def __init__(self, num_neurons, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(num_neurons * self.d_inner, num_neurons * self.d_inner, bias=True, kernel_size=d_conv, groups=num_neurons * self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, L, D = x.shape
        x_and_res = self.in_proj(x)
        (x_in, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        x_conv = x_in.permute(0, 1, 3, 2).reshape(B, N*self.d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L].reshape(B, N, self.d_inner, L).permute(0, 1, 3, 2)
        x_conv = F.silu(x_conv)
        x_flat = x_conv.reshape(B*N, L, self.d_inner)
        dt_rank_state = self.x_proj(x_flat)
        dt, B_ssm, C_ssm = torch.split(dt_rank_state, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        y = ssm_jit(x_flat, dt, A, B_ssm, C_ssm, self.D)
        y = y.reshape(B, N, L, self.d_inner)
        return self.dropout(self.out_proj(y * F.silu(res)))

class INNv2JIT(nn.Module):
    def __init__(self, vocab_size, num_neurons, d_model, num_layers, dropout=0.1):
        super().__init__()
        self.num_neurons = num_neurons
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([nn.ModuleList([
            MultiMambaBlockJIT(num_neurons, d_model, dropout=dropout),
            nn.MultiheadAttention(d_model, 4, dropout=dropout, batch_first=True)
        ]) for _ in range(num_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.head.weight = self.embedding.weight

    def forward(self, x):
        B, L = x.shape
        x = self.embedding(x).unsqueeze(1).expand(-1, self.num_neurons, -1, -1).contiguous()
        for mamba, attn in self.layers:
            x = x + mamba(x)
            x_flat = x.permute(0, 2, 1, 3).reshape(B*L, self.num_neurons, -1)
            x_flat = x_flat + attn(x_flat, x_flat, x_flat)[0]
            x = x_flat.view(B, L, self.num_neurons, -1).permute(0, 2, 1, 3)
        return self.head(self.norm_f(x.mean(dim=1)))

# === MAIN ===
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    # Ensure data exists
    if not os.path.exists("data/text8"):
        print("Downloading text8...")
        os.makedirs("data", exist_ok=True)
        r = requests.get("http://mattmahoney.net/dc/text8.zip")
        with open("data/text8.zip", "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile("data/text8.zip", "r") as zip_ref:
            zip_ref.extractall("data")
            
    corpus = CharCorpus("data/text8", subset_size=CONFIG['subset_size'])
    
    model = INNv2JIT(CONFIG['vocab_size'], 16, CONFIG['d_model'], CONFIG['n_layers'], CONFIG['dropout']).to(device)
    print(f"INNv2 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    train_data = batchify(corpus.train, CONFIG['batch_size'])
    valid_data = batchify(corpus.valid, CONFIG['batch_size'])
    
    total_steps = (train_data.size(0) // CONFIG['seq_len']) * CONFIG['epochs']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CONFIG['lr'], total_steps=total_steps)
    crit = nn.CrossEntropyLoss()
    
    print(f"Total batches: {total_steps}")
    
    model.train()
    total_loss = 0
    start = time.time()
    tokens = 0
    
    try:
        for batch, i in enumerate(range(0, train_data.size(0)-1, CONFIG['seq_len'])):
            x, y = get_batch(train_data, i, CONFIG['seq_len'])
            loss = crit(model(x).view(-1, CONFIG['vocab_size']), y)
            
            if torch.isnan(loss):
                print(f"❌ NAN DETECTED at Batch {batch}! Stopping.")
                break
                
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            opt.step()
            sched.step()
            
            total_loss += loss.item() * len(x)
            tokens += len(x) * x.size(1)
            
            if (batch+1) % 100 == 0:
                elapsed = time.time() - start
                print(f"Batch {batch+1}/{total_steps} | Loss: {total_loss/tokens:.4f} | BPC: {(total_loss/tokens)/math.log(2):.3f} | Speed: {tokens/elapsed:.0f} tok/s")
            
            # Checkpoint
            if (batch+1) % CONFIG['save_every'] == 0:
                torch.save(model.state_dict(), f"models/inn_text8_20M_step{batch+1}.pth")
                print(f"💾 Checkpoint saved: models/inn_text8_20M_step{batch+1}.pth")
                
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    
    print("Training finished. Saving final model...")
    torch.save(model.state_dict(), "models/inn_text8_20M_final.pth")
    
    # Evaluation
    print("Running Final Evaluation...")
    model.eval()
    val_loss = 0
    val_tokens = 0
    with torch.no_grad():
        for i in range(0, valid_data.size(0)-1, CONFIG['seq_len']):
            x, y = get_batch(valid_data, i, CONFIG['seq_len'])
            val_loss += len(x) * x.size(1) * crit(model(x).view(-1, CONFIG['vocab_size']), y).item()
            val_tokens += len(x) * x.size(1)
    
    bpc = (val_loss / val_tokens) / math.log(2)
    print(f"\n=== FINAL 20M RUN RESULT ===")
    print(f"Valid Loss: {val_loss/val_tokens:.4f}")
    print(f"Valid BPC: {bpc:.4f}")

