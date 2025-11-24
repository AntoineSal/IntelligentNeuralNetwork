import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import requests
import zipfile
import numpy as np

# === CONFIGURATION ===
CONFIG = {
    'dataset': 'text8',
    'vocab_size': 27, # a-z + space
    'd_model': 256,
    'n_layers': 4,    
    'n_head': 4,
    'd_hid': 1024,     
    'lstm_layers': 2, 
    'dropout': 0.1,
    'lr': 1e-3,       # Higher LR for char-level usually works
    'batch_size': 32, # Reduced to prevent OOM with 16 neurons
    'seq_len': 128,   # Reduced context for stability
    'epochs': 2       # Text8 is huge, 2 epochs is enough for trend
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === DATA LOADING (Text8) ===
class CharCorpus:
    def __init__(self, path):
        self.char2idx = {' ': 0}
        for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
            self.char2idx[char] = i + 1
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        
        with open(path, 'r') as f:
            text = f.read()
        
        # Split 90/5/5
        n = len(text)
        train_end = int(n * 0.9)
        val_end = int(n * 0.95)
        
        self.train = self.tokenize(text[:train_end])
        self.valid = self.tokenize(text[train_end:val_end])
        self.test = self.tokenize(text[val_end:])
        
        print(f"Train tokens: {len(self.train):,}")
        print(f"Valid tokens: {len(self.valid):,}")

    def tokenize(self, text):
        ids = [self.char2idx.get(c, 0) for c in text] # 0 for unknown/space
        return torch.tensor(ids, dtype=torch.int64)

def download_text8():
    url = "http://mattmahoney.net/dc/text8.zip"
    if not os.path.exists("data/text8"):
        os.makedirs("data", exist_ok=True)
        print("Downloading text8...")
        r = requests.get(url)
        with open("data/text8.zip", "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile("data/text8.zip", "r") as zip_ref:
            zip_ref.extractall("data")
            
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

# === MODELS ===

# 1. INNv2 (Our Hero)
class MultiMambaBlock(nn.Module):
    def __init__(self, num_neurons, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(num_neurons * self.d_inner, num_neurons * self.d_inner, bias=True, kernel_size=d_conv, groups=num_neurons * self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.d_state = d_state

    def forward(self, x): # x: (B, N, L, D)
        B, N, L, D = x.shape
        x_and_res = self.in_proj(x)
        (x_in, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        x_conv = self.conv1d(x_in.permute(0, 1, 3, 2).reshape(B, N*self.d_inner, L))[:, :, :L].reshape(B, N, self.d_inner, L).permute(0, 1, 3, 2)
        y = self.ssm(F.silu(x_conv))
        return self.dropout(self.out_proj(y * F.silu(res)))

    def ssm(self, x):
        x_flat = x.reshape(-1, x.size(2), x.size(3))
        dt_rank_state = self.x_proj(x_flat)
        dt, B, C = torch.split(dt_rank_state, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        y = []
        h = torch.zeros(x_flat.size(0), self.d_inner, self.d_state, device=x.device)
        dA = torch.exp(torch.einsum('bld,ds->blds', dt, A))
        dB = torch.einsum('bld,bls->blds', dt, B)
        for t in range(x.size(2)):
            h = dA[:, t, :, :] * h + dB[:, t, :, :] * x_flat[:, t, :].unsqueeze(-1)
            y.append(torch.einsum('bds,bs->bd', h, C[:, t, :]))
        return (torch.stack(y, dim=1) + x_flat * self.D).reshape(x.shape)

class INNv2(nn.Module):
    def __init__(self, vocab_size, num_neurons, d_model, num_layers, dropout=0.1):
        super().__init__()
        self.num_neurons = num_neurons
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([nn.ModuleList([
            MultiMambaBlock(num_neurons, d_model, dropout=dropout),
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

# 2. LSTM Baseline
class LSTMBaseline(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)
        self.fc.weight = self.embedding.weight

    def forward(self, x):
        return self.fc(self.lstm(self.embedding(x))[0])

# 3. Transformer Baseline
class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_hid, n_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 5000, d_model))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_head, d_hid, dropout, batch_first=True), n_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.fc.weight = self.embedding.weight

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model) + self.pos_encoder[:, :x.size(1)]
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        return self.fc(self.transformer(x, mask, is_causal=True))

# === BENCHMARK ENGINE ===
def train(model, name, corpus):
    print(f"\n>>> Training {name} ({sum(p.numel() for p in model.parameters()):,} params)")
    torch.cuda.empty_cache() # Clear memory before start
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CONFIG['lr'], steps_per_epoch=len(corpus.train)//(CONFIG['batch_size']*CONFIG['seq_len']), epochs=CONFIG['epochs'])
    crit = nn.CrossEntropyLoss()
    
    train_data = batchify(corpus.train, CONFIG['batch_size'])
    valid_data = batchify(corpus.valid, CONFIG['batch_size'])
    
    history = []
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        start = time.time()
        for batch, i in enumerate(range(0, train_data.size(0)-1, CONFIG['seq_len'])):
            x, y = get_batch(train_data, i, CONFIG['seq_len'])
            loss = crit(model(x).view(-1, CONFIG['vocab_size']), y)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, valid_data.size(0)-1, CONFIG['seq_len']):
                x, y = get_batch(valid_data, i, CONFIG['seq_len'])
                val_loss += len(x) * crit(model(x).view(-1, CONFIG['vocab_size']), y).item()
        val_loss /= (len(valid_data)-1)
        
        bpc = val_loss / math.log(2) # Bits Per Character
        print(f"Epoch {epoch+1} | Loss: {val_loss:.3f} | BPC: {bpc:.3f} | Time: {time.time()-start:.0f}s")
        history.append(bpc)
    return history[-1]

if __name__ == "__main__":
    download_text8()
    corpus = CharCorpus("data/text8")
    
    print("\n=== BENCHMARK RESULTS (Bits Per Character) ===")
    # INNv2
    inn = INNv2(CONFIG['vocab_size'], 16, CONFIG['d_model'], CONFIG['n_layers'], CONFIG['dropout']).to(device)
    inn_res = train(inn, "INNv2", corpus)
    
    # LSTM
    lstm = LSTMBaseline(CONFIG['vocab_size'], CONFIG['d_model'], 6, CONFIG['dropout']).to(device) # 6 layers to match params
    lstm_res = train(lstm, "LSTM", corpus)
    
    # Transformer
    tf = TransformerBaseline(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_head'], CONFIG['d_hid'], CONFIG['n_layers'], CONFIG['dropout']).to(device)
    tf_res = train(tf, "Transformer", corpus)
    
    print("\n--- FINAL LEADERBOARD ---")
    print(f"INNv2:       {inn_res:.3f} BPC")
    print(f"LSTM:        {lstm_res:.3f} BPC")
    print(f"Transformer: {tf_res:.3f} BPC")

