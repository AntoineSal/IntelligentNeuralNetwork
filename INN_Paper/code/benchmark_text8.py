import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import requests
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURATION ===
CONFIG = {
    'dataset': 'text8',
    'vocab_size': 27,  # a-z + space
    'd_model': 256,
    'n_layers': 4,    
    'n_head': 4,
    'd_hid': 1024,     
    'dropout': 0.1,
    'lr': 1e-3,
    'batch_size': 8,       # SAFE BATCH SIZE
    'seq_len': 128,        # SAFE SEQ LEN
    'epochs': 1,           
    'subset_size': 5000000 
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === DATA LOADING (Text8) ===
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
        train_end = int(n * 0.9)
        val_end = int(n * 0.95)
        
        self.train = self.tokenize(text[:train_end])
        self.valid = self.tokenize(text[train_end:val_end])
        self.test = self.tokenize(text[val_end:])
        
        print(f"Train tokens: {len(self.train):,}")
        print(f"Valid tokens: {len(self.valid):,}")

    def tokenize(self, text):
        ids = [self.char2idx.get(c, 0) for c in text]
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

# === JIT OPTIMIZED SSM ===
@torch.jit.script
def ssm_jit(x, dt, A, B, C, D):
    dt = torch.clamp(dt, max=4.0) # Clamp for stability
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

# === BASELINES ===
class LSTMBaseline(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)
        self.fc.weight = self.embedding.weight
    def forward(self, x):
        return self.fc(self.lstm(self.embedding(x))[0])

class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_hid, n_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 5000, d_model))
        nn.init.normal_(self.pos_encoder, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_hid, dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.fc.weight = self.embedding.weight
    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model) + self.pos_encoder[:, :x.size(1)]
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        return self.fc(self.transformer(x, mask, is_causal=True))

# === TRAINING LOOP ===
def train(model, name, corpus):
    print(f"\n>>> Training {name} ({sum(p.numel() for p in model.parameters()):,} params)")
    torch.cuda.empty_cache()
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    
    train_data = batchify(corpus.train, CONFIG['batch_size'])
    valid_data = batchify(corpus.valid, CONFIG['batch_size'])
    
    total_steps = (train_data.size(0) // CONFIG['seq_len']) * CONFIG['epochs']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CONFIG['lr'], total_steps=total_steps)
    crit = nn.CrossEntropyLoss()
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        start = time.time()
        tokens = 0
        
        for batch, i in enumerate(range(0, train_data.size(0)-1, CONFIG['seq_len'])):
            x, y = get_batch(train_data, i, CONFIG['seq_len'])
            loss = crit(model(x).view(-1, CONFIG['vocab_size']), y)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            if sched.last_epoch < total_steps:
                sched.step()
            
            total_loss += loss.item() * len(x)
            tokens += len(x) * x.size(1)
            
            if (batch+1) % 100 == 0:
                print(f"Batch {batch+1} | Loss: {total_loss/tokens:.3f} | Speed: {tokens/(time.time()-start):.0f} tok/s")
        
        model.eval()
        val_loss = 0
        val_tokens = 0
        with torch.no_grad():
            for i in range(0, valid_data.size(0)-1, CONFIG['seq_len']):
                x, y = get_batch(valid_data, i, CONFIG['seq_len'])
                val_loss += len(x) * x.size(1) * crit(model(x).view(-1, CONFIG['vocab_size']), y).item()
                val_tokens += len(x) * x.size(1)
        
        bpc = (val_loss / val_tokens) / math.log(2)
        print(f"Epoch {epoch+1} Valid BPC: {bpc:.3f}")
        return bpc

# === VISUALIZATION ===
def visualize_inn(model, corpus, text_sample="the quick brown fox jumps over the lazy dog"):
    print("\n🧠 Generating Neuron Activity Heatmap...")
    model.eval()
    ids = corpus.tokenize(text_sample).to(device).unsqueeze(0)
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            if len(output.shape) == 4:
                act = output.norm(dim=-1).squeeze(0).detach().cpu().numpy()
                activations[name] = act
        return hook
    
    handle = model.layers[-1][0].register_forward_hook(get_activation("mamba_last"))
    with torch.no_grad():
        model(ids)
    handle.remove()
    
    plt.figure(figsize=(15, 6))
    sns.heatmap(activations["mamba_last"], cmap="magma", 
                xticklabels=list(text_sample), 
                yticklabels=[f"N{i}" for i in range(model.num_neurons)])
    plt.title(f"INNv2 Neuron Specialization (Character Level)\nInput: '{text_sample}'")
    plt.tight_layout()
    plt.savefig("inn_text8_heatmap.png")
    print("✓ Saved inn_text8_heatmap.png")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    download_text8()
    corpus = CharCorpus("data/text8", subset_size=CONFIG['subset_size'])
    
    results = {}
    
    # 1. INNv2
    inn = INNv2JIT(CONFIG['vocab_size'], 16, CONFIG['d_model'], CONFIG['n_layers'], CONFIG['dropout']).to(device)
    results['INNv2'] = train(inn, "INNv2 (JIT Safe)", corpus)
    torch.save(inn.state_dict(), "models/inn_text8.pth")
    visualize_inn(inn, corpus) # Visualize BEFORE deleting
    del inn
    
    # 2. LSTM
    lstm = LSTMBaseline(CONFIG['vocab_size'], CONFIG['d_model'], 6, CONFIG['dropout']).to(device)
    results['LSTM'] = train(lstm, "LSTM Baseline", corpus)
    torch.save(lstm.state_dict(), "models/lstm_text8.pth")
    del lstm
    
    # 3. Transformer
    tf = TransformerBaseline(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_head'], CONFIG['d_hid'], CONFIG['n_layers'], CONFIG['dropout']).to(device)
    results['Transformer'] = train(tf, "Transformer Baseline", corpus)
    torch.save(tf.state_dict(), "models/transformer_text8.pth")
    del tf
    
    print("\n=== FINAL LEADERBOARD (BPC) ===")
    for k, v in sorted(results.items(), key=lambda item: item[1]):
        print(f"{k}: {v:.3f}")
