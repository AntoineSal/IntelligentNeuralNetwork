import torch
import torch.nn as nn
import math
import time
import os
import requests
import zipfile
import argparse

# === CONFIGURATION ===
CONFIG = {
    'dataset': 'text8',
    'vocab_size': 27,
    'seq_len': 256,
    'batch_size': 16,   # Same as INN
    'epochs': 1,
    'lr': 4e-4,         # Same as INN
    'grad_clip': 1.0,
    'subset_size': 100000000, # Full dataset
    'd_model': 256,     # Align width
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DATA LOADING (Shared with INN) ===
class Text8Corpus:
    def __init__(self, path, subset_size=None):
        if not os.path.exists(path):
            self.download(path)
        print("Loading text8...")
        with open(path, 'r') as f: data = f.read()
        if subset_size: data = data[:subset_size]
        print(f"Data loaded. Length: {len(data):,}")
        
        chars = sorted(list(set(data)))
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

# === BASELINE MODELS ===

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 3 layers of 512 hidden units gives approx 4M params
        self.lstm = nn.LSTM(d_model, 512, num_layers=3, dropout=0.1) 
        self.head = nn.Linear(512, vocab_size)
        
        # Tie weights? Usually not for LSTM hidden size != emb size
        # But to keep it fair with INN, we rely on standard LSTM practice
        
    def forward(self, x):
        # x: (L, B) for LSTM default
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        return self.head(out)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(5000, d_model)) # Simple pos enc
        
        # Same depth/width as INN (6 layers, 256 dim) -> Approx 4M params
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=4*d_model, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.head.bias.data.zero_()
        self.head.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # x: (L, B)
        # Transformer expects (L, B, D) by default or batch_first=False
        src = self.embedding(x) * math.sqrt(self.d_model)
        src = src + self.pos_encoder[:src.size(0), :].unsqueeze(1)
        
        mask = nn.Transformer.generate_square_subsequent_mask(src.size(0)).to(device)
        
        output = self.transformer(src, mask=mask)
        return self.head(output)

# === MAIN LOOP ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'transformer'])
    args, _ = parser.parse_known_args()
    
    print(f"=== BENCHMARKING {args.model.upper()} ON TEXT8 ===")
    
    corpus = Text8Corpus("data/text8", subset_size=CONFIG['subset_size'])
    train_data = batchify(corpus.train, CONFIG['batch_size'])
    valid_data = batchify(corpus.valid, CONFIG['batch_size'])
    
    if args.model == 'lstm':
        model = LSTMModel(CONFIG['vocab_size'], CONFIG['d_model'], n_layers=3).to(device)
    else:
        model = TransformerModel(CONFIG['vocab_size'], CONFIG['d_model'], n_layers=6).to(device)
        
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    batches_per_epoch = len(range(0, train_data.size(0)-1, CONFIG['seq_len']))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CONFIG['lr'], total_steps=batches_per_epoch)
    crit = nn.CrossEntropyLoss()
    
    model.train()
    total_loss = 0
    start = time.time()
    tokens = 0
    
    try:
        for batch, i in enumerate(range(0, train_data.size(0)-1, CONFIG['seq_len'])):
            # Data is (B, L) on CPU -> get_batch -> (L, B) on GPU
            x, y = get_batch(train_data, i, CONFIG['seq_len'])
            
            opt.zero_grad()
            logits = model(x) # (L, B, Vocab)
            loss = crit(logits.view(-1, CONFIG['vocab_size']), y.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            opt.step()
            sched.step()
            
            total_loss += loss.item() * x.numel()
            tokens += x.numel()
            
            if (batch+1) % 100 == 0:
                elapsed = time.time() - start
                print(f"Batch {batch+1} | Loss: {loss.item():.4f} | BPC: {loss.item()/math.log(2):.3f} | Speed: {tokens/elapsed:.0f} tok/s")
                start = time.time()
                tokens = 0
                
    except KeyboardInterrupt:
        print("Stop.")
        
    print("Evaluating...")
    model.eval()
    val_loss = 0
    val_tokens = 0
    with torch.no_grad():
        for i in range(0, valid_data.size(0)-1, CONFIG['seq_len']):
            x, y = get_batch(valid_data, i, CONFIG['seq_len'])
            logits = model(x)
            loss = crit(logits.view(-1, CONFIG['vocab_size']), y.view(-1))
            val_loss += loss.item() * x.numel()
            val_tokens += x.numel()
            
    print(f"Valid BPC: {(val_loss/val_tokens)/math.log(2):.4f}")

