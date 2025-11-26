import argparse
import torch
import torch.nn as nn
import math
import time
import sys
import os
import requests
import zipfile
from datasets import load_dataset

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Error: mamba-ssm is required for this benchmark.")
    sys.exit(1)

# === ARGS ===
parser = argparse.ArgumentParser(description='Mamba Baseline Benchmark (Text8)')
parser.add_argument('--subset_size', type=int, default=20_000_000, help='Number of characters to train on')
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--n_layers', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seq_len', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DATA LOADING ===
class Text8Subset:
    def __init__(self, subset_size):
        print("Downloading Text8...")
        if not os.path.exists("text8.zip"):
            r = requests.get("http://mattmahoney.net/dc/text8.zip")
            with open("text8.zip", "wb") as f: f.write(r.content)
            with zipfile.ZipFile("text8.zip", "r") as z: z.extractall(".")
        
        print(f"Loading first {subset_size:,} characters...")
        with open("text8", "r") as f:
            text = f.read()[:subset_size]
        
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        
        self.data = torch.tensor([self.char2idx[ch] for ch in text], dtype=torch.long)
        split = int(len(self.data) * 0.9)
        self.train = self.data[:split]
        self.valid = self.data[split:]
        print(f"Vocab: {self.vocab_size} | Train: {len(self.train):,} | Valid: {len(self.valid):,}")

# === MODEL ===
class MambaBaseline(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = x + layer(x) # Residual connection is standard in Mamba stacks
        return self.head(self.norm_f(x))

# === TRAIN ===
def run():
    corpus = Text8Subset(args.subset_size)
    
    # Batchify
    def get_batches(data):
        n_batch = len(data) // args.batch_size
        data = data[:n_batch * args.batch_size]
        return data.view(args.batch_size, -1)
    
    train_data = get_batches(corpus.train).to(device)
    valid_data = get_batches(corpus.valid).to(device)
    
    model = MambaBaseline(corpus.vocab_size, args.d_model, args.n_layers).to(device)
    print(f"Mamba Params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    start_time = time.time()
    steps_per_epoch = train_data.size(1) // args.seq_len
    
    print(f"Training for 1 epoch ({steps_per_epoch} steps)...")
    
    for i in range(steps_per_epoch):
        x = train_data[:, i*args.seq_len:(i+1)*args.seq_len]
        y = train_data[:, i*args.seq_len+1:(i+1)*args.seq_len+1]
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.reshape(-1, corpus.vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0 and i > 0:
            bpc = loss.item() / math.log(2)
            print(f"Step {i} | Loss: {loss.item():.4f} | BPC: {bpc:.3f}")

    # Evaluate
    model.eval()
    total_loss = 0
    steps = valid_data.size(1) // args.seq_len
    with torch.no_grad():
        for i in range(steps):
            x = valid_data[:, i*args.seq_len:(i+1)*args.seq_len]
            y = valid_data[:, i*args.seq_len+1:(i+1)*args.seq_len+1]
            loss = criterion(model(x).reshape(-1, corpus.vocab_size), y.reshape(-1))
            total_loss += loss.item()
            
    final_bpc = (total_loss / steps) / math.log(2)
    print(f"=== Final Mamba Baseline BPC: {final_bpc:.4f} ===")

if __name__ == "__main__":
    run()

