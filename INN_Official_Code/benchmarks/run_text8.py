import argparse
import torch
import torch.nn as nn
import math
import time
import sys
import os
import requests
import zipfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import INN

# === ARGS ===
parser = argparse.ArgumentParser(description='INN Text8 Benchmark')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seq_len', type=int, default=512) # Longer sequence for Text8
parser.add_argument('--epochs', type=int, default=1)    # Text8 is huge, 1 epoch is standard
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--subset', type=int, default=100_000_000) # Full dataset
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DATA LOADING ===
class Text8Corpus:
    def __init__(self, path, subset_size=None):
        if not os.path.exists(path):
            self.download(path)
        print("Loading text8...")
        with open(path, 'r') as f: data = f.read()
        if subset_size: data = data[:subset_size]
        
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        
        print(f"Vocab: {self.vocab_size} | Total Len: {len(data):,}")
        self.data = torch.tensor([self.char2idx[ch] for ch in data], dtype=torch.long)
        
        n = len(self.data)
        self.train = self.data[:int(n*0.9)]
        self.valid = self.data[int(n*0.9):]

    def download(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = "http://mattmahoney.net/dc/text8.zip"
        print("Downloading Text8...")
        r = requests.get(url)
        with open("text8.zip", "wb") as f: f.write(r.content)
        with zipfile.ZipFile("text8.zip", "r") as z: z.extractall(os.path.dirname(path))

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    return data, target

# === TRAIN ===
def train():
    corpus = Text8Corpus("data/text8", args.subset)
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, args.batch_size)
    
    model = INN(corpus.vocab_size, dropout=args.dropout).to(device)
    print(f"INN Params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, 
        total_steps=len(train_data) // args.seq_len
    )
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    total_loss = 0
    
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.seq_len)):
        data, targets = get_batch(train_data, i, args.seq_len)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.reshape(-1, corpus.vocab_size), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if batch % 100 == 0 and batch > 0:
            cur_loss = total_loss / 100
            bpc = cur_loss / math.log(2)
            print(f"Batch {batch} | Loss {cur_loss:.4f} | BPC {bpc:.4f}")
            total_loss = 0

    # Final Validation
    model.eval()
    val_loss = 0
    total_len = 0
    with torch.no_grad():
        for i in range(0, val_data.size(0) - 1, args.seq_len):
            data, targets = get_batch(val_data, i, args.seq_len)
            output = model(data)
            loss = criterion(output.reshape(-1, corpus.vocab_size), targets.reshape(-1))
            val_loss += loss.item() * len(data)
            total_len += len(data)
            
    final_bpc = (val_loss / total_len) / math.log(2)
    print(f"=== Final Valid BPC: {final_bpc:.4f} ===")

if __name__ == "__main__":
    train()

