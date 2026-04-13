import argparse
import torch
import torch.nn as nn
import math
import time
import sys
import os
import requests

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import INN

# === ARGS ===
parser = argparse.ArgumentParser(description='INN PTB Benchmark (Word-Level)')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seq_len', type=int, default=64)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--dropout', type=float, default=0.4) # Higher dropout for word-level
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DATA LOADING (PTB Word Level) ===
class PTBCorpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        if not os.path.exists(path):
             os.makedirs(path, exist_ok=True)
             self.download(path)
             
        self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
        print(f"Vocab: {len(self.dictionary)} | Train Words: {len(self.train):,}")

    def download(self, path):
        print("Downloading PTB...")
        base_url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/"
        for file in ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']:
            r = requests.get(base_url + file)
            with open(os.path.join(path, file), "wb") as f: f.write(r.content)

    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, 'r', encoding="utf-8") as f:
            ids = []
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)
                    ids.append(self.dictionary.word2idx[word])
            return torch.tensor(ids, dtype=torch.long)

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

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
    corpus = PTBCorpus("data/ptb")
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, args.batch_size)
    
    model = INN(len(corpus.dictionary), dropout=args.dropout).to(device)
    print(f"INN Params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, 
        total_steps=(len(train_data)//args.seq_len + 1) * args.epochs
    )
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.seq_len)):
            data, targets = get_batch(train_data, i, args.seq_len)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.reshape(-1, len(corpus.dictionary)), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if batch % 100 == 0 and batch > 0:
                cur_loss = total_loss / 100
                print(f"| Epoch {epoch} | Batch {batch} | Loss {cur_loss:.4f} | PPL {math.exp(cur_loss):.2f}")
                total_loss = 0

        # Validation
        model.eval()
        val_loss = 0
        total_len = 0
        with torch.no_grad():
            for i in range(0, val_data.size(0) - 1, args.seq_len):
                data, targets = get_batch(val_data, i, args.seq_len)
                output = model(data)
                loss = criterion(output.reshape(-1, len(corpus.dictionary)), targets.reshape(-1))
                val_loss += loss.item() * len(data)
                total_len += len(data)
        
        final_loss = val_loss / total_len
        print(f"=== End Epoch {epoch} | Valid PPL {math.exp(final_loss):.2f} ===")

if __name__ == "__main__":
    train()

