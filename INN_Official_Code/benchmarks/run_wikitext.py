import argparse
import torch
import torch.nn as nn
import math
import time
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import INN
from datasets import load_dataset

# === ARGS ===
parser = argparse.ArgumentParser(description='INN WikiText-2 Benchmark')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seq_len', type=int, default=128)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--save_path', type=str, default='best_inn_wikitext.pth')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DATA LOADING ===
def get_data():
    print("Loading WikiText-2 (Hugging Face)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    train_text = "\n".join(dataset['train']['text'])
    valid_text = "\n".join(dataset['validation']['text'])
    test_text = "\n".join(dataset['test']['text'])
    
    chars = sorted(list(set(train_text + valid_text + test_text)))
    vocab_size = len(chars)
    char2idx = {ch: i for i, ch in enumerate(chars)}
    
    def numericalize(text):
        return torch.tensor([char2idx[ch] for ch in text], dtype=torch.long)
        
    train_data = numericalize(train_text)
    valid_data = numericalize(valid_text)
    
    print(f"Vocab: {vocab_size} | Train Tokens: {len(train_data):,}")
    return train_data, valid_data, vocab_size

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

# === TRAIN LOOP ===
def train():
    train_raw, valid_raw, vocab_size = get_data()
    train_data = batchify(train_raw, args.batch_size)
    val_data = batchify(valid_raw, args.batch_size)
    
    model = INN(vocab_size, dropout=args.dropout).to(device)
    print(f"INN Params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, 
        total_steps=(len(train_data)//args.seq_len + 1) * args.epochs
    )
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.seq_len)):
            data, targets = get_batch(train_data, i, args.seq_len)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
            if batch % 200 == 0 and batch > 0:
                cur_loss = total_loss / 200
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
                loss = criterion(output.reshape(-1, vocab_size), targets.reshape(-1))
                val_loss += loss.item() * len(data)
                total_len += len(data)
        
        final_val_loss = val_loss / total_len
        print(f"=== End Epoch {epoch} | Valid Loss {final_val_loss:.4f} | Valid PPL {math.exp(final_val_loss):.2f} ===")
        
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved model to {args.save_path}")

if __name__ == "__main__":
    train()

