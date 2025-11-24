import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import requests

# === CONFIGURATION TO MATCH INNv2 (5.5M Params) ===
# INNv2: d_model=256, n_layers=4, vocab=10000
CONFIG = {
    'vocab_size': 10000,
    'd_model': 256,
    'n_layers': 4,    # Transformer layers
    'n_head': 4,
    'd_hid': 512,     # Transformer FFN dim
    'lstm_layers': 2, # LSTM needs fewer layers to match params usually
    'dropout': 0.1,
    'lr': 3e-4,
    'batch_size': 16,
    'seq_len': 64,
    'epochs': 20
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DATA LOADING (Same as INN) ===
class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    def __len__(self): return len(self.idx2word)

class Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
    def tokenize(self, path):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words: self.dictionary.add_word(word)
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = [self.dictionary.word2idx[w] for w in words]
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        return ids

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

def download_ptb():
    base_url = "https://raw.githubusercontent.com/tomsercu/lstm/master/data"
    files = ["ptb.train.txt", "ptb.valid.txt"]
    os.makedirs("data/ptb", exist_ok=True)
    for filename in files:
        path = f"data/ptb/{filename}"
        if not os.path.exists(path):
            r = requests.get(f"{base_url}/{filename}")
            with open(path, 'wb') as f: f.write(r.content)

# === BASELINE MODELS ===

class LSTMBaseline(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Weight tying
        self.fc.weight = self.embedding.weight
        
    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        return self.fc(out)

class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_hid, n_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        
        # Weight tying
        self.fc.weight = self.embedding.weight

    def forward(self, src):
        mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        src = self.embedding(src) * math.sqrt(CONFIG['d_model'])
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask, is_causal=True)
        return self.fc(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# === TRAINING LOOP ===
def train_model(model, name, corpus):
    print(f"\n=== Training {name} Baseline ===")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()
    train_data = batchify(corpus.train, CONFIG['batch_size'])
    val_data = batchify(corpus.valid, CONFIG['batch_size'])
    
    model.train()
    for epoch in range(1, CONFIG['epochs'] + 1):
        total_loss = 0
        start_time = time.time()
        
        for batch, i in enumerate(range(0, train_data.size(0) - 1, CONFIG['seq_len'])):
            data, targets = get_batch(train_data, i, CONFIG['seq_len'])
            output = model(data)
            loss = criterion(output.reshape(-1, CONFIG['vocab_size']), targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / (len(train_data) // CONFIG['seq_len'])
        print(f"Epoch {epoch} | Loss: {avg_loss:.2f} | PPL: {math.exp(avg_loss):.2f}")

if __name__ == "__main__":
    download_ptb()
    corpus = Corpus('data/ptb')
    
    # Run LSTM
    lstm = LSTMBaseline(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['lstm_layers']).to(device)
    train_model(lstm, "LSTM", corpus)
    
    # Run Transformer
    transformer = TransformerBaseline(
        CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_head'], 
        CONFIG['d_hid'], CONFIG['n_layers']
    ).to(device)
    train_model(transformer, "Transformer", corpus)

