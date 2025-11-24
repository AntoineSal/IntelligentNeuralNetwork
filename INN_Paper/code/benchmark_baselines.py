import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import requests

# === CONFIGURATION ===
CONFIG = {
    'vocab_size': 10000,
    'd_model': 256,
    'n_layers': 4,    
    'n_head': 4,
    'd_hid': 1024,     # Standard FFN ratio
    'lstm_layers': 2, 
    'dropout': 0.1,
    'lr': 3e-4,       # Standard Adam LR
    'batch_size': 16,
    'seq_len': 64,
    'epochs': 20
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DATA LOADING ===
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
        self.fc.weight = self.embedding.weight # Tying
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        return self.fc(out)

# STANDARD TRANSFORMER (FIXED INIT)
class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_hid, n_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Standard PyTorch Transformer Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        
        self.fc.weight = self.embedding.weight # Tying
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # Mask for causality (crucial for LM)
        mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        
        src = self.embedding(src) * math.sqrt(self.d_model)
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
def run_benchmark(model_class, name, **kwargs):
    print(f"\n>>> STARTING BENCHMARK: {name}")
    model = model_class(CONFIG['vocab_size'], CONFIG['d_model'], **kwargs).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model Params: {params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG['lr'], 
        steps_per_epoch=len(corpus.train) // (CONFIG['batch_size']*CONFIG['seq_len']), 
        epochs=CONFIG['epochs']
    )
    criterion = nn.CrossEntropyLoss()
    
    train_data = batchify(corpus.train, CONFIG['batch_size'])
    val_data = batchify(corpus.valid, CONFIG['batch_size'])
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch, i in enumerate(range(0, train_data.size(0) - 1, CONFIG['seq_len'])):
            data, targets = get_batch(train_data, i, CONFIG['seq_len'])
            output = model(data)
            loss = criterion(output.reshape(-1, CONFIG['vocab_size']), targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, val_data.size(0) - 1, CONFIG['seq_len']):
                data, targets = get_batch(val_data, i, CONFIG['seq_len'])
                output = model(data)
                val_loss += len(data) * criterion(output.reshape(-1, CONFIG['vocab_size']), targets).item()
        val_loss /= (len(val_data) - 1)
        
        train_ppl = math.exp(total_loss / (len(train_data) // CONFIG['seq_len']))
        valid_ppl = math.exp(val_loss)
        
        print(f"| Epoch {epoch:2d} | Train PPL: {train_ppl:6.2f} | Valid PPL: {valid_ppl:6.2f} | Time: {time.time() - start_time:.0f}s")

if __name__ == "__main__":
    download_ptb()
    corpus = Corpus('data/ptb')
    
    # 1. LSTM (6 layers -> ~5.6M Params)
    # run_benchmark(LSTMBaseline, "LSTM Baseline", n_layers=6)
    
    # 2. Transformer (4 layers, 1024 FFN -> ~5.8M Params)
    run_benchmark(TransformerBaseline, "Transformer Baseline (FIXED)", n_layers=4, n_head=4, d_hid=1024)
