import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import math
import requests
from torch.amp import autocast, GradScaler

# Utilisation de la configuration EXACTE de INNv4 pour valider la boucle
CONFIG = {
    'batch_size': 8,        
    'accum_steps': 4,       
    'seq_len': 64,          
    'epochs': 2,            # Court pour le test
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'd_model': 256,         
    'learning_rate': 5e-4,
    'warmup_steps': 200
}

# --- DATA LOADING (Identique) ---
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
        self.train = self.tokenize(os.path.join(path, 'train.txt'), build_vocab=True)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
    
    def tokenize(self, path, build_vocab=False):
        if not os.path.exists(path): return torch.LongTensor([])
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                if build_vocab:
                    for word in words: self.dictionary.add_word(word)
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if word in self.dictionary.word2idx:
                        ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids

def batchify(data, bsz, device):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

# --- DUMMY MODEL (Standard Transformer) ---
class DummyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 512, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        src = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
        mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        output = self.transformer(src, mask=mask, is_causal=True)
        return self.decoder(output)

def main():
    print("=== SANITY CHECK: PIPELINE VALIDATION ===")
    
    data_path = "data/wikitext-2"
    if not os.path.exists(f"{data_path}/train.txt"):
        print("Downloading WikiText-2...")
        os.makedirs(data_path, exist_ok=True)
        base_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/"
        for f in ['train.txt', 'valid.txt', 'test.txt']:
            r = requests.get(base_url + f)
            with open(f"{data_path}/{f}", 'w') as out: out.write(r.text)
            
    corpus = Corpus(data_path)
    vocab_size = len(corpus.dictionary)
    print(f"Vocab Size: {vocab_size}")
    
    device = torch.device(CONFIG['device'])
    train_data = batchify(corpus.train, CONFIG['batch_size'], device)
    
    # TEST MODEL
    model = DummyTransformer(vocab_size, CONFIG['d_model']).to(device)
    print(f"Dummy Model Params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    
    # Train Loop (Exact copy of INNv4 loop)
    model.train()
    total_loss = 0
    total_tokens = 0
    
    print("Starting training loop check...")
    
    for batch, i in enumerate(range(0, train_data.size(0)-1, CONFIG['seq_len'])):
        if batch > 200: break # On teste juste 200 batchs
        
        data, targets = get_batch(train_data, i, CONFIG['seq_len'])
        data = data.t()
        
        with autocast('cuda'):
            logits = model(data)
            loss = criterion(logits.reshape(-1, vocab_size), targets)
            loss = loss / CONFIG['accum_steps']
        
        scaler.scale(loss).backward()
        
        if (batch + 1) % CONFIG['accum_steps'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        n_tokens = data.size(0) * data.size(1)
        total_loss += (loss.item() * CONFIG['accum_steps']) * n_tokens
        total_tokens += n_tokens
        
        if (batch + 1) % 20 == 0:
            cur_loss = total_loss / total_tokens
            print(f"| batch {batch} | loss {cur_loss:.2f} | ppl {math.exp(cur_loss):.2f}")
            total_loss = 0
            total_tokens = 0

if __name__ == "__main__":
    main()

