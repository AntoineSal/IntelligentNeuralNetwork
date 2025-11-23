import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import math
import requests
from src.model import INNv3
from torch.cuda.amp import autocast, GradScaler

# ==============================================================================
# CONFIGURATION CORRIGÉE - INNv3 SCALING
# ==============================================================================
CONFIG = {
    'batch_size': 32,        # Augmenté pour stabilité
    'accum_steps': 1,        # Plus besoin avec 32 si la mémoire tient, sinon augmenter
    'seq_len': 64,         
    'epochs': 20,          
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'd_model': 256,
    'num_layers': 4,
    'max_neurons': 128,    
    'growth_interval': 2,
    'learning_rate': 5e-4,
    'dropout': 0.2,          # Crucial pour WikiText
    'grad_clip': 0.25,       # Agressif pour stabilité
    'warmup_steps': 1000     # Crucial
}

# --- DATA LOADING ---
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

def evaluate(model, data_source, criterion, vocab_size, seq_len):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_len):
            data, targets = get_batch(data_source, i, seq_len)
            data = data.t()
            
            with autocast():
                logits = model(data)
                loss = criterion(logits.reshape(-1, vocab_size), targets)
            
            # Accumulate correctly by token count
            n_tokens = data.size(0) * data.size(1)
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            
    return total_loss / total_tokens

def main():
    print("=== TRAINING INNv3: CORRECTED LOOP ===")
    
    # 1. Data
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
    val_data = batchify(corpus.valid, 10, device)
    
    # 2. Model
    model = INNv3(
        vocab_size=vocab_size,
        d_model=CONFIG['d_model'],
        num_layers=CONFIG['num_layers'],
        max_neurons=CONFIG['max_neurons'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    print(f"Model Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Optimizer & Scheduler
    router_params = []
    decoder_params = []
    neuron_params = []
    
    for name, param in model.named_parameters():
        if 'router' in name:
            router_params.append(param)
        elif 'decoder' in name:
            decoder_params.append(param)
        else:
            neuron_params.append(param)
            
    optimizer = optim.AdamW([
        {'params': router_params, 'lr': 1e-3},   
        {'params': neuron_params, 'lr': CONFIG['learning_rate']},   
        {'params': decoder_params, 'lr': 1e-4}   
    ], weight_decay=0.1)
    
    # OneCycleLR
    steps_per_epoch = train_data.size(0) // CONFIG['seq_len']
    total_steps = steps_per_epoch * CONFIG['epochs']
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-3, CONFIG['learning_rate'], 1e-4],
        total_steps=total_steps,
        pct_start=CONFIG['warmup_steps'] / total_steps,
        anneal_strategy='cos'
    )
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # 4. Train Loop
    optimizer.zero_grad()
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        if epoch > 1 and epoch % CONFIG['growth_interval'] == 0:
            model.grow_network()
            
        model.train()
        total_loss = 0
        total_tokens = 0
        start_time = time.time()
        
        for batch, i in enumerate(range(0, train_data.size(0)-1, CONFIG['seq_len'])):
            data, targets = get_batch(train_data, i, CONFIG['seq_len'])
            data = data.t() # (B, L)
            
            with autocast():
                logits = model(data)
                loss = criterion(logits.reshape(-1, vocab_size), targets)
            
            scaler.scale(loss).backward()
            
            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step() # Step per batch
            
            # Accumulate correctly
            n_tokens = data.size(0) * data.size(1)
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            
            if batch % 100 == 0 and batch > 0:
                cur_loss = total_loss / total_tokens
                ppl = math.exp(cur_loss)
                elapsed = time.time() - start_time
                print(f"| epoch {epoch} | {batch}/{steps_per_epoch} batches | lr {scheduler.get_last_lr()[1]:.2e} | loss {cur_loss:.2f} | ppl {ppl:.2f}")
                total_loss = 0
                total_tokens = 0
                start_time = time.time()
                
        # Validation
        val_loss = evaluate(model, val_data, criterion, vocab_size, CONFIG['seq_len'])
        val_ppl = math.exp(val_loss)
        
        print(f"=== Epoch {epoch} | Val Loss {val_loss:.2f} | Val PPL {val_ppl:.2f} ===")
        torch.save(model.state_dict(), "checkpoints/innv3_scaling.pt")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()
