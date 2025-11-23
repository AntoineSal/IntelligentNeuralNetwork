import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import math
import requests
from src.model import INNv3

# ==============================================================================
# CONFIGURATION INNv3 (SCALING)
# ==============================================================================
CONFIG = {
    'batch_size': 20,
    'seq_len': 64,         # Reduced for Mamba/Memory intensity
    'epochs': 10,          
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'd_model': 256,
    'num_layers': 4,
    'max_neurons': 128,    # Base pool size
    'growth_interval': 2,  # Epochs between growth
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

def main():
    print("=== TRAINING INNv3: SCALING DYNAMIC NETWORKS ===")
    
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
        max_neurons=CONFIG['max_neurons']
    ).to(device)
    
    print(f"Model Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Optimizers (Multi-Scale)
    # Split parameters into groups
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
        {'params': router_params, 'lr': 1e-3},   # Fast adaptation
        {'params': neuron_params, 'lr': 5e-4},   # Standard
        {'params': decoder_params, 'lr': 1e-4}   # Careful with vocab
    ])
    
    criterion = nn.CrossEntropyLoss()
    
    # 4. Train Loop
    for epoch in range(1, CONFIG['epochs'] + 1):
        # Progressive Growth
        if epoch > 1 and epoch % CONFIG['growth_interval'] == 0:
            model.grow_network()
            
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch, i in enumerate(range(0, train_data.size(0)-1, CONFIG['seq_len'])):
            data, targets = get_batch(train_data, i, CONFIG['seq_len'])
            data = data.t() # (B, L)
            
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits.reshape(-1, vocab_size), targets)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch % 100 == 0 and batch > 0:
                cur_loss = total_loss / 100
                ppl = math.exp(cur_loss)
                print(f"| epoch {epoch} | batch {batch} | loss {cur_loss:.2f} | ppl {ppl:.2f}")
                total_loss = 0
                
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, val_data.size(0)-1, CONFIG['seq_len']):
                data, targets = get_batch(val_data, i, CONFIG['seq_len'])
                logits = model(data.t())
                val_loss += len(data) * criterion(logits.reshape(-1, vocab_size), targets).item()
        val_loss /= (len(val_data) // CONFIG['seq_len'])
        val_ppl = math.exp(val_loss)
        
        print(f"=== Epoch {epoch} | Val Loss {val_loss:.2f} | Val PPL {val_ppl:.2f} ===")
        torch.save(model.state_dict(), "checkpoints/innv3_scaling.pt")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()

