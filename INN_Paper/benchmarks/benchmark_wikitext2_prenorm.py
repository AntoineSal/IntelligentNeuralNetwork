import torch
import torch.nn as nn
import math
import time
import os
import requests
import zipfile
import gc

# --- AUTO-INSTALL ---
try:
    from mamba_ssm import Mamba
    print("✅ Mamba-SSM detected")
except ImportError:
    print("⚠️ Installing Mamba-SSM...")
    import subprocess
    subprocess.check_call(["pip", "install", "mamba-ssm", "causal-conv1d>=1.2.0"])
    from mamba_ssm import Mamba

# === CONFIGURATION (WIKITEXT-2 BATTLE TESTED) ===
CONFIG = {
    'dataset': 'wikitext-2',
    'vocab_size': 33278,   # Standard WT2 vocab
    'd_model': 256,
    'n_neurons': 32,
    'n_layers': 6,         # On garde 6 layers pour la puissance
    'dropout': 0.4,        # Dropout 0.4 (Solid regularization)
    'weight_decay': 0.1,   # High Weight Decay
    'lr': 3e-4,            # Safe LR
    'batch_size': 32,
    'seq_len': 128,        # Words
    'epochs': 20,          # Enough time to converge
    'grad_clip': 0.25,
    'save_every': 1000
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== LAUNCHING INNv2 PRE-NORM ON WIKITEXT-2 ===")
print(f"Config: {CONFIG}")

# === DATA ===
class WikiText2Corpus:
    def __init__(self, path):
        if not os.path.exists(path):
            self.download(path)
        
        self.dictionary = self.Dictionary()
        self.train = self.tokenize(os.path.join(path, 'wiki.train.tokens'))
        self.valid = self.tokenize(os.path.join(path, 'wiki.valid.tokens'))
        self.test = self.tokenize(os.path.join(path, 'wiki.test.tokens'))
        
        print(f"Vocab size: {len(self.dictionary)}")
        print(f"Train tokens: {self.train.size(0):,}")

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

    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids

    def download(self, path):
        print("Downloading WikiText-2...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
        r = requests.get(url)
        with open("data/wikitext-2-v1.zip", "wb") as f: f.write(r.content)
        with zipfile.ZipFile("data/wikitext-2-v1.zip", "r") as z: z.extractall("data")
        # Move files
        import shutil
        if os.path.exists("data/wikitext-2"):
             for f in os.listdir("data/wikitext-2"):
                 shutil.move(os.path.join("data/wikitext-2", f), "data/")

def get_batch(data, i, seq_len):
    seq_len = min(seq_len, len(data) - 1 - i)
    x = data[i:i+seq_len]
    y = data[i+1:i+1+seq_len]
    return x.to(device), y.to(device)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data

# === INNv2 PRE-NORM + DROPOUT ===
class INNv2Wiki(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.num_neurons = CONFIG['n_neurons']
        self.d_model = CONFIG['d_model']
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.dropout = nn.Dropout(CONFIG['dropout'])
        
        self.layers = nn.ModuleList()
        for _ in range(CONFIG['n_layers']):
            neuron_pop = Mamba(d_model=self.d_model, d_state=16, d_conv=4, expand=2)
            attn = nn.MultiheadAttention(self.d_model, 4, dropout=CONFIG['dropout'], batch_first=True)
            norm1 = nn.LayerNorm(self.d_model)
            norm2 = nn.LayerNorm(self.d_model)
            self.layers.append(nn.ModuleList([neuron_pop, attn, norm1, norm2]))
            
        self.norm_f = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, vocab_size)
        self.head.weight = self.embedding.weight

    def forward(self, x):
        B, L = x.shape
        x = self.embedding(x)
        x = self.dropout(x)
        x = x.unsqueeze(1).expand(-1, self.num_neurons, -1, -1).reshape(B*self.num_neurons, L, -1)
        
        for mamba, attn, norm1, norm2 in self.layers:
            x_norm = norm1(x)
            x_mem = mamba(x_norm)
            x = x + self.dropout(x_mem)
            x_norm2 = norm2(x)
            x_comm = x_norm2.view(B, self.num_neurons, L, -1).permute(0, 2, 1, 3).reshape(B*L, self.num_neurons, -1)
            comm_out, _ = attn(x_comm, x_comm, x_comm)
            comm_out = comm_out.view(B, L, self.num_neurons, -1).permute(0, 2, 1, 3).reshape(B*self.num_neurons, L, -1)
            x = x + self.dropout(comm_out)
            
        x = x.view(B, self.num_neurons, L, -1).mean(dim=1)
        return self.head(self.norm_f(x))

# === TRAINING ===
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    os.makedirs("models", exist_ok=True)

    corpus = WikiText2Corpus("data")
    train_data = batchify(corpus.train, CONFIG['batch_size'])
    valid_data = batchify(corpus.valid, CONFIG['batch_size'])

    model = INNv2Wiki(len(corpus.dictionary)).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    steps_per_epoch = len(range(0, train_data.size(0)-1, CONFIG['seq_len']))
    total_steps = steps_per_epoch * CONFIG['epochs']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CONFIG['lr'], total_steps=total_steps + 100)
    crit = nn.CrossEntropyLoss()

    print(f"Starting training for {CONFIG['epochs']} epochs...")
    model.train()
    start = time.time()
    best_val_ppl = float('inf')

    try:
        for epoch in range(CONFIG['epochs']):
            total_loss = 0
            tokens = 0
            for batch, i in enumerate(range(0, train_data.size(0)-1, CONFIG['seq_len'])):
                x, y = get_batch(train_data, i, CONFIG['seq_len'])
                opt.zero_grad()
                logits = model(x)
                loss = crit(logits.reshape(-1, len(corpus.dictionary)), y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                opt.step()
                sched.step()
                total_loss += loss.item()
                
                if batch % 100 == 0:
                    print(f"Epoch {epoch+1} | Batch {batch} | Loss: {loss.item():.4f} | PPL: {math.exp(loss.item()):.2f}")

            # Eval
            model.eval()
            val_loss = 0
            cnt = 0
            with torch.no_grad():
                for i in range(0, valid_data.size(0)-1, CONFIG['seq_len']):
                    x, y = get_batch(valid_data, i, CONFIG['seq_len'])
                    loss = crit(model(x).view(-1, len(corpus.dictionary)), y.view(-1))
                    val_loss += loss.item()
                    cnt += 1
            val_ppl = math.exp(val_loss/cnt)
            print(f"=== End Epoch {epoch+1} | Valid PPL: {val_ppl:.2f} ===")
            
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                torch.save(model.state_dict(), "models/inn_wikitext_best.pth")
                print(f"🔥 New Best PPL: {best_val_ppl:.2f}")
            model.train()
            
    except KeyboardInterrupt:
        print("Stop.")
    
    print(f"🏆 BEST WIKITEXT PPL: {best_val_ppl:.2f}")

