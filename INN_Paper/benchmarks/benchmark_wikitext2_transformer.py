import torch
import torch.nn as nn
import math
import time
import argparse
from datasets import load_dataset

# === CONFIGURATION ===
CONFIG = {
    'dataset': 'wikitext-2-raw-v1',
    'd_model': 256,       # Same as INN
    'n_layers': 6,        # Same as INN
    'nhead': 4,           # 256/4 = 64 dim per head
    'dim_feedforward': 768, # Reduced from 1024 to match INN param count (4.5M)
    'dropout': 0.1,
    'lr': 4e-4,           
    'batch_size': 32,     
    'seq_len': 128,       
    'epochs': 20,         
    'grad_clip': 1.0,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DATA LOADING (Hugging Face) ===
class CharLevelWikiText:
    def __init__(self):
        print("Loading WikiText-2 via Hugging Face datasets...")
        # Load raw version
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        
        self.train_text = "\n".join(dataset['train']['text'])
        self.valid_text = "\n".join(dataset['validation']['text'])
        self.test_text = "\n".join(dataset['test']['text'])
        
        print("Building vocabulary...")
        chars = set(self.train_text + self.valid_text + self.test_text)
        self.vocab = sorted(list(chars))
        self.vocab_size = len(self.vocab)
        self.char2idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx2char = {i: ch for i, ch in enumerate(self.vocab)}
        
        print(f"Vocab size: {self.vocab_size}")
        
        self.train_data = self.numericalize(self.train_text)
        self.valid_data = self.numericalize(self.valid_text)
        self.test_data = self.numericalize(self.test_text)
        
        print(f"Train tokens: {len(self.train_data):,} | Valid: {len(self.valid_data):,}")

    def numericalize(self, text):
        ids = [self.char2idx[ch] for ch in text]
        return torch.tensor(ids, dtype=torch.long)

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

# === TRANSFORMER MODEL (FAIR & ROBUST) ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [SeqLen, Batch, D]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, dim_feedforward, n_layers, dropout=0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Standard TransformerEncoderLayer (default batch_first=False -> Expects [Seq, Batch, Dim])
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        # src inputs: [SeqLen, Batch]
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src) # No unsqueeze needed, PE handles [Seq, Batch, Dim]
        
        output = self.transformer_encoder(src, mask=src_mask)
        output = self.decoder(output)
        return output

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)

# === MAIN LOOP ===
if __name__ == "__main__":
    print("=== BENCHMARKING TRANSFORMER (FAIR: ~4.5M PARAMS) ===")
    
    corpus = CharLevelWikiText()
    train_data = batchify(corpus.train_data, CONFIG['batch_size'])
    val_data = batchify(corpus.valid_data, CONFIG['batch_size'])
    
    ntokens = corpus.vocab_size
    model = TransformerModel(
        ntokens, 
        CONFIG['d_model'], 
        CONFIG['nhead'], 
        CONFIG['dim_feedforward'], 
        CONFIG['n_layers'], 
        CONFIG['dropout']
    ).to(device)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    
    # Calculate exact steps
    batches_per_epoch = len(range(0, train_data.size(0) - 1, CONFIG['seq_len']))
    total_steps = batches_per_epoch * CONFIG['epochs']
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=CONFIG['lr'], 
        total_steps=total_steps + 100
    )
    
    best_val_loss = float('inf')
    
    try:
        for epoch in range(1, CONFIG['epochs'] + 1):
            epoch_start_time = time.time()
            model.train()
            total_loss = 0.
            start_time = time.time()
            
            src_mask = generate_square_subsequent_mask(CONFIG['seq_len'])
            
            for batch, i in enumerate(range(0, train_data.size(0) - 1, CONFIG['seq_len'])):
                data, targets = get_batch(train_data, i, CONFIG['seq_len'])
                
                if data.size(0) != CONFIG['seq_len']:
                    src_mask = generate_square_subsequent_mask(data.size(0))
                
                optimizer.zero_grad()
                output = model(data, src_mask)
                loss = criterion(output.view(-1, ntokens), targets.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                
                if batch % 200 == 0 and batch > 0:
                    cur_loss = total_loss / 200
                    elapsed = time.time() - start_time
                    print(f'| epoch {epoch:3d} | {batch:5d} batches | '
                          f'lr {scheduler.get_last_lr()[0]:02.5f} | '
                          f'ms/batch {elapsed * 1000 / 200:5.2f} | '
                          f'loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
                    total_loss = 0
                    start_time = time.time()
            
            # Validation Loop
            model.eval()
            total_loss = 0.
            total_len = 0
            with torch.no_grad():
                for i in range(0, val_data.size(0) - 1, CONFIG['seq_len']):
                    data, targets = get_batch(val_data, i, CONFIG['seq_len'])
                    if data.size(0) != CONFIG['seq_len']:
                        src_mask = generate_square_subsequent_mask(data.size(0))
                    else:
                        src_mask = generate_square_subsequent_mask(CONFIG['seq_len'])
                        
                    output = model(data, src_mask)
                    loss = criterion(output.view(-1, ntokens), targets.view(-1)).item()
                    total_loss += loss * len(data)
                    total_len += len(data)
            
            val_loss = total_loss / total_len
            val_ppl = math.exp(val_loss)
            
            print('-' * 89)
            print(f'| End of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | '
                  f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
            print('-' * 89)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss

    except KeyboardInterrupt:
        print("Exiting training early")
