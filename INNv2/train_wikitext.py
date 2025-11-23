import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import math
import json
from collections import Counter
from src.network_parallel import ParallelINN

# ============================================
# CONFIGURATION WIKITEXT-2
# ============================================
CONFIG = {
    'batch_size': 20,       # Standard pour WikiText
    'seq_len': 64,         # Contexte raisonnable pour word-level
    'learning_rate': 1e-3,
    'epochs': 10,           # WikiText converge vite
    'num_neurons': 64,
    'd_model': 64,
    'num_layers': 4,
    'log_interval': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_path': 'checkpoints/innv2_wikitext.pt'
}

class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        # Train est utilisé pour construire le vocab
        self.train = self.tokenize(os.path.join(path, 'train.txt'), build_vocab=True)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path, build_vocab=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        
        # Add words to dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                # WikiText has <eos> markers implicitly at newlines usually handled by dataset classes
                # Here we explicitly add <eos> token
                words = line.split() + ['<eos>']
                tokens += len(words)
                if build_vocab:
                    for word in words:
                        self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if word in self.dictionary.word2idx:
                        ids[token] = self.dictionary.word2idx[word]
                    else:
                        # Handle unknown words (should not happen in valid/test for standard split if vocab is full)
                        # For simplicity, we map to a random existing token or create <unk>
                        # But standard WikiText usually assumes fixed vocab.
                        # If valid set has new words, it's an issue. 
                        # We will map unknown to <unk> if we had one, or skip.
                        # Let's assume standard closed vocabulary for now (common in these benchmarks).
                        # Actually, let's just use the word if in vocab, else <unk> (we need to add it).
                        pass 
                    token += 1
        return ids

def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (residuals).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(model, data_source, criterion, vocab_size):
    model.eval()
    total_loss = 0.
    
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, CONFIG['seq_len']):
            data, targets = get_batch(data_source, i, CONFIG['seq_len'])
            output = model(data.t()) # ParallelINN expects (B, L)
            output_flat = output.view(-1, vocab_size)
            total_loss += len(data) * criterion(output_flat, targets).item()
            
    return total_loss / (len(data_source) - 1)

def train():
    print(f"Loading Corpus from INNv2/data/wikitext-2...")
    corpus = Corpus('INNv2/data/wikitext-2')
    vocab_size = len(corpus.dictionary)
    print(f"Vocab Size: {vocab_size}")
    
    device = torch.device(CONFIG['device'])
    
    train_data = batchify(corpus.train, CONFIG['batch_size'], device)
    val_data = batchify(corpus.valid, 10, device) # Eval batch size 10
    test_data = batchify(corpus.test, 10, device)
    
    print("Building INNv2 Model...")
    model = ParallelINN(
        vocab_size=vocab_size,
        num_neurons=CONFIG['num_neurons'],
        d_model=CONFIG['d_model'],
        num_layers=CONFIG['num_layers']
    ).to(device)
    
    print(f"Model Params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
    
    best_val_loss = float('inf')
    history = []
    
    print("\nStarting Training...")
    for epoch in range(1, CONFIG['epochs'] + 1):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.
        start_time = time.time()
        
        # Training Loop
        # Note: standard LM training iterates sequentially through the corpus
        # source is (L, B) from batchify
        for batch, i in enumerate(range(0, train_data.size(0) - 1, CONFIG['seq_len'])):
            data, targets = get_batch(train_data, i, CONFIG['seq_len'])
            # data shape is (seq_len, batch_size)
            # ParallelINN expects (batch_size, seq_len)
            data = data.t() 
            
            optimizer.zero_grad()
            output = model(data) # (B, L, Vocab)
            loss = criterion(output.reshape(-1, vocab_size), targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch % CONFIG['log_interval'] == 0 and batch > 0:
                cur_loss = total_loss / CONFIG['log_interval']
                elapsed = time.time() - start_time
                print(f"| epoch {epoch:3d} | {batch:5d}/{train_data.size(0)//CONFIG['seq_len']:5d} batches | "
                      f"lr {optimizer.param_groups[0]['lr']:.2e} | ms/batch {elapsed * 1000 / CONFIG['log_interval']:5.2f} | "
                      f"loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}")
                total_loss = 0
                start_time = time.time()
                
        # Validation
        val_loss = evaluate(model, val_data, criterion, vocab_size)
        val_ppl = math.exp(val_loss)
        print('-' * 89)
        print(f"| End of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | "
              f"valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}")
        print('-' * 89)
        
        scheduler.step(val_loss)
        
        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG['save_path'])
            
        history.append({'epoch': epoch, 'val_loss': val_loss, 'val_ppl': val_ppl})
        
    # Test
    print("Running Test...")
    model.load_state_dict(torch.load(CONFIG['save_path']))
    test_loss = evaluate(model, test_data, criterion, vocab_size)
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print(f"| End of training | test loss {test_loss:5.2f} | test ppl {test_ppl:8.2f}")
    print('=' * 89)
    
    # Save results
    with open('results/wikitext_innv2.json', 'w') as f:
        json.dump(history, f)

if __name__ == "__main__":
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    train()

