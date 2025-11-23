import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import json
import random
from src.mamba_pure_model import MambaPure

# ============================================
# CONFIGURATION - IDENTIQUE À TRAIN_PARALLEL.PY
# ============================================
# Params INNv2:
# BATCH_SIZE = 32
# SEQ_LEN = 128
# LEARNING_RATE = 1e-3
# EPOCHS = 10
# NUM_LAYERS = 4
# NUM_NEURONS = 64, D_MODEL = 64 -> 8.6M params total

CONFIG = {
    'batch_size': 32,
    'seq_len': 128,
    'learning_rate': 1e-3,
    'epochs': 10,
    'n_layers': 4,
    
    # Tuning pour matcher 8.6M params :
    # INNv2 params ~ 8.6M
    # MambaPure avec d_model=480 -> ~8.5M params
    'd_model': 480, 
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'log_file': 'results/mamba_pure_results.json'
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    return text, chars, char2idx, idx2char, vocab_size

def get_batch(text, char2idx, batch_size, seq_len, vocab_size):
    input_seqs = []
    target_seqs = []
    for _ in range(batch_size):
        start_idx = random.randint(0, len(text) - seq_len - 1)
        chunk = text[start_idx : start_idx + seq_len + 1]
        input_indices = [char2idx[c] for c in chunk[:-1]]
        target_indices = [char2idx[c] for c in chunk[1:]]
        input_seqs.append(torch.tensor(input_indices, dtype=torch.long))
        target_seqs.append(torch.tensor(target_indices, dtype=torch.long))
    return torch.stack(input_seqs), torch.stack(target_seqs)

def main():
    print("=== TRAINING MAMBA PURE BASELINE ===")
    device = torch.device(CONFIG['device'])
    torch.manual_seed(42)
    
    # 1. Data
    paths = ['test/input.txt', '../INNv1/test/input.txt', 'INNv1/test/input.txt']
    dataset_file = None
    for p in paths:
        if os.path.exists(p):
            dataset_file = p
            break
    if not dataset_file:
        print("Erreur: input.txt non trouvé")
        return

    text, chars, char2idx, idx2char, vocab_size = load_data(dataset_file)
    print(f"Vocab size: {vocab_size}")
    
    # 2. Model
    model = MambaPure(
        vocab_size=vocab_size,
        d_model=CONFIG['d_model'],
        n_layers=CONFIG['n_layers']
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"Mamba Pure Params: {num_params:,} ({num_params/1e6:.2f}M)")
    print("Target: ~8.60M")
    
    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    history = []
    num_batches = 200 # Comme dans train_parallel.py
    
    for epoch in range(CONFIG['epochs']):
        start_time = time.time()
        total_loss = 0
        model.train()
        
        for i in range(num_batches):
            inputs, targets = get_batch(text, char2idx, CONFIG['batch_size'], CONFIG['seq_len'], vocab_size)
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 50 == 0:
                print(f"Epoch {epoch+1} Batch {i} Loss {loss.item():.4f}")
                
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        
        print(f"=== Epoch {epoch+1} Done | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s ===")
        
        history.append({
            'epoch': epoch + 1,
            'val_loss': avg_loss, # On utilise train loss comme val loss proxy ici (dataset simple)
            'time': epoch_time,
            'val_ppl': torch.exp(torch.tensor(avg_loss)).item()
        })
        
        # Save checkpoints
        if epoch == CONFIG['epochs'] - 1:
            torch.save(model.state_dict(), 'checkpoints/mamba_pure_final.pt')

    # Save results
    os.makedirs('results', exist_ok=True)
    with open(CONFIG['log_file'], 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Results saved to {CONFIG['log_file']}")

if __name__ == "__main__":
    main()

