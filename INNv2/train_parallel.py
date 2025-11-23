import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os
from src.network_parallel import ParallelINN

# --- Config ---
BATCH_SIZE = 32
SEQ_LEN = 128
LEARNING_RATE = 1e-3
EPOCHS = 10
NUM_NEURONS = 64
D_MODEL = 64
NUM_LAYERS = 4
PRINT_EVERY = 50

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

def generate_text(model, start_str, char2idx, idx2char, vocab_size, length=100, temperature=0.8):
    model.eval()
    device = next(model.parameters()).device
    input_indices = [char2idx[c] for c in start_str]
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
    generated_str = start_str
    
    with torch.no_grad():
        for _ in range(length):
            if input_tensor.size(1) > SEQ_LEN:
                context = input_tensor[:, -SEQ_LEN:]
            else:
                context = input_tensor
                
            logits = model(context)
            last_logits = logits[:, -1, :]
            probs = torch.softmax(last_logits / temperature, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            
            generated_str += idx2char[next_idx]
            next_tensor = torch.tensor([[next_idx]], device=device)
            input_tensor = torch.cat([input_tensor, next_tensor], dim=1)
            
    model.train()
    return generated_str

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device : {device}")
    
    # Recherche du fichier input.txt
    paths = ['test/input.txt', '../INNv1/test/input.txt', 'INNv1/test/input.txt']
    dataset_file = None
    for p in paths:
        if os.path.exists(p):
            dataset_file = p
            break
            
    if not dataset_file:
        # Fallback création dummy
        print("Info: input.txt non trouvé, création d'un dummy pour le test.")
        dataset_file = 'dummy_input.txt'
        with open(dataset_file, 'w') as f:
            f.write("To be or not to be " * 1000)

    text, chars, char2idx, idx2char, vocab_size = load_data(dataset_file)
    print(f"Dataset: {dataset_file} ({len(text)} chars)")
    
    model = ParallelINN(vocab_size, NUM_NEURONS, D_MODEL, NUM_LAYERS).to(device)
    print(f"Modèle Parallel INN: {sum(p.numel() for p in model.parameters())} params")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    num_batches = 200 
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        
        for i in range(num_batches):
            inputs, targets = get_batch(text, char2idx, BATCH_SIZE, SEQ_LEN, vocab_size)
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs) 
            
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % PRINT_EVERY == 0:
                print(f"Batch {i}, Loss: {loss.item():.4f}")
                
        duration = time.time() - start_time
        avg_loss = total_loss / num_batches
        print(f"=== Epoch {epoch+1} en {duration:.2f}s | Loss: {avg_loss:.4f} ===")
        print(generate_text(model, "The ", char2idx, idx2char, vocab_size))
        print("-" * 30)

if __name__ == "__main__":
    train()

