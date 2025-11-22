import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from src.intelligent_network import IntelligentNetwork

# --- Configuration ---
BATCH_SIZE = 32
SEQ_LEN = 64
LEARNING_RATE = 0.001  # Réduit pour meilleure convergence
EPOCHS = 20           # Augmenté
HIDDEN_DIM = 256      # Augmenté
KEY_QUERY_DIM = 256   # Augmenté
SIGNAL_DIM = 256      # Augmenté pour améliorer la communication
NUM_INPUT = 5
NUM_TRANS = 40        # Fortement augmenté pour donner de la capacité de calcul
NUM_ACTION = 5
PRINT_EVERY = 100

def load_data(filepath):
    # On suppose que le fichier existe et est accessible
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    print(f"Texte chargé : {len(text)} caractères")
    print(f"Vocabulaire : {vocab_size} caractères uniques")
    return text, chars, char2idx, idx2char, vocab_size

def get_batch(text, char2idx, batch_size, seq_len, vocab_size):
    input_seqs = []
    target_seqs = []
    
    for _ in range(batch_size):
        start_idx = random.randint(0, len(text) - seq_len - 1)
        chunk = text[start_idx : start_idx + seq_len + 1]
        
        # Input : One-hot encoding
        input_indices = [char2idx[c] for c in chunk[:-1]]
        target_indices = [char2idx[c] for c in chunk[1:]]
        
        # Conversion en tenseurs one-hot pour l'input
        input_tensor = torch.zeros(seq_len, vocab_size)
        for t, idx in enumerate(input_indices):
            input_tensor[t][idx] = 1.0
            
        input_seqs.append(input_tensor)
        target_seqs.append(torch.tensor(target_indices, dtype=torch.long))
        
    # Stack -> (Batch, Time, Input_Dim)
    inputs = torch.stack(input_seqs)
    # Stack -> (Batch, Time)
    targets = torch.stack(target_seqs)
    
    return inputs, targets

def generate_text(model, start_str, char2idx, idx2char, vocab_size, length=100, temperature=0.8):
    # Température baissée à 0.8 pour réduire le bruit
    model.eval()
    device = next(model.parameters()).device
    
    input_seq = torch.zeros(1, 1, vocab_size).to(device)
    
    # On chauffe la mémoire avec le start_str
    model.reset_memory(batch_size=1)
    
    current_char = start_str[0]
    input_seq[0, 0, char2idx[current_char]] = 1.0
    
    generated_str = start_str
    
    with torch.no_grad():
        # Pre-fill memory
        seed_tensor = torch.zeros(1, len(start_str), vocab_size).to(device)
        for t, char in enumerate(start_str):
            seed_tensor[0, t, char2idx[char]] = 1.0
            
        outputs = model(seed_tensor) 
        last_output = outputs[:, -1, :] 
        
        probs = torch.softmax(last_output / temperature, dim=1)
        next_idx = torch.multinomial(probs, 1).item()
        next_char = idx2char[next_idx]
        generated_str += next_char
        
        # Génération boucle
        current_input = torch.zeros(1, 1, vocab_size).to(device)
        current_input[0, 0, next_idx] = 1.0
        
        for _ in range(length - 1):
            output = model(current_input.squeeze(1)) 
            
            probs = torch.softmax(output / temperature, dim=1)
            next_idx = torch.multinomial(probs, 1).item()
            next_char = idx2char[next_idx]
            generated_str += next_char
            
            # Prepare next input
            current_input.zero_()
            current_input[0, 0, next_idx] = 1.0
            
    model.train()
    return generated_str

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device : {device}")
    
    text, chars, char2idx, idx2char, vocab_size = load_data('test/input.txt')
    
    model = IntelligentNetwork(
        input_size=vocab_size,
        output_size=vocab_size,
        num_input_neurons=NUM_INPUT,
        num_transmission_neurons=NUM_TRANS,
        num_action_neurons=NUM_ACTION,
        neuron_hidden_dim=HIDDEN_DIM,
        key_query_dim=KEY_QUERY_DIM,
        signal_dim=SIGNAL_DIM
    ).to(device)
    
    print(f"Modèle créé avec {NUM_INPUT + NUM_TRANS + NUM_ACTION} neurones.")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print("Début de l'entraînement...")
    
    # On utilise plus de batches pour que l'entraînement soit plus significatif
    # (dans Colab vous pourrez ajuster si trop long, mais il faut de la donnée)
    batches_per_epoch = 200 
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for i in range(batches_per_epoch):
            inputs, targets = get_batch(text, char2idx, BATCH_SIZE, SEQ_LEN, vocab_size)
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs) 
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % PRINT_EVERY == 0:
                print(f"Epoch {epoch+1}, Batch {i}/{batches_per_epoch}, Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / batches_per_epoch
        print(f"=== Fin Epoch {epoch+1}, Moyenne Loss: {avg_loss:.4f} ===")
        print("Génération de texte :")
        print(generate_text(model, "The ", char2idx, idx2char, vocab_size))
        print("===============================================================")

if __name__ == "__main__":
    train()
