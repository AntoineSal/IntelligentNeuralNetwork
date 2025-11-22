import torch
import torch.nn as nn
import torch.optim as optim
import random
from src.vectorized_network import VectorizedIntelligentNetwork
from src.visualization import plot_attention_matrix
import os
import time

# --- Config ---
BATCH_SIZE = 64
SEQ_LEN = 64
LEARNING_RATE = 0.001
EPOCHS = 20
HIDDEN_DIM = 256
KEY_QUERY_DIM = 256
SIGNAL_DIM = 256

NUM_INPUT = 5
NUM_TRANS_STATIC = 40
NUM_TRANS_DYNAMIC = 20
NUM_ACTION = 5

SPARSITY_WEIGHT = 0.001 # Pénalité d'entropie pour forcer la spécialisation

PRINT_EVERY = 100

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
        
        input_tensor = torch.zeros(seq_len, vocab_size)
        for t, idx in enumerate(input_indices):
            input_tensor[t][idx] = 1.0
            
        input_seqs.append(input_tensor)
        target_seqs.append(torch.tensor(target_indices, dtype=torch.long))
        
    inputs = torch.stack(input_seqs)
    targets = torch.stack(target_seqs)
    return inputs, targets

def generate_text(model, start_str, char2idx, idx2char, vocab_size, length=100, temperature=0.8):
    model.eval()
    device = next(model.parameters()).device
    
    input_seq = torch.zeros(1, 1, vocab_size).to(device)
    model.reset_memory(batch_size=1)
    
    current_char = start_str[0]
    input_seq[0, 0, char2idx[current_char]] = 1.0
    
    generated_str = start_str
    
    with torch.no_grad():
        seed_tensor = torch.zeros(1, len(start_str), vocab_size).to(device)
        for t, char in enumerate(start_str):
            seed_tensor[0, t, char2idx[char]] = 1.0
            
        outputs = model(seed_tensor) 
        last_output = outputs[:, -1, :] 
        
        probs = torch.softmax(last_output / temperature, dim=1)
        next_idx = torch.multinomial(probs, 1).item()
        
        current_input = torch.zeros(1, 1, vocab_size).to(device)
        current_input[0, 0, next_idx] = 1.0
        
        for _ in range(length - 1):
            output = model(current_input.squeeze(1)) 
            
            probs = torch.softmax(output / temperature, dim=1)
            next_idx = torch.multinomial(probs, 1).item()
            next_char = idx2char[next_idx]
            generated_str += next_char
            
            current_input.zero_()
            current_input[0, 0, next_idx] = 1.0
            
    model.train()
    return generated_str

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device : {device}")
    
    # Changement ici : on peut utiliser 'test/input_algo.txt' pour l'expérience Reverse
    # ou 'test/input2.txt' pour le mixte.
    # Par défaut, je laisse input2 (mixte) car c'est le setup courant.
    # L'utilisateur peut changer manuellement le nom du fichier.
    dataset_file = 'test/input2.txt' 
    if not os.path.exists(dataset_file):
        print(f"Attention: {dataset_file} introuvable, fallback sur input.txt")
        dataset_file = 'test/input.txt'
        
    text, chars, char2idx, idx2char, vocab_size = load_data(dataset_file)
    print(f"Dataset chargé: {dataset_file}")
    
    model = VectorizedIntelligentNetwork(
        input_size=vocab_size,
        output_size=vocab_size,
        num_input_neurons=NUM_INPUT,
        num_transmission_neurons=NUM_TRANS_STATIC,
        num_dynamic_neurons=NUM_TRANS_DYNAMIC,
        num_action_neurons=NUM_ACTION,
        neuron_hidden_dim=HIDDEN_DIM,
        key_query_dim=KEY_QUERY_DIM,
        signal_dim=SIGNAL_DIM
    ).to(device)
    
    print(f"Modèle Hybride créé avec {NUM_INPUT} In, {NUM_TRANS_STATIC} Stat, {NUM_TRANS_DYNAMIC} Dyn, {NUM_ACTION} Act.")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print("Début de l'entraînement HYBRIDE VECTORISÉ avec Sparsity...")
    if not os.path.exists("plots_vec"): os.makedirs("plots_vec")

    batches_per_epoch = 200 
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        total_entropy = 0
        
        for i in range(batches_per_epoch):
            inputs, targets = get_batch(text, char2idx, BATCH_SIZE, SEQ_LEN, vocab_size)
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs) 
            
            # 1. Loss de prédiction
            main_loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            
            # 2. Loss de Sparsité (Minimisation de l'Entropie d'Attention)
            sparsity_loss = 0
            if model.last_Q is not None and model.last_K is not None:
                # On prend le premier échantillon du batch pour calculer l'entropie
                q = model.last_Q[0] # (N, Dim)
                k = model.last_K[0] # (N, Dim)
                
                # Calcul manuel de l'attention (car le modèle ne la retourne pas directement)
                scores = torch.matmul(q, k.t()) / (KEY_QUERY_DIM ** 0.5)
                attn = torch.softmax(scores, dim=1) # (N_receivers, N_senders)
                
                # Entropie H(p) = - sum(p * log(p))
                # On veut minimiser H -> rendre la distribution "pointue" (choix clair)
                entropy = -torch.sum(attn * torch.log(attn + 1e-9)) / attn.size(0) # Moyenne par neurone
                
                sparsity_loss = SPARSITY_WEIGHT * entropy
                total_entropy += entropy.item()
            
            loss = main_loss + sparsity_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += main_loss.item() # On log la loss "pure" pour comparer
            
            if i % PRINT_EVERY == 0:
                print(f"Batch {i}, Loss: {main_loss.item():.4f}, Entropy: {entropy.item() if 'entropy' in locals() else 0:.4f}")
                
        avg_loss = total_loss / batches_per_epoch
        avg_entropy = total_entropy / batches_per_epoch
        duration = time.time() - start_time
        print(f"=== Epoch {epoch+1} en {duration:.2f}s | Loss: {avg_loss:.4f} | Avg Entropy: {avg_entropy:.4f} ===")
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            plot_attention_matrix(model, filename=f"plots_vec/attention_epoch_{epoch+1}.png", 
                                  title=f"Sparse Attention (Ent={avg_entropy:.2f}) - Epoch {epoch+1}")

        print("Génération :")
        # Pour l'algo Reverse, il faudrait un prompt adapté, ex: "REV: "
        # Si on est sur input2 (Mixte), on garde "The " ou "MATH: "
        prompt = "MATH: " if "input2" in dataset_file else "REV: "
        if "input2" not in dataset_file and "input_algo" not in dataset_file: prompt = "The "
            
        print(generate_text(model, prompt, char2idx, idx2char, vocab_size))
        print("="*30)

if __name__ == "__main__":
    train()
