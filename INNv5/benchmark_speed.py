import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

# --- Transformer Baseline ---
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        return self.lm_head(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_inference(model, input_ids, steps=50):
    model.eval()
    # Warmup
    for _ in range(5):
        with torch.no_grad(): _ = model(input_ids)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(steps):
            _ = model(input_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    return steps / (end_time - start_time)

# --- CONFIGURATION DU DUEL ---
VOCAB_SIZE = 256 # Char level
D_MODEL = 256
SEQ_LEN = 64
BATCH_SIZE = 1 # Inférence temps réel

print(f"⚔️ DUEL: INNv5 vs TRANSFORMER ⚔️")
print(f"Config: d_model={D_MODEL}, seq_len={SEQ_LEN}, vocab={VOCAB_SIZE}")
print("-" * 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)

# 1. TRANSFORMER (Le Champion en titre)
transformer = SimpleTransformer(VOCAB_SIZE, D_MODEL, nhead=4, num_layers=6).to(device)
params_trans = count_parameters(transformer)
print(f"🤖 Transformer Baseline instantiated.")
speed_trans = benchmark_inference(transformer, input_ids)
print(f"   Params: {params_trans/1e6:.2f}M")
print(f"   Vitesse: {speed_trans:.2f} runs/sec")
print("-" * 50)

# 2. INNv5 (Le Challenger Sparse)
# Note: This requires MonolithicINNv5 class to be available in the context
# We will import it from the notebook/script if this is run as a script
# For standalone execution, we need to define it or assume it's pasted
try:
    # Assuming MonolithicINNv5 is defined in the notebook context where this is run
    # If running as standalone script, you need to import it.
    # For now, we'll assume this snippet is pasted AFTER the model definition.
    
    # On simule un grand réseau (512 neurones) mais avec peu d'actifs (32)
    inn_model = MonolithicINNv5(VOCAB_SIZE, d_model=D_MODEL, num_layers=6, num_neurons=512, top_k=32).to(device)
    params_inn = count_parameters(inn_model)
    print(f"🧠 INNv5 (Massive Sparse) instantiated.")
    speed_inn = benchmark_inference(inn_model, input_ids)
    
    print(f"   Params Totaux: {params_inn/1e6:.2f}M")
    print(f"   Neurones: 512 | Actifs: 32 (Sparsity: {1 - 32/512:.1%})")
    print(f"   Vitesse: {speed_inn:.2f} runs/sec")
    
    print("=" * 50)
    ratio = speed_inn / speed_trans
    print(f"🏆 RÉSULTAT FINAL:")
    if ratio > 1:
        print(f"INNv5 est {ratio:.2f}x PLUS RAPIDE que le Transformer !")
    else:
        print(f"INNv5 est {1/ratio:.2f}x plus lent (Optimisation requise).")
        
except NameError:
    print("⚠️ Erreur: La classe MonolithicINNv5 n'est pas définie.")
    print("Veuillez exécuter la cellule contenant la définition du modèle INNv5 avant ce benchmark.")

