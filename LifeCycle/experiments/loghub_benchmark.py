# @title 📉 BENCHMARK: Interleaved HDFS Log Monitoring (LogHub-based)
# @markdown Tâche "Real-World" : Détection d'anomalies sur des logs HDFS entrelacés.
# @markdown Le modèle doit suivre l'état de plusieurs Block_IDs simultanément dans un flux bruité.
# @markdown Vocabulaire réel extrait du dataset HDFS de LogHub.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from collections import OrderedDict, deque

# ==========================================
# CONFIGURATION
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📉 LOGHUB BENCHMARK: Interleaved HDFS on {DEVICE}")

# Fix seed
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Hyperparamètres
D_MODEL = 128
D_STATE = 256
VOCAB_SIZE = 50  
SEQ_LEN = 64

# ==========================================
# DATASET: HDFS TEMPLATES (LogHub)
# ==========================================

HDFS_TEMPLATES = {
    "ALLOCATE": 10,  "RECEIVE": 11,   "OPEN": 12,      "WRITE": 13,     
    "CLOSE": 14,     "REPLICATE": 15, "SUCCESS": 16,   
    # Anomalies contextuelles
    "TIMEOUT": 20,   "EXCEPTION": 21, "WARN": 22,      
    # Bruit de fond
    "INFO_A": 30,    "INFO_B": 31,    "INFO_C": 32,    "DEBUG": 33
}

# Séquence normale d'un bloc HDFS
NORMAL_FLOW = ["ALLOCATE", "OPEN", "WRITE", "REPLICATE", "SUCCESS", "CLOSE"]

class HDFSStream:
    """
    Génère un flux de logs HDFS entrelacés.
    La distance entre 'OPEN' et 'TIMEOUT' est rendue aléatoire par :
    1. L'entrelacement de 4 processus concurrents
    2. L'insertion aléatoire de bruit (60% du temps)
    """
    def __init__(self, num_concurrent_blocks=4, noise_ratio=0.6):
        self.num_concurrent = num_concurrent_blocks
        self.noise_ratio = noise_ratio
        self.block_counter = 0
        
    def generate_chunk(self, batch_size, seq_len):
        x = torch.zeros((batch_size, seq_len), dtype=torch.long)
        y = torch.zeros((batch_size, seq_len), dtype=torch.long) # 0=OK, 1=ANOMALY
        mask = torch.zeros((batch_size, seq_len), dtype=torch.float)
        
        for b in range(batch_size):
            local_blocks = {} # id -> step_index
            
            for t in range(seq_len):
                # 1. BRUIT ALÉATOIRE (Casse les patterns de position)
                if random.random() < self.noise_ratio:
                    # PIÈGE : On ajoute aussi des TIMEOUT et EXCEPTION dans le bruit (Faux Positifs)
                    # Le modèle doit utiliser sa mémoire pour savoir si c'est un vrai TIMEOUT lié à un bloc actif.
                    noise_token = random.choice([
                        HDFS_TEMPLATES["INFO_A"], 
                        HDFS_TEMPLATES["INFO_B"], 
                        HDFS_TEMPLATES["DEBUG"],
                        HDFS_TEMPLATES["TIMEOUT"],   # <--- Le piège
                        HDFS_TEMPLATES["EXCEPTION"]  # <--- Le piège
                    ])
                    x[b, t] = noise_token
                    y[b, t] = 0  # C'est du bruit, pas une anomalie structurelle
                    mask[b, t] = 1.0 # On veut qu'il apprenne à IGNORER ces faux positifs
                    continue
                
                # 2. GESTION DES BLOCS
                finished_ids = [bid for bid, step in local_blocks.items() if step >= len(NORMAL_FLOW)]
                for bid in finished_ids: del local_blocks[bid]
                
                if len(local_blocks) < self.num_concurrent:
                    new_id = self.block_counter
                    self.block_counter += 1
                    local_blocks[new_id] = 0
                
                if not local_blocks: continue
                
                # Avancement aléatoire d'un des blocs actifs (Entrelacement)
                current_bid = random.choice(list(local_blocks.keys()))
                step_idx = local_blocks[current_bid]
                
                # 3. GÉNÉRATION D'ÉVÉNEMENT
                is_anomaly = (random.random() < 0.05)
                
                if is_anomaly:
                    # ANOMALIE CONTEXTUELLE
                    token = HDFS_TEMPLATES["TIMEOUT"]
                    x[b, t] = token
                    y[b, t] = 1 # ALERTE
                    mask[b, t] = 1.0 
                    del local_blocks[current_bid]
                else:
                    # NORMAL FLOW
                    if step_idx < len(NORMAL_FLOW):
                        template_name = NORMAL_FLOW[step_idx]
                        token = HDFS_TEMPLATES[template_name]
                        x[b, t] = token
                        y[b, t] = 0
                        mask[b, t] = 1.0
                        local_blocks[current_bid] += 1
                    else:
                        x[b, t] = HDFS_TEMPLATES["INFO_C"]
                        
        return x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)

# ==========================================
# MODÈLES
# ==========================================

# --- 1. INN V10.1 (HDFS Specialized) ---
class LockedMemoryNeuron(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_state = d_state
        self.gru = nn.GRU(d_model, d_state, batch_first=True)
        self.out_proj = nn.Linear(d_state, d_model)
        with torch.no_grad():
            self.gru.weight_hh_l0.zero_()
            i_start, i_end = 2 * d_state, 3 * d_state
            self.gru.weight_hh_l0[i_start:i_end].copy_(torch.eye(d_state))
            self.gru.bias_hh_l0.zero_()
            self.gru.bias_hh_l0[d_state:2*d_state].fill_(5.0) # Locked Memory Bias
            self.gru.bias_ih_l0.zero_()
        self.gru.weight_hh_l0.requires_grad = False
        self.gru.bias_hh_l0.requires_grad = False
    def forward(self, x, h):
        if h is None: h = torch.zeros(1, x.size(0), self.d_state, device=x.device)
        o, h_new = self.gru(x, h)
        return self.out_proj(o), h_new

class ComputeNeuron(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.gru = nn.GRU(d_model, d_state, batch_first=True)
        self.out_proj = nn.Linear(d_state, d_model)
    def forward(self, x, h):
        if h is None: h = torch.zeros(1, x.size(0), self.gru.hidden_size, device=x.device)
        o, h_new = self.gru(x, h)
        return self.out_proj(o), h_new

class INN_HDFS(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.neurons = nn.ModuleList([
            LockedMemoryNeuron(D_MODEL, D_STATE),
            LockedMemoryNeuron(D_MODEL, D_STATE),
            LockedMemoryNeuron(D_MODEL, D_STATE),
            LockedMemoryNeuron(D_MODEL, D_STATE),
            ComputeNeuron(D_MODEL, D_STATE),
            ComputeNeuron(D_MODEL, D_STATE)
        ])
        self.readout = nn.Linear(D_MODEL * 6, 2) 

    def forward(self, x, h=None):
        if h is None: h = [None] * 6
        x_emb = self.embedding(x)
        outs, new_h = [], []
        for i, n in enumerate(self.neurons):
            o, hh = n(x_emb, h[i])
            outs.append(o); new_h.append(hh)
        stack = torch.cat(outs, dim=-1)
        return self.readout(stack), new_h

# --- 2. BASELINES ---
class BaselineLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.lstm = nn.LSTM(D_MODEL, D_STATE, batch_first=True)
        self.readout = nn.Linear(D_STATE, 2)
    def forward(self, x, h=None):
        x = self.embedding(x)
        o, h = self.lstm(x, h)
        return self.readout(o), h

class BaselineTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(128, D_MODEL)
        enc = nn.TransformerEncoderLayer(D_MODEL, 4, D_MODEL*2, batch_first=True)
        self.trans = nn.TransformerEncoder(enc, 1)
        self.lstm = nn.LSTM(D_MODEL, D_STATE, batch_first=True) 
        self.readout = nn.Linear(D_STATE, 2)
    def forward(self, x, h=None):
        if x.size(1) > 128: x = x[:, -128:] 
        x_emb = self.embedding(x) + self.pos(torch.arange(x.size(1), device=x.device))
        x_trans = self.trans(x_emb)
        if h is None:
            h = (torch.zeros(1, x.size(0), D_STATE, device=x.device),
                 torch.zeros(1, x.size(0), D_STATE, device=x.device))
        o, h = self.lstm(x_trans, h)
        return self.readout(o), h

class BaselineMambaSim(nn.Module):
    """
    Simulation Mamba (SSM) via GRU Orthogonal.
    Représente l'état de l'art des modèles récurrents modernes.
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.gru = nn.GRU(D_MODEL, D_STATE, batch_first=True)
        self.readout = nn.Linear(D_STATE, 2)
        # Initialisation Orthogonale (Crucial pour simuler HiPPO/S4)
        for name, p in self.gru.named_parameters():
            if 'weight_hh' in name: nn.init.orthogonal_(p)

    def forward(self, x, h=None):
        x = self.embedding(x)
        o, h = self.gru(x, h)
        return self.readout(o), h

# ==========================================
# TRAINING & BENCHMARKING
# ==========================================

def train_model(model, name, steps=3000): # Augmenté à 3000 steps pour le Curriculum
    print(f"👷 Training {name} on Interleaved HDFS Logs (Curriculum)...")
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    
    history = []
    
    # Curriculum Phases
    # Phase 1: Facile (1 process, peu de bruit) -> Apprendre la logique de base
    # Phase 2: Moyen (2 process, bruit moyen) -> Apprendre l'entrelacement
    # Phase 3: Expert (4 process, bruit fort) -> Robustesse
    
    for step in range(steps):
        # Ajustement dynamique de la difficulté
        if step < 1000:
            concurrent = 1
            noise = 0.1
            phase = "Easy"
        elif step < 2000:
            concurrent = 2
            noise = 0.3
            phase = "Medium"
        else:
            concurrent = 4
            noise = 0.6
            phase = "Hard"
            
        # Instancier le stream avec la difficulté actuelle
        stream = HDFSStream(num_concurrent_blocks=concurrent, noise_ratio=noise)
        x, y, mask = stream.generate_chunk(32, SEQ_LEN)
        
        if name == "INN":
            logits, _ = model(x, None)
        else:
            logits, _ = model(x)
            
        # Renforcement du poids des anomalies pour éviter la paresse (0 Recall)
        # On passe de 5.0 à 15.0 pour forcer le modèle à tenter des détections
        weights = torch.tensor([1.0, 15.0]).to(DEVICE) 
        loss = F.cross_entropy(logits.view(-1, 2), y.view(-1), weight=weights, reduction='none')
        loss = (loss * mask.view(-1)).sum() / (mask.sum() + 1e-6)
        
        opt.zero_grad()
        loss.backward()
        if name == "INN": torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        history.append(loss.item())
        if step % 200 == 0:
            avg = sum(history[-50:]) / 50 if len(history) > 50 else loss.item()
            print(f"  Step {step} [{phase}] | Loss: {avg:.4f}")
            
    return model

def benchmark_model(model, name, horizon=5000):
    print(f"\n🔍 Auditing {name} on {horizon} lines of continuous logs...")
    model.eval()
    stream = HDFSStream(num_concurrent_blocks=4, noise_ratio=0.6)
    
    total_anomalies = 0
    detected_anomalies = 0
    false_positives = 0
    
    h = None
    chunk_size = 32
    num_chunks = horizon // chunk_size
    
    for _ in range(num_chunks):
        x, y, mask = stream.generate_chunk(1, chunk_size)
        
        with torch.no_grad():
            if name == "INN": logits, h = model(x, h)
            else: logits, h = model(x, h)
                
        preds = logits.argmax(dim=-1).view(-1)
        targets = y.view(-1)
        
        for p, t in zip(preds, targets):
            if t == 1:
                total_anomalies += 1
                if p == 1: detected_anomalies += 1
            elif t == 0:
                if p == 1: false_positives += 1
                
    recall = detected_anomalies / (total_anomalies + 1e-6)
    precision = detected_anomalies / (detected_anomalies + false_positives + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    print(f"  📊 Results for {name}:")
    print(f"     Recall: {recall:.1%}")
    print(f"     Precision: {precision:.1%} (FP: {false_positives})")
    print(f"     🏆 F1-Score: {f1:.3f}")
    return f1

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    inn = INN_HDFS()
    lstm = BaselineLSTM()
    trans = BaselineTransformer()
    mamba = BaselineMambaSim()
    
    print(f"INN Params: {count_params(inn):,}")
    print(f"Mamba Params: {count_params(mamba):,}")
    
    train_model(inn, "INN")
    train_model(lstm, "LSTM")
    train_model(trans, "Transformer")
    train_model(mamba, "Mamba (Sim)")
    
    print("\n" + "="*60)
    print("FINAL SHOWDOWN: 10,000 Log Lines Continuous Stream")
    print("="*60)
    
    f1_inn = benchmark_model(inn, "INN", horizon=10000)
    f1_lstm = benchmark_model(lstm, "LSTM", horizon=10000)
    f1_trans = benchmark_model(trans, "Transformer", horizon=10000)
    f1_mamba = benchmark_model(mamba, "Mamba (Sim)", horizon=10000)
