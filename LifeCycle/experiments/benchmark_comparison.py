# @title ⚔️ BENCHMARK COMPARISON: INN vs LSTM vs Transformer
# @markdown Teste les 3 modèles sur les 2 expériences qui ont marché :
# @markdown 1. Calcul Temporel (V5) : Stockage Key-Value
# @markdown 2. Locked Vault (V10.1) : Calcul Récursif A+B

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from collections import OrderedDict

# ==========================================
# CONFIGURATION GLOBALE
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 BENCHMARK COMPARISON on {DEVICE}")

# Fix seed for reproducibility
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Hyperparamètres
D_MODEL = 128
D_STATE = 256
VOCAB_SIZE = 100
SEQ_LEN = 64
N_NEURONS = 4

# Learning rates
LR_EXP1 = 1e-3  # V5 utilisait 1e-3
LR_EXP2_INN = 2e-3  # V10.1 utilisait 2e-3
LR_EXP2_BASELINE = 1e-3  # Plus bas pour LSTM/Transformer

# Tokens pour Exp1
KEY_START = 10
KEY_END = 19
QUERY_TOKEN = 99
NOISE_START_EXP1 = 20

# Tokens pour Exp2
STORE_A_TOKEN = 20
STORE_B_TOKEN = 21
QUERY_TOKEN_EXP2 = 99
QUERY_KEY_EXP2 = 98
NOISE_START_EXP2 = 30

# ==========================================
# ARCHITECTURES INN
# ==========================================

# --- EXPERIMENT 1: INN V5 (Orthogonal Neurons) ---

class OrthoNeuron(nn.Module):
    """Neurone avec initialisation orthogonale pour stabilité."""
    def __init__(self, d_model, d_state):
        super().__init__()
        self.gru = nn.GRU(d_model, d_state, batch_first=True)
        self.out_proj = nn.Linear(d_state, d_model)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)

    def forward(self, x, h_prev):
        if h_prev is None:
            batch_size = x.size(0)
            h_prev = torch.zeros(1, batch_size, self.gru.hidden_size, device=x.device)
        out_gru, h_new = self.gru(x, h_prev)
        return self.out_proj(out_gru), h_new

class GlobalWorkspace(nn.Module):
    """Global Workspace pour communication entre neurones."""
    def __init__(self, d_model, n_neurons):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, neuron_states):
        B, S, N, D = neuron_states.shape
        flat = neuron_states.view(B * S, N, D)
        q = self.query(flat)
        k = self.key(flat)
        v = self.value(flat)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D), dim=-1)
        out = self.output(torch.matmul(attn, v))
        return self.norm(out + flat).view(B, S, N, D)

class INN_V5(nn.Module):
    """INN V5: 4 OrthoNeurons + Global Workspace pour Key-Value Storage."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.neurons = nn.ModuleList([OrthoNeuron(D_MODEL, D_STATE) for _ in range(N_NEURONS)])
        self.workspace = GlobalWorkspace(D_MODEL, N_NEURONS)
        self.readout = nn.Linear(D_MODEL * N_NEURONS, VOCAB_SIZE)

    def forward(self, x, hidden_states=None):
        B, S = x.shape
        x_emb = self.embedding(x)
        if hidden_states is None: 
            hidden_states = [None] * N_NEURONS
        outs, new_h = [], []
        for i, n in enumerate(self.neurons):
            o, h = n(x_emb, hidden_states[i])
            outs.append(o)
            new_h.append(h)
        stack = torch.stack(outs, dim=2)
        integrated = self.workspace(stack)
        return self.readout(integrated.view(B, S, -1)), new_h

# --- EXPERIMENT 2: INN V10.1 (Locked Memory + Compute) ---

class LockedMemoryNeuron(nn.Module):
    """Neurone avec matrice récurrente figée à l'Identité pour rétention parfaite."""
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.gru = nn.GRU(d_model, d_state, batch_first=True)
        self.out_proj = nn.Linear(d_state, d_model)
        
        with torch.no_grad():
            self.gru.weight_hh_l0.zero_()
            i_start, i_end = 2 * d_state, 3 * d_state
            self.gru.weight_hh_l0[i_start:i_end].copy_(torch.eye(d_state))
            self.gru.bias_hh_l0.zero_()
            self.gru.bias_hh_l0[d_state:2*d_state].fill_(5.0)  # +5.0 comme V10.1
            self.gru.bias_ih_l0.zero_()
        
        self.gru.weight_hh_l0.requires_grad = False
        self.gru.bias_hh_l0.requires_grad = False
        
    def forward(self, x, h):
        if h is None: 
            h = torch.zeros(1, x.size(0), self.d_state, device=x.device)
        o, h_new = self.gru(x, h)
        return self.out_proj(o), h_new

class ComputeNeuron(nn.Module):
    """Neurone standard, entièrement entraînable, dédié au calcul."""
    def __init__(self, d_model, d_state):
        super().__init__()
        self.gru = nn.GRU(d_model, d_state, batch_first=True)
        self.out_proj = nn.Linear(d_state, d_model)
        
        # Initialisation orthogonale pour la stabilité
        for name, param in self.gru.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
                
    def forward(self, x, h):
        if h is None: 
            h = torch.zeros(1, x.size(0), self.gru.hidden_size, device=x.device)
        o, h_new = self.gru(x, h)
        return self.out_proj(o), h_new

class INN_V10_1(nn.Module):
    """INN V10.1: 2 Locked Memory + 2 Compute Neurons pour calcul récursif."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.neurons = nn.ModuleList([
            LockedMemoryNeuron(D_MODEL, D_STATE),
            LockedMemoryNeuron(D_MODEL, D_STATE),
            ComputeNeuron(D_MODEL, D_STATE),
            ComputeNeuron(D_MODEL, D_STATE)
        ])
        self.fusion_net = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL), 
            nn.GELU(), 
            nn.Linear(D_MODEL, D_MODEL)
        )
        self.q = nn.Linear(D_MODEL, D_MODEL)
        self.k = nn.Linear(D_MODEL, D_MODEL)
        self.v = nn.Linear(D_MODEL, D_MODEL)
        self.w_out = nn.Linear(D_MODEL, D_MODEL)
        self.norm = nn.LayerNorm(D_MODEL)
        self.readout = nn.Linear(D_MODEL * N_NEURONS, VOCAB_SIZE)

    def run_workspace(self, stack):
        B, S, N, D = stack.shape
        flat = stack.view(B*S, N, D)
        q = self.q(flat)
        k = self.k(flat)
        v = self.v(flat)
        attn = F.softmax(torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(D), dim=-1)
        mixed = torch.matmul(attn, v)
        processed = self.fusion_net(mixed)
        out = self.norm(flat + self.w_out(processed))
        return out.view(B, S, N, D)

    def forward(self, x, h=None):
        if h is None: 
            h = [None] * N_NEURONS
        x_emb = self.embedding(x)
        outs, new_h = [], []
        for i, n in enumerate(self.neurons):
            o, hh = n(x_emb, h[i])
            outs.append(o)
            new_h.append(hh)
        stack = torch.stack(outs, dim=2)
        integ = self.run_workspace(stack)
        return self.readout(integ.view(x.size(0), x.size(1), -1)), new_h

# ==========================================
# BASELINES (LSTM & Transformer)
# ==========================================

class LSTM_Baseline(nn.Module):
    """LSTM baseline pour comparaison."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.lstm = nn.LSTM(D_MODEL, 512, batch_first=True)
        self.head = nn.Linear(512, VOCAB_SIZE)

    def forward(self, x, h=None):
        x = self.embedding(x)
        out, h = self.lstm(x, h)
        return self.head(out), h

class Transformer_Baseline(nn.Module):
    """Transformer baseline pour comparaison."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_encoder = nn.Embedding(512, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, 
            nhead=4, 
            dim_feedforward=D_MODEL*4, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x, h=None):
        B, S = x.shape
        if S > 512:
            x = x[:, -512:]
            S = 512
        x = self.embedding(x)
        positions = torch.arange(0, S, device=x.device).unsqueeze(0)
        x = x + self.pos_encoder(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(S).to(x.device)
        out = self.transformer(x, mask=mask, is_causal=True)
        return self.head(out), None

# ==========================================
# EXPERIMENT 1: KEY-VALUE STORAGE
# ==========================================

class KeyValueStream:
    """Générateur de flux pour Key-Value Storage (V5 Protocol)."""
    def generate_chunk(self, batch_size, seq_len):
        x = torch.randint(NOISE_START_EXP1, VOCAB_SIZE, (batch_size, seq_len))
        y = x.clone()
        mask = torch.zeros_like(x).float()
        
        for b in range(batch_size):
            k = random.randint(KEY_START, KEY_END)
            v = random.randint(1, 9)
            x[b, 5], x[b, 6] = k, v  # Store early
            qp = random.randint(20, seq_len - 2)
            x[b, qp], x[b, qp+1] = QUERY_TOKEN, k  # Query
            y[b, qp+1] = v
            mask[b, qp+1] = 1.0
        
        return x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)

def train_exp1(model, name, num_steps=1501):
    """Entraînement pour Exp1: Key-Value Storage."""
    print(f"🏋️ Training {name} on Exp1 (Key-Value Storage)...")
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR_EXP1)
    stream = KeyValueStream()
    
    for step in range(num_steps):
        x, y, mask = stream.generate_chunk(32, SEQ_LEN)
        logits, _ = model(x, None)
        
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), reduction='none')
        loss = (loss * mask.view(-1)).sum() / (mask.sum() + 1e-6)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if step % 100 == 0:
            print(f"  Step {step} | Loss: {loss.item():.4f}")
            if loss.item() < 0.01:
                print("  ✅ Converged.")
                break
    
    return model

def benchmark_exp1(model, name, horizons=[2000, 5000, 10000]):
    """Benchmark pour Exp1: Key-Value Storage."""
    print(f"\n🧐 Testing {name} on Exp1...")
    model.eval()
    results = OrderedDict()
    
    for H in horizons:
        key, val = 12, 7
        h = None
        
        # Store
        _, h = model(torch.tensor([[key, val]]).to(DEVICE), h)
        
        # Wait (chunks of 100)
        for _ in range(H // 100):
            noise = torch.randint(NOISE_START_EXP1, QUERY_TOKEN, (1, 100)).to(DEVICE)
            _, h = model(noise, h)
        
        # Query
        logits, _ = model(torch.tensor([[QUERY_TOKEN, key]]).to(DEVICE), h)
        
        pred = logits[0, -1].argmax().item()
        conf = F.softmax(logits[0, -1], dim=0)[pred].item()
        success = (pred == val)
        res = "✅" if success else f"❌ ({pred})"
        print(f"  H={H} | {res} | Conf: {conf:.1%}")
        results[H] = success
    
    return results

# ==========================================
# EXPERIMENT 2: A+B CALCULATION
# ==========================================

class RecursiveCalculationStream:
    """Générateur de flux pour Calcul Récursif A+B (V10.1 Protocol)."""
    def generate_chunk(self, batch_size, seq_len):
        x = torch.randint(NOISE_START_EXP2, VOCAB_SIZE, (batch_size, seq_len))
        y = x.clone()
        mask = torch.zeros_like(x).float()
        
        for b in range(batch_size):
            va = random.randint(0, 9)
            vb = random.randint(0, 9)
            
            # Store A
            pos_a = random.randint(0, seq_len // 4)
            x[b, pos_a], x[b, pos_a + 1] = STORE_A_TOKEN, va
            
            # Store B
            pos_b = random.randint(seq_len // 3, 2 * seq_len // 3)
            x[b, pos_b], x[b, pos_b + 1] = STORE_B_TOKEN, vb
            
            # Query
            q = random.randint(3 * seq_len // 4, seq_len - 2)
            x[b, q], x[b, q+1] = QUERY_TOKEN_EXP2, QUERY_KEY_EXP2
            y[b, q+1] = va + vb
            mask[b, q+1] = 1.0
        
        return x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)

def train_exp2(model, name, num_steps=1001):
    """Entraînement pour Exp2: A+B Calculation."""
    print(f"🏋️ Training {name} on Exp2 (A+B Calculation)...")
    model.to(DEVICE)
    
    if name == "INN":
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        lr = LR_EXP2_INN
    else:
        trainable_params = list(model.parameters())
        lr = LR_EXP2_BASELINE
    
    opt = optim.Adam(trainable_params, lr=lr)
    stream = RecursiveCalculationStream()
    min_steps = 600  # V10.1 original a fait 600 steps
    
    for step in range(num_steps):
        x, y, mask = stream.generate_chunk(32, SEQ_LEN)
        logits, _ = model(x, None)
        
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), reduction='none')
        loss = (loss * mask.view(-1)).sum() / (mask.sum() + 1e-6)
        opt.zero_grad()
        loss.backward()
        
        # Gradient clipping pour stabilité (surtout pour INN)
        if name == "INN":
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        
        opt.step()
        
        if step % 200 == 0:
            print(f"  Step {step} | Loss: {loss.item():.4f}")
        
        # Arrêter seulement après min_steps ET loss < 0.05
        if step >= min_steps and loss.item() < 0.05:
            print(f"  ✅ Converged at step {step} with loss {loss.item():.4f}")
            break
    
    return model

def benchmark_exp2(model, name, horizons=[2000, 5000, 10000]):
    """Benchmark pour Exp2: A+B Calculation."""
    print(f"\n🧐 Testing {name} on Exp2...")
    model.eval()
    results = OrderedDict()
    
    for H in horizons:
        va, vb, tgt = 4, 3, 7
        h = None
        
        # Store A
        _, h = model(torch.tensor([[STORE_A_TOKEN, va]]).to(DEVICE), h)
        
        # Wait
        for _ in range(H // 200):
            noise = torch.randint(NOISE_START_EXP2, QUERY_TOKEN_EXP2, (1, 100)).to(DEVICE)
            _, h = model(noise, h)
        
        # Store B
        _, h = model(torch.tensor([[STORE_B_TOKEN, vb]]).to(DEVICE), h)
        
        # Wait
        for _ in range(H // 200):
            noise = torch.randint(NOISE_START_EXP2, QUERY_TOKEN_EXP2, (1, 100)).to(DEVICE)
            _, h = model(noise, h)
        
        # Query
        logits, _ = model(torch.tensor([[QUERY_TOKEN_EXP2, QUERY_KEY_EXP2]]).to(DEVICE), h)
        
        pred = logits[0, -1].argmax().item()
        conf = F.softmax(logits[0, -1], dim=0)[pred].item()
        success = (pred == tgt)
        res = "✅" if success else f"❌ ({pred})"
        print(f"  H={H} | {res} | Conf: {conf:.1%}")
        results[H] = success
    
    return results

# ==========================================
# EXECUTION PRINCIPALE
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENT 1: KEY-VALUE STORAGE (V5 Protocol)")
    print("=" * 60)
    
    # Train all models
    inn1 = INN_V5()
    lstm1 = LSTM_Baseline()
    trans1 = Transformer_Baseline()
    
    train_exp1(inn1, "INN")
    train_exp1(lstm1, "LSTM")
    train_exp1(trans1, "Transformer")
    
    # Benchmark all models
    print("\n" + "=" * 60)
    print("RESULTS EXPERIMENT 1")
    print("=" * 60)
    r_inn1 = benchmark_exp1(inn1, "INN")
    r_lstm1 = benchmark_exp1(lstm1, "LSTM")
    r_trans1 = benchmark_exp1(trans1, "Transformer")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: RECURSIVE CALCULATION (V10.1 Protocol)")
    print("=" * 60)
    
    # Train all models (new instances)
    inn2 = INN_V10_1()
    lstm2 = LSTM_Baseline()
    trans2 = Transformer_Baseline()
    
    train_exp2(inn2, "INN")
    train_exp2(lstm2, "LSTM")
    train_exp2(trans2, "Transformer")
    
    # Benchmark all models
    print("\n" + "=" * 60)
    print("RESULTS EXPERIMENT 2")
    print("=" * 60)
    r_inn2 = benchmark_exp2(inn2, "INN")
    r_lstm2 = benchmark_exp2(lstm2, "LSTM")
    r_trans2 = benchmark_exp2(trans2, "Transformer")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print("\nExp1 (Key-Value Storage):")
    print("  Horizon | INN | LSTM | Transformer")
    for H in [2000, 5000, 10000]:
        print(f"  {H:6d}  | {'✅' if r_inn1.get(H) else '❌':3s} | {'✅' if r_lstm1.get(H) else '❌':4s} | {'✅' if r_trans1.get(H) else '❌':10s}")
    
    print("\nExp2 (A+B Calculation):")
    print("  Horizon | INN | LSTM | Transformer")
    for H in [2000, 5000, 10000]:
        print(f"  {H:6d}  | {'✅' if r_inn2.get(H) else '❌':3s} | {'✅' if r_lstm2.get(H) else '❌':4s} | {'✅' if r_trans2.get(H) else '❌':10s}")
