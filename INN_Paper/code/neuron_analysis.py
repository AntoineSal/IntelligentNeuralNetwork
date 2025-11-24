import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import os

# === CONFIGURATION ===
# Must match the training config exactly
CONFIG = {
    'vocab_size': 10000,
    'd_model': 256,
    'num_neurons': 16,
    'num_layers': 4,
    'dropout': 0.3, # Important: match training dropout for loading keys correctly
    'model_path': 'innv2_ptb_regularized.pth' # Adjust filename if needed
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === MODEL DEFINITION (Must match trained model) ===
class MultiMambaBlock(nn.Module):
    def __init__(self, num_neurons, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state # Store d_state
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=num_neurons * self.d_inner,
            out_channels=num_neurons * self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=num_neurons * self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, L, D = x.shape
        x_and_res = self.in_proj(x) 
        (x_in, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        x_conv = self.conv1d(x_in.permute(0, 1, 3, 2).reshape(B, N*self.d_inner, L))[:, :, :L].reshape(B, N, self.d_inner, L).permute(0, 1, 3, 2)
        y = self.ssm(F.silu(x_conv))
        return self.dropout(self.out_proj(y * F.silu(res)))

    def ssm(self, x): # Naive implementation for analysis (easier to debug/load)
        x_flat = x.reshape(-1, x.size(2), x.size(3)) 
        dt_rank_state = self.x_proj(x_flat) 
        dt, B, C = torch.split(dt_rank_state, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)) 
        A = -torch.exp(self.A_log.float()) 
        y = []
        h = torch.zeros(x_flat.size(0), self.d_inner, self.d_state, device=x.device)
        dA = torch.exp(torch.einsum('bld,ds->blds', dt, A))
        dB = torch.einsum('bld,bls->blds', dt, B)
        for t in range(x.size(2)):
            h = dA[:, t, :, :] * h + dB[:, t, :, :] * x_flat[:, t, :].unsqueeze(-1)
            y.append(torch.einsum('bds,bs->bd', h, C[:, t, :]))
        return (torch.stack(y, dim=1) + x_flat * self.D).reshape(x.shape)

class MultiHeadNeuronAttention(nn.Module):
    def __init__(self, num_neurons, d_model, n_head, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x, _ = self.attn(x, x, x)
        return self.norm(res + self.dropout(x))

class ParallelINN(nn.Module):
    def __init__(self, vocab_size, num_neurons, d_model, num_layers, n_head=4, dropout=0.1):
        super().__init__()
        self.num_neurons = num_neurons
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            mamba = MultiMambaBlock(num_neurons, d_model, dropout=dropout)
            attn = MultiHeadNeuronAttention(num_neurons, d_model, n_head, dropout=dropout)
            self.layers.append(nn.ModuleList([mamba, attn]))
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.embedding(input_ids)
        x = self.dropout(x)
        x = x.unsqueeze(1).expand(-1, self.num_neurons, -1, -1).contiguous()
        for mamba, attn in self.layers:
            x = x + mamba(x)
            x_flat = x.permute(0, 2, 1, 3).reshape(B*L, self.num_neurons, -1)
            x_flat = attn(x_flat)
            x = x_flat.view(B, L, self.num_neurons, -1).permute(0, 2, 1, 3)
        out = x.mean(dim=1)
        out = self.norm_f(out)
        return self.head(out)

# === DATA LOADING ===
class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    def __len__(self): return len(self.idx2word)

class Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
    def tokenize(self, path):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words: self.dictionary.add_word(word)
        return None # We just need the dictionary for analysis

# === VISUALIZATION ENGINE ===
def analyze_brain(model, sentence, corpus):
    print(f"\n🧠 Analyzing INNv2 Brain Activity for: '{sentence}'")
    model.eval()
    
    # 1. Prepare Input
    words = sentence.lower().split()
    ids = []
    valid_words = []
    for w in words:
        if w in corpus.dictionary.word2idx:
            ids.append(corpus.dictionary.word2idx[w])
            valid_words.append(w)
        else:
            # Use <unk> if available, else skip
            if '<unk>' in corpus.dictionary.word2idx:
                ids.append(corpus.dictionary.word2idx['<unk>'])
                valid_words.append(f"{w}*")
    
    if not ids:
        print("Error: No valid tokens found.")
        return

    input_tensor = torch.tensor([ids], dtype=torch.long).to(device)
    
    # 2. Hook Activations
    activations = {}
    hooks = []
    
    def get_activation(name):
        def hook(model, input, output):
            # Output is (B, N, L, D) or (B*L, N, D)
            # We want to measure "Activity Level" = Norm of the vector D
            if len(output.shape) == 4: # Mamba output
                act = output.norm(dim=-1).squeeze(0).detach().cpu().numpy() # (N, L)
                activations[name] = act
            elif len(output.shape) == 3: # Attention output (needs reshape)
                B, L = input_tensor.shape
                act = output.view(B, L, -1, output.shape[-1]).permute(0, 2, 1, 3)
                act = act.norm(dim=-1).squeeze(0).detach().cpu().numpy() # (N, L)
                activations[name] = act
        return hook

    # Register hooks on the last layer (most semantic)
    last_layer = model.layers[-1]
    hooks.append(last_layer[0].register_forward_hook(get_activation("Temporal Activity (Mamba)")))
    hooks.append(last_layer[1].register_forward_hook(get_activation("Communication Activity (Attention)")))
    
    # 3. Run
    with torch.no_grad():
        model(input_tensor)
        
    for h in hooks: h.remove()
    
    # 4. Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot Mamba (Individual Thought)
    sns.heatmap(activations["Temporal Activity (Mamba)"], ax=axes[0], cmap="magma", 
                xticklabels=valid_words, yticklabels=[f"N{i}" for i in range(CONFIG['num_neurons'])])
    axes[0].set_title("Figure 7: Neuron Specialization (Temporal Processing)", fontsize=14)
    axes[0].set_xlabel("Sequence Time")
    axes[0].set_ylabel("Neurons")
    
    # Plot Attention (Collective Thought)
    sns.heatmap(activations["Communication Activity (Attention)"], ax=axes[1], cmap="viridis", 
                xticklabels=valid_words, yticklabels=[f"N{i}" for i in range(CONFIG['num_neurons'])])
    axes[1].set_title("Figure 9: Neuron Communication (Spatial Attention)", fontsize=14)
    axes[1].set_xlabel("Sequence Time")
    axes[1].set_ylabel("Neurons")
    
    plt.tight_layout()
    plt.show()
    
    # 5. Correlation Analysis (Figure 8)
    # Do neurons fire together?
    act_matrix = activations["Temporal Activity (Mamba)"] # (N, L)
    corr_matrix = np.corrcoef(act_matrix)
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, 
                xticklabels=[f"N{i}" for i in range(CONFIG['num_neurons'])],
                yticklabels=[f"N{i}" for i in range(CONFIG['num_neurons'])])
    plt.title("Figure 8: Neuron Independence Matrix (Red=Redundant, Blue=Distinct)", fontsize=14)
    plt.show()

if __name__ == "__main__":
    # Load Corpus
    if not os.path.exists("data/ptb"):
        print("Please download data first (run training script)")
    else:
        corpus = Corpus("data/ptb")
        print(f"Dictionary loaded: {len(corpus.dictionary)} words")
        
        # Load Model
        model = ParallelINN(
            CONFIG['vocab_size'], 
            CONFIG['num_neurons'], 
            CONFIG['d_model'], 
            CONFIG['num_layers'], 
            dropout=CONFIG['dropout']
        ).to(device)
        
        try:
            # Try to load weights if file exists
            if os.path.exists(CONFIG['model_path']):
                model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device, weights_only=True))
                print(f"Loaded model from {CONFIG['model_path']}")
            elif os.path.exists("innv2_ptb_word_level.pth"):
                 model.load_state_dict(torch.load("innv2_ptb_word_level.pth", map_location=device, weights_only=True))
                 print(f"Loaded model from innv2_ptb_word_level.pth")
            else:
                print("Warning: No checkpoint found. Using random weights (Analysis will be noise).")
            
            # Run Analysis
            test_sentence = "the market declined despite strong growth in the technology sector"
            analyze_brain(model, test_sentence, corpus)
            
        except Exception as e:
            print(f"Error loading model: {e}")

