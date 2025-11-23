import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# On réutilise le bloc Mamba de v2 qui est sain
from INNv2.src.mamba import MultiMambaBlock

class SparseINN(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 d_model=256,      # Dimension "Thick" (au lieu de 64)
                 num_neurons=16,   # Moins de neurones (au lieu de 64)
                 num_layers=4,
                 n_head=4,
                 top_k=4,          # Sparsité : on n'écoute que 4 voisins
                 num_static=4):    # Ancrages : 4 neurones ont des clés statiques
        super().__init__()
        
        self.num_neurons = num_neurons
        self.d_model = d_model
        self.top_k = top_k
        self.num_static = num_static
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Input Projection: (B, L, D) -> (B, N, L, D)
        # On projette vers N sous-espaces différents
        self.input_proj = nn.Linear(d_model, num_neurons * d_model)
        
        # Layers
        self.layers = nn.ModuleList([
            SparseINNLayer(num_neurons, d_model, n_head, top_k, num_static) 
            for _ in range(num_layers)
        ])
        
        # Output: Tous les neurones contribuent (Consensus)
        self.out_proj = nn.Linear(num_neurons * d_model, vocab_size)
        self.norm_f = nn.LayerNorm(d_model)
        
        # Init
        self._init_weights()

    def _init_weights(self):
        # Init GPT-style pour la stabilité
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.bias)
        
        # Input proj
        nn.init.xavier_uniform_(self.input_proj.weight)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # 1. Embedding
        x = self.embedding(input_ids) # (B, L, D)
        
        # 2. Projection vers les neurones
        x = self.input_proj(x)
        x = x.view(batch_size, seq_len, self.num_neurons, self.d_model)
        x = x.permute(0, 2, 1, 3) # (B, N, L, D)
        
        # 3. Layers
        for layer in self.layers:
            x = layer(x)
            
        # 4. Output
        x = self.norm_f(x)
        x_flat = x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        logits = self.out_proj(x_flat)
        
        return logits

class SparseINNLayer(nn.Module):
    def __init__(self, num_neurons, d_model, n_head, top_k, num_static):
        super().__init__()
        
        # Phase 1: Intra-Neuron (Time Mixing via Mamba)
        self.mamba = MultiMambaBlock(num_neurons, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Phase 2: Inter-Neuron (Space Mixing via Sparse Attention)
        self.comm_attention = SparseNeuronAttention(num_neurons, d_model, n_head, top_k, num_static)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Phase 3: Computation (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, N, L, D)
        
        # 1. Mamba
        res = x
        x = self.mamba(x)
        x = self.norm1(x + res)
        
        # 2. Sparse Attention
        res = x
        # (B, N, L, D) -> (B, L, N, D) -> (B*L, N, D)
        B, N, L, D = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B*L, N, D)
        
        comm_out = self.comm_attention(x_flat)
        
        # (B*L, N, D) -> (B, L, N, D) -> (B, N, L, D)
        comm_out = comm_out.reshape(B, L, N, D).permute(0, 2, 1, 3)
        x = self.norm2(comm_out + res)
        
        # 3. FFN
        res = x
        x = self.ffn(x)
        x = self.norm3(x + res)
        
        return x

class SparseNeuronAttention(nn.Module):
    def __init__(self, num_neurons, d_model, n_head, top_k, num_static):
        super().__init__()
        self.num_neurons = num_neurons
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.top_k = min(top_k, num_neurons) # Sécurité
        self.num_static = num_static
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # STATIC ANCHORS: Clés fixes pour les premiers neurones
        # Ils agissent comme des "topics" fixes que les autres peuvent requêter
        if num_static > 0:
            self.static_keys = nn.Parameter(torch.randn(num_static, d_model))

    def forward(self, x):
        # x: (Batch, N, D) where Batch = RealBatch * SeqLen
        B, N, D = x.shape
        
        Q = self.q_proj(x).view(B, N, self.n_head, self.head_dim).transpose(1, 2) # (B, H, N, Dh)
        K = self.k_proj(x).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        
        # Override K for static neurons (Hybrid Static/Dynamic)
        # Si num_static > 0, les K des premiers neurones sont remplacés par les K statiques
        if self.num_static > 0:
            # On projette les clés statiques pour avoir les têtes
            K_stat = self.static_keys.view(1, self.num_static, self.n_head, self.head_dim).transpose(1, 2)
            # On remplace dans le tenseur K global
            # K[:, :, :self.num_static, :] = K_stat.expand(B, -1, -1, -1) 
            # Attention: In-place modif sur expand peut être risqué pour autograd -> concatenation
            K_dyn = K[:, :, self.num_static:, :]
            K_stat_exp = K_stat.expand(B, -1, -1, -1)
            K = torch.cat([K_stat_exp, K_dyn], dim=2)
            
        # Attention Scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, H, N, N)
        
        # --- SPARSITY: TOP-K GATING ---
        if self.top_k < N:
            # On garde seulement les Top-K valeurs par query
            top_scores, top_indices = torch.topk(scores, self.top_k, dim=-1)
            
            # On crée un masque de -inf (Compatible old PyTorch)
            mask = torch.full_like(scores, float('-inf'))
            
            # On remplit les valeurs aux indices top-k
            mask.scatter_(-1, top_indices, top_scores)
            scores = mask
            
        attn = torch.softmax(scores, dim=-1)
        
        # Aggregation
        out = torch.matmul(attn, V) # (B, H, N, Dh)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        return self.out_proj(out)

