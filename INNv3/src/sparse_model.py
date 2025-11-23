import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# On réutilise le bloc Mamba de v2 qui est sain
from INNv2.src.mamba import MultiMambaBlock

class OptimizedINN(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 d_embed=128,      # Petit embedding
                 d_model=256,      # Core dimension
                 num_neurons=12,   # N=12 (Suffisant & léger)
                 num_layers=4,
                 n_head=4,
                 top_k=4,
                 num_static=4,
                 max_seq_len=256):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.d_model = d_model
        
        # --- 1. EMBEDDING STAGE ---
        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        self.embed_proj = nn.Linear(d_embed, d_model, bias=False)
        
        # Positional Encoding (Simple learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.input_norm = nn.LayerNorm(d_model)
        
        # --- 2. DISTRIBUTION STAGE ---
        # Projette le vecteur de contexte vers N neurones indépendants
        self.neuron_dist = nn.Linear(d_model, num_neurons * d_model)
        
        # --- 3. INN LAYERS ---
        self.layers = nn.ModuleList([
            SparseINNLayer(num_neurons, d_model, n_head, top_k, num_static) 
            for _ in range(num_layers)
        ])
        
        # --- 4. AGGREGATION STAGE ---
        # Mean Pooling (pas de params) + Final Norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # --- 5. OUTPUT STAGE (TIED) ---
        # On inverse la projection d'embedding: 256 -> 128
        self.output_proj = nn.Linear(d_model, d_embed, bias=False)
        # LM Head: 128 -> Vocab (Tied with token_embedding)
        self.lm_head = nn.Linear(d_embed, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights()

    def _init_weights(self):
        # Standard GPT initialization
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.embed_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.neuron_dist.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        
        # Zero bias for stability
        if self.neuron_dist.bias is not None: nn.init.zeros_(self.neuron_dist.bias)

    def forward(self, input_ids):
        B, L = input_ids.shape
        
        # 1. Embedding
        x = self.token_embedding(input_ids) # (B, L, 128)
        x = self.embed_proj(x)              # (B, L, 256)
        x = x + self.pos_embedding[:, :L, :]
        x = self.input_norm(x)
        
        # 2. Distribution
        x = self.neuron_dist(x)             # (B, L, N*D)
        x = x.view(B, L, self.num_neurons, self.d_model)
        x = x.permute(0, 2, 1, 3)           # (B, N, L, D)
        
        # 3. Layers
        for layer in self.layers:
            x = layer(x)
            
        # 4. Aggregation (Mean Pooling over Neurons)
        # (B, N, L, D) -> (B, L, D)
        x = x.mean(dim=1) 
        x = self.final_norm(x)
        
        # 5. Output
        x = self.output_proj(x)             # (B, L, 128)
        logits = self.lm_head(x)            # (B, L, Vocab)
        
        return logits

class SparseINNLayer(nn.Module):
    def __init__(self, num_neurons, d_model, n_head, top_k, num_static):
        super().__init__()
        
        # Mamba (Intra-Neuron)
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = MultiMambaBlock(num_neurons, d_model)
        
        # Attention (Inter-Neuron)
        self.norm2 = nn.LayerNorm(d_model)
        self.comm_attention = SparseNeuronAttention(num_neurons, d_model, n_head, top_k, num_static)
        
        # FFN
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        # 1. Mamba (Pre-Norm)
        # Note: Mamba paper uses Pre-Norm logic usually
        res = x
        x_norm = self.norm1(x) 
        # Appliquer Mamba sur x_norm ??? 
        # INNv2 faisait post-norm. Gardons post-norm pour compatibilité MultiMambaBlock
        # Ou suivons standard Pre-Norm.
        # Le MultiMambaBlock actuel n'a pas de Norm interne.
        # Faisons Pre-Norm standard : x = x + f(norm(x))
        
        x = res + self.mamba(self.norm1(x))
        
        # 2. Attention
        res = x
        x_norm = self.norm2(x)
        
        # Reshape for attention: (B, N, L, D) -> (B*L, N, D)
        B, N, L, D = x.shape
        x_flat = x_norm.permute(0, 2, 1, 3).reshape(B*L, N, D)
        
        comm_out = self.comm_attention(x_flat)
        comm_out = comm_out.reshape(B, L, N, D).permute(0, 2, 1, 3)
        
        x = res + comm_out
        
        # 3. FFN
        res = x
        x = res + self.ffn(self.norm3(x))
        
        return x

class SparseNeuronAttention(nn.Module):
    def __init__(self, num_neurons, d_model, n_head, top_k, num_static):
        super().__init__()
        self.num_neurons = num_neurons
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.top_k = min(top_k, num_neurons)
        self.num_static = num_static
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        if num_static > 0:
            self.static_keys = nn.Parameter(torch.randn(num_static, d_model))

    def forward(self, x):
        B, N, D = x.shape
        Q = self.q_proj(x).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        
        if self.num_static > 0:
            K_stat = self.static_keys.view(1, self.num_static, self.n_head, self.head_dim).transpose(1, 2)
            K_dyn = K[:, :, self.num_static:, :]
            K_stat_exp = K_stat.expand(B, -1, -1, -1)
            K = torch.cat([K_stat_exp, K_dyn], dim=2)
            
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if self.top_k < N:
            top_scores, top_indices = torch.topk(scores, self.top_k, dim=-1)
            mask = torch.full_like(scores, float('-inf'))
            mask.scatter_(-1, top_indices, top_scores)
            scores = mask
            
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        return self.out_proj(out)
