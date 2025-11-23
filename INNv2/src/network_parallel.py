import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba import MultiMambaBlock

class ParallelINN(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 num_neurons=64, 
                 d_model=64, 
                 num_layers=4,
                 n_head=4):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.d_model = d_model
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Projection d'entrée différenciée par neurone (Brise la symétrie)
        # Transforme l'embedding global en une entrée spécifique pour chaque neurone
        self.input_proj = nn.Linear(d_model, num_neurons * d_model)
        
        # Stack
        self.layers = nn.ModuleList([
            INNLayer(num_neurons, d_model, n_head) for _ in range(num_layers)
        ])
        
        # Output
        # Modification: On utilise TOUS les neurones pour la sortie pour garantir le flux de gradient
        self.out_proj = nn.Linear(num_neurons * d_model, vocab_size)
        self.norm_f = nn.LayerNorm(d_model)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # 1. Embedding (B, L, D)
        x = self.embedding(input_ids)
        
        # 2. Projection Différenciée
        x = self.input_proj(x)
        x = x.view(batch_size, seq_len, self.num_neurons, self.d_model)
        x = x.permute(0, 2, 1, 3)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        
        # 4. Output Massive (Tous les neurones)
        # (B, N, L, D) -> (B, L, N, D) -> (B, L, N*D)
        out_flat = x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        logits = self.out_proj(out_flat)
        return logits

class INNLayer(nn.Module):
    def __init__(self, num_neurons, d_model, n_head):
        super().__init__()
        
        self.mamba = MultiMambaBlock(num_neurons, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.comm_attention = MultiHeadNeuronAttention(num_neurons, d_model, n_head)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, N, L, D = x.shape
        
        # 1. Mamba (Temps - Intra Neuron)
        res = x
        x = self.mamba(x)
        x = self.norm1(x + res)
        
        # 2. Attention (Espace - Inter Neuron)
        res = x
        # (B, N, L, D) -> (B, L, N, D) -> (B*L, N, D)
        x_flat = x.permute(0, 2, 1, 3).reshape(B*L, N, D)
        
        comm_out = self.comm_attention(x_flat)
        
        # (B*L, N, D) -> (B, L, N, D) -> (B, N, L, D)
        comm_out = comm_out.reshape(B, L, N, D).permute(0, 2, 1, 3)
        x = self.norm2(comm_out + res)
        
        # 3. FFN (Calcul)
        res = x
        x = self.ffn(x)
        x = self.norm3(x + res)
        
        return x

class MultiHeadNeuronAttention(nn.Module):
    def __init__(self, num_neurons, d_model, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        
    def forward(self, x):
        # x: (Batch*L, Num_Neurons, D_Model)
        # Attention is applied over the Num_Neurons dimension (as sequence length)
        attn_out, _ = self.attn(x, x, x)
        return attn_out
