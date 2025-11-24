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
        
        # Stack
        self.layers = nn.ModuleList([
            INNLayer(num_neurons, d_model, n_head) for _ in range(num_layers)
        ])
        
        # Output - ACTION NEURONS ONLY
        # C'est cette config qui a donné le score de 1.16
        self.n_action = max(1, num_neurons // 8)
        self.out_proj = nn.Linear(self.n_action * d_model, vocab_size)
        self.norm_f = nn.LayerNorm(d_model)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # (B, L) -> (B, L, D)
        x = self.embedding(input_ids)
        
        # Expansion simple vers les neurones
        # (B, L, D) -> (B, 1, L, D) -> (B, N, L, D)
        x = x.unsqueeze(1).expand(-1, self.num_neurons, -1, -1)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        
        # Action Neurons (Last N/8)
        action_out = x[:, -self.n_action:, :, :]
        action_out = action_out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        logits = self.out_proj(action_out)
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
        
        # 1. Mamba (Temps)
        res = x
        x = self.mamba(x)
        x = self.norm1(x + res)
        
        # 2. Attention (Espace/Neurones)
        res = x
        x_flat = x.permute(0, 2, 1, 3).reshape(B*L, N, D)
        
        comm_out = self.comm_attention(x_flat)
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
        attn_out, _ = self.attn(x, x, x)
        return attn_out
