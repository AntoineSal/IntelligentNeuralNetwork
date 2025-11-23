import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .mamba import MultiMambaBlock
from torch.utils.checkpoint import checkpoint

class AdaptiveRouter(nn.Module):
    def __init__(self, vocab_size, num_neurons, d_model):
        super().__init__()
        self.neuron_keys = nn.Parameter(torch.randn(num_neurons, d_model))
        self.token_router = nn.Embedding(vocab_size, num_neurons)
        self.gate = nn.Sigmoid()
        
    def forward(self, input_ids, embeddings):
        routing_logits = self.token_router(input_ids) 
        routing_weights = self.gate(routing_logits)
        neuron_inputs = torch.einsum('bld,bln->bnld', embeddings, routing_weights)
        return neuron_inputs

class MixtureOfVocabularies(nn.Module):
    def __init__(self, d_model, num_experts=4, vocab_size=33278):
        super().__init__()
        self.num_experts = num_experts
        self.vocab_size = vocab_size
        self.vocab_per_expert = math.ceil(vocab_size / num_experts)
        self.expert_decoders = nn.ModuleList([
            nn.Linear(d_model, self.vocab_per_expert) 
            for _ in range(num_experts)
        ])
        self.expert_router = nn.Linear(d_model, num_experts)
        
    def forward(self, hidden_states):
        router_logits = self.expert_router(hidden_states)
        expert_weights = F.softmax(router_logits, dim=-1)
        vocab_logits_list = []
        for i, expert in enumerate(self.expert_decoders):
            logits = expert(hidden_states)
            weight = expert_weights[..., i:i+1]
            weighted_logits = logits * weight
            vocab_logits_list.append(weighted_logits)
        full_logits = torch.cat(vocab_logits_list, dim=-1)
        if full_logits.size(-1) > self.vocab_size:
            full_logits = full_logits[..., :self.vocab_size]
        return full_logits

class NeuronAttention(nn.Module):
    def __init__(self, num_neurons, d_model, n_head=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)

class ProgressiveINNLayer(nn.Module):
    def __init__(self, max_neurons, d_model, initial_neurons=32, dropout=0.1):
        super().__init__()
        self.max_neurons = max_neurons
        self.active_neurons = min(initial_neurons, max_neurons)
        self.d_model = d_model
        self.mamba_block = MultiMambaBlock(max_neurons, d_model)
        self.attention = NeuronAttention(max_neurons, d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) # Dropout for activations
        self.mix_gate = nn.Parameter(torch.tensor(0.0))
        self.highway_proj = nn.Linear(d_model, d_model)

    def grow_network(self, growth_rate=1.2):
        old = self.active_neurons
        self.active_neurons = min(int(self.active_neurons * growth_rate), self.max_neurons)
        if self.active_neurons > old:
            print(f"  -> Layer grown: {old} -> {self.active_neurons} neurons")

    def _forward_impl(self, x, highway):
        B, N, L, D = x.shape
        mask = torch.zeros(self.max_neurons, device=x.device)
        mask[:self.active_neurons] = 1.0
        mask = mask.view(1, -1, 1, 1)
        
        x = x * mask
        res = x
        x = self.mamba_block(x)
        x = self.dropout(x) # Dropout after Mamba
        x = self.norm1(x + res)
        x = x * mask
        
        res = x
        x_flat = x.permute(0, 2, 1, 3).reshape(B*L, N, D)
        x_flat = self.attention(x_flat)
        x = x_flat.reshape(B, L, N, D).permute(0, 2, 1, 3)
        x = self.norm2(x + res)
        x = x * mask
        
        highway_signal = self.highway_proj(highway)
        highway_signal = highway_signal.unsqueeze(1).expand(-1, N, -1, -1)
        gate = torch.sigmoid(self.mix_gate)
        x = gate * x + (1 - gate) * highway_signal
        return x

    def forward(self, x, highway):
        # Use checkpointing to save memory
        if self.training:
            return checkpoint(self._forward_impl, x, highway, use_reentrant=False)
        else:
            return self._forward_impl(x, highway)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class INNv3(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=4, max_neurons=256, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = PositionalEncoding(d_model, dropout=dropout, max_len=512)
        self.router = AdaptiveRouter(vocab_size, num_neurons=max_neurons, d_model=d_model)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_max_neurons = max_neurons
            if i >= 2: layer_max_neurons = max_neurons // 2
            if i >= 4: layer_max_neurons = max_neurons // 4
            self.layers.append(ProgressiveINNLayer(max_neurons=layer_max_neurons, d_model=d_model, initial_neurons=32, dropout=dropout))
        
        self.decoder = MixtureOfVocabularies(d_model, num_experts=4, vocab_size=vocab_size)
        self.highway_norm = nn.LayerNorm(d_model)
        
    def grow_network(self):
        print(">>> Growing Network Capacity...")
        for layer in self.layers:
            layer.grow_network()
            
    def forward(self, input_ids):
        token_emb = self.token_embeddings(input_ids)
        token_emb = self.pos_embeddings(token_emb.transpose(0, 1)).transpose(0, 1)
        highway = token_emb
        x = self.router(input_ids, token_emb)
        
        for i, layer in enumerate(self.layers):
            if x.shape[1] != layer.max_neurons:
                if x.shape[1] > layer.max_neurons:
                    x = x[:, :layer.max_neurons, :, :]
                else:
                    pass
            x = layer(x, highway)
            
        x_pooled = x.mean(dim=1)
        highway_final = self.highway_norm(highway)
        combined = x_pooled + highway_final
        logits = self.decoder(combined)
        return logits
