import torch
import torch.nn as nn
import math

class NeuronTokensINN(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 d_model=256, 
                 n_neurons=16, 
                 n_layers=4, 
                 n_head=4, 
                 dropout=0.1,
                 max_seq_len=512):
        super().__init__()
        
        self.d_model = d_model
        self.n_neurons = n_neurons
        
        # 1. Embedding & Positional Encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len + n_neurons)
        
        # 2. Neuron Tokens (Les agents persistants)
        self.neuron_tokens = nn.Parameter(torch.randn(1, n_neurons, d_model) * 0.02)
        
        # 3. Transformer Backbone (Cœur du système)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. Aggregation (Sequence queries Neurons)
        self.neuron_aggregation = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )
        
        # 5. Output Head
        self.norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

    def forward(self, input_ids):
        # input_ids: (B, L)
        B, L = input_ids.shape
        
        # 1. Embed Sequence
        x_seq = self.embedding(input_ids) # (B, L, D)
        
        # 2. Expand Neurons
        x_neurons = self.neuron_tokens.expand(B, -1, -1)
        
        # 3. Concatenate: [Neurons | Sequence]
        x_full = torch.cat([x_neurons, x_seq], dim=1) # (B, N+L, D)
        
        # 4. Positional Encoding
        x_full = self.pos_encoder(x_full)
        
        # 5. Custom Mask (Hybrid Graph/Causal)
        mask = self._generate_inn_mask(self.n_neurons, L).to(input_ids.device)
        
        # 6. Transformer Pass
        # is_causal=False car le masque est custom et fourni manuellement
        x_out = self.transformer(x_full, mask=mask)
        
        # 7. Split
        neurons_out = x_out[:, :self.n_neurons, :] # (B, N, D)
        seq_out = x_out[:, self.n_neurons:, :]     # (B, L, D)
        
        # 8. Aggregate: Sequence queries Neurons
        # Cela permet à chaque token de demander "qu'est-ce que les neurones savent ?"
        aggregated, _ = self.neuron_aggregation(
            query=seq_out,
            key=neurons_out,
            value=neurons_out
        )
        
        # 9. Residual & Output
        x_final = seq_out + aggregated
        x_final = self.norm_final(x_final)
        logits = self.lm_head(x_final)
        
        return logits

    def _generate_inn_mask(self, n_neurons, seq_len):
        """
        Masque Hybride Booléen (Plus stable pour PyTorch SDPA)
        False = Visible, True = Masqué
        """
        total_len = n_neurons + seq_len
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)
        
        # Règle 1: Les neurones ne voient pas les tokens (Look-ahead prevention)
        mask[0:n_neurons, n_neurons:] = True
        
        # Règle 2: Tokens Causal (Triangulaire)
        token_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        mask[n_neurons:, n_neurons:] = token_mask
        
        return mask

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
