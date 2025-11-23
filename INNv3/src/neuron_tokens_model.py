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
        # Initialisés comme des vecteurs latents apprenables
        self.neuron_tokens = nn.Parameter(torch.randn(1, n_neurons, d_model) * 0.02)
        
        # 3. Transformer Backbone (Standard & Robuste)
        # batch_first=True est vital pour notre logique
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. Output Head
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
        # (1, N, D) -> (B, N, D)
        x_neurons = self.neuron_tokens.expand(B, -1, -1)
        
        # 3. Concatenate: [Neurons | Sequence]
        # L'ordre est important. Les neurones sont au début, comme un "préfixe mémoire"
        x_full = torch.cat([x_neurons, x_seq], dim=1) # (B, N+L, D)
        
        # 4. Positional Encoding (sur N+L)
        x_full = self.pos_encoder(x_full)
        
        # 5. Causal Masking
        # On veut que token T ne voie que T-1...0 et les Neurones
        # Les Neurones voient-ils tout ? Non, en training causal, on masque le futur pour tout le monde
        mask = self._generate_square_subsequent_mask(self.n_neurons + L).to(input_ids.device)
        
        # 6. Transformer Pass
        x_out = self.transformer(x_full, mask=mask, is_causal=True)
        
        # 7. Extract Sequence Part
        # On ignore les N premiers tokens (les neurones mis à jour)
        # On garde les L derniers (la séquence prédite)
        x_seq_out = x_out[:, self.n_neurons:, :] # (B, L, D)
        
        x_seq_out = self.norm_final(x_seq_out)
        logits = self.lm_head(x_seq_out)
        
        return logits

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
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
        # x: (B, SeqLen, D)
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

