import torch
import torch.nn as nn
import math

class NeuronTokensINN(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 d_model=256, 
                 n_neurons=16, 
                 n_layers=6,       # Un peu plus profond pour WikiText
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
        
        # 3. Transformer Backbone (Cœur du système)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=d_model*4,
            dropout=dropout, 
            batch_first=True,
            norm_first=True # Pre-Norm est plus stable pour la convergence
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. Output Head
        self.norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (Crucial pour WikiText avec petit modèle)
        self.lm_head.weight = self.embedding.weight
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        # Les poids sont liés, donc lm_head est déjà init via embedding
        # Init des neurones
        nn.init.normal_(self.neuron_tokens, mean=0.0, std=0.02)

    def forward(self, input_ids):
        # input_ids: (B, L)
        B, L = input_ids.shape
        
        # 1. Embed Sequence
        x_seq = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # 2. Expand Neurons
        x_neurons = self.neuron_tokens.expand(B, -1, -1)
        
        # 3. Concatenate: [Neurons | Sequence]
        x_full = torch.cat([x_neurons, x_seq], dim=1) # (B, N+L, D)
        
        # 4. Positional Encoding
        x_full = self.pos_encoder(x_full)
        
        # 5. Custom Mask (Hybrid Graph/Causal)
        # ATTENTION: PyTorch attend un masque BOOLÉEN où True = Masqué (Ignoré)
        mask = self._generate_prefix_causal_mask(self.n_neurons, L).to(input_ids.device)
        
        # 6. Transformer Pass
        x_out = self.transformer(x_full, mask=mask)
        
        # 7. Extract Sequence Part only
        # On ignore les N premiers tokens (les neurones mis à jour)
        x_seq_out = x_out[:, self.n_neurons:, :]
        
        x_seq_out = self.norm_final(x_seq_out)
        logits = self.lm_head(x_seq_out)
        
        return logits

    def _generate_prefix_causal_mask(self, n_neurons, seq_len):
        """
        Masque Booléen (True = Ignoré/Masqué).
        - Block [0:N, 0:N] (Neurons->Neurons) : False (Visible)
        - Block [0:N, N:] (Neurons->Tokens)   : True (Masqué, pas de lookahead)
        - Block [N:, 0:N] (Tokens->Neurons)   : False (Visible, mémoire)
        - Block [N:, N:] (Tokens->Tokens)     : Causal (Triangulaire)
        """
        total_len = n_neurons + seq_len
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)
        
        # 1. Neurones ne voient pas le futur (Tokens)
        mask[0:n_neurons, n_neurons:] = True
        
        # 2. Tokens Causal
        # triu(..., 1) met des True au-dessus de la diagonale -> Masqué
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

