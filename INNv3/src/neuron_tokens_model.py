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
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len + n_neurons)
        
        # Neuron Tokens: Latent agents memory
        self.neuron_tokens = nn.Parameter(torch.randn(1, n_neurons, d_model) * 0.02)
        
        # Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output Head
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
        B, L = input_ids.shape
        
        x_seq = self.embedding(input_ids)
        x_neurons = self.neuron_tokens.expand(B, -1, -1)
        
        # [Neurons | Sequence]
        x_full = torch.cat([x_neurons, x_seq], dim=1)
        x_full = self.pos_encoder(x_full)
        
        # Masque "Prefix Causal"
        mask = self._generate_prefix_causal_mask(self.n_neurons, L).to(input_ids.device)
        
        # Transformer Pass
        x_out = self.transformer(x_full, mask=mask)
        
        # Extract Sequence only (Les neurones ont fait leur job en influençant via l'attention)
        x_seq_out = x_out[:, self.n_neurons:, :]
        
        x_seq_out = self.norm_final(x_seq_out)
        logits = self.lm_head(x_seq_out)
        
        return logits

    def _generate_prefix_causal_mask(self, n_neurons, seq_len):
        """
        Masque Prefix Causal :
        - Les N premiers tokens (Neurones) se voient tous entre eux (Fully Connected Prefix).
        - Les tokens de séquence voient tous les neurones + leur passé causal.
        """
        total_len = n_neurons + seq_len
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)
        
        # 1. Neurones entre eux : Visible (False) -> Déjà fait par zeros
        
        # 2. Neurones vers Tokens : Masqué (True)
        # Les neurones ne doivent pas voir le futur de la séquence, ni même la séquence actuelle
        # pour rester une "mémoire d'état" pure ou un "prompt".
        # MAIS ATTENTION : Si on veut qu'ils apprennent du contexte, ils devraient voir le passé.
        # Ici, on implémente un "Soft Prompt" apprenable statique (Prefix Tuning style).
        # Si on veut une mémoire dynamique, il faudrait une architecture récurrente (segment-level).
        # Pour un modèle causal standard, les neurones sont juste un contexte global constant.
        mask[0:n_neurons, n_neurons:] = True
        
        # 3. Tokens vers Neurones : Visible (False) -> Déjà fait par zeros
        # 4. Tokens vers Tokens : Causal
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
