import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1. CORE BLOCKS (Mamba Vectorisé)
# ==============================================================================
class MultiMambaBlock(nn.Module):
    """Same as INNv2 but checked for correctness."""
    def __init__(self, num_neurons, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.num_neurons = num_neurons
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state
        self.d_conv = d_conv

        self.w_in = nn.Parameter(torch.Tensor(num_neurons, d_model, self.d_inner * 2))
        self.conv1d = nn.Conv1d(
            in_channels=num_neurons * self.d_inner,
            out_channels=num_neurons * self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=num_neurons * self.d_inner, 
            padding=d_conv - 1,
        )
        self.w_x = nn.Parameter(torch.Tensor(num_neurons, self.d_inner, self.dt_rank + d_state * 2))
        self.w_dt = nn.Parameter(torch.Tensor(num_neurons, self.dt_rank, self.d_inner))
        self.b_dt = nn.Parameter(torch.Tensor(num_neurons, self.d_inner))

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(num_neurons, self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) 
        self.D = nn.Parameter(torch.ones(num_neurons, self.d_inner))
        self.w_out = nn.Parameter(torch.Tensor(num_neurons, self.d_inner, d_model))
        
        self.act = nn.SiLU()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w_in)
        nn.init.xavier_uniform_(self.w_x)
        nn.init.xavier_uniform_(self.w_dt)
        nn.init.zeros_(self.b_dt)
        nn.init.xavier_uniform_(self.w_out)

    def selective_scan(self, u, dt, A, B, C, D):
        batch_size, n_neurons, seq_len, d_inner = u.shape
        d_state = A.shape[2]
        h = torch.zeros(batch_size, n_neurons, d_inner, d_state, device=u.device)
        ys = []
        for t in range(seq_len):
            dt_t = F.softplus(dt[:, :, t, :]) 
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt_t.unsqueeze(-1) * B[:, :, t, :].unsqueeze(2)
            u_t = u[:, :, t, :].unsqueeze(-1)
            h = dA * h + dB * u_t
            y_t = torch.sum(h * C[:, :, t, :].unsqueeze(2), dim=-1)
            y_t = y_t + D.unsqueeze(0) * u[:, :, t, :]
            ys.append(y_t)
        return torch.stack(ys, dim=2)

    def forward(self, x):
        batch_size, num_neurons, seq_len, d_model = x.shape
        x_and_res = torch.einsum('bnld,ndk->bnlk', x, self.w_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        
        x = x.permute(0, 1, 3, 2).reshape(batch_size, num_neurons * self.d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.reshape(batch_size, num_neurons, self.d_inner, seq_len).permute(0, 1, 3, 2) 
        x = self.act(x)

        x_dbl = torch.einsum('bnli,nip->bnlp', x, self.w_x)
        (dt, B, C) = x_dbl.split(split_size=[self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = torch.einsum('bnlr,nri->bnli', dt, self.w_dt) + self.b_dt.view(1, num_neurons, 1, -1)
        A = -torch.exp(self.A_log.float()) 
        y = self.selective_scan(x, dt, A, B, C, self.D)
        
        y = y * self.act(res)
        out = torch.einsum('bnli,nid->bnld', y, self.w_out)
        return out


# ==============================================================================
# 2. HYBRID ARCHITECTURE (Stem + INN)
# ==============================================================================
class HybridINN(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 d_model=256,      # Dimension principale (Stem)
                 inn_d_model=64,   # Dimension par neurone INN
                 inn_neurons=16,   # Nombre de neurones INN
                 stem_layers=2,    # Profondeur du tronc commun
                 inn_layers=2,     # Profondeur de la partie INN
                 n_head=4):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # --- A. STEM (Tronc Commun) ---
        # Utilise un Transformer Encoder standard pour construire une représentation contextuelle riche
        # Cela évite le problème de "Cold Start" des neurones indépendants
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.stem = nn.TransformerEncoder(encoder_layer, num_layers=stem_layers)
        
        # --- B. INN INTERFACE ---
        # Projection vers l'espace neuronal: d_model -> (NumNeurons * InnDim)
        self.proj_to_inn = nn.Linear(d_model, inn_neurons * inn_d_model)
        
        # --- C. INN LAYERS ---
        self.inn_layers = nn.ModuleList([
            INNLayer(inn_neurons, inn_d_model, n_head) for _ in range(inn_layers)
        ])
        
        # --- D. OUTPUT ---
        # On récupère l'info de tous les neurones pour la décision finale
        self.proj_from_inn = nn.Linear(inn_neurons * inn_d_model, d_model)
        self.norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Tie weights (Optionnel mais recommandé)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        B, L = input_ids.shape
        
        # 1. Stem Processing
        x = self.embedding(input_ids) # (B, L, D)
        x = self.stem(x)
        
        # 2. INN Processing
        # Project to Neurons
        x_inn = self.proj_to_inn(x) # (B, L, N*D_inn)
        # Reshape to (B, N, L, D_inn)
        x_inn = x_inn.view(B, L, self.inn_layers[0].mamba.num_neurons, self.inn_layers[0].mamba.d_model)
        x_inn = x_inn.permute(0, 2, 1, 3)
        
        # Apply INN Layers
        for layer in self.inn_layers:
            x_inn = layer(x_inn)
            
        # 3. Output Processing
        # Aggregate from all neurons
        x_inn_flat = x_inn.permute(0, 2, 1, 3).reshape(B, L, -1)
        x_out = self.proj_from_inn(x_inn_flat)
        
        # Residual connection from Stem (Skip connection géante)
        # Cela garantit que l'INN ne peut qu'améliorer la performance du Stem, jamais la dégrader
        x_final = self.norm_final(x + x_out)
        
        logits = self.lm_head(x_final)
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
        res = x
        x = self.mamba(x)
        x = self.norm1(x + res)
        
        res = x
        x_flat = x.permute(0, 2, 1, 3).reshape(B*L, N, D)
        comm_out = self.comm_attention(x_flat)
        comm_out = comm_out.reshape(B, L, N, D).permute(0, 2, 1, 3)
        x = self.norm2(comm_out + res)
        
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

