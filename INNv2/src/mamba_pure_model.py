import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# On réutilise la fonction de scan exact de votre implémentation pour une comparaison équitable
def selective_scan_ref(u, dt, A, B, C, D):
    """
    Standard Selective Scan (Single Channel / Monolithic).
    u: (B, L, d_inner)
    dt: (B, L, d_inner)
    A: (d_inner, d_state)
    B: (B, L, d_state)
    C: (B, L, d_state)
    D: (d_inner)
    """
    batch_size, seq_len, d_inner = u.shape
    d_state = A.shape[1]
    
    h = torch.zeros(batch_size, d_inner, d_state, device=u.device)
    ys = []
    
    # Scan séquentiel (pour la baseline, la vitesse pure n'est pas critique, la justesse mathématique l'est)
    # Note: Une implémentation CUDA optimisée serait plus rapide mais cette version python suffit pour la validation
    for t in range(seq_len):
        dt_t = F.softplus(dt[:, t, :]) # (B, d_inner)
        
        # Discretization
        dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0)) # (B, d_inner, d_state)
        dB = dt_t.unsqueeze(-1) * B[:, t, :].unsqueeze(1)   # (B, d_inner, d_state)
        
        u_t = u[:, t, :].unsqueeze(-1) # (B, d_inner, 1)
        
        # State Update
        h = dA * h + dB * u_t
        
        # Output
        y_t = torch.sum(h * C[:, t, :].unsqueeze(1), dim=-1) # (B, d_inner)
        y_t = y_t + D.unsqueeze(0) * u[:, t, :]
        
        ys.append(y_t)
        
    return torch.stack(ys, dim=1) # (B, L, d_inner)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state
        
        # 1. In Projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 2. Conv1d
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # 3. SSM Parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # S4D initialization for A
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 4. Out Projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, L, D)
        batch_size, seq_len, d_model = x.shape
        
        # 1. Proj
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        
        x = x.permute(0, 2, 1) # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.permute(0, 2, 1) # (B, L, d_inner)
        
        x = self.act(x)
        
        # 2. SSM
        x_dbl = self.x_proj(x) # (B, L, dt_rank + 2*d_state)
        (dt, B, C) = x_dbl.split(split_size=[self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj(dt) # (B, L, d_inner)
        
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)
        
        y = selective_scan_ref(x, dt, A, B, C, self.D)
        
        # 3. Output
        y = y * self.act(res)
        out = self.out_proj(y)
        
        return out

class MambaPure(nn.Module):
    """
    Architecture Mamba Pure : Stack de MambaBlocks.
    Conçue pour matcher le nombre de paramètres de l'INNv2 (~8.6M).
    """
    def __init__(
        self,
        vocab_size,
        d_model=384, # Ajusté pour ~8.6M params (à vérifier au runtime)
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        max_seq_len=128
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                MambaBlock(d_model, d_state, d_conv, expand),
                nn.LayerNorm(d_model)
            )
            for _ in range(n_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        # input_ids: (B, L)
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x) + x # Residual connection classique
            
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        return logits

