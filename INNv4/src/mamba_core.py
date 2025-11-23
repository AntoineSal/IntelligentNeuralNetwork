import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MambaCore(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        """
        MambaCore: Le moteur de calcul central d'une Colonie.
        Traite un flux vectoriel unique (d_model) avec haute efficacité.
        """
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state
        self.d_conv = d_conv

        # Projections (Standard Mamba)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        x: (Batch, Length, D_Model) -> Output: (Batch, Length, D_Model)
        """
        B, L, D = x.shape
        
        # 1. In Proj
        x_and_res = self.in_proj(x) # (B, L, 2*Inner)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = x.transpose(1, 2) # (B, Inner, L)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2) # (B, L, Inner)
        x = self.act(x)

        # 2. SSM Parameters
        x_dbl = self.x_proj(x) # (B, L, Rank+2*State)
        (dt, B_ssm, C_ssm) = x_dbl.split(split_size=[self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj(dt) # (B, L, Inner)
        
        # 3. Selective Scan
        # On utilise une version séquentielle simple pour la stabilité (ou custom CUDA si dispo)
        # Ici, version python stable
        y = self.selective_scan(x, dt, -torch.exp(self.A_log), B_ssm, C_ssm, self.D)
        
        y = y * self.act(res)
        out = self.out_proj(y)
        
        return out

    def selective_scan(self, u, dt, A, B, C, D):
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[1]
        
        h = torch.zeros(batch, d_inner, d_state, device=u.device)
        ys = []
        
        dt = F.softplus(dt)
        # Clamp pour stabilité
        dt = torch.clamp(dt, max=10.0)
        
        # Discretization pre-calculation (Approximation pour vitesse Python)
        # dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)) # (B, L, D, N)
        
        # Boucle temporelle (Le goulot en Python, mais stable)
        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1) # (B, D, 1)
            dA = torch.exp(dt_t * A) # (B, D, N)
            dB = dt_t * B[:, t, :].unsqueeze(1) # (B, D, N)
            
            u_t = u[:, t, :].unsqueeze(-1) # (B, D, 1)
            
            h = dA * h + dB * u_t
            
            y_t = torch.sum(h * C[:, t, :].unsqueeze(1), dim=-1) # (B, D)
            y_t = y_t + D * u[:, t, :]
            ys.append(y_t)
            
        return torch.stack(ys, dim=1)

