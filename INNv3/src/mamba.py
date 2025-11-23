import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiMambaBlock(nn.Module):
    def __init__(self, num_neurons, d_model, d_state=16, d_conv=4, expand=2):
        """
        MultiMambaBlock: Version vectorisée pour gérer N neurones indépendants en parallèle.
        Adapté de INNv2.
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state
        self.d_conv = d_conv

        # 1. Projection d'entrée (Input -> Inner x 2)
        self.w_in = nn.Parameter(torch.Tensor(num_neurons, d_model, self.d_inner * 2))
        
        # 2. Convolution Locale (1D Depthwise par neurone)
        self.conv1d = nn.Conv1d(
            in_channels=num_neurons * self.d_inner,
            out_channels=num_neurons * self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=num_neurons * self.d_inner, 
            padding=d_conv - 1,
        )

        # 3. Paramètres SSM
        self.w_x = nn.Parameter(torch.Tensor(num_neurons, self.d_inner, self.dt_rank + d_state * 2))
        
        self.w_dt = nn.Parameter(torch.Tensor(num_neurons, self.dt_rank, self.d_inner))
        self.b_dt = nn.Parameter(torch.Tensor(num_neurons, self.d_inner))

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(num_neurons, self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) 
        self.D = nn.Parameter(torch.ones(num_neurons, self.d_inner))

        # 4. Output Proj
        self.w_out = nn.Parameter(torch.Tensor(num_neurons, self.d_inner, d_model))
        
        self.act = nn.SiLU()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w_in)
        nn.init.xavier_uniform_(self.w_x)
        nn.init.xavier_uniform_(self.w_dt)
        nn.init.zeros_(self.b_dt)
        nn.init.xavier_uniform_(self.w_out)

    def forward(self, x):
        """
        x: (Batch, Neurons, Length, D_Model)
        """
        batch_size, num_neurons, seq_len, d_model = x.shape
        
        # 1. In Proj
        x_and_res = torch.einsum('bnld,ndk->bnlk', x, self.w_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        # 2. Conv
        x = x.permute(0, 1, 3, 2).reshape(batch_size, num_neurons * self.d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.reshape(batch_size, num_neurons, self.d_inner, seq_len).permute(0, 1, 3, 2) 
        x = self.act(x)

        # 3. SSM
        x_dbl = torch.einsum('bnli,nip->bnlp', x, self.w_x)
        (dt, B, C) = x_dbl.split(split_size=[self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = torch.einsum('bnlr,nri->bnli', dt, self.w_dt) + self.b_dt.view(1, num_neurons, 1, -1)
        
        A = -torch.exp(self.A_log.float()) 
        
        y = self.selective_scan(x, dt, A, B, C, self.D)
        
        # 4. Output
        y = y * self.act(res)
        out = torch.einsum('bnli,nid->bnld', y, self.w_out)
        
        return out

    def selective_scan(self, u, dt, A, B, C, D):
        """
        Multi-Neuron Scan (Python Ref).
        """
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

