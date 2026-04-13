import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Intelligent Neural Network (INN) - v2 Implementation
Official Code for "Intelligent Neural Networks: From Layered Architectures to Graph-Organized Intelligence"

Key Components:
1. IntelligentNeuron: A combination of Selective SSM (Mamba) for internal memory and Attention for communication.
2. INN: The graph-based architecture orchestrating these neurons.

Note: This implementation uses a JIT-compiled SSM kernel for portability (runs on CPU/CUDA without custom compilation).
For production use, consider replacing `ssm_jit` with `mamba_ssm` kernels.
"""

@torch.jit.script
def ssm_jit(x, dt, A, B, C, D):
    """
    Minimal Selective State-Space Model kernel (JIT compiled).
    """
    dt = torch.clamp(dt, max=2.5) # Stability clamp for long sequences
    dA = torch.exp(torch.einsum('bld,ds->blds', dt, A))
    dB = dt.unsqueeze(-1) * B.unsqueeze(2)
    
    h = torch.zeros(x.size(0), x.size(2), A.size(1), device=x.device, dtype=x.dtype)
    ys = []
    
    for t in range(x.size(1)):
        h = dA[:, t, :, :] * h + dB[:, t, :, :] * x[:, t, :].unsqueeze(-1)
        y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)
        ys.append(y_t)
        
    y = torch.stack(ys, dim=1)
    return y + x * D

class IntelligentNeuronBlock(nn.Module):
    """
    Represents the internal dynamics of a population of Intelligent Neurons.
    Parallelized implementation using Grouped Convolutions and Shared Projections.
    """
    def __init__(self, num_neurons, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state
        
        # Shared projections for efficiency (parameter sharing across neurons is possible but here we use distinct weights per neuron via grouping if needed, or shared)
        # In this version: Weights are standard Mamba-like, but conceptually applied per neuron stream.
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution is grouped to maintain neuron independence in the local mixing phase
        self.conv1d = nn.Conv1d(
            num_neurons * self.d_inner, 
            num_neurons * self.d_inner, 
            bias=True, 
            kernel_size=d_conv, 
            groups=num_neurons * self.d_inner, 
            padding=d_conv - 1
        )
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # S4 Parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Neurons, SeqLen, Dim)
        B, N, L, D = x.shape
        
        # 1. Projection
        x_and_res = self.in_proj(x)
        (x_in, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        
        # 2. Local Conv (Per neuron, Per channel)
        x_conv = x_in.permute(0, 1, 3, 2).reshape(B, N*self.d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L].reshape(B, N, self.d_inner, L).permute(0, 1, 3, 2)
        x_conv = F.silu(x_conv)
        
        # 3. SSM (Selective Scan)
        x_flat = x_conv.reshape(B*N, L, self.d_inner)
        dt_rank_state = self.x_proj(x_flat)
        dt, B_ssm, C_ssm = torch.split(dt_rank_state, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        
        y = ssm_jit(x_flat, dt, A, B_ssm, C_ssm, self.D)
        y = y.reshape(B, N, L, self.d_inner)
        
        # 4. Output
        return self.dropout(self.out_proj(y * F.silu(res)))

class IntelligentNeuralNetwork(nn.Module):
    """
    The INN Architecture: A stack of Intelligent Neuron layers connected by a learned communication graph.
    """
    def __init__(self, vocab_size, num_neurons=16, d_model=256, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_neurons = num_neurons
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # The Neural Graph Layers
        self.layers = nn.ModuleList([nn.ModuleList([
            # Internal Intelligence (When to fire)
            IntelligentNeuronBlock(num_neurons, d_model, dropout=dropout),
            # Communication Intelligence (To whom to fire)
            nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True),
            # Normalization Layers (Pre-Norm Architecture)
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model)
        ]) for _ in range(num_layers)])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        # Weight Tying (Optional but recommended)
        self.head.weight = self.embedding.weight

    def forward(self, x):
        # x: (Batch, SeqLen)
        B, L = x.shape
        
        # Embed and Replicate: Each neuron gets a copy of the input
        x = self.embedding(x) # (B, L, D)
        x = x.unsqueeze(1).expand(-1, self.num_neurons, -1, -1).contiguous() # (B, N, L, D)
        
        for mamba, attn, norm1, norm2 in self.layers:
            # 1. Internal Dynamics (Parallel Mamba with Pre-Norm)
            x_norm = norm1(x)
            x = x + mamba(x_norm)
            
            # 2. Inter-Neuron Communication (Attention with Pre-Norm)
            x_norm2 = norm2(x)
            # Reshape to (Batch*SeqLen, Neurons, Dim) to apply attention over Neurons
            x_flat = x_norm2.permute(0, 2, 1, 3).reshape(B*L, self.num_neurons, -1)
            
            # Self-Attention over the 'Neuron' dimension
            comm_out, _ = attn(x_flat, x_flat, x_flat)
            
            # Reshape back and Add Residual
            x = x + comm_out.view(B, L, self.num_neurons, -1).permute(0, 2, 1, 3)
            
        # Aggregation (Mean Pooling over Neurons)
        out = self.norm_f(x.mean(dim=1))
        
        return self.head(out)

if __name__ == "__main__":
    # Simple Smoke Test
    model = IntelligentNeuralNetwork(vocab_size=100, num_neurons=8, d_model=64, num_layers=2)
    x = torch.randint(0, 100, (2, 32))
    y = model(x)
    print(f"Model created. Output shape: {y.shape}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

