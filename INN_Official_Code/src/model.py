import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class IntelligentNeuron(nn.Module):
    """
    A single Intelligent Neuron based on Mamba-SSM.
    Represents a node in the graph with internal memory.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError("Mamba-SSM is required. Run `pip install mamba-ssm`.")
            
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
    
    def forward(self, x):
        # x: [Batch, SeqLen, Dim]
        return self.mamba(x)

class INN(nn.Module):
    """
    Intelligent Neural Network (INN)
    
    Architecture:
    - Graph-based topology where neurons are first-class entities.
    - Neurons have internal state (Mamba).
    - Neurons communicate via dynamic attention mechanisms.
    """
    def __init__(self, vocab_size, d_model=256, n_neurons=32, n_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            # 1. Internal Intelligence (Parallel Mamba Neurons)
            neuron_pop = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            
            # 2. Communication (Multi-Head Attention)
            attn = nn.MultiheadAttention(d_model, 4, dropout=dropout, batch_first=True)
            
            # 3. Normalization (Pre-Norm)
            norm1 = nn.LayerNorm(d_model)
            norm2 = nn.LayerNorm(d_model)
            
            self.layers.append(nn.ModuleList([neuron_pop, attn, norm1, norm2]))
            
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        # Tie weights for efficiency
        self.head.weight = self.embedding.weight
        
        self._last_attn_weights = None # For visualization

    def forward(self, x):
        B, L = x.shape
        
        # 1. Embed & Replicate to Neurons
        x = self.embedding(x) # (B, L, D)
        # Expand: (B, Neurons, L, D) -> Flatten to (B*Neurons, L, D) for parallel Mamba
        x = x.unsqueeze(1).expand(-1, self.n_neurons, -1, -1).reshape(B*self.n_neurons, L, -1)
        
        for i, (mamba_block, attn_block, norm1, norm2) in enumerate(self.layers):
            # A. Internal Processing (Neurons think)
            x_norm = norm1(x)
            x_mem = mamba_block(x_norm)
            x = x + x_mem # Residual
            
            # B. Communication (Neurons talk)
            x_norm2 = norm2(x)
            
            # Reshape for Attention: (Batch*Seq, Neurons, Dim)
            # We want neurons to attend to other neurons at the SAME timestep
            x_comm = x_norm2.view(B, self.n_neurons, L, -1).permute(0, 2, 1, 3).reshape(B*L, self.n_neurons, -1)
            
            # Self-Attention among neurons
            comm_out, weights = attn_block(x_comm, x_comm, x_comm, average_attn_weights=True)
            
            if i == self.n_layers - 1:
                self._last_attn_weights = weights # Capture for viz
            
            # Restore shape
            comm_out = comm_out.view(B, L, self.n_neurons, -1).permute(0, 2, 1, 3).reshape(B*self.n_neurons, L, -1)
            x = x + comm_out
            
        # Aggregation (Mean over neurons)
        x = x.view(B, self.n_neurons, L, -1).mean(dim=1) # (B, L, D)
        
        return self.head(self.norm_f(x))

