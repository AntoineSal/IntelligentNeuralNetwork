import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import math

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Mamba not found, visualization requires mamba-ssm")
    exit()

# === CONFIGURATION ===
CONFIG = {
    'd_model': 256,
    'n_neurons': 32,
    'n_layers': 6,
    'vocab_size': 1153, # From WikiText run
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === MODEL DEFINITION (Modified to capture Attention) ===
class INNv2Viz(nn.Module):
    def __init__(self, vocab_size, num_neurons, d_model, num_layers):
        super().__init__()
        self.num_neurons = num_neurons
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        
        # Storage for visualization
        self.last_attn_weights = None
        
        for _ in range(num_layers):
            neuron_pop = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            # Note: average_attn_weights=False to get full [Batch, Heads, Q, K] or [Batch, Q, K]
            attn = nn.MultiheadAttention(d_model, 4, batch_first=True) 
            norm1 = nn.LayerNorm(d_model)
            norm2 = nn.LayerNorm(d_model)
            self.layers.append(nn.ModuleList([neuron_pop, attn, norm1, norm2]))
            
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.head.weight = self.embedding.weight

    def forward(self, x):
        B, L = x.shape
        x = self.embedding(x)
        x = x.unsqueeze(1).expand(-1, self.num_neurons, -1, -1).reshape(B*self.num_neurons, L, -1)
        
        for i, (mamba, attn, norm1, norm2) in enumerate(self.layers):
            # Dynamics
            x_norm = norm1(x)
            x = x + mamba(x_norm)
            
            # Communication
            x_norm2 = norm2(x)
            x_comm = x_norm2.view(B, self.num_neurons, L, -1).permute(0, 2, 1, 3).reshape(B*L, self.num_neurons, -1)
            
            # Capture attention weights here!
            # attn_output: [Batch*Seq, Neurons, Dim]
            # attn_weights: [Batch*Seq, Neurons, Neurons] (averaged over heads if average_attn_weights=True default)
            comm_out, attn_weights = attn(x_comm, x_comm, x_comm, average_attn_weights=True)
            
            # Store the weights of the LAST layer for visualization
            if i == len(self.layers) - 1:
                self.last_attn_weights = attn_weights
            
            x = x + comm_out.view(B, L, self.num_neurons, -1).permute(0, 2, 1, 3).reshape(B*self.num_neurons, L, -1)
            
        x = x.view(B, self.num_neurons, L, -1).mean(dim=1)
        return self.head(self.norm_f(x))

# === VISUALIZATION LOGIC ===
def visualize():
    print("Searching for checkpoints...")
    # Find latest checkpoint
    list_of_files = glob.glob('models/*.pth')
    if not list_of_files:
        print("No checkpoints found in models/!")
        return
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading {latest_file}...")
    
    # Initialize model
    model = INNv2Viz(CONFIG['vocab_size'], CONFIG['n_neurons'], CONFIG['d_model'], CONFIG['n_layers']).to(device)
    
    # Load weights (strict=False because dropout masks etc might vary slightly, but keys should match)
    state_dict = torch.load(latest_file, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print("Model loaded. Generating visualization...")
    
    # Create dummy input (Sequence of random tokens representing a sentence)
    # We just need the structure, the exact tokens don't matter for the connectivity pattern
    dummy_input = torch.randint(0, CONFIG['vocab_size'], (1, 64)).to(device) # Batch 1, Seq 64
    
    with torch.no_grad():
        _ = model(dummy_input)
        
    # Get weights: [SeqLen, Neurons, Neurons] (since Batch=1)
    # We average over the Sequence Length to see the "Global Topology" of the graph
    attn_matrix = model.last_attn_weights.squeeze(0).cpu().numpy() # [Seq, N, N]
    avg_attn = attn_matrix.mean(axis=0) # [N, N] - Average communication map
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attn, cmap="viridis", square=True)
    plt.title("Learned Neuron Connectivity (INN Graph Topology)")
    plt.xlabel("Target Neuron")
    plt.ylabel("Source Neuron")
    
    output_path = "INN_Paper/visualizations/neuron_heatmap.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_path}")
    
    # Also plot a specific timestep (e.g., middle of sequence) to see dynamic routing
    plt.clf()
    mid_step = 32
    step_attn = attn_matrix[mid_step]
    sns.heatmap(step_attn, cmap="magma", square=True)
    plt.title(f"Dynamic Routing at Step {mid_step}")
    plt.xlabel("Target Neuron")
    plt.ylabel("Source Neuron")
    plt.savefig("INN_Paper/visualizations/neuron_heatmap_step.png", dpi=300, bbox_inches='tight')
    print("Dynamic step heatmap saved.")

if __name__ == "__main__":
    visualize()

