import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import INN

def visualize(model_path):
    print(f"Loading model from {model_path}...")
    
    # Load strict=False to handle slight variations in state dict keys
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        # Infer config from weights if possible, otherwise use defaults matching paper
        model = INN(vocab_size=1153) # WikiText default
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # Dummy forward pass to generate attention weights
    dummy_input = torch.randint(0, 1153, (1, 64)) # Batch 1, Seq 64
    with torch.no_grad():
        _ = model(dummy_input)
        
    if model._last_attn_weights is None:
        print("Error: No attention weights captured.")
        return

    # [Batch*Seq, N, N] -> [N, N] (Average over time)
    attn_matrix = model._last_attn_weights.squeeze(0).numpy()
    avg_attn = attn_matrix.mean(axis=0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attn, cmap="viridis", square=True)
    plt.title("Learned Neuron Connectivity (Average)")
    plt.xlabel("Target Neuron")
    plt.ylabel("Source Neuron")
    
    out_file = "neuron_connectivity.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth checkpoint')
    args = parser.parse_args()
    
    visualize(args.model_path)

