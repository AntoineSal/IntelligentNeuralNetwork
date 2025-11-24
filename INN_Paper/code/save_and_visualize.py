import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os

print("=== SAVING MODELS & GENERATING VISUALS ===")

# 1. SAVE WEIGHTS
os.makedirs("models", exist_ok=True)
# We assume 'inn', 'lstm', 'tf' variables are still in scope from the previous cell
if 'inn' in locals():
    torch.save(inn.state_dict(), "models/inn_text8_optimized.pth")
    print("✓ Saved models/inn_text8_optimized.pth")
if 'lstm' in locals():
    torch.save(lstm.state_dict(), "models/lstm_text8_baseline.pth")
    print("✓ Saved models/lstm_text8_baseline.pth")
if 'tf' in locals():
    torch.save(tf.state_dict(), "models/transformer_text8_baseline.pth")
    print("✓ Saved models/transformer_text8_baseline.pth")

# 2. VISUALIZATION (Neuron Activity on Text8)
# We use the INN model trained on Text8 to see character-level specialization
if 'inn' in locals():
    print("\n🧠 Generating Neuron Activity Heatmap (Text8)...")
    inn.eval()
    
    # Test sequence (Character level)
    text_sample = "the quick brown fox jumps over the lazy dog"
    # Need corpus from previous cell
    ids = corpus.tokenize(text_sample).to(device).unsqueeze(0) # (1, L)
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            # Output: (B, N, L, D)
            # Norm over D -> (B, N, L)
            if len(output.shape) == 4:
                act = output.norm(dim=-1).squeeze(0).detach().cpu().numpy()
                activations[name] = act
        return hook
    
    # Hook the last Mamba block (most semantic/abstract)
    handle = inn.layers[-1][0].register_forward_hook(get_activation("mamba_last"))
    
    with torch.no_grad():
        inn(ids)
    
    handle.remove()
    
    # Plot
    plt.figure(figsize=(15, 6))
    sns.heatmap(activations["mamba_last"], cmap="magma", 
                xticklabels=list(text_sample), 
                yticklabels=[f"N{i}" for i in range(inn.num_neurons)])
    plt.title(f"INNv2 Neuron Specialization (Character Level)\nInput: '{text_sample}'")
    plt.xlabel("Character Sequence")
    plt.ylabel("Neuron ID")
    plt.tight_layout()
    plt.savefig("inn_text8_heatmap.png", dpi=300)
    plt.show()
    print("✓ Saved inn_text8_heatmap.png")

else:
    print("⚠️ INN model not found in scope. Run training first.")

