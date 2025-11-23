import torch
import torch.nn as nn
from .mamba_core import MambaCore

class NeuralColony(nn.Module):
    def __init__(self, d_model, n_neurons, colony_id=0):
        """
        NeuralColony: Un groupe de neurones partageant un Cœur Mamba.
        Structure:
        - Adapters (Individualité): Projettent l'input dans l'espace du cœur.
        - MambaCore (Puissance): Traite l'info.
        - Router (Attention): Distribue le résultat.
        """
        super().__init__()
        self.d_model = d_model
        self.n_neurons = n_neurons
        self.id = colony_id
        
        # 1. Le Cœur (Shared Brain)
        self.core = MambaCore(d_model)
        
        # 2. Adapters (Individual personalities)
        # Chaque neurone a une petite modulation unique (LoRA-like)
        # Au lieu de N Mambas, on a 1 Mamba + N petits biases/scales
        self.neuron_biases = nn.Parameter(torch.randn(1, 1, n_neurons, d_model) * 0.01)
        self.neuron_scales = nn.Parameter(torch.ones(1, 1, n_neurons, d_model))
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, active_mask=None):
        """
        x: (B, L, D) - Input global pour la colonie
        active_mask: (B, L) - Si la colonie est active pour ce token
        """
        # 1. Shared Processing
        # Le coeur traite le flux principal
        core_out = self.core(x) # (B, L, D)
        
        # 2. Individualization (Expansion virtuelle)
        # La colonie génère N variantes de sa pensée
        # (B, L, 1, D) * (1, 1, N, D) -> (B, L, N, D)
        expanded = core_out.unsqueeze(2) * self.neuron_scales + self.neuron_biases
        
        # 3. Aggregation interne (Consensus de la colonie)
        # Pour l'instant, on retourne la moyenne pondérée par l'apprentissage
        # Dans le futur, on pourra router vers des sorties spécifiques
        colony_out = expanded.mean(dim=2) # (B, L, D)
        
        # Residual + Norm
        out = self.norm(x + colony_out)
        
        # Gating (Si la colonie est inactive, elle ne contribue pas)
        if active_mask is not None:
            out = out * active_mask.unsqueeze(-1)
            
        return out

