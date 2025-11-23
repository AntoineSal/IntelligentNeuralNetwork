import torch
import torch.nn as nn
import torch.nn.functional as F
from .colony import NeuralColony
from .workspace import GlobalWorkspace

class INNv4(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_colonies=4, neurons_per_colony=32):
        super().__init__()
        self.d_model = d_model
        self.n_colonies = n_colonies
        
        # 1. Input / Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        
        # 2. Router Central (Le Manager)
        # Décide quelle colonie travaille sur quel token
        self.router = nn.Linear(d_model, n_colonies)
        
        # 3. Colonies (Les Départements)
        self.colonies = nn.ModuleList([
            NeuralColony(d_model, neurons_per_colony, colony_id=i)
            for i in range(n_colonies)
        ])
        
        # 4. Global Workspace (La Salle de Réunion)
        # Un par couche (ou partagé, ici un par couche pour profondeur)
        self.workspaces = nn.ModuleList([
            GlobalWorkspace(d_model, n_colonies)
            for _ in range(n_layers)
        ])
        
        self.n_layers = n_layers
        self.norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight # Weight Tying

    def forward(self, input_ids):
        B, L = input_ids.shape
        
        # Embed
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:, :L, :]
        
        # Routing Loop (Layer by Layer)
        # Dans cette V4, les colonies sont réutilisées (Recurrent Depth) ou empilées ?
        # Empilons pour la stabilité, mais conceptuellement c'est le même graphe.
        # Pour simplifier l'implémentation "Colab", on fait une boucle temporelle simulée par layers.
        
        current_state = x
        
        for layer_idx in range(self.n_layers):
            # 1. Routing (Qui travaille ?)
            # (B, L, D) -> (B, L, N_Col)
            router_logits = self.router(current_state)
            routing_weights = F.softmax(router_logits, dim=-1)
            
            # 2. Colony Execution
            colony_outputs = []
            for i, colony in enumerate(self.colonies):
                # Input pondéré par le router pour cette colonie
                # Soft Gating: La colonie reçoit l'input complet mais sait si elle est importante
                weight = routing_weights[:, :, i] # (B, L)
                
                out = colony(current_state, active_mask=weight)
                colony_outputs.append(out)
            
            # 3. Global Workspace Integration
            # Les colonies s'échangent de l'info via le bus
            workspace_signal = self.workspaces[layer_idx](colony_outputs)
            
            # 4. Update State (Residual)
            # Somme pondérée des sorties de colonies + Signal Global
            # (B, L, D)
            aggregated_colonies = torch.stack(colony_outputs, dim=2) # (B, L, N, D)
            # On somme pondéré par les poids du router (ceux qui devaient bosser contribuent le plus)
            weighted_sum = torch.sum(aggregated_colonies * routing_weights.unsqueeze(-1), dim=2)
            
            current_state = current_state + weighted_sum + workspace_signal
            
        x_final = self.norm_final(current_state)
        logits = self.lm_head(x_final)
        
        return logits

