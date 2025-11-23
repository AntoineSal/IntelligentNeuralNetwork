import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalWorkspace(nn.Module):
    def __init__(self, d_model, n_colonies):
        """
        GlobalWorkspace: Le bus de communication entre colonies.
        Implémente le cycle Broadcast -> Dispatch.
        """
        super().__init__()
        self.d_model = d_model
        self.n_colonies = n_colonies
        
        # Le Token de Conscience (Partagé)
        self.workspace_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Attention Mechanism (Cross-Attention)
        # Query: Workspace
        # Key/Value: Colonies
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
        # Dispatch (Update des colonies)
        # Chaque colonie a une projection pour lire le workspace
        self.dispatch_proj = nn.Linear(d_model, d_model)

    def forward(self, colony_outputs):
        """
        colony_outputs: Liste de (B, L, D) tensors, un par colonie.
        """
        # 1. Stack Colonies: (B, L, N_Colonies, D)
        # On traite chaque pas de temps ou la séquence entière ?
        # Pour efficacité transformer, on traite la séquence.
        # On fusionne L et N_Colonies pour l'attention ?
        # Non, l'attention est spatiale (entre colonies) à chaque pas de temps.
        
        # Stratégie Efficace: Somme pondérée par Attention
        # On crée un tenseur (B*L, N_Colonies, D)
        B, L, D = colony_outputs[0].shape
        stack = torch.stack(colony_outputs, dim=2) # (B, L, N, D)
        flat_stack = stack.view(B*L, self.n_colonies, D)
        
        # Workspace Token étendu: (B*L, 1, D)
        ws_query = self.workspace_token.expand(B*L, -1, -1)
        
        # 2. Broadcast (Colonies -> Workspace)
        # Le Workspace "regarde" les colonies et met à jour son état
        ws_out, _ = self.attn(query=ws_query, key=flat_stack, value=flat_stack)
        ws_out = self.norm(ws_out) # (B*L, 1, D)
        
        # 3. Dispatch (Workspace -> Colonies)
        # Le Workspace renvoie son info à tout le monde
        # (B*L, 1, D) -> (B, L, D)
        global_signal = self.dispatch_proj(ws_out).view(B, L, D)
        
        return global_signal

