import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_attention_matrix(model, filename="attention_matrix.png", title="Matrice d'Attention des Neurones"):
    """
    Version standard (Liste de neurones).
    """
    if hasattr(model, 'last_Q'):
        return plot_vectorized_attention_matrix(model, filename, title)
        
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        all_queries = []
        all_keys = []
        for neuron in model.neurons:
            q, k = neuron.get_attention_params(batch_size=1)
            all_queries.append(q.unsqueeze(1))
            all_keys.append(k.unsqueeze(1))
        Q_matrix = torch.cat(all_queries, dim=1).to(device)
        K_matrix = torch.cat(all_keys, dim=1).to(device)
        scores = torch.bmm(Q_matrix, K_matrix.transpose(1, 2)) / (model.key_query_dim ** 0.5)
        attention_weights = torch.nn.functional.softmax(scores, dim=2)
        attention_matrix = attention_weights[0].cpu().numpy()
        
    _save_plot(attention_matrix, filename, title, model)

def plot_vectorized_attention_matrix(model, filename="attention_matrix.png", title="Matrice d'Attention"):
    """
    Version pour VectorizedIntelligentNetwork.
    """
    model.eval()
    
    if model.last_Q is None:
         print("Attention: Pas de Q/K en cache. Visualisation ignorée.")
         return

    with torch.no_grad():
        Q = model.last_Q[0:1] 
        K = model.last_K[0:1] 
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / (model.key_query_dim ** 0.5)
        attention_weights = torch.nn.functional.softmax(scores, dim=2)
        attention_matrix = attention_weights[0].cpu().numpy()
        
    _save_plot(attention_matrix, filename, title, model)

def _save_plot(matrix, filename, title, model):
    plt.figure(figsize=(14, 12))
    
    labels = []
    # Detection des nombres de neurones selon le type de modele
    if hasattr(model, 'num_input'): # Vectorized
        n_in = model.num_input
        n_ts = getattr(model, 'num_trans_static', 0)
        n_td = getattr(model, 'num_trans_dynamic', 0)
        n_act = model.num_action
        
        # Si ancienne version vectorisée sans distinction static/dynamic
        if n_ts == 0 and n_td == 0:
             # Fallback sur l'attribut générique num_trans s'il existe
             n_ts = getattr(model, 'num_trans', 0)
        
        for i in range(n_in): labels.append(f"I{i}")
        for i in range(n_ts): labels.append(f"TS{i}")
        for i in range(n_act): labels.append(f"A{i}")
        for i in range(n_td): labels.append(f"TD{i}")
    else: # Standard
        n_in, n_tr, n_act = model.num_input_neurons, model.num_transmission_neurons, model.num_action_neurons
        for i in range(n_in): labels.append(f"I{i}")
        for i in range(n_tr): labels.append(f"T{i}")
        for i in range(n_act): labels.append(f"A{i}")
        
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="viridis", vmin=0, vmax=1)
    
    plt.title(title)
    plt.xlabel("Émetteur (Source)")
    plt.ylabel("Receveur (Destination)")
    plt.tight_layout()
    
    if os.path.dirname(filename) and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
        
    plt.savefig(filename)
    plt.close()
    print(f"Matrice d'attention sauvegardée dans '{filename}'")
