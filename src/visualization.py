import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_attention_matrix(model, filename="attention_matrix.png", title="Matrice d'Attention des Neurones"):
    """
    Récupère les Queries et Keys actuelles du modèle et affiche la matrice d'attention.
    Axe Y : Receveurs (Qui écoute ?)
    Axe X : Emetteurs (Qui parle ?)
    """
    model.eval()
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        all_queries = []
        all_keys = []
        
        for neuron in model.neurons:
            q = neuron.query.unsqueeze(0) 
            k = neuron.key.unsqueeze(0)
            all_queries.append(q)
            all_keys.append(k)
            
        Q_matrix = torch.stack(all_queries, dim=1).to(device)
        K_matrix = torch.stack(all_keys, dim=1).to(device)
        
        scores = torch.bmm(Q_matrix, K_matrix.transpose(1, 2))
        scores = scores / (model.key_query_dim ** 0.5)
        attention_weights = torch.nn.functional.softmax(scores, dim=2)
        
        attention_matrix = attention_weights[0].cpu().numpy()
        
    plt.figure(figsize=(14, 12))
    
    labels = []
    # Input
    for i in range(model.num_input_neurons):
        labels.append(f"I{i}")
    # Trans
    for i in range(model.num_transmission_neurons):
        labels.append(f"T{i}")
    # Action
    for i in range(model.num_action_neurons):
        labels.append(f"A{i}")
        
    sns.heatmap(attention_matrix, xticklabels=labels, yticklabels=labels, cmap="viridis", vmin=0, vmax=1)
    
    plt.title(title)
    plt.xlabel("Émetteur (Source)")
    plt.ylabel("Receveur (Destination)")
    plt.tight_layout()
    
    # Créer le dossier plots s'il n'existe pas
    if os.path.dirname(filename) and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
        
    plt.savefig(filename)
    plt.close()
    print(f"Matrice d'attention sauvegardée dans '{filename}'")

