import torch
import torch.nn as nn
import torch.nn.functional as F
from .intelligent_neuron import IntelligentNeuron

class IntelligentNetwork(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 num_input_neurons, 
                 num_transmission_neurons, 
                 num_action_neurons,
                 neuron_hidden_dim=32,
                 key_query_dim=16,
                 signal_dim=16):
        """
        Réseau de neurones intelligents (Version Dynamique).
        """
        super(IntelligentNetwork, self).__init__()
        
        self.num_input_neurons = num_input_neurons
        self.num_transmission_neurons = num_transmission_neurons
        self.num_action_neurons = num_action_neurons
        
        self.signal_dim = signal_dim
        self.key_query_dim = key_query_dim
        
        # --- Création des Neurones ---
        self.neurons = nn.ModuleList()
        
        # 1. Input Neurons
        for _ in range(num_input_neurons):
            self.neurons.append(IntelligentNeuron(input_dim=input_size, 
                                                  hidden_dim=neuron_hidden_dim, 
                                                  key_query_dim=key_query_dim, 
                                                  output_dim=signal_dim))
            
        # 2. Transmission Neurons
        for _ in range(num_transmission_neurons):
            self.neurons.append(IntelligentNeuron(input_dim=signal_dim, 
                                                  hidden_dim=neuron_hidden_dim, 
                                                  key_query_dim=key_query_dim, 
                                                  output_dim=signal_dim))
            
        # 3. Action Neurons
        for _ in range(num_action_neurons):
            self.neurons.append(IntelligentNeuron(input_dim=signal_dim, 
                                                  hidden_dim=neuron_hidden_dim, 
                                                  key_query_dim=key_query_dim, 
                                                  output_dim=signal_dim))
        
        # Projection finale
        self.action_projection = nn.Linear(num_action_neurons * signal_dim, output_size)
        
        # Stockage des outputs précédents
        self.previous_outputs = None 
        
    def reset_memory(self, batch_size):
        """Réinitialise la mémoire de tous les neurones et le buffer d'outputs."""
        for neuron in self.neurons:
            neuron.init_state(batch_size)
        
        device = self.action_projection.weight.device
        total_neurons = len(self.neurons)
        self.previous_outputs = torch.zeros(batch_size, total_neurons, self.signal_dim).to(device)

    def forward(self, x):
        """Passe avant (Forward pass)."""
        if x.dim() == 2: 
            return self.step(x)
        elif x.dim() == 3: 
            outputs = []
            self.reset_memory(x.size(0))
            
            for t in range(x.size(1)):
                input_t = x[:, t, :]
                out_t = self.step(input_t)
                outputs.append(out_t.unsqueeze(1)) 
            
            return torch.cat(outputs, dim=1) 
        else:
            raise ValueError("Input doit être 2D (step) ou 3D (sequence)")

    def step(self, x_input):
        """Exécute un pas de temps t avec Attention Dynamique."""
        batch_size = x_input.size(0)
        
        if self.previous_outputs is None or self.previous_outputs.size(0) != batch_size:
            self.reset_memory(batch_size)

        # --- 1. Récupération des Keys et Queries DYNAMIQUES ---
        # Basées sur l'état h_{t-1} (mémoire courante avant update)
        all_queries = []
        all_keys = []
        
        for neuron in self.neurons:
            q, k = neuron.get_attention_params(batch_size)
            all_queries.append(q.unsqueeze(1))
            all_keys.append(k.unsqueeze(1))
            
        Q_matrix = torch.cat(all_queries, dim=1) # (Batch, N, Dim)
        K_matrix = torch.cat(all_keys, dim=1)    # (Batch, N, Dim)
        
        # --- 2. Calcul de l'Attention ---
        attention_scores = torch.bmm(Q_matrix, K_matrix.transpose(1, 2))
        attention_scores = attention_scores / (self.key_query_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=2)
        
        # --- 3. Agrégation des signaux ---
        context_inputs = torch.bmm(attention_weights, self.previous_outputs)
        
        # --- 4. Forward des Neurones (Update h_{t-1} -> h_t) ---
        new_outputs = []
        idx_input_end = self.num_input_neurons
        
        for i, neuron in enumerate(self.neurons):
            if i < idx_input_end:
                inp = x_input
            else:
                inp = context_inputs[:, i, :]
            
            out_signal = neuron(inp)
            new_outputs.append(out_signal)
            
        self.previous_outputs = torch.stack(new_outputs, dim=1)
        
        # --- 5. Sortie Système ---
        start_action = self.num_input_neurons + self.num_transmission_neurons
        action_outputs = self.previous_outputs[:, start_action:, :]
        
        action_outputs_flat = action_outputs.reshape(batch_size, -1)
        system_output = self.action_projection(action_outputs_flat)
        
        return system_output
