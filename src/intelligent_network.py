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
        Réseau de neurones intelligents.
        
        Args:
            input_size (int): Dimension globale de l'entrée du système.
            output_size (int): Dimension globale de la sortie du système.
            num_input_neurons (int): Nombre de neurones dédiés à la réception de l'entrée.
            num_transmission_neurons (int): Nombre de neurones cachés de traitement.
            num_action_neurons (int): Nombre de neurones produisant la sortie finale.
            neuron_hidden_dim (int): Taille de la mémoire interne (LSTM) de chaque neurone.
            key_query_dim (int): Taille des vecteurs d'attention.
            signal_dim (int): Taille du vecteur échangé entre les neurones.
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
        # Ils reçoivent l'input externe directement.
        for _ in range(num_input_neurons):
            self.neurons.append(IntelligentNeuron(input_dim=input_size, 
                                                  hidden_dim=neuron_hidden_dim, 
                                                  key_query_dim=key_query_dim, 
                                                  output_dim=signal_dim))
            
        # 2. Transmission Neurons
        # Ils reçoivent des signaux d'autres neurones (taille signal_dim).
        for _ in range(num_transmission_neurons):
            self.neurons.append(IntelligentNeuron(input_dim=signal_dim, 
                                                  hidden_dim=neuron_hidden_dim, 
                                                  key_query_dim=key_query_dim, 
                                                  output_dim=signal_dim))
            
        # 3. Action Neurons
        # Ils reçoivent des signaux internes et produisent un signal interne + contribuent à la sortie.
        for _ in range(num_action_neurons):
            self.neurons.append(IntelligentNeuron(input_dim=signal_dim, 
                                                  hidden_dim=neuron_hidden_dim, 
                                                  key_query_dim=key_query_dim, 
                                                  output_dim=signal_dim))
        
        # Projection finale pour les Action Neurons vers la sortie système
        self.action_projection = nn.Linear(num_action_neurons * signal_dim, output_size)
        
        # Stockage des outputs précédents pour la récurrence (mémoire tampon)
        self.previous_outputs = None 
        
    def reset_memory(self, batch_size):
        """Réinitialise la mémoire de tous les neurones et le buffer d'outputs."""
        for neuron in self.neurons:
            neuron.init_state(batch_size)
        
        device = self.neurons[0].query.device
        total_neurons = len(self.neurons)
        # Initialisation à 0 des signaux précédents
        self.previous_outputs = torch.zeros(batch_size, total_neurons, self.signal_dim).to(device)

    def forward(self, x):
        """
        Passe avant (Forward pass) sur un pas de temps ou une séquence.
        
        Args:
            x: Input externe. (Batch, Input_Size) ou (Batch, Time, Input_Size)
        """
        if x.dim() == 2: # (Batch, Input_Size) -> Un seul pas de temps
            return self.step(x)
        elif x.dim() == 3: # (Batch, Time, Input_Size) -> Séquence
            outputs = []
            # On reset la mémoire au début d'une nouvelle séquence
            self.reset_memory(x.size(0))
            
            for t in range(x.size(1)):
                input_t = x[:, t, :]
                out_t = self.step(input_t)
                outputs.append(out_t.unsqueeze(1)) # Ajouter dimension temps
            
            return torch.cat(outputs, dim=1) # (Batch, Time, Output_Size)
        else:
            raise ValueError("Input doit être 2D (step) ou 3D (sequence)")

    def step(self, x_input):
        """Exécute un pas de temps t."""
        batch_size = x_input.size(0)
        
        # Si premier pas, initialisation
        if self.previous_outputs is None or self.previous_outputs.size(0) != batch_size:
            self.reset_memory(batch_size)

        # --- 1. Récupération des Keys et Queries ---
        all_queries = []
        all_keys = []
        
        for neuron in self.neurons:
            q = neuron.query.unsqueeze(0).expand(batch_size, -1)
            k = neuron.key.unsqueeze(0).expand(batch_size, -1)
            all_queries.append(q)
            all_keys.append(k)
            
        Q_matrix = torch.stack(all_queries, dim=1) # (Batch, N, KQ_Dim)
        K_matrix = torch.stack(all_keys, dim=1)    # (Batch, N, KQ_Dim)
        
        # --- 2. Calcul de l'Attention ---
        # Score(i, j) : i écoute j
        attention_scores = torch.bmm(Q_matrix, K_matrix.transpose(1, 2))
        attention_scores = attention_scores / (self.key_query_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=2)
        
        # --- 3. Agrégation des signaux ---
        context_inputs = torch.bmm(attention_weights, self.previous_outputs)
        
        # --- 4. Forward des Neurones ---
        new_outputs = []
        idx_input_end = self.num_input_neurons
        
        for i, neuron in enumerate(self.neurons):
            if i < idx_input_end:
                # Input Neurons : Input externe
                inp = x_input
            else:
                # Autres : Contexte agrégé
                inp = context_inputs[:, i, :]
            
            out_signal, _, _ = neuron(inp)
            new_outputs.append(out_signal)
            
        self.previous_outputs = torch.stack(new_outputs, dim=1)
        
        # --- 5. Sortie Système ---
        # On récupère les outputs des derniers (Action Neurons)
        start_action = self.num_input_neurons + self.num_transmission_neurons
        action_outputs = self.previous_outputs[:, start_action:, :]
        
        action_outputs_flat = action_outputs.reshape(batch_size, -1)
        system_output = self.action_projection(action_outputs_flat)
        
        return system_output

