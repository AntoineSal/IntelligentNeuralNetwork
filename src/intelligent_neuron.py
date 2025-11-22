import torch
import torch.nn as nn

class IntelligentNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, key_query_dim, output_dim=None):
        """
        Initialise un 'Intelligent Neuron' avec Key/Query DYNAMIQUES.
        """
        super(IntelligentNeuron, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        
        # --- DYNAMIQUE : Générateurs de Query et Key ---
        self.query_generator = nn.Linear(hidden_dim, key_query_dim)
        self.key_generator = nn.Linear(hidden_dim, key_query_dim)
        
        # Receiver Feed Forward
        self.receiver_ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU() 
        )
        
        # Core (Mémoire)
        self.lstm_cell = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        
        # Sender Feed Forward
        self.sender_ff = nn.Sequential(
            nn.Linear(hidden_dim, self.output_dim),
            nn.Identity() 
        )
        
        # État interne
        self.h_t = None
        self.c_t = None

    def init_state(self, batch_size=1):
        """Réinitialise l'état interne (mémoire) du neurone."""
        device = self.query_generator.weight.device
        self.h_t = torch.zeros(batch_size, self.hidden_dim).to(device)
        self.c_t = torch.zeros(batch_size, self.hidden_dim).to(device)

    def get_attention_params(self, batch_size):
        """
        Génère K et Q basés sur l'état caché actuel (avant mise à jour).
        Utilisé par le réseau pour calculer la matrice d'attention AVANT de faire le forward.
        """
        if self.h_t is None or self.h_t.size(0) != batch_size:
            self.init_state(batch_size)
            
        q = self.query_generator(self.h_t)
        k = self.key_generator(self.h_t)
        return q, k

    def forward(self, input_signal):
        """
        Effectue un pas de temps pour le neurone.
        """
        batch_size = input_signal.size(0)
        
        if self.h_t is None or self.h_t.size(0) != batch_size:
            self.init_state(batch_size)
            
        # 1. Receiver
        processed_input = self.receiver_ff(input_signal)
        
        # 2. Core : Mise à jour de la mémoire
        self.h_t, self.c_t = self.lstm_cell(processed_input, (self.h_t, self.c_t))
        
        # 3. Sender
        output_signal = self.sender_ff(self.h_t)
        
        # Note: On pourrait retourner les Q/K mis à jour ici pour analyse, 
        # mais le réseau utilise ceux générés par get_attention_params (h_t-1).
        
        return output_signal
