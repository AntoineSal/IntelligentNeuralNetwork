import torch
import torch.nn as nn

class IntelligentNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, key_query_dim, output_dim=None):
        """
        Initialise un 'Intelligent Neuron'.

        Args:
            input_dim (int): Dimension du signal d'entrée brut (avant le Feed Forward du receiver).
                             Si c'est un neurone de transmission, cela correspond souvent à la dimension de sortie des autres neurones.
            hidden_dim (int): Dimension de la mémoire interne (état caché du LSTM).
            key_query_dim (int): Dimension des vecteurs Key et Query pour le mécanisme d'attention.
            output_dim (int, optional): Dimension du signal de sortie. Si None, par défaut à hidden_dim.
        """
        super(IntelligentNeuron, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        
        # --- Signal Receiver ---
        # Vecteur Query : Ce que le neurone "veut" écouter.
        # On l'initialise comme un paramètre apprenable.
        self.query = nn.Parameter(torch.randn(key_query_dim))
        
        # Receiver Feed Forward : Prépare l'input pour le LSTM.
        self.receiver_ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU() # Activation non-linéaire
        )
        
        # --- Core (Mémoire) ---
        # LSTM Cell : Gère la mémoire à long terme et l'état courant.
        # input_size est hidden_dim car le receiver_ff a déjà transformé l'input.
        self.lstm_cell = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        
        # --- Signal Sender ---
        # Vecteur Key : Ce que le neurone "annonce" contenir comme information.
        self.key = nn.Parameter(torch.randn(key_query_dim))
        
        # Sender Feed Forward : Prépare la sortie du LSTM pour être envoyée.
        self.sender_ff = nn.Sequential(
            nn.Linear(hidden_dim, self.output_dim),
            nn.Identity() # On peut ajouter une activation ici si nécessaire (ex: Tanh, Sigmoid)
        )
        
        # État interne (h_t, c_t) initialisé à None (sera initialisé au premier forward)
        self.h_t = None
        self.c_t = None

    def init_state(self, batch_size=1):
        """Réinitialise l'état interne (mémoire) du neurone."""
        device = self.query.device
        self.h_t = torch.zeros(batch_size, self.hidden_dim).to(device)
        self.c_t = torch.zeros(batch_size, self.hidden_dim).to(device)

    def forward(self, input_signal):
        """
        Effectue un pas de temps pour le neurone.
        
        Args:
            input_signal (torch.Tensor): Le signal d'entrée agrégé (batch_size, input_dim).
                                         Note: L'agrégation (filtrage par attention) est généralement 
                                         gérée par le réseau avant d'appeler ce forward, 
                                         car elle dépend des Keys des autres neurones.
        
        Returns:
            output_signal (torch.Tensor): Le signal émis par le neurone (batch_size, output_dim).
            query (torch.Tensor): Le vecteur query actuel du neurone (pour le prochain pas de filtrage).
            key (torch.Tensor): Le vecteur key actuel du neurone.
        """
        batch_size = input_signal.size(0)
        
        # Initialisation de l'état si nécessaire
        if self.h_t is None or self.h_t.size(0) != batch_size:
            self.init_state(batch_size)
            
        # 1. Receiver : Traitement de l'input
        processed_input = self.receiver_ff(input_signal)
        
        # 2. Core : Mise à jour de la mémoire LSTM
        self.h_t, self.c_t = self.lstm_cell(processed_input, (self.h_t, self.c_t))
        
        # 3. Sender : Génération de l'output
        output_signal = self.sender_ff(self.h_t)
        
        # Note: Dans cette architecture simple, Key et Query sont des vecteurs statiques (paramètres).
        # On pourrait imaginer qu'ils soient dynamiques (générés par le LSTM), 
        # mais pour l'instant on suit la définition de base.
        # On les retourne pour faciliter le calcul d'attention au niveau du réseau.
        
        # Expansion des dimensions pour correspondre au batch_size si nécessaire
        # (batch_size, key_query_dim)
        current_key = self.key.unsqueeze(0).expand(batch_size, -1)
        current_query = self.query.unsqueeze(0).expand(batch_size, -1)
        
        return output_signal, current_query, current_key

