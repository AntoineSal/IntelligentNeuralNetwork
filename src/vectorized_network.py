import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorizedIntelligentNetwork(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 num_input_neurons, 
                 num_transmission_neurons, 
                 num_action_neurons,
                 neuron_hidden_dim=256,
                 key_query_dim=256,
                 signal_dim=256):
        super().__init__()
        
        self.num_input = num_input_neurons
        self.num_trans = num_transmission_neurons
        self.num_action = num_action_neurons
        self.total_neurons = num_input_neurons + num_transmission_neurons + num_action_neurons
        
        self.hidden_dim = neuron_hidden_dim
        self.key_query_dim = key_query_dim
        self.signal_dim = signal_dim
        
        # --- POIDS DES NEURONES ---
        # On utilise nn.Parameter pour stocker les poids de TOUS les neurones dans des tenseurs uniques.
        # Forme typique : (Num_Neurons, Dim_In, Dim_Out)
        
        # 1. RECEIVER (Input -> Hidden)
        # Groupe 1 : Input Neurons (Input Size -> Hidden)
        self.w_rec_input = nn.Parameter(torch.Tensor(num_input_neurons, input_size, neuron_hidden_dim))
        self.b_rec_input = nn.Parameter(torch.Tensor(num_input_neurons, neuron_hidden_dim))
        
        # Groupe 2 : Trans & Action (Signal Dim -> Hidden)
        num_others = num_transmission_neurons + num_action_neurons
        self.w_rec_other = nn.Parameter(torch.Tensor(num_others, signal_dim, neuron_hidden_dim))
        self.b_rec_other = nn.Parameter(torch.Tensor(num_others, neuron_hidden_dim))
        
        # 2. LSTM CELL MANUELLE
        # Poids pour Input (Hidden -> 4*Hidden) car input est déjà projeté par receiver
        self.w_lstm_ih = nn.Parameter(torch.Tensor(self.total_neurons, neuron_hidden_dim, 4 * neuron_hidden_dim))
        self.b_lstm_ih = nn.Parameter(torch.Tensor(self.total_neurons, 4 * neuron_hidden_dim))
        
        # Poids pour Hidden (Hidden -> 4*Hidden)
        self.w_lstm_hh = nn.Parameter(torch.Tensor(self.total_neurons, neuron_hidden_dim, 4 * neuron_hidden_dim))
        self.b_lstm_hh = nn.Parameter(torch.Tensor(self.total_neurons, 4 * neuron_hidden_dim))
        
        # 3. DYNAMIQUE KEY/QUERY
        self.w_query = nn.Parameter(torch.Tensor(self.total_neurons, neuron_hidden_dim, key_query_dim))
        self.b_query = nn.Parameter(torch.Tensor(self.total_neurons, key_query_dim))
        
        self.w_key = nn.Parameter(torch.Tensor(self.total_neurons, neuron_hidden_dim, key_query_dim))
        self.b_key = nn.Parameter(torch.Tensor(self.total_neurons, key_query_dim))
        
        # 4. SENDER
        self.w_sender = nn.Parameter(torch.Tensor(self.total_neurons, neuron_hidden_dim, signal_dim))
        self.b_sender = nn.Parameter(torch.Tensor(self.total_neurons, signal_dim))
        
        # 5. PROJECTION FINALE
        self.action_projection = nn.Linear(num_action_neurons * signal_dim, output_size)
        
        self._init_weights()
        
        # États
        self.h_state = None
        self.c_state = None
        self.prev_outputs = None
        
        # Pour la visualisation (stockage temporaire des dernières Q/K)
        self.last_Q = None
        self.last_K = None

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def reset_memory(self, batch_size):
        device = self.w_rec_input.device
        self.h_state = torch.zeros(batch_size, self.total_neurons, self.hidden_dim, device=device)
        self.c_state = torch.zeros(batch_size, self.total_neurons, self.hidden_dim, device=device)
        self.prev_outputs = torch.zeros(batch_size, self.total_neurons, self.signal_dim, device=device)

    def forward(self, x):
        if x.dim() == 2:
            return self.step(x)
        
        batch_size, seq_len, _ = x.size()
        self.reset_memory(batch_size)
        outputs = []
        
        for t in range(seq_len):
            out_t = self.step(x[:, t, :])
            outputs.append(out_t)
            
        return torch.stack(outputs, dim=1)

    def step(self, x_input):
        batch_size = x_input.size(0)
        if self.h_state is None: self.reset_memory(batch_size)
        
        # --- 1. Attention Dynamique (Parallèle) ---
        # Calcul de Q et K pour tous les neurones en une fois via einsum
        # (Batch, N, Hidden) * (N, Hidden, KQ) -> (Batch, N, KQ)
        Q = torch.einsum('bnh,nhk->bnk', self.h_state, self.w_query) + self.b_query
        K = torch.einsum('bnh,nhk->bnk', self.h_state, self.w_key) + self.b_key
        
        # Sauvegarde pour visu
        self.last_Q = Q
        self.last_K = K
        
        # Matrice d'attention (Batch, N, N)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.key_query_dim ** 0.5)
        weights = F.softmax(scores, dim=2)
        
        # Agrégation (Batch, N, Signal)
        context_input = torch.bmm(weights, self.prev_outputs)
        
        # --- 2. Receiver Inputs ---
        # Groupe 1 : Inputs externes
        # (Batch, In) -> (Batch, N_In, In)
        x_in_expanded = x_input.unsqueeze(1).expand(-1, self.num_input, -1)
        rec_out_input = torch.einsum('bni,nih->bnh', x_in_expanded, self.w_rec_input) + self.b_rec_input
        
        # Groupe 2 : Inputs internes
        context_others = context_input[:, self.num_input:, :]
        rec_out_others = torch.einsum('bni,nih->bnh', context_others, self.w_rec_other) + self.b_rec_other
        
        # Concat (Batch, Total_N, Hidden)
        lstm_input = torch.cat([rec_out_input, rec_out_others], dim=1)
        lstm_input = F.relu(lstm_input)
        
        # --- 3. LSTM Update (Parallèle) ---
        # Gates = Input_Gate + Hidden_Gate
        gates_ih = torch.einsum('bnh,nhg->bng', lstm_input, self.w_lstm_ih) + self.b_lstm_ih
        gates_hh = torch.einsum('bnh,nhg->bng', self.h_state, self.w_lstm_hh) + self.b_lstm_hh
        gates = gates_ih + gates_hh
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=2)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        self.c_state = (forgetgate * self.c_state) + (ingate * cellgate)
        self.h_state = outgate * torch.tanh(self.c_state)
        
        # --- 4. Sender ---
        self.prev_outputs = torch.einsum('bnh,nhs->bns', self.h_state, self.w_sender) + self.b_sender
        
        # --- 5. Output ---
        action_out = self.prev_outputs[:, -self.num_action:, :]
        final = self.action_projection(action_out.reshape(batch_size, -1))
        
        return final

    # Helper pour le script de visu
    @property
    def neurons(self):
        """
        Dummy property pour compatibilité avec le script de visualisation.
        Renvoie une liste d'objets fictifs si besoin, ou on adapte la visu.
        """
        # Cette version vectorisée n'a pas de liste self.neurons.
        # Il faudra adapter la visualisation.
        return [] 

