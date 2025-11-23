import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorizedIntelligentNetwork(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 num_input_neurons, 
                 num_transmission_neurons, # Statiques
                 num_dynamic_neurons,      # NOUVEAU : Dynamiques
                 num_action_neurons, 
                 neuron_hidden_dim=256,
                 key_query_dim=256,
                 signal_dim=256):
        super().__init__()
        
        self.num_input = num_input_neurons
        self.num_trans_static = num_transmission_neurons
        self.num_trans_dynamic = num_dynamic_neurons
        self.num_action = num_action_neurons
        
        # Ordre en mémoire : [Input, Trans_Static, Action, Trans_Dynamic]
        # 1. Les Statiques (Input + Trans_Static + Action)
        # 2. Les Dynamiques (Trans_Dynamic)
        
        self.num_static_total = num_input_neurons + num_transmission_neurons + num_action_neurons
        self.total_neurons = self.num_static_total + num_dynamic_neurons
        
        self.hidden_dim = neuron_hidden_dim
        self.key_query_dim = key_query_dim
        self.signal_dim = signal_dim
        
        # --- POIDS COMMUNS ---
        
        # 1. RECEIVER
        self.w_rec_input = nn.Parameter(torch.Tensor(num_input_neurons, input_size, neuron_hidden_dim))
        self.b_rec_input = nn.Parameter(torch.Tensor(num_input_neurons, neuron_hidden_dim))
        
        num_others = self.total_neurons - num_input_neurons
        self.w_rec_other = nn.Parameter(torch.Tensor(num_others, signal_dim, neuron_hidden_dim))
        self.b_rec_other = nn.Parameter(torch.Tensor(num_others, neuron_hidden_dim))
        
        # 2. LSTM
        self.w_lstm_ih = nn.Parameter(torch.Tensor(self.total_neurons, neuron_hidden_dim, 4 * neuron_hidden_dim))
        self.b_lstm_ih = nn.Parameter(torch.Tensor(self.total_neurons, 4 * neuron_hidden_dim))
        self.w_lstm_hh = nn.Parameter(torch.Tensor(self.total_neurons, neuron_hidden_dim, 4 * neuron_hidden_dim))
        self.b_lstm_hh = nn.Parameter(torch.Tensor(self.total_neurons, 4 * neuron_hidden_dim))
        
        # 3. ATTENTION HYBRIDE
        # Partie Statique (Input + Trans_Static + Action)
        self.q_static = nn.Parameter(torch.Tensor(self.num_static_total, key_query_dim))
        self.k_static = nn.Parameter(torch.Tensor(self.num_static_total, key_query_dim))
        
        # Partie Dynamique (Trans_Dynamic)
        if num_dynamic_neurons > 0:
            self.w_query_dyn = nn.Parameter(torch.Tensor(num_dynamic_neurons, neuron_hidden_dim, key_query_dim))
            self.b_query_dyn = nn.Parameter(torch.Tensor(num_dynamic_neurons, key_query_dim))
            self.w_key_dyn = nn.Parameter(torch.Tensor(num_dynamic_neurons, neuron_hidden_dim, key_query_dim))
            self.b_key_dyn = nn.Parameter(torch.Tensor(num_dynamic_neurons, key_query_dim))
        
        # 4. SENDER
        self.w_sender = nn.Parameter(torch.Tensor(self.total_neurons, neuron_hidden_dim, signal_dim))
        self.b_sender = nn.Parameter(torch.Tensor(self.total_neurons, signal_dim))
        
        # 5. PROJECTION
        self.action_projection = nn.Linear(num_action_neurons * signal_dim, output_size)
        
        self._init_weights()
        
        self.h_state = None
        self.c_state = None
        self.prev_outputs = None
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
        if x.dim() == 2: return self.step(x)
        batch_size, seq_len, _ = x.size()
        self.reset_memory(batch_size)
        outputs = []
        for t in range(seq_len):
            outputs.append(self.step(x[:, t, :]))
        return torch.stack(outputs, dim=1)

    def step(self, x_input):
        batch_size = x_input.size(0)
        if self.h_state is None: self.reset_memory(batch_size)
        
        # --- 1. ATTENTION HYBRIDE ---
        Q_stat = self.q_static.unsqueeze(0).expand(batch_size, -1, -1)
        K_stat = self.k_static.unsqueeze(0).expand(batch_size, -1, -1)
        
        if self.num_trans_dynamic > 0:
            h_dyn = self.h_state[:, self.num_static_total:, :]
            Q_dyn = torch.einsum('bnh,nhk->bnk', h_dyn, self.w_query_dyn) + self.b_query_dyn
            K_dyn = torch.einsum('bnh,nhk->bnk', h_dyn, self.w_key_dyn) + self.b_key_dyn
            
            Q = torch.cat([Q_stat, Q_dyn], dim=1)
            K = torch.cat([K_stat, K_dyn], dim=1)
        else:
            Q, K = Q_stat, K_stat
            
        self.last_Q = Q
        self.last_K = K
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.key_query_dim ** 0.5)
        weights = F.softmax(scores, dim=2)
        context_input = torch.bmm(weights, self.prev_outputs)
        
        # --- 2. RECEIVER ---
        x_in_expanded = x_input.unsqueeze(1).expand(-1, self.num_input, -1)
        rec_out_input = torch.einsum('bni,nih->bnh', x_in_expanded, self.w_rec_input) + self.b_rec_input
        
        context_others = context_input[:, self.num_input:, :]
        rec_out_others = torch.einsum('bni,nih->bnh', context_others, self.w_rec_other) + self.b_rec_other
        
        lstm_input = torch.cat([rec_out_input, rec_out_others], dim=1)
        lstm_input = F.relu(lstm_input)
        
        # --- 3. LSTM ---
        gates_ih = torch.einsum('bnh,nhg->bng', lstm_input, self.w_lstm_ih) + self.b_lstm_ih
        gates_hh = torch.einsum('bnh,nhg->bng', self.h_state, self.w_lstm_hh) + self.b_lstm_hh
        gates = gates_ih + gates_hh
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=2)
        
        self.c_state = (torch.sigmoid(forgetgate) * self.c_state) + (torch.sigmoid(ingate) * torch.tanh(cellgate))
        self.h_state = torch.sigmoid(outgate) * torch.tanh(self.c_state)
        
        # --- 4. SENDER ---
        self.prev_outputs = torch.einsum('bnh,nhs->bns', self.h_state, self.w_sender) + self.b_sender
        
        # --- 5. OUTPUT ---
        idx_start = self.num_input + self.num_trans_static
        idx_end = idx_start + self.num_action
        action_out = self.prev_outputs[:, idx_start:idx_end, :]
        
        final = self.action_projection(action_out.reshape(batch_size, -1))
        return final
