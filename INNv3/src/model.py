import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .mamba import MultiMambaBlock

class AdaptiveRouter(nn.Module):
    def __init__(self, vocab_size, num_neurons, d_model):
        super().__init__()
        # Neuron embeddings specialized for routing
        self.neuron_keys = nn.Parameter(torch.randn(num_neurons, d_model))
        
        # Token specific routing: Embeds tokens into a space to query neurons
        self.token_router = nn.Embedding(vocab_size, num_neurons)
        self.gate = nn.Sigmoid()
        
    def forward(self, input_ids, embeddings):
        """
        input_ids: (B, L)
        embeddings: (B, L, D) - Contextual embeddings (or just simple embeddings)
        """
        # Routing logits based on Token ID directly (static routing preference)
        # (B, L) -> (B, L, N)
        routing_logits = self.token_router(input_ids) 
        
        # Convert to weights
        routing_weights = self.gate(routing_logits) # (B, L, N)
        
        # Distribute information: (B, L, D) * (B, L, N) -> (B, N, L, D)
        # We broadcast D and N appropriately
        # embeddings: B, L, D
        # weights: B, L, N
        # einsum: 'bld,bln->bnld'
        neuron_inputs = torch.einsum('bld,bln->bnld', embeddings, routing_weights)
        
        return neuron_inputs

class MixtureOfVocabularies(nn.Module):
    def __init__(self, d_model, num_experts=4, vocab_size=33278):
        super().__init__()
        self.num_experts = num_experts
        self.vocab_size = vocab_size
        
        # Ensure divisible, if not, we handle it
        self.vocab_per_expert = math.ceil(vocab_size / num_experts)
        
        # Experts: Each projects to a subset of vocab
        self.expert_decoders = nn.ModuleList([
            nn.Linear(d_model, self.vocab_per_expert) 
            for _ in range(num_experts)
        ])
        
        # Router: Selects which expert contributes
        self.expert_router = nn.Linear(d_model, num_experts)
        
    def forward(self, hidden_states):
        # hidden_states: (B, L, D)
        
        # Router logits: (B, L, NumExperts)
        router_logits = self.expert_router(hidden_states)
        expert_weights = F.softmax(router_logits, dim=-1) # (B, L, E)
        
        vocab_logits_list = []
        for i, expert in enumerate(self.expert_decoders):
            # (B, L, D) -> (B, L, VocabPerExpert)
            logits = expert(hidden_states)
            
            # Weighted: (B, L, VocabPerExpert) * (B, L, 1)
            weight = expert_weights[..., i:i+1]
            weighted_logits = logits * weight
            vocab_logits_list.append(weighted_logits)
            
        # Concatenate to reform full vocab: (B, L, TotalVocab)
        full_logits = torch.cat(vocab_logits_list, dim=-1)
        
        # Crop if we padded vocab
        if full_logits.size(-1) > self.vocab_size:
            full_logits = full_logits[..., :self.vocab_size]
            
        return full_logits

class NeuronAttention(nn.Module):
    def __init__(self, num_neurons, d_model, n_head=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B*L, N, D)
        # Self-attention over Neurons dimension
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)

class ProgressiveINNLayer(nn.Module):
    def __init__(self, max_neurons, d_model, initial_neurons=32):
        super().__init__()
        self.max_neurons = max_neurons
        self.active_neurons = min(initial_neurons, max_neurons)
        self.d_model = d_model
        
        # Mamba Block (Processes each neuron's timeline)
        self.mamba_block = MultiMambaBlock(max_neurons, d_model)
        
        # Attention Block (Communication between neurons)
        self.attention = NeuronAttention(max_neurons, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Highway integration (Residual Token Stream)
        # Mix gate learnable
        self.mix_gate = nn.Parameter(torch.tensor(0.0)) # Start biased towards 0.5 (sigmoid(0) = 0.5)
        self.highway_proj = nn.Linear(d_model, d_model)

    def grow_network(self, growth_rate=1.2):
        """Gradually activate more neurons"""
        old = self.active_neurons
        self.active_neurons = min(
            int(self.active_neurons * growth_rate),
            self.max_neurons
        )
        if self.active_neurons > old:
            print(f"  -> Layer grown: {old} -> {self.active_neurons} neurons")

    def forward(self, x, highway):
        """
        x: (B, N, L, D) - Neuron states
        highway: (B, L, D) - Token stream
        """
        B, N, L, D = x.shape
        
        # 1. Create Mask for Active Neurons
        # We mask computations for inactive neurons to force sparsity/progressive learning
        # (In practice, Mamba processes all, but we zero out inactive ones to simulate growth)
        mask = torch.zeros(self.max_neurons, device=x.device)
        mask[:self.active_neurons] = 1.0
        mask = mask.view(1, -1, 1, 1) # (1, N, 1, 1)
        
        x = x * mask
        
        # 2. Mamba Step (Temporal)
        res = x
        x = self.mamba_block(x)
        x = self.norm1(x + res)
        x = x * mask # Re-mask
        
        # 3. Attention Step (Spatial/Inter-neuron)
        res = x
        # Flatten for attention: (B*L, N, D)
        x_flat = x.permute(0, 2, 1, 3).reshape(B*L, N, D)
        x_flat = self.attention(x_flat)
        x = x_flat.reshape(B, L, N, D).permute(0, 2, 1, 3)
        x = self.norm2(x + res)
        x = x * mask
        
        # 4. Highway Interaction (RTS)
        # Inject highway info into neurons
        highway_signal = self.highway_proj(highway) # (B, L, D)
        highway_signal = highway_signal.unsqueeze(1).expand(-1, N, -1, -1) # (B, N, L, D)
        
        gate = torch.sigmoid(self.mix_gate)
        x = gate * x + (1 - gate) * highway_signal
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class INNv3(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=4, max_neurons=256):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = PositionalEncoding(d_model, max_len=512)
        
        # Adaptive Routing (ATN)
        self.router = AdaptiveRouter(vocab_size, num_neurons=max_neurons, d_model=d_model)
        
        # Hierarchical Layers (HNO + PNA)
        # Levels: 0-1: Feature (Max), 2-3: Concept (Max/2), 4+: Reasoning (Max/4)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_max_neurons = max_neurons
            if i >= 2: layer_max_neurons = max_neurons // 2
            if i >= 4: layer_max_neurons = max_neurons // 4
            
            self.layers.append(
                ProgressiveINNLayer(
                    max_neurons=layer_max_neurons,
                    d_model=d_model,
                    initial_neurons=32 # Start small
                )
            )
        
        # Decoder (MoV)
        self.decoder = MixtureOfVocabularies(d_model, num_experts=4, vocab_size=vocab_size)
        
        self.highway_norm = nn.LayerNorm(d_model)
        
    def grow_network(self):
        """Trigger growth in all layers"""
        print(">>> Growing Network Capacity...")
        for layer in self.layers:
            layer.grow_network()
            
    def forward(self, input_ids):
        # input_ids: (B, L)
        
        # 1. Embeddings
        token_emb = self.token_embeddings(input_ids) # (B, L, D)
        token_emb = self.pos_embeddings(token_emb.transpose(0, 1)).transpose(0, 1)
        
        # 2. Highway Stream (Starts as just embeddings)
        highway = token_emb
        
        # 3. Adaptive Routing
        # Transforms (B, L, D) -> (B, N, L, D) based on token preferences
        # Note: We route to the MAX dimension of the first layer
        x = self.router(input_ids, token_emb)
        
        # 4. Process Layers
        for i, layer in enumerate(self.layers):
            # Handle dimension mismatch between hierarchy levels
            if x.shape[1] != layer.max_neurons:
                # Simple pooling/projection to resize neuron dimension
                # (B, PrevN, L, D) -> (B, NewN, L, D)
                # We just slice or pool. Slicing is consistent with 'specialization'.
                if x.shape[1] > layer.max_neurons:
                    x = x[:, :layer.max_neurons, :, :]
                else:
                    # Should not happen in this decreasing hierarchy
                    pass
            
            x = layer(x, highway)
            
            # Update highway for next layer? 
            # In the prompt RTS diagram, highway is parallel. 
            # We can optionally update highway with aggregated neuron info.
            # Here we keep highway 'pure' but let it interact inside the layer.
            pass
            
        # 5. Final Projection
        # Pool across neurons to get back to sequence stream
        x_pooled = x.mean(dim=1) # (B, L, D)
        highway_final = self.highway_norm(highway)
        
        combined = x_pooled + highway_final
        
        # 6. Mixture of Vocabularies
        logits = self.decoder(combined)
        
        return logits

