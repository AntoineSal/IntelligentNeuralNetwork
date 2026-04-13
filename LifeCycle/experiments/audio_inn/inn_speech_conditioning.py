"""
INN Speech-to-Speech: Semantic Resynthesis
==========================================
Objectif : Prouver que l'INN peut effectuer une tâche S2S (Speech-to-Speech)
en convertissant des tokens sémantiques (sens) en tokens acoustiques (voix).

Tâche : Semantic Resynthesis (HuBERT -> INN -> EnCodec)
Input : Tokens sémantiques (HuBERT, 50Hz, vocab 500)
Output : Tokens acoustiques (EnCodec, 75Hz, vocab 1024)

Architecture "Cross-Modal INN" :
1. Semantic Encoder : Transforme la séquence HuBERT en mémoire contextuelle.
2. INN Core : Neurones récurrents avec Cross-Attention sur la mémoire sémantique.
3. Alignement : Géré par l'attention (le modèle apprend à s'aligner temporellement).
"""

import os
import math
import torch
import torch.nn as nn
import torchaudio
import glob
from torch.utils.data import DataLoader, Dataset
from transformers import EncodecModel, AutoProcessor, Wav2Vec2Model

# ==============================================================================
# CONFIGURATION S2S
# ==============================================================================
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sample_rate": 24000,
    "vocab_audio": 1024,    # EnCodec
    "vocab_semantic": 500,  # HuBERT K-Means (Hypothèse standard)
    "d_model": 512,
    "n_neurons": 16,
    "d_rnn": 512,
    "batch_size": 32,
    "seq_len_audio": 450,   # ~6s
    "seq_len_sem": 300,     # ~6s (50Hz vs 75Hz)
    "lr": 5e-4,
    "log_dir": "./s2s_logs"
}

if not os.path.exists(CONFIG["log_dir"]):
    os.makedirs(CONFIG["log_dir"])

# ==============================================================================
# 1. SEMANTIC ENCODER (The "Ear")
# ==============================================================================
class SemanticEncoder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Convolutions pour capturer le contexte local sémantique
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x: (B, L_sem)
        x = self.embedding(x).transpose(1, 2) # (B, D, L)
        x = self.conv(x).transpose(1, 2)      # (B, L, D)
        return self.proj(x)

# ==============================================================================
# 2. INN SPEECH GENERATOR (The "Voice")
# ==============================================================================
class INN_S2S(nn.Module):
    def __init__(self, vocab_audio, vocab_semantic, d_model, n_neurons, d_rnn):
        super().__init__()
        self.d_model = d_model
        self.n_neurons = n_neurons
        self.d_rnn = d_rnn
        
        # Components
        self.semantic_encoder = SemanticEncoder(vocab_semantic, d_model)
        self.audio_emb = nn.Embedding(vocab_audio, d_model)
        
        # Neurones
        self.neurons = nn.ModuleList([
            nn.LSTMCell(d_model, d_rnn) for _ in range(n_neurons)
        ])
        
        # Attention
        self.mixer_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        self.norm_mix = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_audio)

    def scan_loop(self, x_emb, memory, hx_list, cx_list):
        seq_len = x_emb.size(1)
        batch_size = x_emb.size(0)
        outputs_list = []
        
        for t in range(seq_len):
            xt = x_emb[:, t, :] # (B, D)
            
            # A. Update Neurons
            new_hx, new_cx = [], []
            neuron_outputs = []
            
            for i in range(self.n_neurons):
                h, c = self.neurons[i](xt, (hx_list[i], cx_list[i]))
                new_hx.append(h)
                new_cx.append(c)
                neuron_outputs.append(h)
            
            hx_list = new_hx
            cx_list = new_cx
            
            # Stack: (B, N, D)
            stack = torch.stack(neuron_outputs, dim=1)
            
            # B. Cross-Attention (Sémantique)
            # Les neurones demandent : "Quel est le sens à exprimer ?"
            sem_context, _ = self.cross_attn(stack, memory, memory)
            
            # C. Self-Attention (Acoustique)
            # Les neurones s'accordent en intégrant le sens
            query = xt.unsqueeze(1)
            enriched_stack = stack + sem_context
            attn_out, _ = self.mixer_attn(query, enriched_stack, enriched_stack)
            
            # D. Predict
            mixed = self.norm_mix(attn_out.squeeze(1) + xt)
            outputs_list.append(self.head(mixed))
            
        return torch.stack(outputs_list, dim=1)

    def forward(self, semantic_tokens, audio_tokens):
        # semantic_tokens: (B, L_sem)
        # audio_tokens: (B, L_audio) -> Shifted Input
        
        B = semantic_tokens.size(0)
        device = semantic_tokens.device
        
        # 1. Encode Semantic Context
        memory = self.semantic_encoder(semantic_tokens) # (B, L_sem, D)
        
        # 2. Prepare Audio
        x_emb = self.audio_emb(audio_tokens)
        
        # 3. Init States
        hx_list = [torch.zeros(B, self.d_rnn, device=device) for _ in range(self.n_neurons)]
        cx_list = [torch.zeros(B, self.d_rnn, device=device) for _ in range(self.n_neurons)]
        
        # 4. Generate
        return self.scan_loop(x_emb, memory, hx_list, cx_list)

# ==============================================================================
# MOCK DATASET (Pour tester la compilation sans HuBERT pour l'instant)
# ==============================================================================
# Note: Pour le vrai run, on aura besoin d'extraire les tokens HuBERT.
# Pour l'instant, ce script valide l'architecture.

def main():
    print("🚀 Verifying S2S Architecture...")
    model = INN_S2S(1024, 500, 512, 16, 512).to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Model created. Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Fake Batch
    B = 4
    sem = torch.randint(0, 500, (B, 300)).to(model.audio_emb.weight.device)
    aud = torch.randint(0, 1024, (B, 450)).to(model.audio_emb.weight.device)
    
    # Compilation Check
    try:
        model.scan_loop = torch.compile(model.scan_loop, mode="reduce-overhead")
        print("⚡ Compilation Triggered...")
        out = model(sem, aud)
        print(f"✅ Forward Pass OK. Output shape: {out.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

