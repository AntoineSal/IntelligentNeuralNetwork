"""
INN Multi-Codebook "Hi-Fi": The Hybrid Architecture
===================================================
Architecture Finale pour la génération Audio Haute-Fidélité.

Concept "Hybrid INN" :
1. Parallel Experts : N neurones LSTM indépendants traitent la séquence (Vitesse max).
2. Interleaved Input : Les 4 codebooks sont décalés temporellement pour respecter la causalité.
3. Dynamic Mixing : Une Attention finale sélectionne quel neurone écouter pour chaque codebook.

Objectif : 
- Générer de l'audio "propre" (4 codebooks).
- Battre les baselines en vitesse et en qualité.
"""

import os
import shutil
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import soundfile as sf
import glob
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import EncodecModel, AutoProcessor

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sample_rate": 24000,
    "vocab_size": 1024,
    "n_codebooks": 4,     # Hi-Fi Audio (Bandwidth 6kbps)
    "seq_len": 450,       # ~6 sec of audio (450 * 4 codebooks = 1800 tokens logic)
    "batch_size": 64,     # A100 Optimized
    "epochs": 10,          
    "lr": 1e-3,           # Validé par le benchmark Codebook 0
    "log_dir": "./multicodebook_logs"
}

if not os.path.exists(CONFIG["log_dir"]):
    os.makedirs(CONFIG["log_dir"])

# ==============================================================================
# 1. INN HYBRID ARCHITECTURE (Multi-Codebook)
# ==============================================================================
class INN_Neuron(nn.Module):
    def __init__(self, d_model, d_rnn):
        super().__init__()
        # Le neurone "voit" une somme d'embeddings (C1+C2+C3+C4)
        self.lstm = nn.LSTM(d_model, d_rnn, batch_first=True)
        self.out_proj = nn.Linear(d_rnn, d_model)
        self.ln_out = nn.LayerNorm(d_model) # Stabilization pour l'Attention Mixer
        
    def forward(self, x):
        # x: (batch, seq, d_model)
        lstm_out, _ = self.lstm(x)
        proj = self.out_proj(lstm_out)
        return self.ln_out(proj)

class INN_MultiCodebook(nn.Module):
    def __init__(self, vocab_size, d_model, n_neurons, d_rnn, n_heads, n_codebooks=4):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.d_model = d_model
        self.n_neurons = n_neurons
        
        # Embeddings pour chaque codebook
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, d_model, padding_idx=vocab_size) 
            for _ in range(n_codebooks)
        ])
        
        # Normalisation de l'entrée fusionnée (Crucial car Somme de 4 embeddings)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Parallel Experts (Le Cerveau)
        self.neurons = nn.ModuleList([INN_Neuron(d_model, d_rnn) for _ in range(n_neurons)])
        
        # Global Mixer (Le Juge)
        self.mixer_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_mix = nn.LayerNorm(d_model)
        
        # Têtes de sortie (une par codebook)
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(n_codebooks)
        ])

    def forward(self, x):
        # x: (Batch, N_Codebooks, Seq_Len)
        B, K, T = x.size()
        
        # 1. Interleaving Logic (Delay Pattern)
        # On décale les entrées pour que C_k(t) dépende de C_{k-1}(t) et C_k(t-1)
        # Stratégie simplifiée "Summed Embeddings" avec Masking Causal implicite via LSTM
        
        # Somme des embeddings des K codebooks (Input Fusion)
        x_fused = torch.zeros(B, T, self.d_model, device=x.device)
        for k in range(self.n_codebooks):
            x_fused += self.embeddings[k](x[:, k, :])
            
        # Normalisation Post-Fusion (Stabilization)
        x_fused = self.input_norm(x_fused)
            
        # 2. Parallel Processing
        neuron_outputs = [n(x_fused) for n in self.neurons]
        stack = torch.stack(neuron_outputs, dim=2) # (B, T, N, D)
        
        # 3. Dynamic Mixing
        stack_flat = stack.reshape(-1, self.n_neurons, self.d_model)
        query_flat = x_fused.reshape(-1, 1, self.d_model)
        
        attn_out, _ = self.mixer_attn(query_flat, stack_flat, stack_flat)
        mixed = self.norm_mix(attn_out.squeeze(1) + query_flat.squeeze(1)) # (B*T, D)
        mixed = mixed.view(B, T, self.d_model)
        
        # 4. Parallel Prediction (Multi-Head)
        # Chaque tête prédit le prochain token pour son codebook
        logits_list = []
        for k in range(self.n_codebooks):
            logits_list.append(self.heads[k](mixed))
            
        return torch.stack(logits_list, dim=1) # (B, K, T, Vocab)

# ==============================================================================
# 2. BASELINES (Pour Référence)
# ==============================================================================
class Transformer_Baseline(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers, n_codebooks=4):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, d_model, padding_idx=vocab_size) 
            for _ in range(n_codebooks)
        ])
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.heads = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in range(n_codebooks)])
        
    def forward(self, x):
        B, K, T = x.size()
        x_fused = torch.zeros(B, T, 512, device=x.device) # d_model hardcoded for baseline match
        for k in range(self.n_codebooks):
            x_fused += self.embeddings[k](x[:, k, :])
        
        mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(x.device)
        feat = self.transformer(x_fused, mask=mask, is_causal=True)
        
        return torch.stack([h(feat) for h in self.heads], dim=1)

# ==============================================================================
# DATASET
# ==============================================================================
class LibriSpeechMultiCodebook(Dataset):
    def __init__(self, root, url="train-clean-100"):
        self.files = sorted(glob.glob(os.path.join(root, "LibriSpeech", url, "**", "*.flac"), recursive=True))
        
        # Robust Download Logic (Force Cleanup if Empty/Corrupted)
        if len(self.files) == 0:
            print(f"Dataset not found or corrupted in {root}. Initiating cleanup and download...")
            
            # 1. Clean up potential corrupted partial files
            if os.path.exists(root):
                try:
                    shutil.rmtree(root)
                    print("🧹 Cleaned up existing directory to remove partial files.")
                except Exception as e:
                    print(f"⚠️ Warning: Could not clean directory: {e}")
            
            os.makedirs(root, exist_ok=True)
            
            # 2. Download Fresh
            try:
                print("📥 Downloading LibriSpeech... (This takes a few minutes)")
                torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=True)
                self.files = sorted(glob.glob(os.path.join(root, "LibriSpeech", url, "**", "*.flac"), recursive=True))
            except Exception as e:
                print(f"❌ Download failed: {e}")
                raise e
                
        print(f"📚 Dataset: {len(self.files)} files (Multi-Codebook Mode)")
        self.resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=CONFIG["sample_rate"])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            wav, _ = sf.read(self.files[idx])
            waveform = torch.from_numpy(wav).float()
            if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
            else: waveform = waveform.t()
            return self.resampler(waveform)
        except: return torch.zeros(1, CONFIG["sample_rate"])

class AudioTokenizer:
    def __init__(self, device):
        self.device = device
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.model.eval()

    @torch.no_grad()
    def encode(self, waveform):
        raw_audio = [w.squeeze(0).cpu().numpy() for w in waveform]
        inputs = self.processor(raw_audio=raw_audio, sampling_rate=CONFIG["sample_rate"], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Force bandwidth to match n_codebooks
        # 4 codebooks => 3.0 kbps
        # 8 codebooks => 6.0 kbps
        target_bandwidth = CONFIG["n_codebooks"] * 0.75 
        
        outputs = self.model.encode(**inputs, bandwidth=target_bandwidth)
        # (1, 1, n_codebooks, seq_len) -> (n_codebooks, seq_len)
        codes = outputs.audio_codes[0].squeeze(0).squeeze(0) 
        return codes

# ==============================================================================
# TRAINING LOOP
# ==============================================================================
def train():
    print("🚀 LAUNCHING HYBRID INN MULTI-CODEBOOK RUN")
    torch.set_float32_matmul_precision('high')
    
    tokenizer = AudioTokenizer(CONFIG["device"])
    ds = LibriSpeechMultiCodebook(root="./librispeech_data")
    
    def collate_fn(batch):
        max_len = min(max([w.size(-1) for w in batch]), 24000*10)
        padded = torch.zeros(len(batch), 1, max_len)
        for i, w in enumerate(batch):
            sl = min(w.size(-1), max_len)
            padded[i, :, :sl] = w[:, :sl]
        return padded

    dl = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    # INN Model (Scaled Up for Hi-Fi)
    model = INN_MultiCodebook(
        vocab_size=CONFIG["vocab_size"],
        d_model=512,      # Plus large pour gérer la complexité
        n_neurons=24,     # Plus d'experts
        d_rnn=256,
        n_heads=8,
        n_codebooks=4
    ).to(CONFIG["device"])
    
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"] * len(dl))
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    step = 0
    start_time = time.time()
    
    for epoch in range(CONFIG["epochs"]):
        print(f"🏁 Epoch {epoch+1}/{CONFIG['epochs']}")
        for batch_waves in dl:
            batch_waves = batch_waves.to(CONFIG["device"])
            
            with torch.no_grad():
                # (Batch, 4, Seq)
                codes = torch.stack([tokenizer.encode(w.unsqueeze(0)) for w in batch_waves])
                codes = codes.to(CONFIG["device"])
            
            if codes.size(-1) > CONFIG["seq_len"]:
                codes = codes[:, :, :CONFIG["seq_len"]]
                
            inputs = codes[:, :, :-1]
            targets = codes[:, :, 1:]
            
            if inputs.size(-1) < 10: continue

            optimizer.zero_grad()
            logits = model(inputs) # (B, 4, T, Vocab)
            
            # Loss sum over all codebooks
            loss = 0
            for k in range(CONFIG["n_codebooks"]):
                loss += criterion(logits[:, k, :, :].reshape(-1, CONFIG["vocab_size"]), targets[:, k, :].reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            step += 1
            if step % 10 == 0:
                elapsed = time.time() - start_time
                avg_loss = loss.item() / 4 # Average per codebook
                ppl = math.exp(avg_loss)
                print(f"Step {step} | Total Loss: {loss.item():.4f} | Avg Ppl: {ppl:.2f} | Time: {elapsed:.0f}s")
                
            if step % 1000 == 0:
                torch.save(model.state_dict(), f"{CONFIG['log_dir']}/inn_multicodebook_{step}.pt")
                print("💾 Checkpoint saved.")

if __name__ == "__main__":
    train()
