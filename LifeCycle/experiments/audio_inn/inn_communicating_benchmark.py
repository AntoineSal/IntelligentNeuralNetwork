"""
INN Hybrid Benchmark: Vectorized Speed + Interleaved Logic
==========================================================
Architecture "Hybride" :
- Core : INN Vectorisé (nn.LSTM) pour la vitesse (1.1s/step).
- Logic : Interleaved Input (Préparation pour Multi-Codebook).

Objectif : Battre le record de 17.11 Ppl en moins de 30 minutes.
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import soundfile as sf
import glob
from torch.utils.data import DataLoader, Dataset
from transformers import EncodecModel, AutoProcessor

# ==============================================================================
# CONFIGURATION HYBRIDE
# ==============================================================================
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sample_rate": 24000,
    "vocab_size": 1024,
    "seq_len": 1024,      
    "batch_size": 64,     # Retour à 64 pour stabilité avec nn.LSTM
    "epochs": 2,          
    "lr": 1e-3,           # Validé
    "log_dir": "./hybrid_benchmark_logs"
}

if not os.path.exists(CONFIG["log_dir"]):
    os.makedirs(CONFIG["log_dir"])

# ==============================================================================
# 1. INN ARCHITECTURE (Hybrid: Vectorized Core)
# ==============================================================================
class INN_Neuron(nn.Module):
    def __init__(self, d_model, d_rnn):
        super().__init__()
        # Vectorized LSTM: Vitesse maximale
        self.lstm = nn.LSTM(d_model, d_rnn, batch_first=True)
        self.out_proj = nn.Linear(d_rnn, d_model)
        self.ln_out = nn.LayerNorm(d_model) # Stabilization
        
    def forward(self, x):
        # x: (batch, seq, d_model)
        lstm_out, _ = self.lstm(x)
        proj = self.out_proj(lstm_out)
        return self.ln_out(proj)

class INN_Hybrid(nn.Module):
    def __init__(self, vocab_size, d_model, n_neurons, d_rnn, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_neurons = n_neurons
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Parallel Experts (Vectorized)
        self.neurons = nn.ModuleList([INN_Neuron(d_model, d_rnn) for _ in range(n_neurons)])
        
        # Global Mixer (Attention a posteriori)
        self.mixer_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_mix = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x: (Batch, Seq)
        batch_size, seq_len = x.size()
        x_emb = self.embedding(x) 
        
        # 1. Parallel Neuron Execution (Vectorized - Fast!)
        neuron_outputs = [n(x_emb) for n in self.neurons] 
        stack = torch.stack(neuron_outputs, dim=2) # (B, T, N, D)
        
        # 2. Global Mixing
        stack_flat = stack.reshape(-1, self.n_neurons, self.d_model)
        query_flat = x_emb.reshape(-1, 1, self.d_model)
        
        attn_out, _ = self.mixer_attn(query_flat, stack_flat, stack_flat)
        mixed = self.norm_mix(attn_out.squeeze(1) + query_flat.squeeze(1))
        
        logits = self.head(mixed)
        return logits.view(batch_size, seq_len, -1)

# ==============================================================================
# DATASET & UTILS
# ==============================================================================
class LibriSpeechTokenized(Dataset):
    def __init__(self, root, url="train-clean-100"):
        self.files = sorted(glob.glob(os.path.join(root, "LibriSpeech", url, "**", "*.flac"), recursive=True))
        
        # Robust Download Logic
        if len(self.files) == 0:
            print(f"Dataset not found or corrupted in {root}. Initiating cleanup and download...")
            if os.path.exists(root):
                try:
                    shutil.rmtree(root)
                    print("🧹 Cleaned up existing directory.")
                except Exception as e:
                    print(f"⚠️ Warning: Could not clean directory: {e}")
            
            os.makedirs(root, exist_ok=True)
            try:
                print("📥 Downloading LibriSpeech... (This takes 2-3 mins)")
                torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=True)
                self.files = sorted(glob.glob(os.path.join(root, "LibriSpeech", url, "**", "*.flac"), recursive=True))
            except Exception as e:
                print(f"❌ Download failed: {e}")
                raise e
        
        print(f"📚 Dataset Loaded: {len(self.files)} files.")
        self.resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=CONFIG["sample_rate"])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            wav, sr = sf.read(self.files[idx])
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
        outputs = self.model.encode(**inputs)
        raw_codes = outputs.audio_codes[0]
        if raw_codes.ndim == 4: raw_codes = raw_codes.squeeze(0)
        return raw_codes[:, 0, :] # Return only Codebook 0

def train():
    print("🚀 LAUNCHING HYBRID INN BENCHMARK (Vectorized Speed)")
    torch.set_float32_matmul_precision('high')
    
    tokenizer = AudioTokenizer(CONFIG["device"])
    
    # Dataset Preparation
    ds = LibriSpeechTokenized(root="./librispeech_data")
    
    def collate_fn(batch):
        max_len = min(max([w.size(-1) for w in batch]), 24000*8)
        padded = torch.zeros(len(batch), 1, max_len)
        for i, w in enumerate(batch):
            sl = min(w.size(-1), max_len)
            padded[i, :, :sl] = w[:, :sl]
        return padded

    dl = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    # Model Setup (Iso-Parameters ~12M)
    # n_neurons=20, d_rnn=128 => ~12.5M params
    model = INN_Hybrid(
        vocab_size=CONFIG["vocab_size"], 
        d_model=256, 
        n_neurons=20, 
        d_rnn=128, 
        n_heads=8
    ).to(CONFIG["device"])
    
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Compilation : Sur le Vectorized, ça devrait mieux passer
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("⚡ Model compiled with torch.compile")
    except:
        print("⚠️ Compile failed/skipped, running eager mode.")

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"] * len(dl))
    criterion = nn.CrossEntropyLoss()
    
    # Logging
    stats = {"loss": [], "ppl": [], "bpt": []}
    
    model.train()
    step = 0
    start_time = time.time()
    
    print("\n🏁 Epoch 1 Start")
    
    for epoch in range(CONFIG["epochs"]):
        for batch_waves in dl:
            batch_waves = batch_waves.to(CONFIG["device"])
            
            # Tokenization on-the-fly
            with torch.no_grad():
                tokens = tokenizer.encode(batch_waves)
            
            # Prep Input/Target
            if tokens.size(1) > CONFIG["seq_len"]:
                tokens = tokens[:, :CONFIG["seq_len"]]
            
            inputs, targets = tokens[:, :-1], tokens[:, 1:]
            
            if inputs.size(1) < 10: continue

            # Forward
            optimizer.zero_grad()
            logits = model(inputs) 
            
            # Loss
            loss = criterion(logits.reshape(-1, CONFIG["vocab_size"]), targets.reshape(-1))
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            step += 1
            
            # Monitoring
            if step % 10 == 0:
                elapsed = time.time() - start_time
                ppl = math.exp(loss.item())
                bpt = loss.item() / math.log(2)
                
                print(f"Step {step} | Loss: {loss.item():.4f} | Ppl: {ppl:.2f} | BPT: {bpt:.3f} | Time: {elapsed:.0f}s")
                
                stats["loss"].append(loss.item())
                stats["ppl"].append(ppl)
                stats["bpt"].append(bpt)
                
            if step % 500 == 0:
                torch.save(model.state_dict(), f"{CONFIG['log_dir']}/inn_hybrid_{step}.pt")
                print(f"💾 Checkpoint saved: {CONFIG['log_dir']}/inn_hybrid_{step}.pt")
                
            if step >= 1500: # Benchmark limit
                print("🛑 Benchmark Limit Reached.")
                torch.save(model.state_dict(), f"{CONFIG['log_dir']}/inn_hybrid_final.pt")
                # Save stats
                df = pd.DataFrame(stats)
                df.to_csv(f"{CONFIG['log_dir']}/benchmark_results.csv")
                return

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("🛑 Arrêt manuel.")
