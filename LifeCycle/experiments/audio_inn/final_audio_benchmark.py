
"""
Top-Tier Paper Benchmark: Efficient Audio Modeling
==================================================
Comparaison rigoureuse "Iso-Parameter" (~12M) entre :
1. INN (Intelligent Neural Network) - Ours (Vectorized Implementation)
2. WaveNet (Causal Dilated Convolutions) - Baseline SOTA Efficiency
3. Transformer (Attention Only) - Baseline SOTA Generalist

Task: Discrete Audio Token Modeling (LibriSpeech -> EnCodec)
Metric: Bits per Dimension (BPD) & Perplexity (Ppl)
"""

import os
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
# CONFIGURATION SCIENTIFIQUE
# ==============================================================================
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sample_rate": 24000,
    "vocab_size": 1024,
    "seq_len": 1024,      # 1024 tokens
    "batch_size": 64,     # A100 optimized
    "epochs": 10,         # 10 Epochs possible thanks to Vectorization (~4h runtime)
    "lr": 1e-3,           # Aggressive start
    "seed": 42,
    "target_params": 12_000_000, 
    "log_dir": "./benchmark_logs"
}

if not os.path.exists(CONFIG["log_dir"]):
    os.makedirs(CONFIG["log_dir"])

# ==============================================================================
# 1. INN ARCHITECTURE (Ours - Vectorized & Optimized)
# ==============================================================================
class INN_Neuron(nn.Module):
    def __init__(self, d_model, d_rnn):
        super().__init__()
        # Vectorized LSTM: Processes (Batch, Seq, Dim) in one cuDNN call
        self.lstm = nn.LSTM(d_model, d_rnn, batch_first=True)
        self.out_proj = nn.Linear(d_rnn, d_model)
        
    def forward(self, x):
        # x: (batch, seq, d_model)
        lstm_out, _ = self.lstm(x)
        return self.out_proj(lstm_out)

class INN_Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_neurons, d_rnn, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_neurons = n_neurons
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Parallel Experts
        self.neurons = nn.ModuleList([INN_Neuron(d_model, d_rnn) for _ in range(n_neurons)])
        
        # Global Mixer (Attention)
        self.mixer_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_mix = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        x_emb = self.embedding(x) # (batch, seq, d_model)
        
        # 1. Parallel Neuron Execution (Vectorized)
        # Each neuron processes the full sequence independently
        # Result: List of (batch, seq, d_model)
        neuron_outputs = [n(x_emb) for n in self.neurons] 
        
        # Stack: (batch, seq, n_neurons, d_model)
        stack = torch.stack(neuron_outputs, dim=2)
        
        # 2. Global Mixing (Time-step independent mixing)
        # Flatten batch and seq to treat each timestep as a sample
        stack_flat = stack.reshape(-1, self.n_neurons, self.d_model) # (batch*seq, n_neurons, d_model)
        query_flat = x_emb.reshape(-1, 1, self.d_model)              # (batch*seq, 1, d_model)
        
        # Attention: "Given context (Query), which Neuron (Key/Value) should I listen to?"
        attn_out, _ = self.mixer_attn(query_flat, stack_flat, stack_flat)
        
        # Residual + Norm
        mixed = self.norm_mix(attn_out.squeeze(1) + query_flat.squeeze(1))
        
        # Output Head
        logits = self.head(mixed)
        
        # Reshape back to sequence
        return logits.view(batch_size, seq_len, -1)

# ==============================================================================
# 2. WAVENET ARCHITECTURE (Baseline 1 - Corrected Causal Padding)
# ==============================================================================
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)

    def forward(self, x):
        # Explicit Left Padding for strict causality
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0))
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        super().__init__()
        self.filter_conv = CausalConv1d(res_channels, res_channels, 2, dilation)
        self.gate_conv = CausalConv1d(res_channels, res_channels, 2, dilation)
        self.res_conv = nn.Conv1d(res_channels, res_channels, 1)
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, 1)

    def forward(self, x):
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        x_gated = filter_out * gate_out
        res_out = self.res_conv(x_gated)
        skip_out = self.skip_conv(x_gated)
        return (x + res_out) * math.sqrt(0.5), skip_out

class WaveNet_Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_blocks):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_blocks):
            for i in range(n_layers):
                dilation = 2 ** i
                self.layers.append(ResidualBlock(d_model, d_model, dilation))
        
        self.end_conv_1 = nn.Conv1d(d_model, d_model, 1)
        self.end_conv_2 = nn.Conv1d(d_model, vocab_size, 1)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        skip_connections = []
        for layer in self.layers:
            x, skip = layer(x)
            skip_connections.append(skip)
        out = sum(skip_connections)
        out = F.relu(out)
        out = F.relu(self.end_conv_1(out))
        logits = self.end_conv_2(out)
        return logits.transpose(1, 2)

# ==============================================================================
# 3. TRANSFORMER ARCHITECTURE (Baseline 2 - Generalist King)
# ==============================================================================
class Transformer_Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 2048, d_model)) # Max len 2048
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dim_feedforward=d_model*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(x)

# ==============================================================================
# DATASET & UTILS
# ==============================================================================
class LibriSpeechTokenized(Dataset):
    def __init__(self, root, url="train-clean-100"):
        self.files = sorted(glob.glob(os.path.join(root, "LibriSpeech", url, "**", "*.flac"), recursive=True))
        if len(self.files) == 0:
            print(f"Downloading LibriSpeech {url} (6.3GB)... This may take a few minutes.")
            torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=True)
            self.files = sorted(glob.glob(os.path.join(root, "LibriSpeech", url, "**", "*.flac"), recursive=True))
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
        return raw_codes[:, 0, :]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_benchmark(model_name, model, dl, tokenizer):
    print(f"\n🥊 STARTING BENCHMARK: {model_name}")
    print(f"📊 Parameters: {count_parameters(model):,}")
    
    model = model.to(CONFIG["device"])
    
    # COMPILATION: Safe now with Vectorized layers (standard cuDNN)
    try:
        print("⚡ Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print(f"⚠️ Compilation failed: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"] * len(dl))
    criterion = nn.CrossEntropyLoss()
    
    stats = []
    start_time = time.time()
    step = 0
    
    model.train()
    
    for epoch in range(CONFIG["epochs"]):
        for batch_waves in dl:
            batch_waves = batch_waves.to(CONFIG["device"])
            with torch.no_grad():
                tokens = tokenizer.encode(batch_waves)
            
            if tokens.size(1) > CONFIG["seq_len"]: tokens = tokens[:, :CONFIG["seq_len"]]
            inputs, targets = tokens[:, :-1], tokens[:, 1:]
            if inputs.size(1) < 10: continue

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, CONFIG["vocab_size"]), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            step += 1
            if step % 10 == 0:
                ppl = math.exp(loss.item())
                bits_per_token = loss.item() / math.log(2)
                elapsed = time.time() - start_time
                print(f"{model_name} | Step {step} | Loss: {loss.item():.4f} | Ppl: {ppl:.2f} | Bits/Tok: {bits_per_token:.3f} | Time: {elapsed:.0f}s")
                stats.append({
                    "model": model_name,
                    "step": step,
                    "loss": loss.item(),
                    "ppl": ppl,
                    "bits_per_token": bits_per_token,
                    "time": elapsed
                })
    
    # Save Model
    torch.save(model.state_dict(), f"{CONFIG['log_dir']}/{model_name}_final.pt")
    return stats

def main():
    # Optimization for A100
    torch.set_float32_matmul_precision('high')
    
    print("🚀 LAUNCHING TOP-TIER AUDIO BENCHMARK (Vectorized INN - 10 Epochs)")
    
    tokenizer = AudioTokenizer(CONFIG["device"])
    ds = LibriSpeechTokenized(root="./librispeech_data")
    
    def collate_fn(batch):
        max_len = min(max([w.size(-1) for w in batch]), 24000*8)
        padded = torch.zeros(len(batch), 1, max_len)
        for i, w in enumerate(batch):
            sl = min(w.size(-1), max_len)
            padded[i, :, :sl] = w[:, :sl]
        return padded

    dl = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    # --- MODEL SETUP (CALIBRATED FOR ~12M PARAMS) ---
    
    # 1. INN (Reference - Efficient Distributed)
    # 20 Neurones (256 dim).
    # Removed dead code inside neurons -> More params for active neurons.
    # Params: 20 * (LSTM(256,256) + Linear) + Mixer + Embed ~= 12M
    inn_model = INN_Model(
        vocab_size=CONFIG["vocab_size"], d_model=256, n_neurons=20, d_rnn=256, n_heads=8
    )
    
    # 2. WaveNet (Medium)
    wavenet_model = WaveNet_Model(
        vocab_size=CONFIG["vocab_size"], d_model=256, n_layers=10, n_blocks=3
    )
    
    # 3. Transformer (Standard GPT-2 Small size)
    transformer_model = Transformer_Model(
        vocab_size=CONFIG["vocab_size"], d_model=512, n_head=8, n_layers=4
    )
    
    models = [
        ("INN", inn_model),
        ("WaveNet", wavenet_model),
        ("Transformer", transformer_model)
    ]
    
    all_stats = []
    
    for name, model in models:
        stats = run_benchmark(name, model, dl, tokenizer)
        all_stats.extend(stats)
        
    # Save CSV
    df = pd.DataFrame(all_stats)
    df.to_csv(f"{CONFIG['log_dir']}/benchmark_results.csv", index=False)
    print("\n✅ Benchmark Complete. Results saved to benchmark_results.csv")

if __name__ == "__main__":
    main()
