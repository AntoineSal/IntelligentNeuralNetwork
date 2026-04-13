
"""
Rerun Transformer Baseline (Calibration ~3.8M Params)
=====================================================
Ajustement pour matcher la taille de l'INN (3.7M).
Config précédente (5.5M) était injuste.

Nouvelle Config:
- d_model = 256 (Inchangé)
- n_layers = 4 (Au lieu de 6)
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import soundfile as sf
import glob
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import EncodecModel, AutoProcessor

# ==============================================================================
# CONFIG IDENTIQUE AU BENCHMARK PRECEDENT
# ==============================================================================
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sample_rate": 24000,
    "vocab_size": 1024,
    "seq_len": 600,
    "batch_size": 64,
    "epochs": 20, # On vise ~850 steps comme la run précédente
    "lr": 5e-4,
    "log_dir": "./benchmark_logs"
}

# ... (Imports des classes Dataset et Tokenizer identiques au script principal) ...
# Pour faire court, je réutilise les imports si possible ou je redéfinis minialement
# Je redéfinis pour autonomie totale du script

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

class LibriSpeechTokenized(Dataset):
    def __init__(self, root, url="dev-clean"): # On reste sur dev-clean pour comparer à la run précédente
        self.files = sorted(glob.glob(os.path.join(root, "LibriSpeech", url, "**", "*.flac"), recursive=True))
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

# ==============================================================================
# TRANSFORMER (RE-CALIBRATED)
# ==============================================================================
class Transformer_Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1024, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dim_feedforward=d_model*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("🚀 RELAUNCHING TRANSFORMER (ISO-PARAMS ~3.8M)")
    
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
    
    # NEW CONFIG: 4 Layers instead of 6
    transformer_model = Transformer_Model(
        vocab_size=CONFIG["vocab_size"], d_model=256, n_head=4, n_layers=4
    )
    
    print(f"📊 New Transformer Params: {count_parameters(transformer_model):,}")
    
    model = transformer_model.to(CONFIG["device"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"] * len(dl))
    criterion = nn.CrossEntropyLoss()
    
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
            if step % 50 == 0:
                ppl = math.exp(loss.item())
                bits_per_token = loss.item() / math.log(2)
                elapsed = time.time() - start_time
                print(f"Transformer-4L | Step {step} | Loss: {loss.item():.4f} | Ppl: {ppl:.2f} | Bits/Tok: {bits_per_token:.3f} | Time: {elapsed:.0f}s")

if __name__ == "__main__":
    main()

