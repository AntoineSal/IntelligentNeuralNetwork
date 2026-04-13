
"""
Audio Benchmark: INN vs Transformer vs LSTM
===========================================
Ce script compare l'INN (Intelligent Neural Network) contre deux baselines classiques
sur la tâche de Speech Language Modeling (LibriSpeech Tokens).

Critère d'équité : Même nombre de paramètres (~3.5M - 4M) pour tous les modèles.
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
from torch.utils.data import DataLoader, Dataset
from transformers import EncodecModel, AutoProcessor

# ==============================================================================
# CONFIGURATION COMMUNE
# ==============================================================================
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sample_rate": 24000,
    "vocab_size": 1024,
    "seq_len": 600,       # Contexte A100
    "batch_size": 64,     # Batch A100
    "epochs": 15,         # Suffisant pour voir la convergence
    "lr": 5e-4,
    "seed": 42,
    "target_params": 4000000 # On vise ~4M params pour tout le monde (Taille INN 12 neurons)
}

# ==============================================================================
# DATASET (Copie conforme du script INN pour équité)
# ==============================================================================
# ... (Je réutilise la logique robuste soundfile/glob du script précédent) ...
class LibriSpeechTokenized(Dataset):
    def __init__(self, root, url="dev-clean"):
        self.files = sorted(glob.glob(os.path.join(root, "LibriSpeech", url, "**", "*.flac"), recursive=True))
        if len(self.files) == 0:
            # Fallback si pas encore téléchargé
            print("Downloading LibriSpeech...")
            torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=True)
            self.files = sorted(glob.glob(os.path.join(root, "LibriSpeech", url, "**", "*.flac"), recursive=True))
            
        self.resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=CONFIG["sample_rate"])
        # Tokenizer setup (On assume EnCodec pré-chargé ou on le charge ici si besoin)
        # Pour le benchmark pur "Model vs Model", on peut pré-calculer ou le faire à la volée.
        # Ici on simplifie : on charge le tokenizer dans le main.

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            wav, sr = sf.read(self.files[idx])
            waveform = torch.from_numpy(wav).float()
            if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
            else: waveform = waveform.t()
            return self.resampler(waveform)
        except:
            return torch.zeros(1, CONFIG["sample_rate"])

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

# ==============================================================================
# BASELINE 1: TRANSFORMER (GPT-Style)
# ==============================================================================
class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1024, d_model)) # Max len 1024
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        seq_len = x.size(1)
        # Causal Mask
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(x)

# ==============================================================================
# BASELINE 2: LSTM (RNN Standard)
# ==============================================================================
class LSTMBaseline(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_size, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, hidden_size, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.head(out)

# ==============================================================================
# INN ARCHITECTURE (Rappel pour comparaison params)
# ==============================================================================
# On n'a pas besoin de la redéfinir si on l'importe, mais pour l'autonomie du script:
class INN_Neuron(nn.Module):
    def __init__(self, d_model, d_rnn, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.lstm = nn.LSTMCell(d_model, d_rnn)
        self.out_proj = nn.Linear(d_rnn, d_model)
    def forward(self, x, hidden_state): # Dummy for param count
        pass

# ==============================================================================
# BENCHMARK ENGINE
# ==============================================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_training(model_name, model, dl, tokenizer):
    print(f"\n🥊 Training {model_name}...")
    print(f"📊 Params: {count_parameters(model):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"] * len(dl))
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    history = []
    
    start_time = time.time()
    step = 0
    
    for epoch in range(CONFIG["epochs"]):
        for batch_waves in dl:
            batch_waves = batch_waves.to(CONFIG["device"])
            
            # Tokenize
            with torch.no_grad():
                tokens = tokenizer.encode(batch_waves)
            
            if tokens.size(1) > CONFIG["seq_len"]:
                tokens = tokens[:, :CONFIG["seq_len"]]
            
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            
            if inputs.size(1) < 10: continue

            # Forward
            optimizer.zero_grad()
            logits = model(inputs) # Baseline signature
            
            loss = criterion(logits.reshape(-1, CONFIG["vocab_size"]), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            step += 1
            if step % 50 == 0:
                ppl = math.exp(loss.item())
                elapsed = time.time() - start_time
                print(f"{model_name} | Step {step} | Ppl: {ppl:.2f} | Time: {elapsed:.1f}s")
                history.append((step, ppl))
                
    return history

def main():
    print("🚀 Initialisation Benchmark Audio...")
    tokenizer = AudioTokenizer(CONFIG["device"])
    ds = LibriSpeechTokenized(root="./librispeech_data")
    
    def collate_fn(batch): # Simplified collate
        max_len = min(max([w.size(-1) for w in batch]), 24000*5)
        padded = torch.zeros(len(batch), 1, max_len)
        for i, w in enumerate(batch):
            sl = min(w.size(-1), max_len)
            padded[i, :, :sl] = w[:, :sl]
        return padded

    dl = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=0)

    # 1. Configurer les modèles pour avoir ~4M params (Comme INN 12 Neurones)
    
    # INN 12 Neurones ~ 4.2M params (Estimé: 12 * (LSTM(512) + Attn(512)) + Embed)
    # Baseline Transformer:
    # d_model=512, nhead=8. 
    # 1 Layer ~ 3M params (SelfAttn + FFN). Donc ~2 Layers max pour matcher.
    baseline_transformer = TransformerBaseline(
        vocab_size=CONFIG["vocab_size"], 
        d_model=512, 
        n_head=8, 
        n_layers=2 # Light Transformer
    ).to(CONFIG["device"])

    # Baseline LSTM:
    # LSTM monolithique. 
    # LSTM(input=512, hidden=1024) -> ~6M params. Trop gros.
    # LSTM(input=512, hidden=768) -> ~4M params. Parfait.
    baseline_lstm = LSTMBaseline(
        vocab_size=CONFIG["vocab_size"], 
        d_model=512, 
        hidden_size=800, # Ajusté pour matcher
        n_layers=1
    ).to(CONFIG["device"])

    print("--- CONCOURS DE TAILLE ---")
    print(f"Transformer Params: {count_parameters(baseline_transformer):,}")
    print(f"LSTM Params       : {count_parameters(baseline_lstm):,}")
    # print(f"INN Params        : ~4,200,000 (Reference)") 
    print("--------------------------")

    # 2. Run Benchmarks
    hist_trans = run_training("Transformer", baseline_transformer, dl, tokenizer)
    hist_lstm = run_training("LSTM", baseline_lstm, dl, tokenizer)
    
    print("\n🏁 BENCHMARK FINISHED")
    print("Copiez ces résultats pour le papier:")
    print("Step,Transformer_Ppl,LSTM_Ppl")
    for i in range(len(hist_trans)):
        s, p_t = hist_trans[i]
        _, p_l = hist_lstm[i] if i < len(hist_lstm) else (0, 0)
        print(f"{s},{p_t:.2f},{p_l:.2f}")

if __name__ == "__main__":
    main()

