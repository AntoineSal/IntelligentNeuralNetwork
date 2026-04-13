
"""
INN Audio Benchmark - LibriSpeech -> EnCodec -> INN -> Audio
============================================================
Ce script implémente une pipeline complète pour entraîner un Intelligent Neural Network (INN)
sur de la génération audio (Speech Modeling).

Architecture :
1. Input Audio (WAV) -> Neural Audio Codec (EnCodec by Meta) -> Tokens Discrets (Integers)
2. INN Model -> Prédit le prochain Token
3. Output Tokens -> Neural Audio Decoder -> Audio (WAV)

Auteur : INN Team
"""

import os
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import EncodecModel, AutoProcessor

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "exp_name": "INN_Audio_LibriSpeech_A100",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sample_rate": 24000,   # Standard EnCodec 24khz
    "vocab_size": 1024,     # EnCodec codebook size
    "seq_len": 600,         # ~8 secondes d'audio (Contexte long pour A100)
    "batch_size": 64,       # A100 Power (Stabilité du gradient)
    "epochs": 20,           # Plus d'epochs car plus rapide
    "lr": 5e-4,             # LR un peu plus agressif au début
    "save_every": 500,      # Steps
    
    # INN Architecture Params (Scaled Up)
    "n_neurons": 12,        # Doublé pour capacité cognitive
    "d_model": 512,         # Dimension interne
    "d_rnn": 512,           # Taille LSTM interne
    "n_heads": 8,           # Plus de têtes d'attention
}

# ==============================================================================
# 0. INSTALLATION & UTILS (Colab Friendly)
# ==============================================================================
def install_dependencies():
    """Installe les dépendances si non présentes (pour Colab)."""
    import subprocess
    import sys
    try:
        import transformers
        import torchaudio
        import soundfile
    except ImportError:
        print("🔧 Installation des dépendances Audio (Transformers, Torchaudio, SoundFile)...")
        # Installation système de libsndfile1 pour soundfile
        subprocess.run(["apt-get", "install", "-y", "libsndfile1"], check=False)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torchaudio", "accelerate", "soundfile"])

# ==============================================================================
# 1. DATASET & TOKENIZATION
# ==============================================================================
class AudioTokenizer:
    """Wrapper autour de Meta EnCodec pour transformer Audio <-> Tokens."""
    def __init__(self, device):
        print("🔊 Chargement du Neural Codec (EnCodec)...")
        self.device = device
        # Utilisation de la version HF officielle
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.model.eval()
        
    @torch.no_grad()
    def encode(self, waveform):
        """Audio (batch, 1, samples) -> Tokens (batch, seq_len)"""
        # Conversion du batch tensor en LISTE de numpy arrays (Mono)
        # Cela force le processeur à traiter chaque exemple indépendamment
        # waveform est (batch, 1, samples)
        raw_audio = [w.squeeze(0).cpu().numpy() for w in waveform]
        
        inputs = self.processor(raw_audio=raw_audio, 
                                sampling_rate=CONFIG["sample_rate"], 
                                return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.encode(**inputs)
        
        # codes shape logic:
        # outputs.audio_codes est un tuple/list (souvent len=1 pour 1 stream)
        # Le contenu est un tensor (batch, n_codebooks, seq_len) OU (1, batch, n_codebooks, seq_len)
        raw_codes = outputs.audio_codes[0] # Tensor
        
        # On s'assure d'avoir (batch, n_codebooks, seq_len)
        if raw_codes.ndim == 4:
            raw_codes = raw_codes.squeeze(0)
            
        # On extrait le codebook 0 : (batch, seq_len)
        codes = raw_codes[:, 0, :] 
        return codes

    @torch.no_grad()
    def decode(self, tokens):
        """Tokens (batch, seq_len) -> Audio (batch, 1, samples)"""
        # EnCodec (24khz) attend souvent au moins 2 codebooks.
        # Shape attendue par HF decode: (batch, n_codebooks, seq_len)
        
        batch_size, seq_len = tokens.shape
        # On crée un tenseur avec 2 codebooks (le min souvent pour 24khz)
        # Codebook 0 = nos prédictions
        # Codebook 1 = zéros (silence/bruit)
        
        codes = torch.zeros(batch_size, 2, seq_len, dtype=torch.long, device=self.device)
        codes[:, 0, :] = tokens # Fill first codebook
        
        # Decode
        # HF EncodecModel.decode returns [audio_values]
        audio_values = self.model.decode(codes, [None]) 
        return audio_values[0]

def keep_colab_alive():
    """Fonction pour simuler une activité et empêcher le timeout Colab."""
    pass # Placeholder, généralement géré par le user via JS console ou clicker


class LibriSpeechTokenized(Dataset):
    """Dataset robuste qui scanne les fichiers FLAC et utilise SoundFile directement."""
    def __init__(self, root, url="dev-clean", tokenizer=None, seq_len=300):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # 1. Trigger Download via Torchaudio (juste pour récupérer les fichiers)
        print(f"📚 Vérification/Téléchargement de LibriSpeech ({url})...")
        if not os.path.exists(root):
            os.makedirs(root)
        try:
            # On laisse torchaudio gérer le download/extract
            torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=True)
        except Exception as e:
            print(f"Note: Download warning (peut-être déjà présent): {e}")

        # 2. Scan manuel des fichiers FLAC (Bypass du loader Torchaudio buggé)
        import glob
        search_path = os.path.join(root, "LibriSpeech", url, "**", "*.flac")
        self.files = sorted(glob.glob(search_path, recursive=True))
        print(f"📂 Trouvé {len(self.files)} fichiers audio dans {search_path}")
        
        if len(self.files) == 0:
            raise RuntimeError("Aucun fichier FLAC trouvé ! Vérifiez le téléchargement.")

        # Resampler
        self.resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=CONFIG["sample_rate"])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            # Chargement robuste avec SoundFile
            import soundfile as sf
            wav_numpy, sr = sf.read(path)
            # Soundfile retourne (samples, channels) ou (samples,)
            waveform = torch.from_numpy(wav_numpy).float()
            
            # Ensure (1, samples)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t() # (channels, samples)
            
            # Resample 16k -> 24k
            waveform = self.resampler(waveform)
            
            return waveform
            
        except Exception as e:
            print(f"⚠️ Erreur fichier {path}: {e}")
            return torch.zeros(1, CONFIG["sample_rate"])


def audio_collate_fn(batch):
    """Tokenisation par batch pour efficacité GPU."""
    # Batch est une liste de waveforms
    # On pad les waveforms pour en faire un tenseur
    max_len = max([w.size(-1) for w in batch])
    # Cap max len pour éviter OOM
    max_len = min(max_len, CONFIG["sample_rate"] * 5) 
    
    padded_waves = torch.zeros(len(batch), 1, max_len)
    for i, w in enumerate(batch):
        sl = min(w.size(-1), max_len)
        padded_waves[i, :, :sl] = w[:, :sl]
        
    return padded_waves

# ==============================================================================
# 2. INN ARCHITECTURE (Clean Implementation for Sequence Modeling)
# ==============================================================================
class INN_Neuron(nn.Module):
    """
    Un neurone intelligent spécialisé :
    - Possède son propre état récurrent (LSTM)
    - Possède un mécanisme d'attention (MultiHead)
    - Lit le contexte global, met à jour son état, propose une sortie
    """
    def __init__(self, d_model, d_rnn, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.lstm = nn.LSTMCell(d_model, d_rnn)
        
        self.out_proj = nn.Linear(d_rnn, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, hidden_state):
        """
        x: (batch, d_model) - Input courant (embedding du token)
        hidden_state: (hx, cx) tuple pour LSTM
        """
        # 1. Self-Attention "Instantannée" ou sur le contexte passé ?
        # Dans cette version simplifiée INN, le neurone regarde l'input x
        # En réalité, dans un transformer complet, il regarderait la séquence.
        # Ici, on traite token par token (RNN style).
        
        # On triche un peu : on utilise l'attention pour "processer" l'input x
        # x_unsq = x.unsqueeze(1) # (batch, 1, d_model)
        # attn_out, _ = self.attn(x_unsq, x_unsq, x_unsq)
        # x = x + attn_out.squeeze(1)
        # x = self.norm1(x)
        
        # Simplification RNN pur pour stabilité initiale :
        hx, cx = hidden_state
        hx_new, cx_new = self.lstm(x, (hx, cx))
        
        output = self.out_proj(hx_new)
        return output, (hx_new, cx_new)

class INN_Brain(nn.Module):
    """
    Le cerveau qui connecte les neurones.
    Architecture :
    - Embedding Layer (Token -> Vector)
    - N Neurones Indépendants qui traitent l'info en parallèle
    - Mixing Layer (Attention "All-to-All" implicite ou Aggregation)
    - Projection Layer (Vector -> Logits)
    """
    def __init__(self, vocab_size, d_model, n_neurons, d_rnn, n_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.neurons = nn.ModuleList([
            INN_Neuron(d_model, d_rnn, n_heads) for _ in range(n_neurons)
        ])
        
        # Le "Global Workspace" via Attention
        # Au lieu d'un Linear simple, on utilise une Attention pour mixer les avis des neurones
        # Query = Input context, Key/Value = Neurons Outputs
        self.mixer_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_mix = nn.LayerNorm(d_model)
        
        self.head = nn.Linear(d_model, vocab_size)
        
        self.n_neurons = n_neurons
        self.d_rnn = d_rnn
        self.d_model = d_model

    def init_states(self, batch_size, device):
        states = []
        for _ in range(self.n_neurons):
            hx = torch.zeros(batch_size, self.d_rnn, device=device)
            cx = torch.zeros(batch_size, self.d_rnn, device=device)
            states.append((hx, cx))
        return states

    def forward(self, x, states):
        """
        x: (batch, seq_len) - indices
        states: list of states per neuron
        """
        batch_size, seq_len = x.size()
        x_emb = self.embedding(x) # (batch, seq_len, d_model)
        
        # On doit boucler sur le temps car c'est un RNN
        # Pour optimiser, on pourrait utiliser nn.LSTM standard si les neurones n'étaient pas séparés
        # Ici, on fait une boucle manuelle (unroll) ou on utilise TorchScript.
        # Pour la lisibilité Python :
        
        logits_seq = []
        
        # Transpose pour iterer sur seq
        x_emb = x_emb.transpose(0, 1) # (seq_len, batch, d_model)
        
        for t in range(seq_len):
            xt = x_emb[t]
            
            # 1. Chaque neurone traite l'input
            neuron_outputs = []
            new_states = []
            
            for i, neuron in enumerate(self.neurons):
                out, state = neuron(xt, states[i])
                neuron_outputs.append(out)
                new_states.append(state)
            
            states = new_states # Update states
            
            # 2. Aggregation (Le "Conseil des Neurones" avec Attention)
            # neuron_outputs: liste de (batch, d_model) -> Stack -> (batch, n_neurons, d_model)
            neuron_stack = torch.stack(neuron_outputs, dim=1)
            
            # Attention Mixing :
            # Query : L'input courant xt (context) -> (batch, 1, d_model)
            # Key/Value : Les sorties des neurones -> (batch, n_neurons, d_model)
            # Le système demande : "Quel neurone a la meilleure info pour compléter ce contexte xt ?"
            query = xt.unsqueeze(1)
            attn_out, _ = self.mixer_attn(query, neuron_stack, neuron_stack)
            
            # attn_out est (batch, 1, d_model)
            mixed = attn_out.squeeze(1)
            
            mixed = self.norm_mix(mixed + xt) # Residual connection avec l'input
            
            # 3. Prediction
            logit = self.head(mixed)
            logits_seq.append(logit)
            
        # Stack outputs
        logits = torch.stack(logits_seq, dim=1) # (batch, seq_len, vocab_size)
        return logits, states

# ==============================================================================
# 3. TRAINING LOOP
# ==============================================================================
def train():
    install_dependencies()
    
    print(f"🚀 Initialisation INN Audio sur {CONFIG['device']}")
    
    # 1. Init Utils
    tokenizer = AudioTokenizer(CONFIG["device"])
    
    # 2. Dataset
    # On utilise un dataset custom qui wrap LibriSpeech
    # num_workers=0 est CRUCIAL dans Colab pour éviter les erreurs d'import dans les sous-processus
    ds = LibriSpeechTokenized(root="./librispeech_data", tokenizer=tokenizer)
    dl = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=True, 
                    collate_fn=audio_collate_fn, num_workers=0)
    
    # 3. Model
    model = INN_Brain(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_neurons=CONFIG["n_neurons"],
        d_rnn=CONFIG["d_rnn"],
        n_heads=CONFIG["n_heads"]
    ).to(CONFIG["device"])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"] * len(dl))
    criterion = nn.CrossEntropyLoss()
    
    print(f"🧠 Modèle INN créé avec {CONFIG['n_neurons']} neurones intelligents.")
    print(f"🚀 A100 Mode Activated: Batch {CONFIG['batch_size']}, Seq {CONFIG['seq_len']}")
    print(f"📂 Dataset LibriSpeech prêt. Training starts...")
    
    step = 0
    model.train()
    
    for epoch in range(CONFIG["epochs"]):
        print(f"\n🏁 Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch_waves in dl:
            # 1. Tokenization à la volée (Audio -> Codes)
            batch_waves = batch_waves.to(CONFIG["device"])
            with torch.no_grad():
                # encode retourne (batch, seq_len)
                tokens = tokenizer.encode(batch_waves)
                
            # Limiter la séquence pour le training (BPTT limit)
            if tokens.size(1) > CONFIG["seq_len"]:
                # Random crop
                start = random.randint(0, tokens.size(1) - CONFIG["seq_len"])
                tokens = tokens[:, start:start+CONFIG["seq_len"]]
            
            # Input = tokens[:-1], Target = tokens[1:]
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            
            if inputs.size(1) < 10: continue # Skip trop court
            
            # 2. Forward INN
            states = model.init_states(inputs.size(0), CONFIG["device"])
            logits, _ = model(inputs, states)
            
            # 3. Loss
            # Flatten pour CrossEntropy: (batch*seq, vocab)
            loss = criterion(logits.reshape(-1, CONFIG["vocab_size"]), targets.reshape(-1))
            
            # 4. Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step() # Update LR
            
            step += 1
            if step % 10 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f} | Ppl: {math.exp(loss.item()):.2f}")
                
            if step % CONFIG["save_every"] == 0:
                save_checkpoint(model, step)
                generate_sample(model, tokenizer, step)

def save_checkpoint(model, step):
    path = f"./audio_inn_checkpoints/inn_audio_{step}.pt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"💾 Checkpoint saved: {path}")

def generate_sample(model, tokenizer, step):
    """Génère un petit bout d'audio pour monitorer le progrès."""
    model.eval()
    print("🎤 Génération d'un sample...")
    
    # Prompt vide ou aléatoire
    prompt = torch.zeros(1, 1).long().to(CONFIG["device"]) # Start token 0
    states = model.init_states(1, CONFIG["device"])
    
    generated_tokens = []
    
    with torch.no_grad():
        curr_input = prompt
        for _ in range(150): # ~2 secondes
            logits, states = model(curr_input, states)
            # Last token logits
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sampling
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token.item())
            
            curr_input = next_token
            
    # Decode to Audio
    token_tensor = torch.tensor([generated_tokens], device=CONFIG["device"])
    # Il faut peut-être adapter le décodeur pour qu'il accepte ce format brut
    # Pour ce script, on tente le decode direct
    try:
        # Note: ceci échouera si le tokenizer attend 2 codebooks.
        # Il faudra adapter la fonction decode pour padder avec des zéros.
        # Voir AudioTokenizer.decode
        pass 
        # TODO: Implementer la reconstruction audio complète
        # Pour l'instant on print juste les tokens
        print(f"Sample Tokens: {generated_tokens[:20]}...")
    except Exception as e:
        print(f"Erreur génération audio: {e}")
        
    model.train()

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("🛑 Arrêt manuel.")

