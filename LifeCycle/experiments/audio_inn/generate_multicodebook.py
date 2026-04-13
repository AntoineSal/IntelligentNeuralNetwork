"""
Generate Multi-Codebook Audio with SoundTrack v1
================================================
Generates audio from a trained INN Multi-Codebook model ("SoundTrack").

Logic:
1. Loads the trained model.
2. Generates tokens frame-by-frame (Autoregressive) for all 4 codebooks.
3. Decodes tokens to audio using EnCodec.
"""

import os
import torch
import torch.nn as nn
import torchaudio
from transformers import EncodecModel, AutoProcessor
import soundfile as sf

# ==============================================================================
# CONFIG (Must match Training)
# ==============================================================================
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sample_rate": 24000,
    "vocab_size": 1024,
    "n_codebooks": 4,
    "d_model": 512,
    "n_neurons": 24,
    "d_rnn": 256,
    "n_heads": 8,
    "checkpoint_path": "./multicodebook_logs/inn_hybrid_final.pt", # Or intermediate checkpoint
    "output_path": "./generated_sample.wav",
    "generation_len": 225  # Tokens (225 * 4 codebooks / 75Hz = 3 seconds)
}

# ==============================================================================
# MODEL ARCHITECTURE (Copy-Pasted for Standalone Execution)
# ==============================================================================
class INN_Neuron(nn.Module):
    def __init__(self, d_model, d_rnn):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_rnn, batch_first=True)
        self.out_proj = nn.Linear(d_rnn, d_model)
        self.ln_out = nn.LayerNorm(d_model)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        proj = self.out_proj(lstm_out)
        return self.ln_out(proj)

class INN_MultiCodebook(nn.Module):
    def __init__(self, vocab_size, d_model, n_neurons, d_rnn, n_heads, n_codebooks=4):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.d_model = d_model
        self.n_neurons = n_neurons
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, d_model, padding_idx=vocab_size) 
            for _ in range(n_codebooks)
        ])
        
        self.input_norm = nn.LayerNorm(d_model)
        self.neurons = nn.ModuleList([INN_Neuron(d_model, d_rnn) for _ in range(n_neurons)])
        self.mixer_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_mix = nn.LayerNorm(d_model)
        
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(n_codebooks)
        ])

    def forward(self, x):
        B, K, T = x.size()
        x_fused = torch.zeros(B, T, self.d_model, device=x.device)
        for k in range(self.n_codebooks):
            x_fused += self.embeddings[k](x[:, k, :])
            
        x_fused = self.input_norm(x_fused)
        neuron_outputs = [n(x_fused) for n in self.neurons]
        stack = torch.stack(neuron_outputs, dim=2)
        
        stack_flat = stack.reshape(-1, self.n_neurons, self.d_model)
        query_flat = x_fused.reshape(-1, 1, self.d_model)
        
        attn_out, _ = self.mixer_attn(query_flat, stack_flat, stack_flat)
        mixed = self.norm_mix(attn_out.squeeze(1) + query_flat.squeeze(1))
        mixed = mixed.view(B, T, self.d_model)
        
        logits_list = []
        for k in range(self.n_codebooks):
            logits_list.append(self.heads[k](mixed))
            
        return torch.stack(logits_list, dim=1)

# ==============================================================================
# GENERATION LOGIC
# ==============================================================================
@torch.no_grad()
def generate():
    print(f"🚀 Loading SoundTrack Model from {CONFIG['checkpoint_path']}...")
    
    # 1. Load Model
    model = INN_MultiCodebook(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_neurons=CONFIG["n_neurons"],
        d_rnn=CONFIG["d_rnn"],
        n_heads=CONFIG["n_heads"],
        n_codebooks=CONFIG["n_codebooks"]
    ).to(CONFIG["device"])
    
    try:
        model.load_state_dict(torch.load(CONFIG["checkpoint_path"], map_location=CONFIG["device"]))
        print("✅ Weights loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Checkpoint not found at {CONFIG['checkpoint_path']}. Using random weights (garbage output).")

    model.eval()
    
    # 2. Autoregressive Generation Loop
    # Start with a "Silence" or specific prompt. Here: Zero/Silence initialization
    # Shape: (Batch=1, n_codebooks, seq_len)
    
    # Prompt: A few silence frames to warm up
    generated_codes = torch.zeros(1, CONFIG["n_codebooks"], 1, dtype=torch.long).to(CONFIG["device"])
    
    print(f"🎹 Generating {CONFIG['generation_len']} frames (~{CONFIG['generation_len']/75:.1f}s)...")
    
    for _ in range(CONFIG["generation_len"]):
        # Input is the full sequence generated so far
        # Note: In production, we would use KV-Cache or State-Passing to be O(1) per step.
        # Here we do O(T) re-computation for simplicity as T is small.
        
        logits = model(generated_codes) # (1, 4, T, Vocab)
        
        # Take the last timestep's logits
        next_token_logits = logits[:, :, -1, :] # (1, 4, Vocab)
        
        # Sampling (Greedy or Temperature)
        # For Codebook 0 (Structure): Temperature Sampling
        # For Codebook 1-3 (Texture): Greedy or Low Temp usually works better to avoid artifacts
        
        next_tokens_list = []
        for k in range(CONFIG["n_codebooks"]):
            probs = torch.softmax(next_token_logits[:, k, :] / 1.0, dim=-1) # Temp=1.0
            token = torch.multinomial(probs, 1) # Sample
            next_tokens_list.append(token)
            
        next_col = torch.stack(next_tokens_list, dim=1).squeeze(-1) # (1, 4)
        
        # Append
        generated_codes = torch.cat([generated_codes, next_col.unsqueeze(2)], dim=2)
        
    print("✨ Generation Complete.")
    
    # 3. Decode to Audio
    print("🔊 Decoding to Waveform...")
    audio_decoder = EncodecModel.from_pretrained("facebook/encodec_24khz").to(CONFIG["device"])
    audio_decoder.eval()
    
    # Prepare codes for EnCodec: (1, 1, n_codebooks, seq_len)
    final_codes = generated_codes.unsqueeze(0) 
    
    # Decode
    # Warning: EnCodec expects specific bandwidth.
    # 4 codebooks = 3.0 kbps
    # If trained on 4 codebooks, we must decode with 4 codebooks.
    # EnCodec decode() takes 'audio_codes' and 'audio_scales'. 
    # Usually model.decode(audio_codes, audio_scales, padding_mask)
    # But huggingface API is simpler.
    
    with torch.no_grad():
        # Using the huggingface forward method which decodes internally if audio_codes is passed?
        # Actually, let's use the low-level decode
        # audio_values = audio_decoder.decode(final_codes, [None], None)[0] 
        # API might vary. Let's use the standard flow:
        
        # audio_codes shape must be (batch, n_chunks, n_codebooks, seq_len) -> (1, 1, 4, T)
        # We also need scale. EnCodec is scale-invariant usually or uses None.
        
        # Hack: Pass it through the model structure
        audio_out = audio_decoder.decode(final_codes, [None]) # Returns (audio_values, )
        waveform = audio_out[0].squeeze(0).cpu() # (1, samples)
        
    # 4. Save
    sf.write(CONFIG["output_path"], waveform.numpy().T, 24000)
    print(f"💾 Saved to {CONFIG['output_path']}")

if __name__ == "__main__":
    generate()

