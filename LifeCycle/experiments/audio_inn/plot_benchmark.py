import re
import matplotlib
matplotlib.use('Agg') # Force backend without display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

LOG_FILE = "LifeCycle/experiments/audio_inn/full_libir_logs.txt"
OUTPUT_IMG = "LifeCycle/experiments/audio_inn/benchmark_comparison.png"

def parse_logs(filepath):
    print(f"Reading logs from {filepath}...")
    data = []
    # Regex updated to be flexible with spaces
    pattern = re.compile(r"(\w+)\s+\|\s+Step\s+(\d+)\s+\|\s+Loss:\s+([\d\.]+)\s+\|\s+Ppl:\s+([\d\.]+)")
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            print(f"File has {len(lines)} lines.")
            for line in lines:
                match = pattern.search(line)
                if match:
                    model = match.group(1)
                    step = int(match.group(2))
                    loss = float(match.group(3))
                    ppl = float(match.group(4))
                    
                    if "INN" in model: model = "INN"
                    if "WaveNet" in model: model = "WaveNet"
                    if "Transformer" in model: model = "Transformer"
                    
                    data.append({
                        "Model": model,
                        "Step": step,
                        "Loss": loss,
                        "Ppl": ppl
                    })
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame()

    print(f"Parsed {len(data)} data points.")
    return pd.DataFrame(data)

def plot_benchmark(df):
    print("Generating plot...")
    plt.figure(figsize=(12, 8))
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        print("Style seaborn-v0_8-darkgrid not found, using default.")
        plt.style.use('ggplot')
    
    models = df['Model'].unique()
    print(f"Models found: {models}")
    colors = {'INN': '#2ca02c', 'WaveNet': '#1f77b4', 'Transformer': '#d62728'}
    
    WINDOW = 50 
    
    for model in models:
        subset = df[df['Model'] == model].sort_values('Step')
        if len(subset) == 0: continue
        
        plt.plot(subset['Step'], subset['Ppl'], color=colors.get(model, 'gray'), alpha=0.1)
        
        # Exponential moving average
        smoothed = subset['Ppl'].ewm(span=WINDOW).mean()
        
        min_ppl = smoothed.min()
        final_ppl = smoothed.iloc[-1]
        
        label = f"{model} (Best: {min_ppl:.2f} | Final: {final_ppl:.2f})"
        plt.plot(subset['Step'], smoothed, label=label, linewidth=2.5, color=colors.get(model, 'black'))

    plt.title("LibriSpeech Benchmark: INN vs Baselines (Iso-Parameters ~12M)", fontsize=16, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Perplexity (Lower is Better)", fontsize=12)
    plt.ylim(15, 60)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"✅ Graphique généré : {OUTPUT_IMG}")

if __name__ == "__main__":
    try:
        df = parse_logs(LOG_FILE)
        if not df.empty:
            plot_benchmark(df)
        else:
            print("❌ Aucune donnée trouvée.")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

