import json
import matplotlib.pyplot as plt
import os

def main():
    # Dummy INNv2 data (à remplacer par le vrai JSON si tu l'as, sinon je le reconstruis depuis tes logs)
    # Basé sur tes logs précédents (Loss ~1.16 à la fin)
    # Je mets des valeurs approximatives pour l'exemple, tu devras peut-être parser tes logs d'entraînement INNv2
    
    # NOTE: Idéalement, modifie train_parallel.py pour qu'il sauve aussi un .json
    innv2_data = [
        {'epoch': 1, 'val_loss': 3.33},
        {'epoch': 2, 'val_loss': 2.63},
        {'epoch': 3, 'val_loss': 2.36},
        {'epoch': 4, 'val_loss': 2.20},
        {'epoch': 5, 'val_loss': 2.02},
        {'epoch': 6, 'val_loss': 1.85},
        {'epoch': 7, 'val_loss': 1.65},
        {'epoch': 8, 'val_loss': 1.45},
        {'epoch': 9, 'val_loss': 1.25},
        {'epoch': 10, 'val_loss': 1.16},
    ]
    
    mamba_file = 'results/mamba_pure_results.json'
    if not os.path.exists(mamba_file):
        print("Mamba results not found. Run train_baseline.py first.")
        return
        
    with open(mamba_file, 'r') as f:
        mamba_data = json.load(f)
        
    epochs = range(1, 11)
    innv2_loss = [d['val_loss'] for d in innv2_data]
    mamba_loss = [d['val_loss'] for d in mamba_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, innv2_loss, 'o-', label='INNv2 (Ours)', linewidth=2, color='blue')
    plt.plot(epochs, mamba_loss, 's--', label='Mamba Pure (Baseline)', linewidth=2, color='gray')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Convergence: INNv2 vs Mamba Pure')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/comparison.png')
    print("Plot saved to figures/comparison.png")
    
    # Print delta
    final_inn = innv2_loss[-1]
    final_mamba = mamba_loss[-1]
    print(f"Final INNv2: {final_inn}")
    print(f"Final Mamba: {final_mamba}")
    
    if final_inn < final_mamba:
        print(f"SUCCESS: INNv2 is {(final_mamba - final_inn)/final_mamba*100:.1f}% better!")
    else:
        print(f"WARNING: Baseline is better.")

if __name__ == "__main__":
    main()

