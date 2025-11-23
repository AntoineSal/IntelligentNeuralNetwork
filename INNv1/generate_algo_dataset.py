import random
import string

def generate_reverse_task(min_len=5, max_len=20):
    """Génère une tâche de renversement de chaîne."""
    length = random.randint(min_len, max_len)
    # On utilise des lettres minuscules pour limiter le vocabulaire
    chars = string.ascii_lowercase
    sequence = ''.join(random.choice(chars) for _ in range(length))
    reversed_seq = sequence[::-1]
    
    # Format : "REV: sequence -> reversed"
    return f"REV: {sequence} -> {reversed_seq}"

def create_algo_dataset(output_path, num_samples=10000):
    with open(output_path, 'w', encoding='utf-8') as f:
        for _ in range(num_samples):
            line = generate_reverse_task()
            f.write(line + '\n')
            
    print(f"Dataset algorithmique (Reverse) généré : {num_samples} exemples dans {output_path}")
    print("Exemple :")
    print(generate_reverse_task())

if __name__ == "__main__":
    # On crée le dossier test si besoin
    import os
    if not os.path.exists('test'):
        os.makedirs('test')
    create_algo_dataset('test/input_algo.txt')

