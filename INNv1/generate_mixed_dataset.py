import random

def generate_equation():
    """Génère une équation simple 'a+b=c'."""
    # Opérations : +, -, * (pour varier un peu)
    op = random.choice(['+', '-'])
    
    if op == '+':
        a = random.randint(0, 999)
        b = random.randint(0, 999)
        c = a + b
        return f"MATH: {a}+{b}={c}" # On ajoute un préfixe clair pour aider au début
    else:
        a = random.randint(0, 999)
        b = random.randint(0, a) # Pour rester positif simple
        c = a - b
        return f"MATH: {a}-{b}={c}"

def create_mixed_dataset(shakespeare_path, output_path):
    try:
        with open(shakespeare_path, 'r', encoding='utf-8') as f:
            shakespeare_lines = f.readlines()
    except FileNotFoundError:
        print("Fichier shakespeare non trouvé, création d'un dummy.")
        shakespeare_lines = ["To be or not to be", "That is the question"]

    mixed_lines = []
    
    # On garde les lignes non vides
    shakespeare_lines = [line.strip() for line in shakespeare_lines if line.strip()]
    
    # On limite la taille si c'est trop gros (pour équilibrer)
    # Disons qu'on veut 50% maths / 50% texte
    
    for line in shakespeare_lines:
        mixed_lines.append(line) 
        mixed_lines.append(generate_equation())
    
    # Écriture
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in mixed_lines:
            f.write(line + '\n')
            
    print(f"Dataset mixte généré : {len(mixed_lines)} lignes dans {output_path}")
    print("Exemple de contenu :")
    for l in mixed_lines[:6]:
        print(f"  {l}")

if __name__ == "__main__":
    create_mixed_dataset('test/input.txt', 'test/input2.txt')

