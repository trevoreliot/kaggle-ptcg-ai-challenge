import os
import numpy as np

# Total known card ID space is around ~1500 for this challenge.
MAX_CARD_ID = 2000 
ARCHETYPES = ["aggro", "control", "combo", "versatile"]

def generate_likelihood_matrix():
    # Shape: [Num Cards, Num Archetypes]
    # Initialize with Laplace smoothing (alpha = 1) so P(Card|Archetype) > 0 for all cards.
    counts = np.ones((MAX_CARD_ID, len(ARCHETYPES)))
    
    # Base directory
    base_dir = "assets/decks"
    
    for arch_idx, arch in enumerate(ARCHETYPES):
        arch_dir = os.path.join(base_dir, arch)
        if not os.path.exists(arch_dir):
            continue
            
        # Read all deck CSVs in the archetype folder
        for filename in os.listdir(arch_dir):
            if filename.endswith(".csv"):
                filepath = os.path.join(arch_dir, filename)
                with open(filepath, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.isdigit():
                            card_id = int(line)
                            if card_id < MAX_CARD_ID:
                                counts[card_id, arch_idx] += 1
                                
    # Calculate P(Card | Archetype) = count / sum_of_counts_in_archetype
    # We sum down the columns (axis=0)
    totals = np.sum(counts, axis=0)
    
    likelihood_matrix = counts / totals
    
    # Save the matrix to disk
    out_path = os.path.join("assets", "likelihood_matrix.npy")
    np.save(out_path, likelihood_matrix)
    print(f"Likelihood matrix generated and saved to {out_path}")
    print(f"Archetypes mapping: {ARCHETYPES}")

if __name__ == "__main__":
    generate_likelihood_matrix()
