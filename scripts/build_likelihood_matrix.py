import os
import csv
import numpy as np

def build_matrix():
    # 1. Load EN_Card_Data.csv mapping
    card_name_to_id = {}
    en_card_path = "assets/decks/EN_Card_Data.csv"
    with open(en_card_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            if not row: continue
            card_id = int(row[0])
            card_name = row[1].strip().lower()
            card_name_to_id[card_name] = card_id

    # Initialize a 2000x4 matrix with Laplace smoothing base (0.25 uniform)
    # Archetype order: ["aggro", "control", "combo", "versatile"]
    matrix = np.full((2000, 4), 0.25)
    
    # 2. Parse deck folders directly to guarantee exact ID matches
    archetype_dirs = {
        "aggro": 0,
        "control": 1,
        "combo": 2,
        "versatile": 3
    }
    
    import glob
    matched_cards = 0
    for arch, arch_idx in archetype_dirs.items():
        arch_dir = os.path.join("assets", "decks", arch)
        if not os.path.exists(arch_dir):
            continue
            
        for csv_file in glob.glob(os.path.join(arch_dir, "*.csv")):
            with open(csv_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        card_id = int(line.split(',')[0])
                        if card_id < 2000:
                            # Boost this card's likelihood for this archetype
                            matrix[card_id][arch_idx] += 2.0
                            matched_cards += 1
                    except ValueError:
                        pass
    
    print(f"Matched {matched_cards} meta cards to the database.")
    
    # 3. Normalize columns so the sum of probabilities in each archetype = 1.0
    col_sums = matrix.sum(axis=0)
    matrix = matrix / col_sums
    
    # 4. Save to assets/prob/
    out_dir = os.path.join("assets", "prob")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "likelihood_matrix.npy")
    np.save(out_path, matrix)
    print(f"Successfully generated matrix at: {out_path}")
    print(f"Shape: {matrix.shape}")
    
    # Verification Test
    print("\n--- Bayesian Verification Test ---")
    snorlax_id = card_name_to_id.get("snorlax", 0) # Fallback to 0 if not found
    charizard_id = card_name_to_id.get("charizard ex", 0)
    
    print(f"Snorlax (Control) P(Card | Archetype):")
    print(f"Aggro: {matrix[snorlax_id][0]:.6f}, Control: {matrix[snorlax_id][1]:.6f}, Combo: {matrix[snorlax_id][2]:.6f}, Versatile: {matrix[snorlax_id][3]:.6f}")
    
    print(f"\nCharizard ex (Aggro) P(Card | Archetype):")
    print(f"Aggro: {matrix[charizard_id][0]:.6f}, Control: {matrix[charizard_id][1]:.6f}, Combo: {matrix[charizard_id][2]:.6f}, Versatile: {matrix[charizard_id][3]:.6f}")

if __name__ == "__main__":
    build_matrix()
