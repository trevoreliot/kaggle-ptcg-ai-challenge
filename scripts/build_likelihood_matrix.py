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
    
    # 2. Parse pokemon_tcg_meta_cards.csv
    meta_path = "assets/decks/_appendix/pokemon_tcg_meta_cards.csv"
    
    matched_cards = 0
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or len(row) < 4: continue
            
            raw_name = row[0].strip().lower()
            category = row[1].strip()
            
            try:
                weight = float(row[3])
            except ValueError:
                weight = 0.5
                
            # Match cardId (Handle parentheses like "Snorlax (Block)")
            base_name = raw_name.split("(")[0].strip()
            
            found_id = None
            if raw_name in card_name_to_id:
                found_id = card_name_to_id[raw_name]
            elif base_name in card_name_to_id:
                found_id = card_name_to_id[base_name]
            else:
                # Partial match fallback
                for db_name, db_id in card_name_to_id.items():
                    if base_name in db_name or db_name in base_name:
                        found_id = db_id
                        break
                        
            if found_id is not None:
                matched_cards += 1
                # Route weight to archetypes: ["aggro", "control", "combo", "versatile"]
                dist = np.zeros(4)
                
                if "Aggressive" in category:
                    dist[0] = weight
                    remaining = (1.0 - weight) / 3
                    dist[1:] = remaining
                elif "Combo/Control" in category:
                    # Boost both control and combo equally
                    dist[1] = weight / 2
                    dist[2] = weight / 2
                    remaining = (1.0 - weight) / 2
                    dist[0] = remaining / 2
                    dist[3] = remaining / 2
                elif "Versatile/Toolbox" in category:
                    dist[3] = weight
                    remaining = (1.0 - weight) / 3
                    dist[:3] = remaining
                elif "Engine" in category:
                    # Universal, flat boost
                    dist = np.full(4, 0.25)
                else:
                    dist = np.full(4, 0.25)
                    
                matrix[found_id] = dist
    
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
