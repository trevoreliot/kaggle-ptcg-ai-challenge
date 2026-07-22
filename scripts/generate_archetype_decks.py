import os
import csv

def write_deck(folder, name, card_counts):
    # Ensure folder exists
    os.makedirs(folder, exist_ok=True)
    
    # Expand counts into a flat list
    deck = []
    for card_id, count in card_counts.items():
        deck.extend([card_id] * count)
        
    assert len(deck) == 60, f"Deck {name} must have exactly 60 cards. Got {len(deck)}."
    
    # Check 4-copy rule (basic energies are 1-8)
    for card_id in set(deck):
        if card_id > 8:
            assert deck.count(card_id) <= 4, f"Deck {name} violates 4-copy rule for card {card_id}."
            
    # Write to CSV
    path = os.path.join(folder, f"{name}.csv")
    with open(path, "w", newline="") as f:
        for card_id in deck:
            f.write(f"{card_id}\n")
    print(f"Generated {path}")

def generate_decks():
    base_dir = os.path.join("assets", "decks")
    
    # --- AGGRO DECKS ---
    aggro_dir = os.path.join(base_dir, "aggro")
    write_deck(aggro_dir, "Aggro_Charizard", {
        790: 4, # Mega Charizard X ex
        928: 4, # Mega Charizard Y ex
        1121: 4, # Ultra Ball
        1102: 4, # Dusk Ball
        1218: 4, # Giovanni
        1119: 4, # Energy Search
        2: 36 # Fire Energy
    })
    
    write_deck(aggro_dir, "Aggro_Fighting", {
        678: 4, # Mega Lucario ex
        1121: 4, # Ultra Ball
        1218: 4, # Giovanni
        1119: 4, # Energy Search
        1182: 4, # Boss's Orders
        6: 40 # Fighting Energy
    })
    
    write_deck(aggro_dir, "Aggro_Swarm", {
        945: 4, # Froakie
        1072: 4, # Snorlax
        1121: 4, # Ultra Ball
        1218: 4, # Giovanni
        1118: 4, # Energy Retrieval
        3: 40 # Water Energy
    })
    
    # --- CONTROL DECKS ---
    control_dir = os.path.join(base_dir, "control")
    write_deck(control_dir, "Control_Snorlax_Block", {
        1072: 4, # Snorlax
        1182: 4, # Boss's Orders
        1120: 4, # Crushing Hammer
        1081: 4, # Enhanced Hammer
        1218: 4, # Giovanni
        1: 40 # Grass Energy (Snorlax uses generic)
    })
    
    write_deck(control_dir, "Control_Energy_Denial", {
        533: 4, # Crustle
        1120: 4, # Crushing Hammer
        1081: 4, # Enhanced Hammer
        1182: 4, # Boss's Orders
        1119: 4, # Energy Search
        6: 40 # Fighting Energy
    })
    
    write_deck(control_dir, "Control_Poison_Lock", {
        680: 4, # Toxicroak
        1072: 4, # Snorlax
        1182: 4, # Boss's Orders
        1120: 4, # Crushing Hammer
        1218: 4, # Giovanni
        6: 40 # Fighting Energy
    })
    
    # --- COMBO DECKS ---
    combo_dir = os.path.join(base_dir, "combo")
    write_deck(combo_dir, "Combo_Dragapult_ex", {
        120: 4, # Drakloak
        121: 4, # Dragapult ex
        1121: 4, # Ultra Ball
        1182: 4, # Boss's Orders
        1218: 4, # Giovanni
        5: 40 # Psychic Energy
    })
    
    write_deck(combo_dir, "Combo_Team_Rocket", {
        1072: 4, # Snorlax
        1218: 4, # Team Rocket's Giovanni
        1121: 4, # Ultra Ball
        15: 4, # Team Rocket's Energy
        7: 44 # Dark Energy
    })
    
    write_deck(combo_dir, "Combo_Water_Engine", {
        33: 4, # Froakie
        945: 4, # Froakie
        1118: 4, # Energy Retrieval
        1119: 4, # Energy Search
        1218: 4, # Giovanni
        1121: 4, # Ultra Ball
        3: 36 # Water Energy
    })
    
    print("All 9 archetype decks generated successfully!")

if __name__ == "__main__":
    generate_decks()
