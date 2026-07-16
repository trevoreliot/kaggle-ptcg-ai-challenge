import os

def create_deck(path, cards):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for card_id in cards:
            f.write(f"{card_id}\n")

# Base Energy IDs: 
# 1 = Grass, 2 = Fire, 3 = Water, 4 = Lightning, 5 = Psychic, 6 = Fighting, 7 = Darkness, 8 = Metal

# Aggro (Charizard Focus)
aggro = [790] * 4 + [2] * 56
create_deck("assets/decks/aggro/Charizard_Aggro.csv", aggro)

# Control (Snorlax Stall)
control = [1072] * 4 + [5] * 56
create_deck("assets/decks/control/Snorlax_Stall.csv", control)

# Combo (Mewtwo/Alakazam)
combo = [245] * 2 + [743] * 2 + [431] * 4 + [5] * 52
create_deck("assets/decks/combo/Mewtwo_Alakazam.csv", combo)

print("Dummy decks generated successfully!")
