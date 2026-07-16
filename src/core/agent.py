import random
import os
from src.core.parser import parse_observation

def load_deck() -> list[int]:
    """Load a deck list from a CSV file."""
    deck_paths = [
        "deck.csv", 
        "assets/decks/versatile/Team_Rockets_Box.csv"
    ]
    for path in deck_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                return [int(line.strip()) for line in f.readlines() if line.strip()]
    return [5] * 60 # Fallback to 60 basic energies

# Load the deck once when the agent module is imported
agent_deck = load_deck()

def agent(obs_dict: dict) -> list[int]:
    """
    A basic agent that randomly selects valid actions.
    """
    # The cabt engine expects the deck as the action for step 0 when running via standard `make`
    if obs_dict.get("step", 0) == 0:
        return agent_deck
        
    # Sometimes the engine passes None when there are no valid actions to take
    select_data = obs_dict.get("select")
    if not select_data:
        return []
        
    # Parse the observation into our typed dataclasses
    obs = parse_observation(obs_dict)
    
    if not obs.select or not obs.select.option:
        return []
        
    # Select randomly from the available options up to maxCount
    return random.sample(list(range(len(obs.select.option))), min(obs.select.maxCount, len(obs.select.option)))
