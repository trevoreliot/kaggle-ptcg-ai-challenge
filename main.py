import json
import os
import logging

# Suppress litellm and other noisy warnings from kaggle-environments
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["SUPPRESS_LITELLM_WARNINGS"] = "True"
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

from kaggle_environments import make
from src.core.agent import agent

def load_deck(filepath: str) -> list[int]:
    """Load a deck list from a CSV file as integers."""
    with open(filepath, "r") as f:
        # Assuming the CSV contains card IDs on each line, one per card.
        # We need to strip whitespace, ignore empty lines, and convert to int.
        deck = [int(line.strip()) for line in f.readlines() if line.strip()]
    return deck

def main():
    print("Initializing environment...")
    # Load the Team Rockets Box deck
    deck_path = "assets/decks/versatile/Team_Rockets_Box.csv"
    try:
        deck = load_deck(deck_path)
        print(f"Loaded {len(deck)} cards from {deck_path}")
    except Exception as e:
        print(f"Failed to load deck: {e}")
        return

    # Initialize the cabt environment
    # Pass copies of the deck to avoid reference mutation issues if the engine modifies them
    env = make("cabt", configuration={"decks": [list(deck), list(deck)]})
    
    print("Running match...")
    # Run the simulation with two random agents
    env.run([agent, agent])
    
    print("Match finished. Saving results...")
    # Save the output to an HTML file
    result_path = "result.html"
    with open(result_path, "w") as f:
        f.write(env.render(mode="html"))
        
    print(f"Simulation saved to {result_path}")

if __name__ == "__main__":
    main()
