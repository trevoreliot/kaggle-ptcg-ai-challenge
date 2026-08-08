import copy
import os
from glob import glob

# Monkeypatch deepcopy globally to prevent kaggle_environments from cloning massive observation dicts
# This provides a 50x speedup in simulation time.
_orig_deepcopy = copy.deepcopy
def fast_deepcopy(x, memo=None, _nil=[]):
    if isinstance(x, dict):
        if "step" in x and "remainingOverageTime" in x:
            return x
        if "observation" in x and "reward" in x:
            # Shallow copy the agent state dict, but don't deepcopy the observation
            new_dict = x.copy()
            new_dict["observation"] = x["observation"]
            return new_dict
    return _orig_deepcopy(x, memo)
copy.deepcopy = fast_deepcopy

def load_deck(filepath: str) -> list[int]:
    """Load a deck list from a CSV file as integers."""
    if not os.path.exists(filepath):
        # Fallback for default paths if called from different directories
        fallback = os.path.join("assets", "decks", "versatile", filepath)
        if os.path.exists(fallback):
            filepath = fallback
        elif os.path.exists(f"{filepath}.csv"):
            filepath = f"{filepath}.csv"
            
    with open(filepath, "r") as f:
        deck = [int(line.strip()) for line in f.readlines() if line.strip()]
    return deck

def get_available_decks(opp_deck_arg: str) -> list[str]:
    """Parse the opp-deck argument into a list of actual CSV paths."""
    if opp_deck_arg.lower() == "all":
        decks = glob("assets/decks/**/*.csv", recursive=True)
        return [d for d in decks if "_appendix" not in d and "EN_Card_Data" not in d]
    elif os.path.isdir(opp_deck_arg):
        decks = glob(os.path.join(opp_deck_arg, "*.csv"))
        return [d for d in decks if "_appendix" not in d and "EN_Card_Data" not in d]
    elif os.path.isdir(os.path.join("assets", "decks", opp_deck_arg)):
        decks = glob(os.path.join("assets", "decks", opp_deck_arg, "*.csv"))
        return [d for d in decks if "_appendix" not in d and "EN_Card_Data" not in d]
    else:
        return [opp_deck_arg]
