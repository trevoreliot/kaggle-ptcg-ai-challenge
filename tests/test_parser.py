from kaggle_environments import make
import json
import logging
import os
import sys

# Add the project root to sys.path so we can import from src regardless of where this script is run
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.agent import load_deck
from src.core.parser import parse_observation
from dataclasses import asdict

os.environ["LITELLM_LOG"] = "ERROR"
os.environ["SUPPRESS_LITELLM_WARNINGS"] = "True"
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

def get_missing_keys(raw_dict, parsed_dict, path=""):
    missing = set()
    if not isinstance(raw_dict, dict) or not isinstance(parsed_dict, dict):
        return missing
        
    for k, v in raw_dict.items():
        if k not in parsed_dict:
            missing.add(f"{path}.{k}" if path else k)
        elif isinstance(v, dict):
            missing.update(get_missing_keys(v, parsed_dict[k], f"{path}.{k}" if path else k))
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            if parsed_dict[k] and len(parsed_dict[k]) > 0 and isinstance(parsed_dict[k][0], dict):
                missing.update(get_missing_keys(v[0], parsed_dict[k][0], f"{path}.{k}[]" if path else f"{k}[]"))
    return missing

def main():
    print("Initializing environment for parser test...")
    deck = load_deck()
    env = make("cabt", configuration={"decks": [list(deck), list(deck)]})
    
    trainer = env.train([None, "random"])
    obs = trainer.reset()
    
    all_missing_keys = set()
    step_count = 0
    
    while not env.done:
        step_count += 1
        
        # We wrap it in a dict because Kaggle's trainer obs is a dotdict or similar sometimes
        raw_dict = dict(obs) 
        
        if "current" in raw_dict:
            parsed_obs = parse_observation(raw_dict)
            parsed_dict = asdict(parsed_obs)
            
            missing = get_missing_keys(raw_dict, parsed_dict)
            all_missing_keys.update(missing)
            
        action = [deck] if raw_dict.get("step") == 0 else []
        if "select" in raw_dict and raw_dict["select"]:
            import random
            options = raw_dict["select"].get("option", [])
            max_count = raw_dict["select"].get("maxCount", 1)
            if options:
                action = random.sample(range(len(options)), min(max_count, len(options)))
                
        try:
            obs, _, _, _ = trainer.step(action)
        except Exception:
            break
            
    print(f"Parsed {step_count} steps.")
    print("Fields present in engine but NOT in our parser:")
    for k in sorted(list(all_missing_keys)):
        print(f" - {k}")

if __name__ == "__main__":
    main()
