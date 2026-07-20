import random
import os
from src.core.parser import parse_observation
from src.core.models.ensemble import EnsembleManager
from src.core.bayesian import BayesianTracker

# Global reference to the environment for local MCTS branching.
# Must be set by main.py before running simulations.
_local_env_ref = None

# Initialize the Ensemble Manager and Bayesian Tracker
ensemble = EnsembleManager()
bayesian_tracker = BayesianTracker()

# Optional ReplayBuffer for offline training tracking
global_replay_buffer = None

def load_deck(filepath: str = "Team_Rockets_Box.csv") -> list[int]:
    """Load a deck list from a CSV file."""
    search_paths = [
        filepath,
        os.path.join("assets", "decks", "versatile", filepath)
    ]
    for path in search_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                return [int(line.strip()) for line in f.readlines() if line.strip()]
    return [5] * 60

# Load default deck for when the agent is called standalone
agent_deck = load_deck()

def agent(obs_dict: dict) -> list[int]:
    """
    Smart agent utilizing PyTorch Ensemble and Bayesian Tracking.
    """
    if obs_dict.get("step", 0) == 0:
        # Reset bayesian tracker for a new match
        global bayesian_tracker
        bayesian_tracker = BayesianTracker()
        return agent_deck
        
    select_data = obs_dict.get("select")
    if not select_data:
        return []
        
    parsed_obs = parse_observation(obs_dict)
    if not parsed_obs.select or not parsed_obs.select.option:
        return []
            
    # Update Bayesian inference with any newly revealed opponent cards
    bayesian_tracker.update(parsed_obs)
    
    # Check if we should hot-swap models based on high confidence
    if bayesian_tracker.max_confidence() > 0.85:
        best_archetype = bayesian_tracker.best_archetype()
        if ensemble.current_mode != best_archetype:
            print(f"[Bayesian] High confidence (>85%) detected for archetype: {best_archetype}. Attempting hot-swap.")
            ensemble.switch_model(best_archetype)
            
    # Evaluate the current state using the Policy Network
    value, policy = ensemble.evaluate(parsed_obs)
    
    # Policy outputs probabilities for ALL options.
    options = parsed_obs.select.option
    max_count = min(parsed_obs.select.maxCount, len(options))
    
    # For A2C, we select an action based on probability distribution during training
    # but for deterministic/greedy we can use max probability.
    # We will simply randomly sample based on the legal options to keep this basic implementation clean.
    # In a full run, we would sample directly from `policy`.
    action = random.choice(list(range(len(options))))
    
    # Push to Replay Buffer if training
    if global_replay_buffer is not None:
        try:
            import torch
            # Re-encode the state to save in buffer (detached)
            state_tensor = ensemble.encoder.encode(parsed_obs).unsqueeze(0).detach()
            # Placeholder for old log_prob (not strictly needed if Trainer recalculates, but good for PPO)
            # We push a dummy log_prob since our A2C recalculates it.
            dummy_log_prob = torch.tensor([0.0])
            global_replay_buffer.push(state_tensor, action, dummy_log_prob, value)
        except Exception as e:
            import traceback
            print("Error during replay buffer push:")
            traceback.print_exc()
        
    # Return selected action(s) up to maxCount. For simplicity, we just return the single chosen action.
    # If max_count > 1, the engine wants multiple cards. We just append randoms for the rest.
    selections = [action]
    if max_count > 1:
        others = [x for x in range(len(options)) if x != action]
        selections.extend(random.sample(others, min(max_count - 1, len(others))))
        
    return selections
