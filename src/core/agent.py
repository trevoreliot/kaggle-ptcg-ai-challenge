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

# Set by main.py during training to prevent hot-swapping
IS_TRAINING = False

# Cache for loaded decks to prevent extreme disk I/O in worker processes
_opp_deck_cache = {}

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
    if not IS_TRAINING and bayesian_tracker.max_confidence() > 0.85:
        best_archetype = bayesian_tracker.best_archetype()
        if ensemble.current_mode != best_archetype:
            print(f"[Bayesian] High confidence (>85%) detected for archetype: {best_archetype}. Attempting hot-swap.")
            ensemble.switch_model(best_archetype)
            
    import numpy as np
    from src.core.mcts import MCTSEngine
    
    # Evaluate the current state using the Policy Network to get value for buffer
    value, policy_logits = ensemble.evaluate(parsed_obs)
    
    # Set up MCTS Evaluator
    def mcts_evaluator(search_state):
        # We can just return a heuristic or 0.0 for now, 
        # since deep state conversion is complex.
        return 0.0
        
    mcts = MCTSEngine(evaluator=mcts_evaluator, num_simulations=5) # 5 simulations for speed
    
    # Approximate opponent deck based on Bayesian Tracker
    best_archetype = bayesian_tracker.best_archetype()
    if best_archetype in _opp_deck_cache:
        opponent_deck_pred = _opp_deck_cache[best_archetype].copy()
    else:
        opp_deck_path = os.path.join("assets", "decks", best_archetype, "default.csv")
        if os.path.exists(opp_deck_path):
            with open(opp_deck_path, "r") as f:
                opponent_deck_pred = [int(line.strip()) for line in f.readlines() if line.strip()]
        else:
            opponent_deck_pred = [5] * 60
        _opp_deck_cache[best_archetype] = opponent_deck_pred

    # Execute Search
    selections = mcts.search(obs_dict, agent_deck, opponent_deck_pred)
    
    # Fallback to policy sampling if MCTS failed
    if not selections:
        options = parsed_obs.select.option
        max_count = min(parsed_obs.select.maxCount, len(options))
        valid_logits = np.array(policy_logits[:len(options)])
        exp_logits = np.exp(valid_logits - np.max(valid_logits))
        valid_probs = exp_logits / exp_logits.sum()
        action = np.random.choice(len(options), p=valid_probs)
        selections = [action]
        if max_count > 1:
            others = [x for x in range(len(options)) if x != action]
            selections.extend(random.sample(others, min(max_count - 1, len(others))))
    
    # Push to Replay Buffer if training
    if global_replay_buffer is not None:
        try:
            import torch
            # Re-encode the state to save in buffer (detached)
            state_tensor = ensemble.encoder.encode(parsed_obs).unsqueeze(0).detach()
            dummy_log_prob = torch.tensor([0.0])
            action = selections[0] if selections else 0
            global_replay_buffer.push(state_tensor, action, dummy_log_prob, value)
        except Exception as e:
            import traceback
            print("Error during replay buffer push:")
            traceback.print_exc()
            
    return selections
