import random
import os
import json
from src.core.parser import parse_observation
from src.core.models.ensemble import EnsembleManager
from src.core.bayesian import BayesianTracker
from src.core.agent.rewards import calculate_step_reward

# Global reference to the environment for local MCTS branching.
# Must be set by main.py before running simulations.
_local_env_ref = None

# Initialize the Ensemble Manager and Bayesian Tracker
ensemble = EnsembleManager()
bayesian_tracker = BayesianTracker()

# Optional ReplayBuffer for offline training tracking
global_replay_buffer = None

# Optional IPC Client for Batched Inference Server
ipc_client = None

# Set by main.py during training to prevent hot-swapping
IS_TRAINING = False

# Set by main.py during evaluate mode for diagnostics
IS_EVALUATING = False
evaluation_telemetry = {
    "entropy": [],
    "hand_sizes": [],
    "action_paralysis": 0
}

# Cache for loaded decks to prevent extreme disk I/O in worker processes
_opp_deck_cache = {}

# State tracking for dense intermediate rewards during training
_state_tracker = {}
for _p in [0, 1]:
    _state_tracker[_p] = {
        "initialized": False,
        "my_prizes": 6,
        "opp_prizes": 6,
        "my_deck": 60,
        "my_energies": 0,
        "my_evolutions": 0,
        "opp_damage": 0,
        "my_damage": 0,
        "my_active_serial": None,
        "last_state": None,
        "last_actions": None,
        "last_log_prob": None,
        "last_value": None
    }

def reset_state_tracking():
    global _state_tracker
    for _p in [0, 1]:
        _state_tracker[_p] = {
            "initialized": False,
            "my_prizes": 6,
            "opp_prizes": 6,
            "my_deck": 60,
            "my_energies": 0,
            "my_evolutions": 0,
            "opp_damage": 0,
            "my_damage": 0,
            "my_active_serial": None,
            "last_state": None,
            "last_actions": None,
            "last_log_prob": None,
            "last_value": None
        }

def load_deck(filepath: str = "Team_Rockets_Box.csv") -> list[int]:
    """Load a deck list from a CSV file."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    search_paths = [
        filepath,
        "deck.csv",
        os.path.join(project_root, "deck.csv"),
        os.path.join(project_root, "assets", "decks", "versatile", filepath),
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
    # The cabt environment does not natively provide 'step' in the observation payload.
    # The deck selection phase (step 0) is uniquely identified by the absence of 'select'.
    if obs_dict.get("select") is None:
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
    from src.core.agent.mcts import MCTSEngine
    
    # Evaluate the current state using the Policy Network to get value for buffer
    if ipc_client is not None:
        state_array = ensemble.encoder.encode(parsed_obs).unsqueeze(0).detach().numpy()
        value, policy_logits = ipc_client.evaluate(state_array)
    else:
        value, policy_logits = ensemble.evaluate(parsed_obs)
    
    if IS_EVALUATING:
        import torch
        import torch.nn.functional as F
        # Calculate entropy
        logits_tensor = torch.tensor(policy_logits, dtype=torch.float32)
        probs = F.softmax(logits_tensor, dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        evaluation_telemetry["entropy"].append(entropy)
        
        # Calculate hand size
        my_player = parsed_obs.current.players[parsed_obs.current.yourIndex]
        my_hand = my_player.hand if my_player.hand is not None else []
        evaluation_telemetry["hand_sizes"].append(len(my_hand))
    
    # Set up MCTS Evaluator
    def mcts_evaluator(search_state):
        if search_state is None or search_state.observation is None:
            return 0.0
        try:
            if ipc_client is not None:
                state_array = ensemble.encoder.encode(search_state.observation).unsqueeze(0).detach().numpy()
                val, _ = ipc_client.evaluate(state_array)
                return val
            else:
                val, _ = ensemble.evaluate(search_state.observation)
                return val
        except Exception:
            return 0.0
        
    num_sims = int(os.environ.get("MCTS_SIMS", 50))
    mcts = MCTSEngine(evaluator=mcts_evaluator, num_simulations=num_sims)
    
    # Approximate opponent deck based on Bayesian Tracker
    best_archetype = bayesian_tracker.best_archetype()
    if best_archetype in _opp_deck_cache:
        opponent_deck_pred = _opp_deck_cache[best_archetype].copy()
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        opp_deck_path = os.path.join(project_root, "assets", "decks", best_archetype, "default.csv")
        if not os.path.exists(opp_deck_path):
            opp_deck_path = os.path.join("assets", "decks", best_archetype, "default.csv")
            
        if os.path.exists(opp_deck_path):
            with open(opp_deck_path, "r") as f:
                opponent_deck_pred = [int(line.strip()) for line in f.readlines() if line.strip()]
        else:
            opponent_deck_pred = [5] * 60
        _opp_deck_cache[best_archetype] = opponent_deck_pred

    temperature = globals().get("CURRENT_TEMPERATURE", 1.0)
    epsilon = globals().get("CURRENT_EPSILON", 0.0)
    
    # Execute Search
    force_explore = False
    if IS_TRAINING and random.random() < epsilon:
        selections = []
        force_explore = True
    else:
        selections = mcts.search(obs_dict, agent_deck, opponent_deck_pred, is_training=IS_TRAINING, temperature=temperature)
    
    # Fallback to policy sampling if MCTS failed
    if not selections:
        options = parsed_obs.select.option
        max_count = min(parsed_obs.select.maxCount, len(options))
        min_count = parsed_obs.select.minCount
        if max_count == 0:
            if IS_EVALUATING:
                evaluation_telemetry["action_paralysis"] += 1
            return []
        valid_logits = np.array(policy_logits[:len(options)])
        
        if IS_TRAINING:
            epsilon = globals().get("CURRENT_EPSILON", 0.01)
            
            # Epsilon-greedy check
            if force_explore or random.random() < epsilon:
                valid_probs = np.ones(len(options)) / len(options)
            else:
                scaled_logits = valid_logits / temperature
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
                valid_probs = exp_logits / exp_logits.sum()
                
            try:
                target_size = max(1, min_count)
                sampled_actions = np.random.choice(len(options), size=target_size, replace=False, p=valid_probs)
                selections = sampled_actions.tolist()
            except ValueError:
                # Fallback if probability array has issues (e.g. fewer non-zero probs than target_size)
                action = np.random.choice(len(options), p=valid_probs)
                selections = [action]
                if min_count > 1:
                    others = [x for x in range(len(options)) if x != action]
                    selections.extend(random.sample(others, min(min_count - 1, len(others))))
        else:
            # Greedy decoding for evaluation/inference
            best_indices = np.argsort(valid_logits)[::-1]
            selections = best_indices[:max_count].tolist()
    
    # Push to Replay Buffer if training (only for Player 1 to avoid mixed rewards in self-play)
    if global_replay_buffer is not None and parsed_obs.current.yourIndex == 0:
        me_idx = parsed_obs.current.yourIndex
        tracker = _state_tracker[me_idx]
        
        # Extracted calculation to rewards.py
        step_reward = calculate_step_reward(parsed_obs, tracker)
        
        try:
            import torch
            my_player = parsed_obs.current.players[me_idx]
            # Re-encode the state to save in buffer (detached)
            state_tensor = ensemble.encoder.encode(parsed_obs).unsqueeze(0).detach()
            dummy_log_prob = torch.tensor([0.0])
            value = torch.tensor([0.0])
            
            actions_to_push = selections if selections else [0]
            
            # Push the previous turn's state/actions to the buffer now that we know its reward
            if tracker.get("last_state") is not None:
                for a in tracker["last_actions"]:
                    global_replay_buffer.push(
                        tracker["last_state"],
                        a,
                        tracker["last_log_prob"],
                        tracker["last_value"],
                        step_reward=step_reward
                    )
                
            # Cache the current turn's state/actions for the next turn
            tracker["last_state"] = state_tensor
            tracker["last_actions"] = actions_to_push
            tracker["last_options"] = parsed_obs.select.option
            tracker["last_hand"] = my_player.hand if my_player.hand is not None else []
            tracker["last_log_prob"] = dummy_log_prob
            tracker["last_value"] = value

        except Exception as e:
            import traceback
            print("Error during replay buffer push:")
            traceback.print_exc()
            
    return selections
