import random
import os
import json
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

# Set by main.py during evaluate mode for diagnostics
IS_EVALUATING = False
evaluation_telemetry = {
    "entropy": [],
    "hand_sizes": [],
    "action_paralysis": 0
}

# Cache for loaded decks to prevent extreme disk I/O in worker processes
_opp_deck_cache = {}

# Load external reward shaping configuration
# Defaults
REWARD_CONFIG = {
    "r_prize_taken": 0.5,
    "r_prize_lost": -0.2,
    "r_deck_out": -2.0,
    "r_energy_attach": 0.05,
    "r_evolution": 0.10,
    "r_damage_dealt_per_10": 0.01
}
try:
    _reward_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "reward", "reward_shaping.json"))
    if os.path.exists(_reward_path):
        with open(_reward_path, "r") as _f:
            REWARD_CONFIG.update(json.load(_f))
except Exception as e:
    print(f"Warning: Could not load reward_shaping.json. Using defaults. ({e})")

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
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    from src.core.mcts import MCTSEngine
    
    # Evaluate the current state using the Policy Network to get value for buffer
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
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    
    # Execute Search
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
            if random.random() < epsilon:
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
        # Calculate step rewards for dense reward shaping
        step_reward = 0.0
        
        me_idx = parsed_obs.current.yourIndex
        my_player = parsed_obs.current.players[me_idx]
        opp_player = parsed_obs.current.players[1 - me_idx]
        my_prizes = len([p for p in my_player.prize if p is not None])
        opp_prizes = len([p for p in opp_player.prize if p is not None])
        my_deck = my_player.deckCount
        
        my_all_poke = []
        if my_player.active: my_all_poke.extend(my_player.active)
        if my_player.bench: my_all_poke.extend(my_player.bench)
        my_all_poke = [p for p in my_all_poke if p is not None]
            
        opp_all_poke = []
        if opp_player.active: opp_all_poke.extend(opp_player.active)
        if opp_player.bench: opp_all_poke.extend(opp_player.bench)
        opp_all_poke = [p for p in opp_all_poke if p is not None]
        
        current_energies = sum(len(p.energyCards) for p in my_all_poke)
        current_evolutions = sum(len(p.preEvolution) for p in my_all_poke)
        current_opp_damage = sum((p.maxHp - p.hp) for p in opp_all_poke)
        current_my_damage = sum((p.maxHp - p.hp) for p in my_all_poke)
        current_my_active_serial = my_player.active[0].serial if my_player.active and my_player.active[0] is not None else None
        
        has_fezandipiti = any(p.id == 140 for p in my_all_poke)
        has_ursulana = any(p.id == 44 for p in my_all_poke)
        
        tracker = _state_tracker[me_idx]
        
        # Calculate deltas if state is initialized
        if tracker["initialized"]:
            prizes_taken = tracker["opp_prizes"] - opp_prizes
            prizes_lost = tracker["my_prizes"] - my_prizes
            
            # 1. Prize conditions
            if prizes_taken > 0:
                step_reward += REWARD_CONFIG["r_prize_taken"] * prizes_taken
                if tracker.get("boss_played_turn") == parsed_obs.current.turn:
                    step_reward += REWARD_CONFIG.get("r_boss_ko_bonus", 0.5) * prizes_taken
            if prizes_lost > 0:
                step_reward += REWARD_CONFIG["r_prize_lost"] * prizes_lost
                if tracker.get("had_fezandipiti", False) and not has_fezandipiti:
                    step_reward += REWARD_CONFIG.get("r_prize_lost_fezandipiti_penalty", -0.5) * prizes_lost
                    
            if my_deck == 0 and tracker["my_deck"] > 0:
                step_reward += REWARD_CONFIG["r_deck_out"]
                
            # Card Play Reward
            if tracker.get("last_options") is not None:
                for act in tracker["last_actions"]:
                    if act < len(tracker["last_options"]):
                        opt = tracker["last_options"][act]
                        if opt.type == 4: # OptionType.PLAY
                            step_reward += REWARD_CONFIG.get("r_play_trainer", 0.05)
                            if opt.area == 2 and tracker.get("last_hand") is not None:
                                try:
                                    played_card = tracker["last_hand"][opt.index]
                                    if played_card.id in (1182, 1088, 1218):
                                        tracker["boss_played_turn"] = parsed_obs.current.turn
                                    elif played_card.id == 1251:
                                        step_reward += REWARD_CONFIG.get("r_play_stadium", 0.05)
                                except:
                                    pass
                
            # 2. Dense Setup Rewards
            energy_delta = current_energies - tracker["my_energies"]
            if energy_delta > 0:
                step_reward += REWARD_CONFIG["r_energy_attach"] * energy_delta
                
            evo_delta = current_evolutions - tracker["my_evolutions"]
            if evo_delta > 0:
                step_reward += REWARD_CONFIG["r_evolution"] * evo_delta
                
            # Retreating Reward
            if current_my_active_serial is not None and tracker.get("my_active_serial") is not None:
                if current_my_active_serial != tracker["my_active_serial"] and prizes_lost == 0:
                    step_reward += REWARD_CONFIG.get("r_retreat", 0.10)
                    
            # Healing Reward
            my_damage_delta = current_my_damage - tracker.get("my_damage", 0)
            if my_damage_delta < 0 and prizes_lost == 0:
                step_reward += REWARD_CONFIG.get("r_healing_per_10", 0.02) * (-my_damage_delta / 10.0)
                
            # 3. Dense Attack Rewards
            damage_delta = current_opp_damage - tracker["opp_damage"]
            if damage_delta > 0:
                base_dmg_reward = REWARD_CONFIG["r_damage_dealt_per_10"] * (damage_delta / 10.0)
                if has_ursulana and my_player.active and my_player.active[0] is not None and my_player.active[0].id == 44:
                    base_dmg_reward += REWARD_CONFIG.get("r_ursulana_attack_bonus_per_prize", 0.02) * (6 - opp_prizes)
                step_reward += base_dmg_reward
                
        tracker["my_prizes"] = my_prizes
        tracker["opp_prizes"] = opp_prizes
        tracker["my_deck"] = my_deck
        tracker["my_energies"] = current_energies
        tracker["my_evolutions"] = current_evolutions
        tracker["opp_damage"] = current_opp_damage
        tracker["my_damage"] = current_my_damage
        tracker["my_active_serial"] = current_my_active_serial
        tracker["had_fezandipiti"] = has_fezandipiti
        tracker["initialized"] = True
        try:
            import torch
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
