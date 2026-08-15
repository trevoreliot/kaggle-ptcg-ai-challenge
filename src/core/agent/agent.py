print("AGENT CALLED!")
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
        "last_value": None,
        "reward_metrics": {}
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
            "last_value": None,
            "reward_metrics": {}
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

def get_global_action_id(opt) -> int:
    """
    Map an option (dict or object) to a fixed global ID between 0 and 2047.
    """
    try:
        def get_val(key, default=0):
            if isinstance(opt, dict):
                return opt.get(key, default)
            return opt.kwargs.get(key, default)
            
        t = opt["type"] if isinstance(opt, dict) else opt.type
        
        if t == 0: # NUMBER
            num = get_val("number", 0)
            return min(60, num)
        elif t == 1: # YES
            return 61
        elif t == 2: # NO
            return 62
        elif t == 3: # CARD
            area = get_val("area", 0)
            idx = get_val("index", 0)
            return 63 + (area * 60) + min(59, idx) # Max 12 * 60 = 720 -> 783
        elif t == 4: # TOOL_CARD
            area = get_val("area", 0)
            idx = get_val("index", 0)
            t_idx = get_val("toolIndex", 0)
            # Area is usually 4 or 5 (Active/Bench), so 2 possibilities * 6 Pokemon * 4 tools
            area_offset = 0 if area == 4 else 1
            return 784 + (area_offset * 24) + (min(5, idx) * 4) + min(3, t_idx) # Max 48 -> 832
        elif t == 5: # ENERGY_CARD
            area = get_val("area", 0)
            idx = get_val("index", 0)
            e_idx = get_val("energyIndex", 0)
            area_offset = 0 if area == 4 else 1
            return 833 + (area_offset * 120) + (min(5, idx) * 20) + min(19, e_idx) # Max 240 -> 1073
        elif t == 6: # ENERGY
            area = get_val("area", 0)
            idx = get_val("index", 0)
            e_idx = get_val("energyIndex", 0)
            area_offset = 0 if area == 4 else 1
            return 1074 + (area_offset * 120) + (min(5, idx) * 20) + min(19, e_idx) # Max 240 -> 1314
        elif t == 7: # PLAY
            idx = get_val("index", 0)
            return 1315 + min(59, idx) # Max 60 -> 1375
        elif t == 8: # ATTACH
            idx = get_val("index", 0)
            in_play_idx = get_val("inPlayIndex", 0)
            in_play_area = get_val("inPlayArea", 4)
            area_offset = 0 if in_play_area == 4 else 1
            # 60 cards * 2 areas * 6 slots = 720 possibilities
            return 1376 + (min(29, idx % 30) * 12) + (area_offset * 6) + min(5, in_play_idx) # Max 360 -> 1736
        elif t == 9: # EVOLVE
            idx = get_val("index", 0)
            in_play_idx = get_val("inPlayIndex", 0)
            in_play_area = get_val("inPlayArea", 4)
            area_offset = 0 if in_play_area == 4 else 1
            return 1737 + (min(9, idx % 10) * 12) + (area_offset * 6) + min(5, in_play_idx) # Max 120 -> 1857
        elif t == 10: # ABILITY
            idx = get_val("index", 0)
            return 1858 + min(59, idx) # Max 60 -> 1918
        elif t == 11: # DISCARD
            idx = get_val("index", 0)
            return 1919 + min(59, idx) # Max 60 -> 1979
        elif t == 12: # RETREAT
            return 1980
        elif t == 13: # ATTACK
            attack_id = get_val("attackId", 0)
            return 1981 + (attack_id % 20)
        elif t == 14: # END
            return 2001
        elif t == 15: # SKILL
            idx = get_val("serial", 0)
            return 2002 + (idx % 20)
        elif t == 16: # SPECIAL_CONDITION
            cond = get_val("specialConditionType", 0)
            return 2022 + min(4, cond)
    except Exception as e:
        pass
    return 2047


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
            
    # ACTION MASKING VALIDATION
    valid_options_dict = []
    valid_options_parsed = []
    
    my_player = parsed_obs.current.players[parsed_obs.current.yourIndex]
    tracker = _state_tracker[parsed_obs.current.yourIndex]
    
    if tracker.get("current_turn") != parsed_obs.current.turn:
        tracker["current_turn"] = parsed_obs.current.turn
        tracker["has_attached_energy_this_turn"] = False
        tracker["current_phase"] = 1
        
    is_main_phase = any(o.type == 14 for o in parsed_obs.select.option)
    
    def get_canonical_phase(t):
        if t == 7: return 1 # PLAY
        if t == 9: return 2 # EVOLVE
        if t == 8: return 3 # ATTACH
        if t in (10, 12, 15): return 4 # ABILITY, RETREAT, SKILL
        if t in (13, 14): return 5 # ATTACK, END
        return 0
        
    valid_mask = []
    original_indices = []
    action_phases = {}
    original_options_dict = obs_dict["select"]["option"]
    original_options_parsed = parsed_obs.select.option
    
    for i, (dict_opt, parsed_opt) in enumerate(zip(obs_dict["select"]["option"], parsed_obs.select.option)):
        opt_type = parsed_opt.type
        is_valid = True
        
        if opt_type == 13: # ATTACK
            active_pkmn = my_player.active[0] if my_player.active and len(my_player.active) > 0 else None
            if active_pkmn:
                energy_count = sum(2 if getattr(e, 'id', None) == 15 else 1 for e in getattr(active_pkmn, 'energyCards', []))
                if energy_count == 0:
                    is_valid = False
        elif opt_type == 8: # ATTACH (Energy)
            if tracker.get("has_attached_energy_this_turn", False):
                is_valid = False
                
        if is_valid and is_main_phase:
            phase = get_canonical_phase(opt_type)
            if phase > 0 and phase < tracker.get("current_phase", 1):
                is_valid = False
                
        valid_mask.append(is_valid)
        if is_valid:
            action_phases[i] = get_canonical_phase(opt_type)
            valid_options_dict.append(dict_opt)
            valid_options_parsed.append(parsed_opt)
            original_indices.append(i)
            
    # Removed observation option mutation here to prevent MCTS index mismatch.
            
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
    alpha = globals().get("CURRENT_ALPHA", 0.0)
    
    # Execute Search
    if globals().get("HUMAN_ASSIST", False):
        if True:
            # We must import at function level because tui.py uses modules that might not be fully initialized or we just avoid top-level dependencies forkaggle
            from src.core.agent.tui import render_board_html, format_option
            
            # Extract mid-game replay JSON if possible
            import json as sys_json
            import src.core.agent.agent as agent_module
            replay_payload = "{}"
            if hasattr(agent_module, "_local_env_ref"):
                try:
                    env = agent_module._local_env_ref
                    env_json = env.toJSON()
                    if "steps" in env_json and len(env_json["steps"]) > 0 and len(env_json["steps"][0]) > 0 and "visualize" in env_json["steps"][0][0]:
                        replay_payload = sys_json.dumps(env_json["steps"][0][0]["visualize"])
                    else:
                        replay_payload = sys_json.dumps(env_json)
                except Exception:
                    pass
            
            html = render_board_html(parsed_obs, replay_payload)
            os.makedirs(os.path.join("assets", "results", "diagnostics"), exist_ok=True)
            path = os.path.join("assets", "results", "diagnostics", "latest_replay.html")
            with open(path, "w") as f:
                f.write(html)
            
            import sys
            from src.core.agent.tui import CONTEXT_MAP
            
            context_id = parsed_obs.select.context if parsed_obs.select else 0
            context_str = CONTEXT_MAP.get(context_id, f"Context {context_id}")
            
            print(f"\n--- HUMAN ASSIST MODE ---", file=sys.__stdout__, flush=True)
            print(f"Turn: {parsed_obs.current.turn}", file=sys.__stdout__, flush=True)
            print(f"Action Context: {context_str}", file=sys.__stdout__, flush=True)
            print(f"Board state rendered to {path}. Refresh your dashboard!", file=sys.__stdout__, flush=True)
            options_path = os.path.join("assets", "results", "diagnostics", "human_options.json")
            action_path = os.path.join("assets", "results", "diagnostics", "human_action.txt")
            
            import json
            import select
            
            opt_texts = [format_option(opt, parsed_obs) for opt in valid_options_parsed]
            min_c = parsed_obs.select.minCount if parsed_obs.select else 1
            max_c = parsed_obs.select.maxCount if parsed_obs.select else 1
            with open(options_path, "w") as f:
                json.dump({
                    "options": opt_texts, 
                    "turn": parsed_obs.current.turn, 
                    "context": context_str,
                    "min_count": min_c,
                    "max_count": max_c
                }, f)
                
            if os.path.exists(action_path):
                os.remove(action_path)

            print("Available Options:", file=sys.__stdout__, flush=True)
            for i, opt in enumerate(valid_options_parsed):
                print(f"[{i}] {opt_texts[i]}", file=sys.__stdout__, flush=True)
                
            sys.__stdout__.write("Select action index in Dashboard, or type here (or type 'agent'): ")
            sys.__stdout__.flush()
            
            user_input = None
            while True:
                if os.path.exists(action_path):
                    with open(action_path, "r") as f:
                        user_input = f.read().strip()
                    os.remove(action_path)
                    break
                
                r, _, _ = select.select([sys.__stdin__], [], [], 0.5)
                if r:
                    user_input = sys.__stdin__.readline().strip()
                    break
                    
            if os.path.exists(options_path):
                os.remove(options_path)
                
            if user_input.lower() != "agent":
                try:
                    choices = [int(c.strip()) for c in user_input.split(',')]
                    valid_choices = [original_indices[c] for c in choices if 0 <= c < len(valid_options_parsed)]
                    if len(valid_choices) == len(choices):
                        return valid_choices
                    print("Invalid index.", file=sys.__stdout__, flush=True)
                except Exception:
                    pass
                    
    force_explore = False
    if IS_TRAINING and alpha > 0.0:
        selections = mcts.search(
            obs_dict, agent_deck, opponent_deck_pred, 
            is_training=IS_TRAINING, temperature=temperature, 
            epsilon=epsilon, alpha=alpha, valid_mask=valid_mask,
            action_phases=action_phases, phase_budgeting=True
        )
    else:
        # MCTS returns index into the original options array (cabt engine indices)
        # We don't map it here since valid_mask handles filtering internally in MCTS
        if IS_TRAINING and random.random() < epsilon:
            selections = []
            force_explore = True
        else:
            selections = mcts.search(
                obs_dict, agent_deck, opponent_deck_pred, 
                is_training=IS_TRAINING, temperature=temperature, 
                valid_mask=valid_mask, action_phases=action_phases, 
                phase_budgeting=True
            )
            
    # Original options were not mutated, so no need to restore.
    
    # Fallback to policy sampling if MCTS failed
    if not selections:
        options = original_options_parsed
        max_count = min(parsed_obs.select.maxCount, len(options))
        min_count = parsed_obs.select.minCount
        if max_count == 0:
            if IS_EVALUATING:
                evaluation_telemetry["action_paralysis"] += 1
            return []
        global_ids = [get_global_action_id(opt) for opt in options]
        valid_logits = np.array([policy_logits[idx] if idx < len(policy_logits) else -1e9 for idx in global_ids])
        
        # Apply strict validity mask to prevent picking illegal actions
        for i, is_v in enumerate(valid_mask):
            if not is_v:
                valid_logits[i] = -1e9
        
        # Count how many VALID options we actually have
        num_valid = sum(1 for is_v in valid_mask if is_v)
        
        # Hard ban on passing the turn if other VALID options exist during training
        # This forces the agent to learn to play its cards instead of stalling
        if IS_TRAINING and num_valid > 1:
            for i, opt in enumerate(options):
                opt_type = opt["type"] if isinstance(opt, dict) else getattr(opt, "type", None)
                if opt_type == 14: # OptionType.END
                    if valid_mask[i]:
                        valid_logits[i] = -1e9
                        valid_mask[i] = False
                    break

        if IS_TRAINING:
            epsilon = globals().get("CURRENT_EPSILON", 0.01)
            alpha = globals().get("CURRENT_ALPHA", 0.0)
            
            # Epsilon-greedy check ONLY if alpha == 0
            if force_explore or (alpha == 0.0 and random.random() < epsilon):
                valid_probs = np.zeros(len(options))
                for i, is_v in enumerate(valid_mask):
                    if is_v:
                        opt = options[i]
                        opt_type = opt["type"] if isinstance(opt, dict) else getattr(opt, "type", None)
                        
                        if opt_type == 13: # ATTACK
                            valid_probs[i] = 100.0 # Combat Bias: Massive attack weight
                        elif opt_type == 8: # ATTACH
                            valid_probs[i] = 5.0 # Combat Bias: Moderate attach weight
                        else:
                            valid_probs[i] = 1.0
                            
                # Failsafe: If everything valid was masked out somehow, just allow the first option
                if valid_probs.sum() == 0:
                    valid_probs[0] = 1.0
                    
                valid_probs = valid_probs / valid_probs.sum()
            else:
                scaled_logits = valid_logits / temperature
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
                valid_probs = exp_logits / exp_logits.sum()
                
            try:
                target_size = max(1, min_count)
                sampled_actions = np.random.choice(len(options), size=target_size, replace=False, p=valid_probs)
                selections = sampled_actions.tolist()
            except ValueError:
                # Fallback if probability array has issues
                action = np.random.choice(len(options), p=valid_probs)
                selections = [action]
                if min_count > 1:
                    others = [x for x in range(len(options)) if x != action and valid_mask[x]]
                    selections.extend(random.sample(others, min(min_count - 1, len(others))))
                    # Failsafe if not enough valid options
                    if len(selections) < min_count:
                        even_more = [x for x in range(len(options)) if x not in selections]
                        selections.extend(random.sample(even_more, min(min_count - len(selections), len(even_more))))
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

            # Track attachments and phase progression
            for a in actions_to_push:
                if a < len(parsed_obs.select.option):
                    opt = parsed_obs.select.option[a]
                    if opt.type == 8:
                        tracker["has_attached_energy_this_turn"] = True
                        
                    if is_main_phase:
                        action_phase = get_canonical_phase(opt.type)
                        if action_phase > 0:
                            tracker["current_phase"] = max(tracker.get("current_phase", 1), action_phase)

        except Exception as e:
            import traceback
            print("Error during replay buffer push:")
            traceback.print_exc()
            
    return selections
