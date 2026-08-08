import os
import sys
import json
import random
from tqdm import tqdm
from kaggle_environments import make
import src.core.agent.agent as agent_module
from src.core.utils.utils import load_deck, get_available_decks

def run_evaluate(episodes, p1_deck_arg, opp_deck_arg, model_name, p2_type, p2_agent_arg):
    print(f"Running {episodes} matches in 'evaluate' mode...")
    
    agent_module.IS_TRAINING = False
    agent_module.IS_EVALUATING = True
    
    opp_decks = get_available_decks(opp_deck_arg)
    
    checkpoint_path = os.path.join("assets", "models", model_name)
    if os.path.exists(checkpoint_path):
        import torch
        agent_module.ensemble.active_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"Loaded PyTorch checkpoint from {checkpoint_path} for evaluation!")
        
    diagnostics = {
        "matches": episodes,
        "wins": 0,
        "avg_length": 0,
        "avg_hand_size": 0,
        "avg_entropy": 0,
        "total_action_paralysis": 0,
        "first_prize_taken_pct": 0,
        "avg_pokemon_kos_received": 0
    }
    
    lengths, hand_sizes, entropies, first_prizes, kos_received = [], [], [], [], []
    
    for ep in tqdm(range(episodes), desc="Evaluating"):
        p1_deck = load_deck(p1_deck_arg)
        p2_deck_path = random.choice(opp_decks)
        p2_deck = load_deck(p2_deck_path)
        
        env = make("cabt", configuration={"decks": [list(p1_deck), list(p2_deck)]})
        agent_module._local_env_ref = env
        
        p1_func = agent_module.agent
        p2_func = agent_module.agent
        
        p2_agent_name = p2_agent_arg
        if p2_type == "rules":
            if p2_agent_arg == "all" or not p2_agent_arg:
                from src.core.rules_agents import get_available_rules_agents
                p2_agent_name = random.choice(get_available_rules_agents())
            from src.core.rules_agents import get_rules_agent
            p2_func = get_rules_agent(p2_agent_name)
            
        agent_module.evaluation_telemetry = {"entropy": [], "hand_sizes": [], "action_paralysis": 0}
        
        # Suppress logs during run
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        
        saved_stdout, saved_stderr = sys.stdout, sys.stderr
        try:
            with open(os.devnull, "w") as devnull:
                sys.stdout = devnull
                sys.stderr = devnull
                env.run([p1_func, p2_func])
        finally:
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            os.close(devnull_fd)
        
        reward = env.state[0].reward if env.state[0].reward is not None else 0
        if reward == 1: diagnostics["wins"] += 1
        lengths.append(len(env.steps))
        
        telemetry = agent_module.evaluation_telemetry
        if telemetry["entropy"]: entropies.append(sum(telemetry["entropy"]) / len(telemetry["entropy"]))
        if telemetry["hand_sizes"]: hand_sizes.append(sum(telemetry["hand_sizes"]) / len(telemetry["hand_sizes"]))
        diagnostics["total_action_paralysis"] += telemetry["action_paralysis"]
        
        p1_prizes, p2_prizes = None, None
        first_prize = None
        kos = 0
        
        for step in env.steps:
            if step[0].observation:
                obs_dict = step[0].observation
                current = obs_dict.get("current") or {}
                players = current.get("players") or []
                if len(players) == 2:
                    p1_idx = current.get("yourIndex", 0)
                    p2_idx = 1 - p1_idx
                    p1_cur = len(players[p1_idx].get("prize", []))
                    p2_cur = len(players[p2_idx].get("prize", []))
                    
                    if p1_prizes is None and p1_cur > 0: p1_prizes = p1_cur
                    if p2_prizes is None and p2_cur > 0: p2_prizes = p2_cur
                        
                    if p1_prizes is not None and p1_cur < p1_prizes:
                        if first_prize is None: first_prize = "p1"
                        p1_prizes = p1_cur
                        
                    if p2_prizes is not None and p2_cur < p2_prizes:
                        if first_prize is None: first_prize = "p2"
                        kos += (p2_prizes - p2_cur)
                        p2_prizes = p2_cur
                
        first_prizes.append(1 if first_prize == "p1" else 0)
        kos_received.append(kos)
        
    diagnostics["avg_length"] = sum(lengths) / len(lengths) if lengths else 0
    diagnostics["avg_hand_size"] = sum(hand_sizes) / len(hand_sizes) if hand_sizes else 0
    diagnostics["avg_entropy"] = sum(entropies) / len(entropies) if entropies else 0
    diagnostics["first_prize_taken_pct"] = (sum(first_prizes) / len(first_prizes)) * 100 if first_prizes else 0
    diagnostics["avg_pokemon_kos_received"] = sum(kos_received) / len(kos_received) if kos_received else 0
    
    out_name = p2_agent_arg if p2_type == "rules" else "rl_mirror"
    os.makedirs(os.path.join("assets", "results", "diagnostics"), exist_ok=True)
    out_path = os.path.join("assets", "results", "diagnostics", f"diagnostics_{out_name}.json")
    with open(out_path, "w") as f:
        json.dump(diagnostics, f, indent=4)
    print(f"Diagnostics saved to {out_path}")
