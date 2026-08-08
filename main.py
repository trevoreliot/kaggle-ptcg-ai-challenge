import json
import os
import sys
import logging
import argparse
import random
import csv
import time
from multiprocessing import Pool
from glob import glob

# Suppress litellm and other noisy warnings from kaggle-environments
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["SUPPRESS_LITELLM_WARNINGS"] = "True"
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

from kaggle_environments import make
import copy

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

def worker_wrapper(args):
    import os
    p1, p2, model_name, debug, p2_type, p2_agent, epsilon, temperature = args
    if debug:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
        
    try:
        res = worker_run_episode(p1, p2, model_name, p2_type, p2_agent, epsilon, temperature)
        if debug:
            pr.disable()
            out_dir = os.path.join("logs", "debug", "worker_profiles")
            os.makedirs(out_dir, exist_ok=True)
            pr.dump_stats(os.path.join(out_dir, f"worker_{os.getpid()}.prof"))
        return res
    except KeyboardInterrupt:
        return None
    except Exception as e:
        import traceback
        print(f"Worker exception: {e}")
        traceback.print_exc()
        return None

def worker_run_episode(p1_deck_path, p2_deck_path, model_name=None, p2_type="rl", p2_agent_name=None, epsilon=0.01, temperature=1.0):
    import sys
    import os
    
    # Force workers to use CPU to avoid massive GPU context switching overhead
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Force workers to use CPU to avoid massive GPU context switching overhead
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Prevent OpenMP thread contention in worker processes
    import torch
    torch.set_num_threads(1)
    
    # Import locally AFTER setting CUDA_VISIBLE_DEVICES to ensure each process has its own isolated CPU agent state
    import src.core.agent as agent_module
    from src.core.models.replay_buffer import ReplayBuffer
    
    agent_module.CURRENT_EPSILON = epsilon
    agent_module.CURRENT_TEMPERATURE = temperature
    
    # Lock model for training
    agent_module.IS_TRAINING = True
    if model_name:
        archetype = model_name.split("_")[0]
        if archetype in agent_module.bayesian_tracker.archetypes:
            if archetype not in agent_module.ensemble.models:
                # Add a reference so switch_model doesn't throw a fallback warning
                agent_module.ensemble.models[archetype] = agent_module.ensemble.models.get("general")
            agent_module.ensemble.switch_model(archetype)
            
    snapshot_path = os.path.join("assets", "models", model_name) if model_name else os.path.join("assets", "models", "latest_snapshot.pt")
    if os.path.exists(snapshot_path):
        try:
            agent_module.ensemble.active_model.load_state_dict(torch.load(snapshot_path, weights_only=True))
        except Exception:
            pass # ignore loading errors if file is being written concurrently
            
    import random
    
    # Resolve 'mixed' early so we can assign the correct deck
    if p2_type == "mixed":
        p2_type = "rl" if random.random() < 0.8 else "rules"
        
    if p2_type == "rules":
        from src.core.rules_agents import get_available_rules_agents
        p2_actual_name = p2_agent_name
        if not p2_actual_name or p2_actual_name == "all":
            p2_actual_name = random.choice(get_available_rules_agents())
        elif p2_actual_name in ["aggro", "control", "prob"]:
            p2_actual_name = random.choice(get_available_rules_agents(p2_actual_name))
        
        p2_agent_name = p2_actual_name
        # Override the deck with the specific rule agent's deck!
        p2_deck_path = os.path.join("assets", "decks", "rules", f"{p2_actual_name}.csv")

    p1_deck = load_deck(p1_deck_path)
    p2_deck = load_deck(p2_deck_path)
    
    # Redirect at OS level to catch C++ prints (e.g. from cg-lib)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            env = make("cabt", configuration={"decks": [p1_deck, p2_deck]})
            agent_module._local_env_ref = env
            
            local_buffer = ReplayBuffer(gamma=0.99)
            agent_module.global_replay_buffer = local_buffer
            
            p1_func = agent_module.agent
            p2_func = agent_module.agent
            
            if p2_type == "rules":
                from src.core.rules_agents import get_rules_agent
                p2_func = get_rules_agent(p2_agent_name)
            else:
                p2_func = agent_module.agent
            
            agent_module.CURRENT_EPSILON = epsilon
            agent_module.CURRENT_TEMPERATURE = temperature
            
            agent_module.reset_state_tracking()
            env.reset()
            env.run([p1_func, p2_func])
            
            # Check for invalid actions to prevent poor training data
            for step_idx, step_data in enumerate(env.steps):
                for p_idx, player_state in enumerate(step_data):
                    status = player_state.status if hasattr(player_state, 'status') else player_state.get('status')
                    if status == 'INVALID':
                        action = player_state.action if hasattr(player_state, 'action') else player_state.get('action')
                        raise AssertionError(f"INVALID ACTION DETECTED by Player {p_idx} (Step {step_idx})!\nAction Taken: {action}\nAborting training to prevent poor data pollution.")
                        
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Restore OS level descriptors
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            os.close(devnull_fd)
    
    if env.state[0].status == "ERROR":
        pass
    
    reward = env.state[0].reward if env.state[0].reward is not None else 0.0
    
    # Push the final dangling action from Player 1's state tracker
    tracker = agent_module._state_tracker[0]
    if tracker.get("last_state") is not None:
        for a in tracker["last_actions"]:
            local_buffer.push(
                tracker["last_state"],
                a,
                tracker["last_log_prob"],
                tracker["last_value"],
                step_reward=0.0
            )
        
    trajectory = local_buffer.finalize_episode(reward)
    
    # Calculate episode length (rough estimate by actions or step count)
    episode_length = len(env.steps)
    
    opponent_name = p2_deck_path
    if p2_type == "rules":
        opponent_name = f"[RULES] {p2_actual_name}"
    else:
        opponent_name = f"[SELF-PLAY] {os.path.basename(p2_deck_path)}"
        
    return opponent_name, reward, episode_length, trajectory

def main():
    parser = argparse.ArgumentParser(description="Pokémon TCG AI Challenge Simulator")
    parser.add_argument("--p1-deck", type=str, default="assets/decks/versatile/Team_Rockets_Box.csv",
                        help="Path or name of Player 1's deck CSV.")
    parser.add_argument("--opp-deck", type=str, default="all",
                        help="Path to specific deck, folder of decks, or 'all'.")
    parser.add_argument("--mode", type=str, choices=["play", "train", "evaluate"], default="play",
                        help="Execution mode: 'play' to save an HTML trace, 'train' for headless fast simulation, 'evaluate' for diagnostics.")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of matches to simulate (only used in 'train' mode).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes for training.")
    parser.add_argument("--model-name", type=str, default="general_model.pt",
                        help="Name of the model file to save/load (e.g. aggro_model.pt).")
    parser.add_argument("--p2-type", type=str, choices=["rl", "rules", "mixed"], default="rl",
                        help="The type of agent Player 2 is.")
    parser.add_argument("--p2-agent", type=str, default="all",
                        help="The specific rules-based agent to use if --p2-type is rules (or 'all', 'aggro', 'control', 'prob' for archetype).")
    parser.add_argument("--debug", action="store_true",
                        help="Enable cProfile worker profiling.")
    args = parser.parse_args()
    
    if args.p2_type in ["rl", "mixed"] and args.p2_agent != "all":
        parser.error(f"You provided a specific --p2-agent ('{args.p2_agent}'), but --p2-type is set to '{args.p2_type}'. To play against a rules-based agent, you must set --p2-type rules.")

    # Play mode (synchronous, 1 match)
    if args.mode == "play":
        print("Initializing environment...")
        opp_decks = get_available_decks(args.opp_deck)
        if not opp_decks:
            print("No opponent decks found.")
            return
            
        p2_deck_path = random.choice(opp_decks)
        
        p2_agent_name = args.p2_agent
        if args.p2_type == "rules":
            from src.core.rules_agents import get_available_rules_agents
            if args.p2_agent == "all" or not args.p2_agent:
                p2_agent_name = random.choice(get_available_rules_agents())
                print(f"Randomly selected rules agent: {p2_agent_name}")
            elif args.p2_agent in ["aggro", "control", "prob"]:
                p2_agent_name = random.choice(get_available_rules_agents(args.p2_agent))
                print(f"Randomly selected rules agent from archetype {args.p2_agent}: {p2_agent_name}")
                
            p2_deck_path = os.path.join("assets", "decks", "rules", f"{p2_agent_name}.csv")
            
        try:
            p1_deck = load_deck(args.p1_deck)
            p2_deck = load_deck(p2_deck_path)
        except Exception as e:
            print(f"Failed to load decks: {e}")
            return
            
        env = make("cabt", configuration={"decks": [list(p1_deck), list(p2_deck)]})
        import src.core.agent as agent_module
        agent_module._local_env_ref = env
        
        p1_func = agent_module.agent
        p2_func = agent_module.agent
        
        if args.p2_type == "rules":
            from src.core.rules_agents import get_rules_agent
            p2_func = get_rules_agent(p2_agent_name)
            
        print(f"Running single match against {p2_deck_path} in 'play' mode...")
        env.run([p1_func, p2_func])
        print("Match finished. Saving results...")
        with open("result.html", "w") as f:
            f.write(env.render(mode="html"))
        print(f"Simulation saved to result.html")
        
    elif args.mode == "train":
        print(f"Running {args.episodes} matches in 'train' mode with {args.workers} workers...")
        opp_decks = get_available_decks(args.opp_deck)
        if not opp_decks:
            print(f"No decks found for filter: {args.opp_deck}")
            return
            
        from src.core.models.replay_buffer import ReplayBuffer
        from src.core.models.trainer import Trainer
        import src.core.agent as agent_module
        
        import torch
        
        master_buffer = ReplayBuffer(gamma=0.99)
        trainer = Trainer(ensemble=agent_module.ensemble, lr=1e-4)
        
        checkpoint_path = os.path.join("assets", "models", args.model_name)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        if os.path.exists(checkpoint_path):
            agent_module.ensemble.active_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            print(f"Loaded existing PyTorch checkpoint from {checkpoint_path}!")
        else:
            print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch!")
        
        model_prefix = args.model_name.replace("_model.pt", "").replace(".pt", "") if args.model_name else "general"
        csv_filename = f"{model_prefix}_training_metrics.csv"
        log_file = os.path.join("assets", "results", "rl_training", csv_filename)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["Episode", "Opponent_Deck", "Reward", "Episode_Length", "Policy_Loss", "Value_Loss"])
            
            if args.p2_type == "rules":
                from src.core.rules_agents import get_available_rules_agents
                if args.p2_agent == "all" or not args.p2_agent:
                    available_p2_agents = get_available_rules_agents()
                elif args.p2_agent in ["aggro", "control", "prob"]:
                    available_p2_agents = get_available_rules_agents(args.p2_agent)
                else:
                    available_p2_agents = [args.p2_agent]
            else:
                available_p2_agents = [None]
                
            # Prepare tasks
            tasks = []
            for i in range(args.episodes):
                progress = i / max(1, args.episodes - 1)
                
                # Decay epsilon from 0.15 down to 0.01
                epsilon = 0.15 - (0.14 * progress)
                # Decay temperature from 2.0 down to 0.5
                temperature = 2.0 - (1.5 * progress)
                
                p2_deck_path = random.choice(opp_decks)
                p2_agent_choice = random.choice(available_p2_agents)
                tasks.append((args.p1_deck, p2_deck_path, args.model_name, args.debug, args.p2_type, p2_agent_choice, epsilon, temperature))
                
            completed = 0
            
            snapshot_path = os.path.join("assets", "models", args.model_name) if args.model_name else os.path.join("assets", "models", "latest_snapshot.pt")
            torch.save(agent_module.ensemble.active_model.state_dict(), snapshot_path)
            
            from tqdm import tqdm
            
            with tqdm(total=args.episodes, desc="Training", unit="match") as pbar:
                if args.workers > 1:
                    # Run in parallel with imap_unordered
                    with Pool(processes=args.workers) as pool:
                        for p2_path, reward, ep_len, trajectory in pool.imap_unordered(worker_wrapper, tasks):
                            completed += 1
                            master_buffer.add_trajectory(trajectory)
                            
                            policy_loss, value_loss = 0.0, 0.0
                            if completed % 5 == 0 or completed == args.episodes:
                                policy_loss, value_loss = trainer.update(master_buffer)
                                master_buffer.clear()
                                pbar.set_postfix({"P_Loss": f"{policy_loss:.3f}", "V_Loss": f"{value_loss:.3f}"})
                                
                                tmp_snapshot = snapshot_path + ".tmp"
                                torch.save(agent_module.ensemble.active_model.state_dict(), tmp_snapshot)
                                os.replace(tmp_snapshot, snapshot_path)
                                
                            if completed % 100 == 0 or completed == args.episodes:
                                import torch
                                torch.save(agent_module.ensemble.active_model.state_dict(), checkpoint_path)
                                
                            writer.writerow([completed, os.path.basename(p2_path), reward, ep_len, policy_loss, value_loss])
                            csvfile.flush()
                            pbar.update(1)
                else:
                    # Run synchronously
                    for task in tasks:
                        p2_path, reward, ep_len, trajectory = worker_run_episode(task[0], task[1], task[2], task[4], task[5])
                        completed += 1
                        master_buffer.add_trajectory(trajectory)
                        
                        policy_loss, value_loss = 0.0, 0.0
                        if completed % 5 == 0 or completed == args.episodes:
                            policy_loss, value_loss = trainer.update(master_buffer)
                            master_buffer.clear()
                            pbar.set_postfix({"P_Loss": f"{policy_loss:.3f}", "V_Loss": f"{value_loss:.3f}"})
                            
                            tmp_snapshot = snapshot_path + ".tmp"
                            torch.save(agent_module.ensemble.active_model.state_dict(), tmp_snapshot)
                            os.replace(tmp_snapshot, snapshot_path)
                            
                        if completed % 100 == 0 or completed == args.episodes:
                            import torch
                            torch.save(agent_module.ensemble.active_model.state_dict(), checkpoint_path)
                            
                        writer.writerow([completed, os.path.basename(p2_path), reward, ep_len, policy_loss, value_loss])
                        csvfile.flush()
                        pbar.update(1)

    elif args.mode == "evaluate":
        print(f"Running {args.episodes} matches in 'evaluate' mode...")
        import src.core.agent as agent_module
        import json
        
        agent_module.IS_TRAINING = False
        agent_module.IS_EVALUATING = True
        
        opp_decks = get_available_decks(args.opp_deck)
        
        checkpoint_path = os.path.join("assets", "models", args.model_name)
        if os.path.exists(checkpoint_path):
            import torch
            agent_module.ensemble.active_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            print(f"Loaded PyTorch checkpoint from {checkpoint_path} for evaluation!")
            
        diagnostics = {
            "matches": args.episodes,
            "wins": 0,
            "avg_length": 0,
            "avg_hand_size": 0,
            "avg_entropy": 0,
            "total_action_paralysis": 0,
            "first_prize_taken_pct": 0,
            "avg_pokemon_kos_received": 0
        }
        
        lengths, hand_sizes, entropies, first_prizes, kos_received = [], [], [], [], []
        
        from tqdm import tqdm
        for ep in tqdm(range(args.episodes), desc="Evaluating"):
            p1_deck = load_deck(args.p1_deck)
            p2_deck_path = random.choice(opp_decks)
            p2_deck = load_deck(p2_deck_path)
            
            env = make("cabt", configuration={"decks": [list(p1_deck), list(p2_deck)]})
            agent_module._local_env_ref = env
            
            p1_func = agent_module.agent
            p2_func = agent_module.agent
            
            p2_agent_name = args.p2_agent
            if args.p2_type == "rules":
                if args.p2_agent == "all" or not args.p2_agent:
                    from src.core.rules_agents import get_available_rules_agents
                    p2_agent_name = random.choice(get_available_rules_agents())
                from src.core.rules_agents import get_rules_agent
                p2_func = get_rules_agent(p2_agent_name)
                
            agent_module.evaluation_telemetry = {"entropy": [], "hand_sizes": [], "action_paralysis": 0}
            
            # Suppress logs during run (OS level and Python level)
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
                        
                        if p1_prizes is None and p1_cur > 0:
                            p1_prizes = p1_cur
                        if p2_prizes is None and p2_cur > 0:
                            p2_prizes = p2_cur
                            
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
        
        out_name = args.p2_agent if args.p2_type == "rules" else "rl_mirror"
        os.makedirs(os.path.join("assets", "results", "diagnostics"), exist_ok=True)
        out_path = os.path.join("assets", "results", "diagnostics", f"diagnostics_{out_name}.json")
        with open(out_path, "w") as f:
            json.dump(diagnostics, f, indent=4)
        print(f"Diagnostics saved to {out_path}")

import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
