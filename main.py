import json
import os
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

# Worker function for multiprocessing
def worker_wrapper(args):
    import cProfile, os
    pr = cProfile.Profile()
    pr.enable()
    try:
        res = worker_run_episode(*args)
        pr.disable()
        pr.dump_stats(f"worker_{os.getpid()}.prof")
        return res
    except KeyboardInterrupt:
        return None
    except Exception as e:
        print(f"Worker exception: {e}")
        return None

def worker_run_episode(p1_deck_path, p2_deck_path, model_name=None):
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
    
    # Lock model for training
    agent_module.IS_TRAINING = True
    if model_name:
        archetype = model_name.split("_")[0]
        if archetype in agent_module.bayesian_tracker.archetypes:
            agent_module.ensemble.switch_model(archetype)
            
    snapshot_path = os.path.join("assets", "models", "latest_snapshot.pt")
    if os.path.exists(snapshot_path):
        try:
            agent_module.ensemble.active_model.load_state_dict(torch.load(snapshot_path, weights_only=True))
        except Exception:
            pass # ignore loading errors if file is being written concurrently
            
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
            
            env.reset()
            env.run([agent_module.agent, agent_module.agent])
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
    trajectory = local_buffer.finalize_episode(reward)
    
    # Calculate episode length (rough estimate by actions or step count)
    episode_length = len(env.steps)
    
    return p2_deck_path, reward, episode_length, trajectory

def main():
    parser = argparse.ArgumentParser(description="Pokémon TCG AI Challenge Simulator")
    parser.add_argument("--p1-deck", type=str, default="assets/decks/versatile/Team_Rockets_Box.csv",
                        help="Path or name of Player 1's deck CSV.")
    parser.add_argument("--opp-deck", type=str, default="all",
                        help="Path to specific deck, folder of decks, or 'all'.")
    parser.add_argument("--mode", type=str, choices=["play", "train"], default="play",
                        help="Execution mode: 'play' to save an HTML trace, 'train' for headless fast simulation.")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of matches to simulate (only used in 'train' mode).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes for training.")
    parser.add_argument("--model-name", type=str, default="general_model.pt",
                        help="Name of the model file to save/load (e.g. aggro_model.pt).")
    args = parser.parse_args()

    # Play mode (synchronous, 1 match)
    if args.mode == "play":
        print("Initializing environment...")
        opp_decks = get_available_decks(args.opp_deck)
        if not opp_decks:
            print("No opponent decks found.")
            return
        p2_deck_path = random.choice(opp_decks)
        
        try:
            p1_deck = load_deck(args.p1_deck)
            p2_deck = load_deck(p2_deck_path)
        except Exception as e:
            print(f"Failed to load decks: {e}")
            return
            
        env = make("cabt", configuration={"decks": [list(p1_deck), list(p2_deck)]})
        import src.core.agent as agent_module
        agent_module._local_env_ref = env
        
        print(f"Running single match against {p2_deck_path} in 'play' mode...")
        env.run([agent_module.agent, agent_module.agent])
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
        
        log_file = os.path.join("assets", "results", "rl_training", "training_metrics.csv")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["Episode", "Opponent_Deck", "Reward", "Episode_Length", "Policy_Loss", "Value_Loss"])
            
            # Prepare tasks
            tasks = []
            for i in range(args.episodes):
                p2_deck_path = random.choice(opp_decks)
                tasks.append((args.p1_deck, p2_deck_path, args.model_name))
                
            completed = 0
            
            snapshot_path = os.path.join("assets", "models", "latest_snapshot.pt")
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
                        p2_path, reward, ep_len, trajectory = worker_run_episode(*task)
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

import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
