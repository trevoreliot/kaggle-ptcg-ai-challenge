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
        return glob("assets/decks/**/*.csv", recursive=True)
    elif os.path.isdir(opp_deck_arg):
        return glob(os.path.join(opp_deck_arg, "*.csv"))
    else:
        return [opp_deck_arg]

# Worker function for multiprocessing
def worker_wrapper(args):
    return worker_run_episode(*args)

def worker_run_episode(p1_deck_path, p2_deck_path):
    # Import locally to ensure each process has its own isolated agent state
    import src.core.agent as agent_module
    from src.core.models.replay_buffer import ReplayBuffer
    
    p1_deck = load_deck(p1_deck_path)
    p2_deck = load_deck(p2_deck_path)
    
    env = make("cabt", configuration={"decks": [p1_deck, p2_deck]})
    agent_module._local_env_ref = env
    
    local_buffer = ReplayBuffer(gamma=0.99)
    agent_module.global_replay_buffer = local_buffer
    
    env.reset()
    env.run([agent_module.agent, agent_module.agent])
    
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
        
        master_buffer = ReplayBuffer(gamma=0.99)
        trainer = Trainer(ensemble=agent_module.ensemble, lr=1e-4)
        
        log_file = "training_metrics.csv"
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["Episode", "Opponent_Deck", "Reward", "Episode_Length", "Policy_Loss", "Value_Loss"])
            
            # Prepare tasks
            tasks = []
            for i in range(args.episodes):
                p2_deck_path = random.choice(opp_decks)
                tasks.append((args.p1_deck, p2_deck_path))
                
            completed = 0
            
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
                                pbar.set_postfix({"P_Loss": f"{policy_loss:.3f}", "V_Loss": f"{value_loss:.3f}"})
                                
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
                            pbar.set_postfix({"P_Loss": f"{policy_loss:.3f}", "V_Loss": f"{value_loss:.3f}"})
                            
                        writer.writerow([completed, os.path.basename(p2_path), reward, ep_len, policy_loss, value_loss])
                        csvfile.flush()
                        pbar.update(1)

import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
