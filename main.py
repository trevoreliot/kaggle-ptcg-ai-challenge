import json
import os
import logging
import argparse

# Suppress litellm and other noisy warnings from kaggle-environments
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["SUPPRESS_LITELLM_WARNINGS"] = "True"
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

from kaggle_environments import make
from src.core.agent import agent

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

def main():
    parser = argparse.ArgumentParser(description="Pokémon TCG AI Challenge Simulator")
    parser.add_argument("--p1-deck", type=str, default="assets/decks/versatile/Team_Rockets_Box.csv",
                        help="Path or name of Player 1's deck CSV.")
    parser.add_argument("--p2-deck", type=str, default="assets/decks/versatile/Team_Rockets_Box.csv",
                        help="Path or name of Player 2's deck CSV.")
    parser.add_argument("--mode", type=str, choices=["play", "train"], default="play",
                        help="Execution mode: 'play' to save an HTML trace, 'train' for headless fast simulation.")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of matches to simulate (only used in 'train' mode).")
    args = parser.parse_args()

    print("Initializing environment...")
    try:
        p1_deck = load_deck(args.p1_deck)
        p2_deck = load_deck(args.p2_deck)
        print(f"Loaded P1 Deck: {len(p1_deck)} cards")
        print(f"Loaded P2 Deck: {len(p2_deck)} cards")
    except Exception as e:
        print(f"Failed to load decks: {e}")
        return

    env = make("cabt", configuration={"decks": [list(p1_deck), list(p2_deck)]})
    
    # Inject the local environment reference for MCTS engine branching
    import src.core.agent as agent_module
    agent_module._local_env_ref = env
    
    if args.mode == "play":
        print("Running single match in 'play' mode...")
        env.run([agent_module.agent, agent_module.agent])
        print("Match finished. Saving results...")
        result_path = "result.html"
        with open(result_path, "w") as f:
            f.write(env.render(mode="html"))
        print(f"Simulation saved to {result_path}")
        
    elif args.mode == "train":
        print(f"Running {args.episodes} matches in 'train' mode...")
        from src.core.models.replay_buffer import ReplayBuffer
        from src.core.models.trainer import Trainer
        
        # Initialize Replay Buffer and Trainer
        buffer = ReplayBuffer(gamma=0.99)
        trainer = Trainer(ensemble=agent_module.ensemble, lr=1e-4)
        
        # Inject buffer into agent module
        agent_module.global_replay_buffer = buffer
        
        for i in range(args.episodes):
            env.reset()
            env.run([agent_module.agent, agent_module.agent])
            
            # The Kaggle environment assigns reward 1 to winner, -1 to loser.
            # We look at agent 0's perspective.
            reward = env.state[0].reward if env.state[0].reward is not None else 0.0
            
            # Finalize the episode in the buffer
            buffer.finalize_episode(reward)
            
            print(f"Episode {i+1}/{args.episodes} complete. Reward: {reward}")
            
            # Optimize every 5 episodes (or at the end)
            if (i + 1) % 5 == 0 or (i + 1) == args.episodes:
                print(f"Optimizing model on {len(buffer)} recorded transitions...")
                trainer.update(buffer)
                
if __name__ == "__main__":
    main()
