import argparse
import multiprocessing

def main():
    parser = argparse.ArgumentParser(description="Pokémon TCG AI Challenge Simulator")
    parser.add_argument("--p1-deck", type=str, default="assets/decks/versatile/Team_Rockets_Box.csv",
                        help="Path or name of Player 1's deck CSV.")
    parser.add_argument("--opp-deck", type=str, default="all",
                        help="Path to specific deck, folder of decks, or 'all'.")
    parser.add_argument("--mode", type=str, choices=["play", "train", "evaluate"], default="play",
                        help="Execution mode: 'play' to save an HTML trace, 'train' for headless fast simulation, 'evaluate' for diagnostics.")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of matches to simulate (only used in 'train' or 'evaluate' mode).")
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
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Starting epsilon for exploration decay. Defaults to 0.0 (no randomness).")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Dirichlet noise alpha parameter for MCTS. If > 0, overrides epsilon-greedy with MCTS noise.")
    args = parser.parse_args()
    
    if args.p2_type in ["rl", "mixed"] and args.p2_agent != "all":
        parser.error(f"You provided a specific --p2-agent ('{args.p2_agent}'), but --p2-type is set to '{args.p2_type}'. To play against a rules-based agent, you must set --p2-type rules.")

    if args.mode == "play":
        from src.core.engine.play import run_play
        run_play(args.p1_deck, args.opp_deck, args.p2_type, args.p2_agent)
    elif args.mode == "train":
        from src.core.engine.train import run_train
        run_train(args.episodes, args.workers, args.p1_deck, args.opp_deck, args.model_name, args.p2_type, args.p2_agent, args.debug, args.epsilon, args.alpha)
    elif args.mode == "evaluate":
        from src.core.engine.evaluate import run_evaluate
        run_evaluate(args.episodes, args.workers, args.p1_deck, args.opp_deck, args.model_name, args.p2_type, args.p2_agent, args.alpha)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    
    # Suppress litellm and other noisy warnings from kaggle-environments
    import os
    import logging
    os.environ["LITELLM_LOG"] = "ERROR"
    os.environ["SUPPRESS_LITELLM_WARNINGS"] = "True"
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    
    main()
