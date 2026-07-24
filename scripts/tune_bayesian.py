import os
import sys
import numpy as np
import random

os.environ['LITELLM_LOG'] = 'ERROR'
os.environ['SUPPRESS_LITELLM_WARNINGS'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from kaggle_environments import make
import src.core.agent as agent_module
from src.core.agent import load_deck

def test_bayesian_accuracy(archetype_name, deck_path, num_matches=5):
    print(f"\n--- Testing Bayesian Detection against {archetype_name} ({num_matches} matches) ---")
    p1_deck = load_deck("assets/decks/versatile/Team_Rockets_Box.csv")
    p2_deck = load_deck(deck_path)
    
    def p2_agent(obs, config):
        if obs.step == 0:
            return p2_deck
        select_data = obs.get("select")
        if select_data:
            options = select_data.get("option", [])
            max_count = min(select_data.get("maxCount", 1), len(options))
            if max_count > 0 and options:
                # Random valid selection
                return random.sample(range(len(options)), max_count)
        return []

    turns_to_detect = []
    
    for match_idx in range(num_matches):
        env = make("cabt")
        trainer = env.train([None, p2_agent])
        obs = trainer.reset()
        
        detected_step = -1
        
        while not env.done:
            action = agent_module.agent(obs)
            
            tracker = agent_module.bayesian_tracker
            best = tracker.best_archetype()
            conf = tracker.max_confidence()
            
            if conf > 0.85 and best == archetype_name and detected_step == -1:
                detected_step = len(env.steps)
                break
                
            obs, reward, done, info = trainer.step(action)
            
        if detected_step != -1:
            turns_to_detect.append(detected_step)
            print(f"Match {match_idx+1}: Correctly detected {archetype_name} at step {detected_step} (Conf: {conf:.2f})")
        else:
            final_best = agent_module.bayesian_tracker.best_archetype()
            final_conf = agent_module.bayesian_tracker.max_confidence()
            print(f"Match {match_idx+1}: FAILED to detect. Final prediction: {final_best} at {final_conf:.2f}")

    if turns_to_detect:
        avg_steps = sum(turns_to_detect) / len(turns_to_detect)
        print(f"\nAverage steps to detect {archetype_name} >85%: {avg_steps:.1f} steps")
    else:
        print(f"\nFailed to detect {archetype_name} in any matches.")

if __name__ == "__main__":
    test_bayesian_accuracy("aggro", "assets/decks/aggro/Aggro_Charizard.csv")
    test_bayesian_accuracy("control", "assets/decks/control/Control_Snorlax_Block.csv")
    test_bayesian_accuracy("combo", "assets/decks/combo/Combo_Water_Engine.csv")
