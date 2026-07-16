import sys
import os

# Ensure src module is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.bayesian import BayesianTracker
from src.core.parser import Observation, CurrentState, PlayerState, Pokemon

def test_bayesian_update():
    tracker = BayesianTracker()
    print("Initial Prior:", tracker.prior)
    
    # 1072 is Snorlax (Control)
    snorlax = Pokemon(id=1072, serial=1, playerIndex=1, hp=150, maxHp=150, appearThisTurn=True)
    
    # Create a dummy observation with Snorlax on opponent's bench
    obs = Observation(step=1, current=CurrentState(
        yourIndex=0,
        players=[PlayerState(), PlayerState(bench=[snorlax])]
    ))
    
    tracker.update(obs)
    
    print("Posterior after Snorlax:", tracker.prior)
    best = tracker.best_archetype()
    confidence = tracker.max_confidence()
    print(f"Detected Archetype: {best} (Confidence: {confidence:.2f})")
    
    assert best == "control", f"Expected 'control', got {best}"
    print("Test passed!")

if __name__ == "__main__":
    test_bayesian_update()
