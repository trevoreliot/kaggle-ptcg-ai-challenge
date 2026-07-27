import sys
import os
import random

# Apply monkeypatch from main.py
import copy
_orig_deepcopy = copy.deepcopy
def fast_deepcopy(x, memo=None, _nil=[]):
    if isinstance(x, dict):
        if "step" in x and "remainingOverageTime" in x: return x
        if "observation" in x and "reward" in x:
            new_dict = x.copy()
            new_dict["observation"] = x["observation"]
            return new_dict
    return _orig_deepcopy(x, memo)
copy.deepcopy = fast_deepcopy

from kaggle_environments import make
import src.core.agent as agent_module
from src.core.rules_agents import get_rules_agent
from main import load_deck

p1_deck = load_deck("assets/decks/versatile/Team_Rockets_Box.csv")
p2_deck = load_deck("assets/decks/aggro/Aggro_Fighting.csv")

env = make("cabt", configuration={"decks": [list(p1_deck), list(p2_deck)]}, debug=True)
agent_module._local_env_ref = env
agent_module.IS_TRAINING = False

p1_func = agent_module.agent
p2_func = get_rules_agent("improved_probabilistic_agent")

env.run([p1_func, p2_func])

print(f"Match finished. Steps: {len(env.steps)}")
print(f"P1 Status: {env.state[0].status}, Reward: {env.state[0].reward}")
print(f"P2 Status: {env.state[1].status}, Reward: {env.state[1].reward}")
