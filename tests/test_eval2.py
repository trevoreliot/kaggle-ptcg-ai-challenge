import sys
import main

sys.argv = ["main.py", "--mode", "evaluate", "--episodes", "1", "--p2-type", "rules", "--p2-agent", "iono_s_deck"]
import src.core.agent as agent_module
from kaggle_environments import make
from src.core.rules_agents import get_rules_agent

env = make("cabt", configuration={"decks": [[5]*60, [5]*60]})
p1_func = agent_module.agent
p2_func = get_rules_agent("iono_s_deck")
agent_module._local_env_ref = env
agent_module.IS_EVALUATING = True
agent_module.IS_TRAINING = False

env.run([p1_func, p2_func])
print("Status:", env.state[0].status)
print("Steps:", len(env.steps))
if len(env.steps) < 10:
    print(env.steps[-1][0].get("observation", {}))
