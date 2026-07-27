import sys
import main
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
obs = env.steps[-1][0].observation
try:
    p1_func(obs)
except Exception as e:
    import traceback
    traceback.print_exc()
