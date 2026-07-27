from kaggle_environments import make
from src.core.agent import agent as p1_func
from src.core.rules_agents import get_rules_agent
p2_func = get_rules_agent("iono_s_deck")

env = make("cabt", configuration={"decks": [[5]*60, [5]*60]})
env.run([p1_func, p2_func])
print(env.state[0].status)
print("Steps:", len(env.steps))
if len(env.steps) < 10:
    for i, s in enumerate(env.steps):
        print(f"Step {i} Reward: {s[0].reward} | p1 action: {s[0].action} | p2 action: {s[1].action}")
