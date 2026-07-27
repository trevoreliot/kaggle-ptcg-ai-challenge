import sys
sys.path.append("vendor/cg-lib")
from src.core.rules_agents.improved_probabilistic_agent import SEARCH_ALGO, simulate_action, AdvancedPolicy
from main import load_deck
from kaggle_environments import make
from cg.api import to_observation_class, SelectContext

p1_deck = load_deck("assets/decks/versatile/Team_Rockets_Box.csv")
p2_deck = load_deck("assets/decks/aggro/Aggro_Fighting.csv")
env = make("cabt", configuration={"decks": [list(p1_deck), list(p2_deck)]})

obs = env.reset()
env.step([list(p1_deck), list(p2_deck)])
env.step([[0], [0]])
while not env.done:
    st = env.state[0]
    obs_dict = st["observation"]
    obs_class = to_observation_class(obs_dict)
    if obs_class.select and obs_class.select.context == SelectContext.MAIN:
        base_order = AdvancedPolicy(obs_class).choose()
        if len(base_order) > 1:
            print(f"Found MAIN context with {len(base_order)} candidates! Testing SEARCH_ALGO...")
            try:
                res = SEARCH_ALGO(obs_dict, obs_class)
                print("SEARCH_ALGO returned:", res)
            except Exception as e:
                import traceback
                traceback.print_exc()
            break
        
    options = obs_class.select.option if obs_class.select else []
    if options:
        env.step([[0], [0]])
    else:
        break
