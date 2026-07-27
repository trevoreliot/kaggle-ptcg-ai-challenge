from kaggle_environments import make

env = make("ptcg", debug=True)
try:
    env.run(["src/core/rules_agents/improved_probabilistic_agent.py", "random"])
    print("improved_probabilistic_agent finished successfully.")
except Exception as e:
    import traceback
    traceback.print_exc()

try:
    env.run(["src/core/rules_agents/mega_lucario_score_agent.py", "random"])
    print("mega_lucario_score_agent finished successfully.")
except Exception as e:
    import traceback
    traceback.print_exc()

try:
    env.run(["src/core/rules_agents/probablity_v2.py", "random"])
    print("probablity_v2 finished successfully.")
except Exception as e:
    import traceback
    traceback.print_exc()
