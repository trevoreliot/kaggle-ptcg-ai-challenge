import os
import random
from kaggle_environments import make
import src.core.agent.agent as agent_module # We will move agent.py to src/core/agent/agent.py soon
from src.core.utils.utils import load_deck, get_available_decks

def run_play(p1_deck_arg, opp_deck_arg, p2_type, p2_agent_arg):
    print("Initializing environment...")
    opp_decks = get_available_decks(opp_deck_arg)
    if not opp_decks:
        print("No opponent decks found.")
        return
        
    p2_deck_path = random.choice(opp_decks)
    
    p2_agent_name = p2_agent_arg
    if p2_type == "rules":
        from src.core.rules_agents import get_available_rules_agents
        if p2_agent_arg == "all" or not p2_agent_arg:
            p2_agent_name = random.choice(get_available_rules_agents())
            print(f"Randomly selected rules agent: {p2_agent_name}")
        elif p2_agent_arg in ["aggro", "control", "prob"]:
            p2_agent_name = random.choice(get_available_rules_agents(p2_agent_arg))
            print(f"Randomly selected rules agent from archetype {p2_agent_arg}: {p2_agent_name}")
            
        p2_deck_path = os.path.join("assets", "decks", "rules", f"{p2_agent_name}.csv")
        
    try:
        p1_deck = load_deck(p1_deck_arg)
        p2_deck = load_deck(p2_deck_path)
    except Exception as e:
        print(f"Failed to load decks: {e}")
        return
        
    env = make("cabt", debug=True, configuration={"decks": [list(p1_deck), list(p2_deck)]})
    agent_module._local_env_ref = env
    
    p1_func = agent_module.agent
    p2_func = agent_module.agent
    
    if p2_type == "rules":
        from src.core.rules_agents import get_rules_agent
        p2_func = get_rules_agent(p2_agent_name)
        
    print(f"Running single match against {p2_deck_path} in 'play' mode...")
    env.run([p1_func, p2_func])
    print("Match finished. Saving results...")
    is_assist = getattr(agent_module, "HUMAN_ASSIST", False)
    if is_assist:
        import time
        import json as sys_json
        assist_dir = os.path.join("assets", "results", "assist_replays")
        os.makedirs(assist_dir, exist_ok=True)
        timestamp = int(time.time())
        base_path = os.path.join(assist_dir, f"assist_replay_{timestamp}")
        
        env_json = env.toJSON()
        payload = ""
        if "steps" in env_json and len(env_json["steps"]) > 0 and len(env_json["steps"][0]) > 0 and "visualize" in env_json["steps"][0][0]:
            payload = sys_json.dumps(env_json["steps"][0][0]["visualize"])
        else:
            payload = sys_json.dumps(env_json)
            
        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8" /></head>
<body>
    <div style="text-align: center; margin-top: 20px; font-family: sans-serif;">Loading visualizer...</div>
    <script>
        const form = document.createElement("form");
        form.method = "POST";
        form.action = "https://ptcgvis.heroz.jp/Visualizer/Replay/0";
        form.target = "_self";
        
        const input = document.createElement("input");
        input.type = "hidden";
        input.name = "json";
        input.value = {sys_json.dumps(payload)};
        
        form.appendChild(input);
        document.body.appendChild(form);
        form.submit();
    </script>
</body>
</html>"""
        
        with open(f"{base_path}.html", "w") as f:
            f.write(html)
            
        with open(f"{base_path}.json", "w") as f:
            sys_json.dump(env.toJSON(), f)
            
        print(f"Assist simulation saved to {base_path}.html and .json")
    else:
        import json as sys_json
        env_json = env.toJSON()
        payload = ""
        if "steps" in env_json and len(env_json["steps"]) > 0 and len(env_json["steps"][0]) > 0 and "visualize" in env_json["steps"][0][0]:
            payload = sys_json.dumps(env_json["steps"][0][0]["visualize"])
        else:
            payload = sys_json.dumps(env_json)
            
        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8" /></head>
<body>
    <div style="text-align: center; margin-top: 20px; font-family: sans-serif;">Loading visualizer...</div>
    <script>
        const form = document.createElement("form");
        form.method = "POST";
        form.action = "https://ptcgvis.heroz.jp/Visualizer/Replay/0";
        form.target = "_self";
        
        const input = document.createElement("input");
        input.type = "hidden";
        input.name = "json";
        input.value = {sys_json.dumps(payload)};
        
        form.appendChild(input);
        document.body.appendChild(form);
        form.submit();
    </script>
</body>
</html>"""
        with open("result.html", "w") as f:
            f.write(html)
        print(f"Simulation saved to result.html")
