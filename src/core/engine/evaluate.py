import os
import sys
import json
import random
import multiprocessing
from tqdm import tqdm
from kaggle_environments import make
import src.core.agent.agent as agent_module
from src.core.utils.utils import load_deck, get_available_decks

def eval_worker_run_episode(p1_deck_path, p2_deck_path, model_name, p2_type, p2_agent_name, alpha):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    torch.set_num_threads(1)
    
    agent_module.IS_TRAINING = False
    agent_module.IS_EVALUATING = True
    agent_module.CURRENT_EPSILON = 0.0
    agent_module.CURRENT_ALPHA = alpha
    
    if model_name:
        archetype = model_name.split("_")[0]
        if archetype in agent_module.bayesian_tracker.archetypes:
            if archetype not in agent_module.ensemble.models:
                agent_module.ensemble.models[archetype] = agent_module.ensemble.models.get("general")
            agent_module.ensemble.switch_model(archetype)
            
    snapshot_path = os.path.join("assets", "models", model_name) if model_name else os.path.join("assets", "models", "latest_snapshot.pt")
    if os.path.exists(snapshot_path):
        try:
            agent_module.ensemble.active_model.load_state_dict(torch.load(snapshot_path, weights_only=True))
        except Exception:
            pass
            
    if p2_type == "rules":
        from src.core.rules_agents import get_available_rules_agents
        if not p2_agent_name or p2_agent_name == "all":
            p2_agent_name = random.choice(get_available_rules_agents())
        elif p2_agent_name in ["aggro", "control", "prob"]:
            p2_agent_name = random.choice(get_available_rules_agents(p2_agent_name))
        elif "," in p2_agent_name:
            p2_agent_name = random.choice([x.strip() for x in p2_agent_name.split(",")])
        p2_deck_path = os.path.join("assets", "decks", "rules", f"{p2_agent_name}.csv")
        
    p1_deck = load_deck(p1_deck_path)
    p2_deck = load_deck(p2_deck_path)
    
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            env = make("cabt", configuration={"decks": [list(p1_deck), list(p2_deck)], "actTimeout": 99999, "runTimeout": 99999})
            agent_module._local_env_ref = env
            
            p1_func = agent_module.agent
            p2_func = agent_module.agent
            
            if p2_type == "rules":
                from src.core.rules_agents import get_rules_agent
                p2_func = get_rules_agent(p2_agent_name)
                
            agent_module.evaluation_telemetry = {"entropy": [], "hand_sizes": [], "action_paralysis": 0}
            agent_module.reset_state_tracking()
            env.reset()
            env.run([p1_func, p2_func])
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)
        
    reward = env.state[0].reward if env.state[0].reward is not None else 0
    ep_len = len(env.steps)
    
    telemetry = agent_module.evaluation_telemetry
    avg_ent = sum(telemetry["entropy"]) / len(telemetry["entropy"]) if telemetry["entropy"] else 0
    avg_hs = sum(telemetry["hand_sizes"]) / len(telemetry["hand_sizes"]) if telemetry["hand_sizes"] else 0
    action_paralysis = telemetry["action_paralysis"]
    
    p1_prizes, p2_prizes = None, None
    first_prize = None
    kos = 0
    
    for step in env.steps:
        if step[0].observation:
            obs_dict = step[0].observation
            current = obs_dict.get("current") or {}
            players = current.get("players") or []
            if len(players) == 2:
                p1_idx = current.get("yourIndex", 0)
                p2_idx = 1 - p1_idx
                p1_cur = len(players[p1_idx].get("prize", []))
                p2_cur = len(players[p2_idx].get("prize", []))
                
                if p1_prizes is None and p1_cur > 0: p1_prizes = p1_cur
                if p2_prizes is None and p2_cur > 0: p2_prizes = p2_cur
                    
                if p1_prizes is not None and p1_cur < p1_prizes:
                    if first_prize is None: first_prize = "p1"
                    p1_prizes = p1_cur
                    
                if p2_prizes is not None and p2_cur < p2_prizes:
                    if first_prize is None: first_prize = "p2"
                    kos += (p2_prizes - p2_cur)
                    p2_prizes = p2_cur
                    
    res = {
        "reward": reward,
        "ep_len": ep_len,
        "avg_ent": avg_ent,
        "avg_hs": avg_hs,
        "action_paralysis": action_paralysis,
        "first_prize": 1 if first_prize == "p1" else 0,
        "kos": kos,
        "opponent": p2_agent_name,
        "html": env.render(mode="html")
    }
    return res

def eval_ipc_worker_loop(actor_id, request_queue, response_pipe, task_queue, result_queue):
    from src.core.engine.ipc_protocol import IPCClient
    agent_module.ipc_client = IPCClient(actor_id, request_queue, response_pipe)
    
    while True:
        try:
            import queue
            task = task_queue.get(timeout=5.0)
        except queue.Empty:
            continue
            
        if task is None:
            break
            
        p1, p2, model_name, p2_type, p2_agent, alpha = task
        try:
            res = eval_worker_run_episode(p1, p2, model_name, p2_type, p2_agent, alpha)
            result_queue.put(res)
        except KeyboardInterrupt:
            break
        except Exception as e:
            import traceback
            print(f"Eval Worker {actor_id} exception: {e}")
            traceback.print_exc()
            result_queue.put(None)

def eval_worker_wrapper(args):
    p1, p2, model_name, p2_type, p2_agent, alpha = args
    try:
        return eval_worker_run_episode(p1, p2, model_name, p2_type, p2_agent, alpha)
    except KeyboardInterrupt:
        return None
    except Exception as e:
        import traceback
        print(f"[Eval Worker Exception] {e}")
        traceback.print_exc()
        return None

def run_evaluate(episodes, workers, p1_deck_arg, opp_deck_arg, model_name, p2_type, p2_agent_arg, alpha=0.0):
    print(f"Running {episodes} matches in 'evaluate' mode with {workers} workers...")
    
    agent_module.IS_TRAINING = False
    agent_module.IS_EVALUATING = True
    
    opp_decks = get_available_decks(opp_deck_arg)
    
    checkpoint_path = os.path.join("assets", "models", model_name)
    if os.path.exists(checkpoint_path):
        import torch
        agent_module.ensemble.active_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"Loaded PyTorch checkpoint from {checkpoint_path} for evaluation!")
        
    diagnostics = {
        "matches": episodes,
        "wins": 0,
        "avg_length": 0,
        "avg_hand_size": 0,
        "avg_entropy": 0,
        "total_action_paralysis": 0,
        "first_prize_taken_pct": 0,
        "avg_pokemon_kos_received": 0
    }
    
    lengths, hand_sizes, entropies, first_prizes, kos_received = [], [], [], [], []
    
    tasks = []
    for _ in range(episodes):
        p2_deck = random.choice(opp_decks)
        tasks.append((p1_deck_arg, p2_deck, model_name, p2_type, p2_agent_arg, alpha))
        
    completed = 0
    latest_res = None
    
    with tqdm(total=episodes, desc="Evaluating", unit="match") as pbar:
        if workers > 1:
            from src.core.engine.worker_pool import WorkerPool
            import torch
            agent_module.ensemble.active_model.cpu()
            pool = WorkerPool(model=agent_module.ensemble.active_model, num_workers=workers)
            agent_module.ensemble.active_model.to(agent_module.ensemble.device)
            
            task_queue = pool.ctx.Queue()
            result_queue = pool.ctx.Queue()
            
            pool.start_server()
            
            for task in tasks:
                task_queue.put(task)
            for _ in range(workers):
                task_queue.put(None)
                
            args_list = [(task_queue, result_queue)] * workers
            os.environ["KAGGLE_AGENT_CPU_ONLY"] = "1"
            pool.spawn_actors(eval_ipc_worker_loop, args_list)
            os.environ["KAGGLE_AGENT_CPU_ONLY"] = "0"
            
            try:
                for _ in range(episodes):
                    res = result_queue.get()
                    if not res: continue
                    latest_res = res
                    completed += 1
                    
                    if res["reward"] == 1: diagnostics["wins"] += 1
                    lengths.append(res["ep_len"])
                    entropies.append(res["avg_ent"])
                    hand_sizes.append(res["avg_hs"])
                    diagnostics["total_action_paralysis"] += res["action_paralysis"]
                    first_prizes.append(res["first_prize"])
                    kos_received.append(res["kos"])
                    
                    pbar.update(1)
            except KeyboardInterrupt:
                print("\n[Ctrl+C] Evaluation interrupted by user.")
            finally:
                pool.stop()
                pool.join_actors()
        else:
            agent_module.ipc_client = None
            try:
                for task in tasks:
                    res = eval_worker_wrapper(task)
                    if not res: continue
                    latest_res = res
                    completed += 1
                    
                    if res["reward"] == 1: diagnostics["wins"] += 1
                    lengths.append(res["ep_len"])
                    entropies.append(res["avg_ent"])
                    hand_sizes.append(res["avg_hs"])
                    diagnostics["total_action_paralysis"] += res["action_paralysis"]
                    first_prizes.append(res["first_prize"])
                    kos_received.append(res["kos"])
                    
                    pbar.update(1)
            except KeyboardInterrupt:
                print("\n[Ctrl+C] Evaluation interrupted by user.")
                
    diagnostics["avg_length"] = sum(lengths) / len(lengths) if lengths else 0
    diagnostics["avg_hand_size"] = sum(hand_sizes) / len(hand_sizes) if hand_sizes else 0
    diagnostics["avg_entropy"] = sum(entropies) / len(entropies) if entropies else 0
    diagnostics["first_prize_taken_pct"] = (sum(first_prizes) / len(first_prizes)) * 100 if first_prizes else 0
    diagnostics["avg_pokemon_kos_received"] = sum(kos_received) / len(kos_received) if kos_received else 0
    
    if latest_res:
        os.makedirs(os.path.join("assets", "results", "diagnostics"), exist_ok=True)
        replay_path = os.path.join("assets", "results", "diagnostics", "latest_replay.html")
        with open(replay_path, "w") as f:
            f.write(latest_res["html"])
        
        meta_path = os.path.join("assets", "results", "diagnostics", "latest_replay_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"reward": latest_res["reward"], "opponent": latest_res["opponent"]}, f)
            
    out_name = p2_agent_arg if p2_type == "rules" else "rl_mirror"
    os.makedirs(os.path.join("assets", "results", "diagnostics"), exist_ok=True)
    out_path = os.path.join("assets", "results", "diagnostics", f"diagnostics_{out_name}.json")
    with open(out_path, "w") as f:
        json.dump(diagnostics, f, indent=4)
    print(f"Diagnostics saved to {out_path}")
