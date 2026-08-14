import os
import sys
import csv
import random
from multiprocessing import Pool
from tqdm import tqdm
import torch
from kaggle_environments import make

import src.core.agent.agent as agent_module # We will move agent.py to src/core/agent/agent.py soon
from src.core.models.replay_buffer import ReplayBuffer
from src.core.models.trainer import Trainer
from src.core.engine.learner import LearnerProcess
from src.core.utils.utils import load_deck, get_available_decks

class CurriculumWrapper:
    def __init__(self, agent_func, target_turns=0):
        self.agent_func = agent_func
        self.target_turns = target_turns
        self.turns_taken = 0
        
    def __call__(self, obs_dict, configuration=None):
        if self.turns_taken < self.target_turns:
            from src.core.parser import parse_observation
            parsed_obs = parse_observation(obs_dict)
            if parsed_obs.select and parsed_obs.select.option:
                options = parsed_obs.select.option
                for i, opt in enumerate(options):
                    opt_type = opt["type"] if isinstance(opt, dict) else getattr(opt, "type", None)
                    if opt_type == 8: # OptionType.ATTACH
                        area = opt.get("inPlayArea", 4) if isinstance(opt, dict) else getattr(opt, "kwargs", {}).get("inPlayArea", 4)
                        idx = opt.get("inPlayIndex", 0) if isinstance(opt, dict) else getattr(opt, "kwargs", {}).get("inPlayIndex", 0)
                        if area == 4 and idx == 0:
                            tracker = agent_module._state_tracker[parsed_obs.current.yourIndex]
                            
                            # Prevent forcing a second energy attachment in the same turn, 
                            # which crashes the environment.
                            if tracker.get("has_attached_energy_this_turn", False) and tracker.get("current_turn") == parsed_obs.current.turn:
                                continue
                                
                            self.turns_taken += 1
                            
                            # Manually update tracker so the agent doesn't try to attach again and crash!
                            tracker["current_turn"] = parsed_obs.current.turn
                            tracker["has_attached_energy_this_turn"] = True
                            tracker["current_phase"] = max(tracker.get("current_phase", 1), 3)
                            
                            return [i]
        import inspect
        if configuration and "configuration" in inspect.signature(self.agent_func).parameters:
            return self.agent_func(obs_dict, configuration)
        return self.agent_func(obs_dict)

def worker_run_episode(p1_deck_path, p2_deck_path, model_name=None, p2_type="rl", p2_agent_name=None, epsilon=0.01, temperature=1.0, curriculum_turns=0, alpha=0.0):
    # Force workers to use CPU to avoid massive GPU context switching overhead
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    torch.set_num_threads(1)
    
    agent_module.CURRENT_EPSILON = epsilon
    agent_module.CURRENT_ALPHA = alpha
    agent_module.CURRENT_TEMPERATURE = temperature
    
    # Lock model for training
    agent_module.IS_TRAINING = True
    if model_name:
        archetype = model_name.split("_")[0]
        if archetype in agent_module.bayesian_tracker.archetypes:
            if archetype not in agent_module.ensemble.models:
                # Add a reference so switch_model doesn't throw a fallback warning
                agent_module.ensemble.models[archetype] = agent_module.ensemble.models.get("general")
            agent_module.ensemble.switch_model(archetype)
            
    snapshot_path = os.path.join("assets", "models", model_name) if model_name else os.path.join("assets", "models", "latest_snapshot.pt")
    if os.path.exists(snapshot_path):
        try:
            agent_module.ensemble.active_model.load_state_dict(torch.load(snapshot_path, weights_only=True))
        except Exception:
            pass # ignore loading errors if file is being written concurrently
            
    # Resolve 'mixed' early so we can assign the correct deck
    if p2_type == "mixed":
        p2_type = "rl" if random.random() < 0.8 else "rules"
        
    if p2_type == "rules":
        from src.core.rules_agents import get_available_rules_agents
        p2_actual_name = p2_agent_name
        if not p2_actual_name or p2_actual_name == "all":
            p2_actual_name = random.choice(get_available_rules_agents())
        elif p2_actual_name in ["aggro", "control", "prob"]:
            p2_actual_name = random.choice(get_available_rules_agents(p2_actual_name))
        
        p2_agent_name = p2_actual_name
        # Override the deck with the specific rule agent's deck!
        p2_deck_path = os.path.join("assets", "decks", "rules", f"{p2_actual_name}.csv")

    p1_deck = load_deck(p1_deck_path)
    p2_deck = load_deck(p2_deck_path)
    
    # Redirect at OS level to catch C++ prints (e.g. from cg-lib)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            env = make("cabt", configuration={"decks": [p1_deck, p2_deck], "actTimeout": 99999, "runTimeout": 99999})
            agent_module._local_env_ref = env
            
            local_buffer = ReplayBuffer(gamma=0.99)
            agent_module.global_replay_buffer = local_buffer
            
            p1_func = agent_module.agent
            p2_func = agent_module.agent
            
            if p2_type == "rules":
                from src.core.rules_agents import get_rules_agent
                p2_func = get_rules_agent(p2_agent_name)
            else:
                p2_func = agent_module.agent
                
            if curriculum_turns > 0:
                p1_func = CurriculumWrapper(agent_module.agent, target_turns=curriculum_turns)
                
            agent_module.reset_state_tracking()
            env.reset()
            env.run([p1_func, p2_func])
            
            # Check for INVALID actions (e.g. agent picked an illegal action during exploration)
            # Instead of throwing an exception and discarding data, we let the episode finish 
            # so the agent learns a massive penalty for the invalid action!
            invalid_warning = None
            for step_idx, step_data in enumerate(env.steps):
                for p_idx, player_state in enumerate(step_data):
                    status = player_state.status if hasattr(player_state, 'status') else player_state.get('status')
                    if status in ['INVALID', 'ERROR', 'TIMEOUT']:
                        if p_idx != 0:
                            # If rules agent crashes, we just abort, it's not our agent's fault
                            raise AssertionError(f"Rules Agent (Player {p_idx}) hit {status} status! Aborting.")
                        else:
                            action = player_state.action if hasattr(player_state, 'action') else player_state.get('action')
                            invalid_warning = f"\n[WARNING] RL Agent (Player 0) took an INVALID action at Step {step_idx}! Action: {action}. Match aborted with penalty."
                        
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            if invalid_warning:
                print(invalid_warning)
            
            # Restore OS level descriptors
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            os.close(devnull_fd)
    
    reward = env.state[0].reward if env.state[0].reward is not None else 0.0
    
    # If the episode ended early because our agent failed, punish heavily
    status = env.state[0].status if hasattr(env.state[0], 'status') else env.state[0].get('status')
    if status in ['INVALID', 'ERROR', 'TIMEOUT']:
        reward = -1.0
    
    from src.core.agent.rewards import REWARD_CONFIG, _track_reward
    tracker = agent_module._state_tracker[0]
    
    if reward > 0:
        scaled_terminal_reward = _track_reward(tracker, "r_win", REWARD_CONFIG.get("r_win", 15.0))
    elif reward < 0:
        scaled_terminal_reward = _track_reward(tracker, "r_loss", REWARD_CONFIG.get("r_loss", -15.0))
        # Extra penalty if it was specifically due to an INVALID action crash
        if status == 'INVALID':
            scaled_terminal_reward += _track_reward(tracker, "r_invalid_action_penalty", -20.0)
    else:
        scaled_terminal_reward = 0.0
    
    # Push the final dangling action from Player 1's state tracker
    if tracker.get("last_state") is not None:
        for a in tracker["last_actions"]:
            local_buffer.push(
                tracker["last_state"],
                a,
                tracker["last_log_prob"],
                tracker["last_value"],
                step_reward=0.0
            )
        
    trajectory = local_buffer.finalize_episode(scaled_terminal_reward)
    episode_length = len(env.steps)
    
    opponent_name = p2_deck_path
    if p2_type == "rules":
        opponent_name = f"[RULES] {p2_actual_name}"
    else:
        opponent_name = f"[SELF-PLAY] {os.path.basename(p2_deck_path)}"
        
    return opponent_name, reward, episode_length, trajectory, tracker.get("reward_metrics", {})

def ipc_worker_loop(actor_id, request_queue, response_pipe, task_queue, result_queue):
    # Initialize IPC Client globally for the worker process
    from src.core.engine.ipc_protocol import IPCClient
    agent_module.ipc_client = IPCClient(actor_id, request_queue, response_pipe)
    
    # Process tasks until poison pill (None)
    while True:
        try:
            import queue
            task = task_queue.get(timeout=5.0)
        except queue.Empty:
            continue
            
        if task is None:
            break
            
        p1, p2, model_name, debug, p2_type, p2_agent, epsilon, temperature, curriculum_turns, alpha = task
        if debug:
            import cProfile
            pr = cProfile.Profile()
            pr.enable()
            
        try:
            res = worker_run_episode(p1, p2, model_name, p2_type, p2_agent, epsilon, temperature, curriculum_turns, alpha)
            if debug:
                pr.disable()
                out_dir = os.path.join("logs", "debug", "worker_profiles")
                os.makedirs(out_dir, exist_ok=True)
                pr.dump_stats(os.path.join(out_dir, f"worker_{os.getpid()}.prof"))
            result_queue.put(res)
        except KeyboardInterrupt:
            break
        except Exception as e:
            import traceback
            print(f"Worker {actor_id} exception: {e}")
            traceback.print_exc()
            result_queue.put(None)

def worker_wrapper(args):
    # Fallback for synchronous mode
    p1, p2, model_name, debug, p2_type, p2_agent, epsilon, temperature, curriculum_turns, alpha = args
    if debug:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
        
    try:
        res = worker_run_episode(p1, p2, model_name, p2_type, p2_agent, epsilon, temperature, curriculum_turns, alpha)
        if debug:
            pr.disable()
            out_dir = os.path.join("logs", "debug", "worker_profiles")
            os.makedirs(out_dir, exist_ok=True)
            pr.dump_stats(os.path.join(out_dir, f"worker_{os.getpid()}.prof"))
        return res
    except KeyboardInterrupt:
        return None
    except Exception as e:
        import traceback
        print(f"\n[Worker Exception] {e}")
        traceback.print_exc()
        return None
    except Exception as e:
        import traceback
        print(f"Worker exception: {e}")
        traceback.print_exc()
        return None

def run_train(episodes, workers, p1_deck_path, p2_deck_path, model_name, p2_type, p2_agent, debug=False, start_epsilon=None, alpha=0.0):
    if not os.path.exists("assets/models"):
        os.makedirs("assets/models")
    print(f"Running {episodes} matches in 'train' mode with {workers} workers...")
    opp_decks = get_available_decks(p2_deck_path)
    if not opp_decks:
        print(f"No decks found for filter: {p2_deck_path}")
        return
        
    master_buffer = ReplayBuffer(gamma=0.99)
    
    checkpoint_path = os.path.join("assets", "models", model_name) if model_name else os.path.join("assets", "models", "general_model.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    if os.path.exists(checkpoint_path):
        agent_module.ensemble.active_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"Loaded existing PyTorch checkpoint from {checkpoint_path}!")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch!")
    
    model_prefix = model_name.replace("_model.pt", "").replace(".pt", "") if model_name else "general"
    csv_filename = f"{model_prefix}_training_metrics.csv"
    log_file = os.path.join("assets", "results", "rl_training", csv_filename)
    reward_csv_filename = f"{model_prefix}_reward_metrics.csv"
    reward_log_file = os.path.join("assets", "results", "rl_training", reward_csv_filename)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    file_exists = os.path.isfile(log_file)
    reward_file_exists = os.path.isfile(reward_log_file)
    
    reward_keys = []
    try:
        import json
        with open("assets/reward/reward_shaping.json", "r") as f:
            reward_keys = sorted(list(json.load(f).keys()))
    except:
        pass
        
    reward_headers = ["Episode", "Opponent_Deck", "Reward"]
    for k in reward_keys:
        reward_headers.extend([f"Count_{k}", f"Total_{k}"])
        
    with open(log_file, "a", newline="") as csvfile, open(reward_log_file, "a", newline="") as reward_csvfile:
        writer = csv.writer(csvfile)
        reward_writer = csv.writer(reward_csvfile)
        if not file_exists:
            writer.writerow(["Episode", "Opponent_Deck", "Reward", "Episode_Length", "Policy_Loss", "Value_Loss"])
        if not reward_file_exists:
            reward_writer.writerow(reward_headers)
        
        if p2_type == "rules":
            from src.core.rules_agents import get_available_rules_agents
            if p2_agent == "all" or not p2_agent:
                available_p2_agents = get_available_rules_agents()
            elif p2_agent in ["aggro", "control", "prob"]:
                available_p2_agents = get_available_rules_agents(p2_agent)
            elif "," in p2_agent:
                available_p2_agents = [x.strip() for x in p2_agent.split(",")]
            else:
                available_p2_agents = [p2_agent]
        else:
            available_p2_agents = [None]
            
        completed = 0
        snapshot_path = os.path.join("assets", "models", model_name) if model_name else os.path.join("assets", "models", "latest_snapshot.pt")
        torch.save(agent_module.ensemble.active_model.state_dict(), snapshot_path)
        
        with tqdm(total=episodes, desc="Training", unit="match") as pbar:
            tasks = []
            for i in range(episodes):
                p2_deck = random.choice(opp_decks)
                p2_agent_choice = random.choice(available_p2_agents) if available_p2_agents[0] else None
                epsilon = 0.0
                curriculum_turns = 0
                if start_epsilon is not None and start_epsilon > 0.0:
                    epsilon = start_epsilon - (start_epsilon * i / max(1, episodes * 0.8))
                    epsilon = max(0.0, epsilon)
                    # Decay curriculum turns from 2 to 0 over the first 50% of training
                    progress = i / max(1, episodes)
                    curriculum_turns = int(max(0, 2 * (1.0 - (progress * 2))))
                    
                tasks.append((p1_deck_path, p2_deck, model_name, debug, p2_type, p2_agent_choice, epsilon, 1.0, curriculum_turns, alpha))

            if workers > 1:
                from src.core.engine.worker_pool import WorkerPool
                # Temporarily move to CPU to avoid massive VRAM usage when 48 actors unpickle the model
                agent_module.ensemble.active_model.cpu()
                pool = WorkerPool(model=agent_module.ensemble.active_model, num_workers=workers)
                
                # Move main process model back to GPU so Learner has a VRAM-bound model
                agent_module.ensemble.active_model.to(agent_module.ensemble.device)
                
                # Spawn background Learner Process
                trajectory_queue = pool.ctx.Queue()
                metrics_queue = pool.ctx.Queue()
                learner = LearnerProcess(agent_module.ensemble, trajectory_queue, metrics_queue, update_freq=100, batch_episodes=5, model_name=model_name)
                learner.start()

                # Use the pool's context for task queues
                task_queue = pool.ctx.Queue()
                result_queue = pool.ctx.Queue()
                
                pool.start_server()
                
                # Enqueue tasks in chunks or dynamically
                for task in tasks:
                    task_queue.put(task)
                    
                # Put poison pills
                for _ in range(workers):
                    task_queue.put(None)
                    
                args_list = [(task_queue, result_queue)] * workers
                
                os.environ["KAGGLE_AGENT_CPU_ONLY"] = "1"
                pool.spawn_actors(ipc_worker_loop, args_list)
                os.environ["KAGGLE_AGENT_CPU_ONLY"] = "0"
                
                policy_loss, value_loss = 0.0, 0.0
                
                try:
                    for _ in range(episodes):
                        res = result_queue.get()
                        if not res: continue
                        p2_path, reward, ep_len, trajectory, reward_metrics = res
                        completed += 1
                        
                        trajectory_queue.put(trajectory)
                        
                        try:
                            import queue
                            policy_loss, value_loss = metrics_queue.get_nowait()
                        except queue.Empty:
                            pass
                            
                        if completed % 5 == 0 or completed == episodes:
                            pbar.set_postfix({"P_Loss": f"{policy_loss:.3f}", "V_Loss": f"{value_loss:.3f}"})
                            
                        writer.writerow([completed, os.path.basename(p2_path), reward, ep_len, policy_loss, value_loss])
                        
                        reward_row = [completed, os.path.basename(p2_path), reward]
                        for k in reward_keys:
                            metrics = reward_metrics.get(k, {"count": 0, "total": 0.0})
                            reward_row.extend([metrics["count"], metrics["total"]])
                        reward_writer.writerow(reward_row)
                        
                        csvfile.flush()
                        reward_csvfile.flush()
                        pbar.update(1)
                except KeyboardInterrupt:
                    print("\n[Ctrl+C] Training interrupted by user. Cleaning up workers...")
                finally:
                    pool.stop()
                    pool.join_actors()
                    learner.stop()
                    learner.join()
                
            else:
                agent_module.ipc_client = None
                trainer = Trainer(ensemble=agent_module.ensemble, lr=1e-4)
                try:
                    for task in tasks:
                        res = worker_wrapper(task)
                        if not res: continue
                        p2_path, reward, ep_len, trajectory, reward_metrics = res
                        completed += 1
                        master_buffer.add_trajectory(trajectory)
                        
                        policy_loss, value_loss = 0.0, 0.0
                        if completed % 5 == 0 or completed == episodes:
                            policy_loss, value_loss = trainer.update(master_buffer)
                            master_buffer.clear()
                            pbar.set_postfix({"P_Loss": f"{policy_loss:.3f}", "V_Loss": f"{value_loss:.3f}"})
                            
                            tmp_snapshot = snapshot_path + ".tmp"
                            torch.save(agent_module.ensemble.active_model.state_dict(), tmp_snapshot)
                            os.replace(tmp_snapshot, snapshot_path)
                            
                        if completed % 100 == 0 or completed == episodes:
                            torch.save(agent_module.ensemble.active_model.state_dict(), checkpoint_path)
                            
                        writer.writerow([completed, os.path.basename(p2_path), reward, ep_len, policy_loss, value_loss])
                        
                        reward_row = [completed, os.path.basename(p2_path), reward]
                        for k in reward_keys:
                            metrics = reward_metrics.get(k, {"count": 0, "total": 0.0})
                            reward_row.extend([metrics["count"], metrics["total"]])
                        reward_writer.writerow(reward_row)
                        
                        csvfile.flush()
                        reward_csvfile.flush()
                        pbar.update(1)
                except KeyboardInterrupt:
                    print("\n[Ctrl+C] Training interrupted by user.")
