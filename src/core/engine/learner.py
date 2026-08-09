import os
import torch
import torch.multiprocessing as mp
import time
import queue

from src.core.models.replay_buffer import ReplayBuffer
from src.core.models.trainer import Trainer

class LearnerProcess(mp.Process):
    """
    Dedicated GPU backpropagation engine.
    Continuously receives trajectories from the main thread, accumulates them,
    and performs gradient updates in the background without blocking actors.
    """
    def __init__(self, ensemble, trajectory_queue, metrics_queue=None, update_freq=100, batch_episodes=5, model_name="general_model.pt"):
        super().__init__()
        self.ensemble = ensemble
        self.trajectory_queue = trajectory_queue
        self.metrics_queue = metrics_queue
        self.update_freq = update_freq
        self.batch_episodes = batch_episodes
        self.running = mp.Event()
        self.checkpoint_path = os.path.join("assets", "models", model_name)
        self.snapshot_path = os.path.join("assets", "models", "latest_snapshot.pt")
        os.makedirs(os.path.dirname(self.snapshot_path), exist_ok=True)
        
    def run(self):
        self.running.set()
        
        # Ensure CUDA initialization happens *inside* the new process
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ensemble.active_model.to(device)
        self.ensemble.active_model.train()
        
        # Performance Tweaks
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
        trainer = Trainer(ensemble=self.ensemble, lr=1e-4)
        
        # A2C is on-policy, so we accumulate recent trajectories, train, and clear.
        # Max capacity protects against memory leaks if the queue backs up.
        buffer = ReplayBuffer(capacity=100000, gamma=0.99)
        
        episodes_collected = 0
        gradient_steps = 0
        
        while self.running.is_set():
            try:
                # Block until we get at least one trajectory
                trajectory = self.trajectory_queue.get(timeout=1.0)
                if trajectory is None: # Poison pill
                    self.running.clear()
                    break
                    
                buffer.add_trajectory(trajectory)
                episodes_collected += 1
                
                # Drain queue of any other ready trajectories up to a reasonable limit
                while not self.trajectory_queue.empty() and episodes_collected < self.batch_episodes * 2:
                    try:
                        traj = self.trajectory_queue.get_nowait()
                        if traj is None:
                            self.running.clear()
                            break
                        buffer.add_trajectory(traj)
                        episodes_collected += 1
                    except queue.Empty:
                        break
                        
            except queue.Empty:
                pass
                
            # Train if we have enough episodes
            if episodes_collected >= self.batch_episodes:
                policy_loss, value_loss = trainer.update(buffer)
                buffer.clear()
                episodes_collected = 0
                gradient_steps += 1
                
                if self.metrics_queue is not None:
                    try:
                        # Clear old metrics if main thread is slow
                        while not self.metrics_queue.empty():
                            self.metrics_queue.get_nowait()
                        self.metrics_queue.put((policy_loss, value_loss))
                    except Exception:
                        pass
                        
                # Periodically sync weights to disk so BatchServer can hot-reload them
                if gradient_steps % self.update_freq == 0:
                    tmp_snapshot = self.snapshot_path + ".tmp"
                    try:
                        torch.save(self.ensemble.active_model.state_dict(), tmp_snapshot)
                        os.replace(tmp_snapshot, self.snapshot_path)
                    except Exception as e:
                        print(f"[Learner] Error saving snapshot: {e}")
                        
        # Final save when stopped
        try:
            torch.save(self.ensemble.active_model.state_dict(), self.checkpoint_path)
            torch.save(self.ensemble.active_model.state_dict(), self.snapshot_path)
        except Exception:
            pass
            
    def stop(self):
        self.running.clear()
        try:
            self.trajectory_queue.put(None)
        except Exception:
            pass
