import torch
from typing import List, Tuple

class ReplayBuffer:
    def __init__(self, capacity: int = 10000, gamma: float = 0.99):
        self.capacity = capacity
        self.gamma = gamma
        self.buffer = []
        self.current_episode = []
        
    def push(self, state: torch.Tensor, action: int, log_prob: torch.Tensor, value: float):
        self.current_episode.append({
            "state": state.detach().cpu().numpy(),
            "action": action,
            "log_prob": log_prob.detach().cpu().numpy(),
            "value": value
        })
        
    def finalize_episode(self, reward: float) -> List[dict]:
        R = reward
        completed_episode = []
        for step in reversed(self.current_episode):
            R = R * self.gamma
            step["return"] = R
            completed_episode.insert(0, step)
            
        self.add_trajectory(completed_episode)
        self.current_episode = []
        return completed_episode
        
    def add_trajectory(self, trajectory: List[dict]):
        for step in trajectory:
            if len(self.buffer) < self.capacity:
                self.buffer.append(step)
            else:
                self.buffer.pop(0)
                self.buffer.append(step)
        
    def get_batch(self):
        return self.buffer
        
    def clear(self):
        self.buffer = []
