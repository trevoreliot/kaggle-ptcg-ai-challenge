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
            "state": state,
            "action": action,
            "log_prob": log_prob,
            "value": value
        })
        
    def finalize_episode(self, reward: float):
        R = reward
        for step in reversed(self.current_episode):
            R = R * self.gamma
            step["return"] = R
            if len(self.buffer) < self.capacity:
                self.buffer.append(step)
            else:
                self.buffer.pop(0)
                self.buffer.append(step)
        self.current_episode = []
        
    def get_batch(self):
        return self.buffer
        
    def clear(self):
        self.buffer = []
