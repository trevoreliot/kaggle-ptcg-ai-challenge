import torch
from typing import List, Tuple

class ReplayBuffer:
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.reset()
        
    def reset(self):
        """Clears all stored trajectories."""
        # Store individual transitions
        self.states: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[float] = []
        
        self.rewards: List[float] = []
        
        # We track episodes by maintaining a list of transition lists before flattening them
        self.current_episode = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "values": []
        }
        
    def push(self, state: torch.Tensor, action: int, log_prob: torch.Tensor, value: float):
        """Appends a single transition from the current turn to the active episode."""
        self.current_episode["states"].append(state)
        self.current_episode["actions"].append(action)
        self.current_episode["log_probs"].append(log_prob)
        self.current_episode["values"].append(value)
        
    def finalize_episode(self, terminal_reward: float):
        """
        Called at the end of a match. Backpropagates the terminal reward through the episode's turns 
        using the discount factor (gamma), and commits the episode to the main buffer.
        """
        ep_len = len(self.current_episode["states"])
        if ep_len == 0:
            return
            
        # Calculate discounted rewards backwards
        discounted_rewards = [0.0] * ep_len
        running_reward = terminal_reward
        
        for i in reversed(range(ep_len)):
            discounted_rewards[i] = running_reward
            running_reward = running_reward * self.gamma
            
        # Commit to main buffer
        self.states.extend(self.current_episode["states"])
        self.actions.extend(self.current_episode["actions"])
        self.log_probs.extend(self.current_episode["log_probs"])
        self.values.extend(self.current_episode["values"])
        self.rewards.extend(discounted_rewards)
        
        # Reset current episode tracker
        self.current_episode = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "values": []
        }
        
    def __len__(self):
        return len(self.states)
