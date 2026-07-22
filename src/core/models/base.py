import torch
import torch.nn as nn
from typing import Tuple
from src.core.parser import Observation

class StateEncoder:
    """
    Serializes our Observation dataclass into a fixed size tensor.
    Currently a skeleton implementation. 
    """
    def __init__(self, feature_size: int = 128):
        self.feature_size = feature_size
        
    def encode(self, obs: Observation) -> torch.Tensor:
        # Trivial encoding for skeleton purposes.
        # In a real scenario, this iterates over HP, damage, bench cards, etc.
        # and populates a 1D tensor representing the state.
        return torch.zeros(self.feature_size)

class BaseNetwork(nn.Module):
    """
    Base Neural Network Architecture for the AI agent.
    Takes in an encoded game state and outputs:
    1. Value: Scalar from -1 to 1 representing win probability.
    2. Policy: Vector representing action prior probabilities.
    """
    def __init__(self, input_size: int = 128, hidden_size: int = 256, policy_size: int = 512):
        super(BaseNetwork, self).__init__()
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), 1),
            nn.Tanh() # Squeeze to [-1, 1]
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, policy_size)
            # Output raw logits so trainer can apply log_softmax without double-softmaxing
        )
        
    def forward(self, state_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared_mlp(state_tensor)
        value = self.value_head(features)
        policy = self.policy_head(features)
        return value, policy
