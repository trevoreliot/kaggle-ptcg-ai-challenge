import torch
import torch.nn.functional as F
from src.core.models.ensemble import EnsembleManager
from src.core.models.replay_buffer import ReplayBuffer

class Trainer:
    def __init__(self, ensemble: EnsembleManager, lr: float = 1e-4):
        self.ensemble = ensemble
        self.device = ensemble.device
        
        # We optimize the 'general' model which is the active model at start
        # If we add specific model training, we would pass the specific model here.
        self.optimizer = torch.optim.Adam(self.ensemble.active_model.parameters(), lr=lr)
        
    def update(self, buffer: ReplayBuffer):
        """
        Runs one step of Advantage Actor-Critic optimization using the accumulated buffer.
        """
        if len(buffer) == 0:
            return
            
        # Convert lists to tensors
        states = torch.cat(buffer.states).to(self.device)
        actions = torch.tensor(buffer.actions, dtype=torch.int64).to(self.device)
        returns = torch.tensor(buffer.rewards, dtype=torch.float32).to(self.device)
        # Calculate current values and action probabilities through the network (with gradients!)
        values, policies = self.ensemble.active_model(states)
        
        # Policy gives probabilities for all possible actions. We want the log_prob of the chosen action.
        # Ensure probabilities are > 0 for log
        log_policies = torch.log(policies + 1e-10)
        
        # Gather the log probability of the action that was actually taken
        action_log_probs = log_policies.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Calculate advantages (Reward - Baseline Value). We detach values so Advantage is treated as a constant scalar for Policy
        values = values.squeeze(-1)
        advantages = returns - values.detach()
        
        # We want to maximize expected reward, which means minimizing: -log_prob * advantage
        policy_loss = -(action_log_probs * advantages).mean()
        
        # We want the Value head to accurately predict the return (MSE Loss)
        value_loss = F.mse_loss(values, returns)
        
        # Total Loss
        loss = policy_loss + value_loss
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.ensemble.active_model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        print(f"[Trainer] Updated Model | Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f}")
        
        # Clear buffer after update
        buffer.reset()
