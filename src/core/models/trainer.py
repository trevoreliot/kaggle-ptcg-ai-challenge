import torch
import torch.nn.functional as F
import torch.optim as optim

class Trainer:
    def __init__(self, ensemble, lr=1e-4):
        self.ensemble = ensemble
        self.model = ensemble.active_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def update(self, buffer):
        batch = buffer.get_batch()
        if len(batch) == 0:
            return 0.0, 0.0
            
        device = next(self.model.parameters()).device
        states = torch.stack([torch.as_tensor(b["state"]).squeeze(0) if len(b["state"].shape) > 1 else torch.as_tensor(b["state"]) for b in batch]).to(device)
        actions = torch.tensor([b["action"] for b in batch]).to(device)
        returns = torch.tensor([b["return"] for b in batch], dtype=torch.float32).to(device)
        
        self.model.train()
        values, policies = self.model(states)
        values = values.squeeze(-1)
        
        value_loss = F.mse_loss(values, returns)
        
        # A2C Policy Loss
        advantages = returns - values.detach()
        
        # Advantage normalization to stabilize training
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Gather the log_probs of the actions that were actually taken
        # policies shape: [batch_size, num_actions]
        # We need to compute log_softmax
        log_probs = F.log_softmax(policies, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Entropy bonus to prevent premature convergence / exploding logits
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        policy_loss = -(action_log_probs * advantages).mean() - 0.01 * entropy
        
        loss = policy_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()
