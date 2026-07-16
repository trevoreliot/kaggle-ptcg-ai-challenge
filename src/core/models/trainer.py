import torch
import torch.nn.functional as F
import torch.optim as optim

class Trainer:
    def __init__(self, ensemble, lr=1e-4):
        self.ensemble = ensemble
        self.model = ensemble.active_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def update(self, replay_buffer):
        batch = replay_buffer.get_batch()
        if not batch:
            return 0.0, 0.0
            
        states = torch.stack([b["state"] for b in batch]).to(self.model.device)
        returns = torch.tensor([b["return"] for b in batch], dtype=torch.float32).to(self.model.device)
        
        self.model.train()
        values, policies = self.model(states)
        values = values.squeeze(-1)
        
        value_loss = F.mse_loss(values, returns)
        
        # A2C Policy Loss
        advantages = returns - values.detach()
        # In a real scenario we'd compute log probs of actions taken.
        # Since this is a skeleton we'll use a dummy policy loss.
        policy_loss = -(advantages * policies.mean(dim=1)).mean()
        
        loss = value_loss + policy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.model.eval()
        return policy_loss.item(), value_loss.item()
