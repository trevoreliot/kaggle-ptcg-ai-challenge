import torch
from typing import Dict, Any, Tuple
from src.core.models.base import BaseNetwork, StateEncoder
from src.core.parser import Observation

class EnsembleManager:
    """
    Manages the suite of Neural Networks (General, Anti-Aggro, Anti-Control).
    Currently defaults to loading an untrained General model for execution structure.
    """
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.encoder = StateEncoder()
        
        # Load the base default general network
        self.active_model = BaseNetwork().to(self.device)
        self.active_model.eval()
        
        self.models: Dict[str, BaseNetwork] = {
            "general": self.active_model
        }
        self.current_mode = "general"
        
    def load_model(self, name: str, path: str):
        """Loads weights from a .pt file into a model slot."""
        model = BaseNetwork().to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        self.models[name] = model
        
    def switch_model(self, name: str) -> bool:
        """Swaps the active model. Returns True if successful, False if fallback used."""
        if name == self.current_mode:
            return True
            
        if name in self.models:
            self.active_model = self.models[name]
            self.current_mode = name
            print(f"[Ensemble] Switched model to: {name}")
            return True
        else:
            print(f"[Ensemble] Warning: Model '{name}' not found. Falling back to 'general'.")
            self.active_model = self.models["general"]
            self.current_mode = "general"
            return False
            
    def evaluate(self, obs: Observation) -> Tuple[float, list]:
        """
        Encodes the observation and passes it through the active neural network.
        Returns the scalar Value, and the Policy list.
        """
        with torch.no_grad():
            state_tensor = self.encoder.encode(obs).unsqueeze(0).to(self.device)
            value_tensor, policy_tensor = self.active_model(state_tensor)
            
            value = value_tensor.item()
            policy = policy_tensor.squeeze(0).tolist()
            
            return value, policy
