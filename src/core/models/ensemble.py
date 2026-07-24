import os
from typing import Dict, Any, Tuple
from src.core.parser import Observation

try:
    import torch
    from src.core.models.base import BaseNetwork, StateEncoder
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

class FallbackStateEncoder:
    """Fallback encoder if torch is missing."""
    def __init__(self, feature_size: int = 6160):
        self.feature_size = feature_size
        
    def encode_numpy(self, obs: Observation):
        import numpy as np
        return np.zeros((1, self.feature_size), dtype=np.float32)
        
    def encode(self, obs: Observation):
        if TORCH_AVAILABLE:
            import torch
            return torch.zeros(self.feature_size)
        else:
            return self.encode_numpy(obs)[0]

class EnsembleManager:
    """
    Manages the suite of Neural Networks (General, Anti-Aggro, Anti-Control).
    Dynamically falls back to ONNX if Torch is unavailable.
    """
    def __init__(self, device: str = None):
        self.device = None
        if TORCH_AVAILABLE:
            self.encoder = StateEncoder()
        else:
            self.encoder = FallbackStateEncoder()
        
        self.models: Dict[str, Any] = {}
        self.current_mode = "general"
        self.use_onnx = False
        
        if TORCH_AVAILABLE:
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            self.active_model = BaseNetwork().to(self.device)
            self.active_model.eval()
            self.models["general"] = self.active_model
        elif ONNX_AVAILABLE:
            self.use_onnx = True
            onnx_path = os.path.join("assets", "models", "general.onnx")
            if os.path.exists(onnx_path):
                self.active_model = ort.InferenceSession(onnx_path)
                self.models["general"] = self.active_model
            else:
                self.active_model = None
        else:
            raise RuntimeError("Neither Torch nor ONNX Runtime is available!")
            
    def load_model(self, name: str, path: str):
        """Loads weights into a model slot."""
        if path.endswith(".onnx") and ONNX_AVAILABLE:
            self.models[name] = ort.InferenceSession(path)
            self.use_onnx = True
        elif path.endswith(".pt") and TORCH_AVAILABLE:
            model = BaseNetwork().to(self.device)
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            self.models[name] = model
            self.use_onnx = False
            
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
            if "general" in self.models:
                self.active_model = self.models["general"]
                self.current_mode = "general"
            return False
            
    def evaluate(self, obs: Observation) -> Tuple[float, list]:
        """
        Encodes the observation and passes it through the active neural network (PyTorch or ONNX).
        Returns the scalar Value, and the Policy list.
        """
        if self.use_onnx or not TORCH_AVAILABLE:
            state_array = self.encoder.encode_numpy(obs)
            input_name = self.active_model.get_inputs()[0].name
            value_out, policy_out = self.active_model.run(None, {input_name: state_array})
            return float(value_out[0][0]), policy_out[0].tolist()
        else:
            with torch.no_grad():
                state_tensor = self.encoder.encode(obs).unsqueeze(0).to(self.device)
                value_tensor, policy_tensor = self.active_model(state_tensor)
                
                value = value_tensor.item()
                policy = policy_tensor.squeeze(0).tolist()
                
                return value, policy
