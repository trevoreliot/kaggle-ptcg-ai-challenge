import torch
import torch.multiprocessing as mp
import time
import os
import queue
from typing import Dict, Any
from src.core.engine.ipc_protocol import InferenceRequest, InferenceResponse

class BatchedInferenceServer(mp.Process):
    """
    Dedicated GPU IPC Inference Engine.
    Receives state tensors from CPU actors via a shared Queue, batches them,
    performs a single vectorized forward pass on the GPU, and sends results back via Pipes.
    """
    def __init__(self, model, request_queue, response_pipes: Dict[int, Any], 
                 batch_size=64, timeout_ms=2.0):
        super().__init__()
        self.model = model
        self.request_queue = request_queue
        self.response_pipes = response_pipes
        self.batch_size = batch_size
        self.timeout_s = timeout_ms / 1000.0
        self.running = mp.Event()
        self.snapshot_path = os.path.join("assets", "models", "latest_snapshot.pt")
        self.last_mtime = 0.0
        self.last_check_time = time.time()

    def run(self):
        self.running.set()
        
        # Ensure CUDA initialization happens *inside* the new process
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        
        # --- PERFORMANCE TWEAKS ---
        if device.type == "cuda":
            # 1. Enable TensorFloat-32 on Tensor Cores for massive math speedups
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # 2. Enable cuDNN benchmark for static input sizes
            torch.backends.cudnn.benchmark = True
            
            # 3. Fuse CUDA kernels to reduce CPU overhead
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                print(f"[BatchServer] torch.compile skipped: {e}")
        # --------------------------
        
        # Pre-allocate batched tensor memory (prevent memory churn)
        # 6160 is the feature_size defined in StateEncoder
        batched_tensor = torch.zeros((self.batch_size, 6160), device=device, dtype=torch.float32)
        
        # Initialize mtime if snapshot exists
        if os.path.exists(self.snapshot_path):
            self.last_mtime = os.path.getmtime(self.snapshot_path)
            
        while self.running.is_set():
            # Periodically check for new weights from the Learner
            current_time = time.time()
            if current_time - self.last_check_time > 5.0:  # Check every 5 seconds
                self.last_check_time = current_time
                if os.path.exists(self.snapshot_path):
                    mtime = os.path.getmtime(self.snapshot_path)
                    if mtime > self.last_mtime:
                        self.last_mtime = mtime
                        try:
                            # Load strictly to device to avoid RAM spikes
                            self.model.load_state_dict(torch.load(self.snapshot_path, map_location=device, weights_only=True))
                        except Exception as e:
                            print(f"[BatchServer] Failed to hot-reload weights: {e}")
                            
            batch_reqs = []
            try:
                # Block until we get at least one request
                first_req = self.request_queue.get(timeout=1.0)
                if first_req is None:
                    self.running.clear()
                    break
                batch_reqs.append(first_req)
            except queue.Empty:
                continue
                
            # Grab any additional requests that arrived concurrently
            while len(batch_reqs) < self.batch_size:
                try:
                    req = self.request_queue.get_nowait()
                    if req is None:
                        self.running.clear()
                        break
                    batch_reqs.append(req)
                except queue.Empty:
                    break
            
            if not batch_reqs:
                continue
                
            current_batch_size = len(batch_reqs)
            
            with torch.no_grad():
                # Fill pre-allocated tensor
                for i, req in enumerate(batch_reqs):
                    batched_tensor[i].copy_(torch.from_numpy(req.state_array).squeeze())
                    
                # Vectorized forward pass
                # We intentionally evaluate the FULL static batch size (self.batch_size) instead of slicing. 
                # This ensures the matrix shape never changes, which allows torch.compile and cudnn.benchmark
                # to run at absolute maximum efficiency without needing to re-profile every cycle!
                values, policies = self.model(batched_tensor)
                
                # Move to CPU and numpy before sending over IPC, only taking the valid outputs
                values = values[:current_batch_size].cpu()
                policies = policies[:current_batch_size].cpu().numpy()
                
            # Dispatch back to actors
            for i, req in enumerate(batch_reqs):
                resp = InferenceResponse(
                    request_id=req.request_id,
                    value=values[i].item(),
                    policy_logits=policies[i]
                )
                try:
                    self.response_pipes[req.actor_id].send(resp)
                except Exception as e:
                    print(f"[BatchServer] Failed to send response to actor {req.actor_id}: {e}")
                    
    def stop(self):
        self.running.clear()
        try:
            self.request_queue.put(None)
        except Exception:
            pass
