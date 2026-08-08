from dataclasses import dataclass
import torch
import numpy as np

@dataclass
class InferenceRequest:
    actor_id: int
    request_id: int
    state_array: np.ndarray

@dataclass
class InferenceResponse:
    request_id: int
    value: float
    policy_logits: np.ndarray

class IPCClient:
    """
    Client for the CPU worker to send inference requests to the Batch Server over IPC queues/pipes.
    """
    def __init__(self, actor_id, request_queue, response_pipe):
        self.actor_id = actor_id
        self.request_queue = request_queue
        self.response_pipe = response_pipe
        self._request_counter = 0
        
    def evaluate(self, state_array: np.ndarray):
        req_id = self._request_counter
        self._request_counter += 1
        
        req = InferenceRequest(self.actor_id, req_id, state_array)
        self.request_queue.put(req)
        
        # Block until response is received. 
        # (A short timeout could be used here to avoid deadlocks, but pipe is generally reliable)
        resp = self.response_pipe.recv()
        
        if resp.request_id != req_id:
            raise RuntimeError(f"Mismatched request ID in actor {self.actor_id}: expected {req_id}, got {resp.request_id}")
            
        return resp.value, resp.policy_logits

