import torch.multiprocessing as mp
import time
import os
from src.core.engine.batch_server import BatchedInferenceServer

class WorkerPool:
    """
    CPU Multi-process orchestration logic.
    Manages the IPC queues, pipes, and spawning of the BatchServer and Actor processes.
    """
    def __init__(self, model, num_workers=8):
        self.num_workers = num_workers
        self.model = model
        
        self.ctx = mp.get_context('spawn')
        self.request_queue = self.ctx.Queue()
        self.server_pipes = {}
        self.actor_pipes = {}
        
        for i in range(num_workers):
            # Server writes to server_pipe, Actor reads from actor_pipe
            p_recv, p_send = self.ctx.Pipe(duplex=False)
            self.server_pipes[i] = p_send
            self.actor_pipes[i] = p_recv
            
        self.server = BatchedInferenceServer(
            model=self.model,
            request_queue=self.request_queue,
            response_pipes=self.server_pipes,
            batch_size=64,
            timeout_ms=2.0
        )
        
        self.actor_processes = []
        
    def start_server(self):
        print("[WorkerPool] Starting GPU Batch Inference Server...")
        self.server.start()
        
    def spawn_actors(self, target_func, args_list):
        """
        Spawn actor processes. target_func should accept (actor_id, request_queue, response_pipe, *args)
        """
        if len(args_list) != self.num_workers:
            raise ValueError(f"args_list length ({len(args_list)}) must match num_workers ({self.num_workers})")
            
        print(f"[WorkerPool] Spawning {self.num_workers} actor processes...")
        for i in range(self.num_workers):
            p_args = (i, self.request_queue, self.actor_pipes[i]) + args_list[i]
            p = self.ctx.Process(target=target_func, args=p_args)
            p.start()
            self.actor_processes.append(p)
            
    def join_actors(self):
        for p in self.actor_processes:
            p.join()
            
    def stop(self):
        print("[WorkerPool] Stopping processes...")
        # Give processes a moment to cleanly exit via poison pill and release shared CUDA memory
        for p in self.actor_processes:
            if p.is_alive():
                p.join(timeout=2.0)
                
        for p in self.actor_processes:
            if p.is_alive():
                print(f"[WorkerPool] Force terminating hanging worker {p.pid}...")
                p.terminate()
                
        self.server.stop()
        self.server.join()
        
        # Clean up queues/pipes
        self.request_queue.close()
        self.request_queue.join_thread()
