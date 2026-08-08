# Pokémon TCG AI Battle Challenge

Welcome to our project repository for the Pokémon TCG AI Challenge.

## Current Project State

We are currently in Phase 7 (Production Training). Our agent utilizes a PyTorch Deep Learning ensemble combined with a lightweight Bayesian Tracker and an MCTS Engine to mathematically infer the opponent's strategy and execute highly optimized turns. We have successfully implemented a highly concurrent, asynchronously batched PyTorch Reinforcement Learning pipeline that perfectly saturates GPU hardware using Inter-Process Communication (IPC).

Below is an ASCII diagram representing the state-of-the-art execution pipeline:

```text
+-------------------------------------------------------------+
|                          main.py                            |
|       (Orchestrates Multi-Processing Training Loop)         |
+----------------------------+--------------------------------+
                             |
    +------------------------v------------------------+
    |             WorkerPool / Train.py               |
    |  - Spawns N isolated Actor Processes            |
    |  - Spawns 1 dedicated GPU Batch Server          |
    |  - Collects Trajectories for RL Trainer         |
    +--------+-------------------------------+--------+
             |                               |
 +-----------v-----------+       +-----------v-----------+
 | Actor Process (1..N)  |       | GPU Batch Server      |
 |  (System RAM / CPU)   |       | (CUDA VRAM Context)   |
 |                       |       |                       |
 | - Evaluates game env  |       | - Listens on Queue    |
 | - Uses IPCClient      | <===> | - Dynamic Batching    |
 | - Requests GPU eval   | (IPC) | - torch.compile()     |
 | - Sends Numpy Arrays  |       | - Static Tensor shape |
 +-----------+-----------+       +-----------+-----------+
             |                               |
             +---------------+---------------+
                             |
                 +-----------v-----------+
                 |    Trainer (CPU/GPU)  |
                 | - Optimizer.step()    |
                 | - Backpropagation     |
                 +-----------------------+
```

### Recent System Design Enhancements
- **Asynchronous IPC Batching**:
  - Replaced the initial synchronous multiprocessing pool with a highly optimized `WorkerPool` architecture. CPU-bound game simulations are entirely decoupled from GPU inferences.
  - Deployed a highly efficient `BatchedInferenceServer` running in a dedicated background process. It dynamically aggregates raw neural network requests from up to 48 independent workers and evaluates them instantly in a single vectorized forward pass.
- **Extreme GPU Optimizations**: 
  - Eliminated CUDA Out of Memory (OOM) crashes during worker `spawn` unpickling by enforcing a strict `KAGGLE_AGENT_CPU_ONLY` flag for the actors. Workers consume 0 bytes of VRAM.
  - Activated NVIDIA Ada Lovelace / Blackwell hardware tweaks including **TensorFloat-32 (TF32)** and `cuDNN` benchmarking.
  - Implemented `torch.compile(mode="reduce-overhead")` with a permanently static input matrix `(self.batch_size, 6160)`. This allows the CUDA Graph to fuse kernels perfectly without ever needing to re-profile due to fluctuating batch sizes.
- **Deep Architecture**:
  - Replaced the initial 3-layer MLP with a massive **5.6M parameter Residual-MLP** (4 Residual Blocks).
  - Designed a high-dimensional **StateEncoder (6,160 dim)** using Bag-of-Words vectors to precisely track Card IDs across Hand, Bench, Active, and Discard zones.

### Directory Structure
- `src/core/`: Contains the core agent logic, state parsers, and machine learning models.
  - `agent.py`: The entrypoint for our custom AI logic.
  - `mcts.py`: Monte Carlo Tree Search engine interfacing with `cg-lib`.
  - `engine/`: Multiprocessing backend.
    - `batch_server.py`: The `BatchedInferenceServer` that compiles and executes GPU batches.
    - `worker_pool.py`: Orchestrates IPC pipes, Queues, and `spawn` contexts.
    - `train.py`: Handles high-level Reinforcement Learning logic and trajectory collection.
  - `models/`: Deep Learning infrastructure.
    - `base.py`: The core `BaseNetwork` architecture (Dual-headed Policy/Value).
    - `ensemble.py`: The `EnsembleManager` that handles loading PyTorch `.pt` or `.onnx` models.
    - `replay_buffer.py`: Temporarily stores trajectory transitions (using Numpy arrays to prevent FD leaks).
    - `trainer.py`: Triggers backpropagation and optimizes model parameters.
- `assets/`: Assorted assets including deck CSVs.
  - `models/`: Checkpoints and `.onnx` models.
  - `decks/`: Valid PTCG deck compositions separated by archetype.
- `main.py`: Our top-level script for initiating local simulations and training workers.

## `main.py` Parameters
The simulator orchestrates jobs via the CLI. Below are the accepted arguments:

| Argument | Description | Default | Choices |
| :--- | :--- | :--- | :--- |
| `--mode` | Execution mode. 'play' renders HTML, 'train' runs headless fast simulation, 'evaluate' runs diagnostics. | `play` | `play`, `train`, `evaluate` |
| `--p1-deck` | Path or name of Player 1's deck CSV. | `assets/decks/versatile/Team_Rockets_Box.csv` | |
| `--opp-deck` | Path to specific deck, folder of decks, or 'all' to randomly pull from all valid decks. | `all` | |
| `--episodes` | Number of matches to simulate (only used in 'train' and 'evaluate' modes). | `1` | |
| `--workers` | Number of parallel worker processes for training. Scale this up to 48 for maximum IPC batch saturation. | `1` | |
| `--model-name` | Name of the PyTorch weights to load/save (e.g. `general_model.pt`). | `general_model.pt` | |
| `--p2-type` | The type of agent Player 2 is. Use 'rules' to play/train against hardcoded agents. | `rl` | `rl`, `rules`, `mixed` |
| `--p2-agent` | The specific rules-based agent to use if `--p2-type` is rules (or 'all', 'aggro', 'control', 'prob'). | `all` | |
| `--debug` | Flag to enable cProfile worker profiling. Output is dumped to `logs/debug/worker_profiles/`. | `False` | |

## Usage Commands

**1. Run Headless Parallel Training (RTX 5080):**
```bash
uv run python main.py --mode train --opp-deck all --workers 48 --episodes 50000
```

**2. Train against Specific Rules-Based Agent (with custom MCTS Sims):**
```bash
MCTS_SIMS=200 uv run python main.py --mode train --episodes 500 --workers 48 --p2-type rules --p2-agent battle_field_audited_alakazam_v8
```

**3. Evaluate Model Diagnostics:**
```bash
uv run python main.py --mode evaluate --episodes 100 --workers 48 --model-name general_model.pt
```
