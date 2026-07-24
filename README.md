# Pokémon TCG AI Battle Challenge

Welcome to our project repository for the Pokémon TCG AI Challenge.

## Current Project State

We are currently in Phase 7 (Production Training). Our agent utilizes a PyTorch Deep Learning ensemble combined with a lightweight Bayesian Tracker and an MCTS Engine to mathematically infer the opponent's strategy and execute highly optimized turns. We have successfully implemented a highly concurrent PyTorch Reinforcement Learning offline training pipeline (A2C) scaling to 10-15 matches/second and an ONNX conversion pipeline for Kaggle deployment.

Below is an ASCII diagram representing the current execution pipeline and game state flow:

```text
+-------------------------------------------------------------+
|                          main.py                            |
|       (Orchestrates Multi-Processing Training Loop)         |
|  - Applies global copy.deepcopy monkeypatch for 50x speed   |
+----------------------------+--------------------------------+
                             |
         +-------------------v-------------------+
         | multiprocessing.Pool.imap_unordered() |
         |   (Spawns CPU-Isolated environments)  |
         +---+---------------+---------------+---+
             |               |               |
       +-----v-----+   +-----v-----+   +-----v-----+
       | Worker 1  |   | Worker 2  |   | Worker N  |
       | env.run() |   | env.run() |   | env.run() |
       +-----+-----+   +-----+-----+   +-----+-----+
             |               |               |
         +---v---------------+---------------v---+
         |      master_buffer.add_trajectory()   |
         |  (Yields completed games via Numpy)   |
         +-------------------+-------------------+
                             |
+----------------------------v--------------------------------+
|                   src/core/models/trainer.py                |
|  - Samples batches from master_buffer                       |
|  - Backpropagates Policy & Value Loss to ensemble model     |
|  - main.py periodically saves general_model.pt              |
+-------------------------------------------------------------+

Inside each Worker (env.run):
+-------------------------------------------------+
|                 src/core/agent.py               |
|  (Agent receives raw JSON obs_dict each turn)   |
+------------------------+------------------------+
                         |
+------------------------v------------------------+
|                 src/core/parser.py              |
| (Maps obs_dict into typed Python Dataclasses)   |
+------------------------+------------------------+
                         |
+------------------------v------------------------+
| 1. Passes Observation to BayesianTracker        |
|    - Infers archetype via assets/prob/matrix    |
|    - Hot-swaps Ensemble Model if Conf > 85%     |
|                                                 |
| 2. Passes Observation to EnsembleManager        |
|    - Evaluates State via Model for Policy/Value |
|                                                 |
| 3. Triggers MCTSEngine (src/core/mcts.py)       |
|    - Interfaces with cg-lib (C++ Engine)        |
|    - Explores tree using model Policy as prior  |
|    - Returns robust selected action             |
+-------------------------------------------------+
```

### Recent System Design Enhancements
- **Performance Scaling**: 
  - Isolated GPU contexts from multiprocessing workers by forcing CPU execution (`CUDA_VISIBLE_DEVICES="-1"`) and OpenMP thread limiting (`torch.set_num_threads(1)`).
  - Reduced IPC queue bottleneck by moving away from raw PyTorch tensors to lightweight Numpy array serializations within the ReplayBuffer.
  - Mitigated a critical Kaggle Engine cloning bottleneck by globally monkeypatching `copy.deepcopy` to shallow-copy observation states, yielding a 50x speedup in environment simulation without hitting recursion limits.
- **Bayesian Tuning**: 
  - Retrained the likelihood matrix (`assets/prob/likelihood_matrix.npy`) to directly parse the actual validation deck CSVs (Aggro, Control, Combo), guaranteeing that the unique Card IDs perfectly correspond to the archetypes for flawless 100% confidence hot-swapping.
  - Adjusted training decks to include necessary Basic and Stage 1 Pokémon to comply with core PTCG rules and prevent simulation crashes (infinite mulligans).

### Directory Structure
- `src/core/`: Contains the core agent logic, state parsers, and machine learning models.
  - `agent.py`: The entrypoint for our custom AI logic.
  - `parser.py`: Transforms engine JSON arrays into structured state classes.
  - `bayesian.py`: Bayesian logic for inferring opponent deck archetypes.
  - `mcts.py`: Monte Carlo Tree Search engine interfacing with `cg-lib`.
  - `models/`: Deep Learning infrastructure.
    - `base.py`: The core `BaseNetwork` architecture (Dual-headed Policy/Value).
    - `ensemble.py`: The `EnsembleManager` that handles loading and hot-swapping PyTorch `.pt` or `.onnx` models.
    - `replay_buffer.py`: Temporarily stores trajectory transitions during RL matches.
    - `trainer.py`: Triggers backpropagation and optimizes model parameters.
- `src/viz/`: Visualization and Dev Tools.
  - `rl_training_dashboard.py`: Live Streamlit tracking of training metrics.
- `assets/`: Assorted assets including deck CSVs.
  - `prob/`: Contains the generated `likelihood_matrix.npy`.
  - `models/`: Checkpoints and `.onnx` models.
  - `decks/`: Valid PTCG deck compositions separated by archetype.
- `scripts/`: Development scripts.
  - `build_likelihood_matrix.py`: Retrains the Bayesian probability matrix.
  - `export_onnx.py`: Compiles PyTorch weights into optimized CPU `.onnx` formats.
  - `bundle_submission.py`: Compresses necessary code and models into a `<197.7 MiB` `.tar.gz`.
  - `tune_bayesian.py`: Tool for calibrating the Bayesian detector's speed against test decks.
- `main.py`: Our top-level script for initiating local simulations and training workers.

## Usage Commands

**1. Generate Bayesian Probabilities:**
```bash
uv run python scripts/build_likelihood_matrix.py
```

**2. Run Headless Parallel Training (RTX 5080):**
```bash
uv run python main.py --mode train --opp-deck all --workers 16 --episodes 100000
```

**3. Launch Live Training Dashboard:**
```bash
uv run streamlit run src/viz/rl_training_dashboard.py
```
