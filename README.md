# Pokémon TCG AI Battle Challenge

Welcome to our project repository for the Pokémon TCG AI Challenge.

## Current Project State

We are currently preparing for Phase 7 (Production Training). Phases 1 through 6 are completely finished. Our agent utilizes a PyTorch Deep Learning ensemble combined with a lightweight Bayesian Tracker to mathematically infer the opponent's strategy and hot-swap to the appropriate counter-model mid-game. We have successfully implemented a PyTorch Reinforcement Learning offline training pipeline (A2C) and an ONNX conversion pipeline for Kaggle deployment.

Below is an ASCII diagram representing the current execution pipeline and game state flow:

```text
+-------------------------------------------------------------+
|                          main.py                            |
|       (Orchestrates Multi-Processing Training Loop)         |
+----------------------------+--------------------------------+
                             |
         +-------------------v-------------------+
         | multiprocessing.Pool.imap_unordered() |
         |   (Spawns parallel environments)      |
         +---+---------------+---------------+---+
             |               |               |
       +-----v-----+   +-----v-----+   +-----v-----+
       | Worker 1  |   | Worker 2  |   | Worker N  |
       | env.run() |   | env.run() |   | env.run() |
       +-----+-----+   +-----+-----+   +-----+-----+
             |               |               |
         +---v---------------+---------------v---+
         |      master_buffer.add_trajectory()   |
         |  (Yields completed games real-time)   |
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
|    - Evaluates State via Model                  |
|    - Returns Policy (Action Probabilities)      |
|                                                 |
| 3. Agent executes highly probable legal move    |
+-------------------------------------------------+
```

### Directory Structure
- `src/core/`: Contains the core agent logic, state parsers, and machine learning models.
  - `agent.py`: The entrypoint for our custom AI logic.
  - `parser.py`: Transforms engine JSON arrays into structured state classes.
  - `bayesian.py` & `bayesian_matrix.py`: Bayesian logic for inferring opponent deck archetypes.
  - `models/`: Deep Learning infrastructure.
    - `base.py`: The core `BaseNetwork` architecture (Dual-headed Policy/Value).
    - `ensemble.py`: The `EnsembleManager` that handles loading and hot-swapping PyTorch `.pt` or `.onnx` models.
    - `replay_buffer.py`: Temporarily stores trajectory transitions during RL matches.
    - `trainer.py`: Triggers backpropagation and optimizes model parameters.
- `tests/`: Contains test suites and scripts for validating modules.
- `assets/`: Assorted assets including deck CSVs.
  - `prob/`: Contains the generated `likelihood_matrix.npy`.
  - `models/`: Checkpoints and `.onnx` models.
- `scripts/`: Development scripts.
  - `build_likelihood_matrix.py`: Retrains the Bayesian probability matrix.
  - `export_onnx.py`: Compiles PyTorch weights into optimized CPU `.onnx` formats.
  - `bundle_submission.py`: Compresses necessary code and models into a `<197.7 MiB` `.tar.gz`.
- `viz/`: Visualization and Dev Tools.
  - `dashboard.py`: Live Streamlit tracking of training metrics.
- `submissions/`: Output directory where `.tar.gz` packages are saved for upload to Kaggle.
- `main.py`: Our top-level script for initiating local simulations and tests.

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
uv run streamlit run viz/dashboard.py
```
