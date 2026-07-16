# Pokémon TCG AI Battle Challenge

Welcome to our project repository for the Pokémon TCG AI Challenge.

## Current Project State

We are currently preparing for Phase 7 (Production Training). Phases 1 through 6 are completely finished. Our agent utilizes a PyTorch Deep Learning ensemble combined with a lightweight Bayesian Tracker to mathematically infer the opponent's strategy and hot-swap to the appropriate counter-model mid-game. We have successfully implemented a PyTorch Reinforcement Learning offline training pipeline (A2C) and an ONNX conversion pipeline for Kaggle deployment.

Below is an ASCII diagram representing the current execution pipeline and game state flow:

```text
+-------------------------------------------------+
|                    main.py                      |
| (Instantiates cabt env and loads deck.csv)      |
+------------------------+------------------------+
                         |
                +--------v---------+
                | env.run([agents])|
                +--------+---------+
                         |
+------------------------+------------------------+
|                 src/core/agent.py               |
|  (Agent receives raw JSON obs_dict each turn)   |
+------------------------+------------------------+
                         |
+------------------------+------------------------+
|                src/core/parser.py               |
| (Maps obs_dict into typed Python Dataclasses)   |
+------------------------+------------------------+
                         |
+------------------------v------------------------+
|                 src/core/agent.py               |
|                                                 |
| 1. Passes Observation to BayesianTracker        |
|    - Infers opponent archetype                  |
|    - Hot-swaps Ensemble Model if Conf > 85%     |
|                                                 |
| 2. Passes Observation to EnsembleManager        |
|    - Evaluates State via ONNX (or PyTorch)      |
|    - Returns Policy (Action Probabilities)      |
|                                                 |
| 3. Trajectory pushing (Training Mode)           |
|    - Saves state/prob/value to ReplayBuffer     |
|                                                 |
| 4. Agent executes a highly probable legal move  |
+------------------------+------------------------+
                         |
+------------------------v------------------------+
|             src/core/models/trainer.py          |
|  (At Game End, pulls from ReplayBuffer)         |
|  - Calculates recursive discounted rewards      |
|  - Backpropagates Policy & Value Loss to model  |
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
- `assets/`: Assorted assets including deck CSVs, `likelihood_matrix.npy`, and `models/` directory for `.onnx` files.
- `scripts/`: Development scripts.
  - `export_onnx.py`: Compiles PyTorch weights into optimized CPU `.onnx` formats.
  - `bundle_submission.py`: Compresses necessary code and models into a `<197.7 MiB` `.tar.gz`.
  - `generate_dummy_decks.py`: Quickly spawns test CSVs.
- `submissions/`: Output directory where `.tar.gz` packages are saved for upload to Kaggle.
- `main.py`: Our top-level script for initiating local simulations and tests.
