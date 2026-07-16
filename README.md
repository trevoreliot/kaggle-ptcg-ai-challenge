# Pokémon TCG AI Battle Challenge

Welcome to our project repository for the Pokémon TCG AI Challenge.

## Current Project State

We are currently in the process of training the Agent. Phase 1 through 4 are complete. Our agent utilizes a PyTorch Deep Learning ensemble combined with a lightweight Bayesian Tracker to mathematically infer the opponent's strategy and hot-swap to the appropriate counter-model mid-game.

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
|    - Evaluates State via Active PyTorch NN      |
|    - Returns Policy (Action Probabilities)      |
|                                                 |
| 3. Agent executes a highly probable legal move  |
+-------------------------------------------------+
```

### Directory Structure
- `src/core/`: Contains the core agent logic, state parsers, and machine learning models.
  - `agent.py`: The entrypoint for our custom AI logic.
  - `parser.py`: Transforms engine JSON arrays into structured state classes.
  - `bayesian.py` & `bayesian_matrix.py`: Bayesian logic for inferring opponent deck archetypes.
  - `models/`: PyTorch Deep Learning infrastructure.
    - `base.py`: The core `BaseNetwork` (Dual-headed Policy/Value).
    - `ensemble.py`: The `EnsembleManager` that handles loading and hot-swapping different meta models.
- `tests/`: Contains test suites and scripts for validating modules (e.g. `test_parser.py`, `test_bayesian.py`).
- `assets/`: Assorted assets including deck CSVs, the Bayesian `likelihood_matrix.npy`, and engine visualizers.
- `scripts/`: Assorted scripts (e.g., generating dummy decks).
- `main.py`: Our top-level script for initiating local simulations and tests.
