# Pokémon TCG AI Battle Challenge

Welcome to our project repository for the Pokémon TCG AI Challenge.

## Current Project State

We are currently building out the architecture for our Agent. Phase 1 (Environment Setup) and Phase 2 (Data Parsing) are fully complete. Our local execution pipeline successfully launches the `cabt` engine, triggers our agent, parses the engine's raw JSON states into typed Python dataclasses, and runs full matches flawlessly. 

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
+------------------------v------------------------+
|                 src/core/agent.py               |
|  (Agent receives raw JSON obs_dict each turn)   |
+------------------------+------------------------+
                         |
+------------------------v------------------------+
|                src/core/parser.py               |
| (Maps obs_dict into typed Python Dataclasses)   |
|                                                 |
|    Observation                                  |
|     |-- current: CurrentState                   |
|     |     |-- players: [PlayerState, PlayerState]
|     |     |      |-- active: [Pokemon]          |
|     |     |      |-- bench: [Pokemon]           |
|     |     |      |-- hand: [Card]               |
|     |     |      +-- prize, discard, status...  |
|     |     +-- turn, stadium, etc.               |
|     |                                           |
|     +-- select: SelectState                     |
|           +-- option: [Legal Moves]             |
+------------------------+------------------------+
                         |
+------------------------v------------------------+
|                 src/core/agent.py               |
| (Evaluates SelectState and returns random move) |
+-------------------------------------------------+
```

### Directory Structure
- `src/core/`: Contains the core agent execution logic and state parsers.
  - `agent.py`: The entrypoint for our custom AI logic.
  - `parser.py`: Transforms engine JSON arrays into structured state classes.
- `tests/`: Contains test suites and scripts for validating modules (e.g. `test_parser.py`).
- `assets/`: Assorted assets including decks and engine visualizers.
- `main.py`: Our top-level script for initiating local simulations with `kaggle_environments`.
