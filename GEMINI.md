# Pokémon TCG AI Challenge: RL Strategy Context

This document maintains alignment on the reinforcement learning architecture, reward structures, and exploration paradigms for our Pokémon TCG AI agent.

## Progressive Reward Shaping
The reward function is configured to densely guide the agent toward winning conditions while heavily penalizing stalling.
- **Energy Attachments (`r_energy_attach`)**: Only granted if the energy is attached to the **Active Pokémon**. This biases the agent toward aggressive setups early in training rather than dumping energy on bench sitters.
- **Damage Dealing (`r_damage_dealt_per_10`)**: Distinct from prize-taking. Deals dense positive rewards per 10 damage dealt to opponents.
- **Missed Attack Penalty (`r_missed_attack_penalty`)**: If the agent's action options include `OptionType.ATTACK` but the agent selects `OptionType.END`, it receives a harsh penalty. This explicitly discourages hesitation when lethal/damage is available.

## Exploration Strategy (Epsilon-Greedy vs Dirichlet)
The agent supports two modes of exploration, toggled by the `--alpha` CLI parameter during training:
1. **Epsilon-Greedy (Default, `--alpha 0.0`)**: 
   - With probability `epsilon`, the agent bypasses the MCTS and executes a uniformly sampled random valid action.
2. **MCTS Dirichlet Noise (`--alpha > 0.0`, e.g., `--alpha 0.3`)**:
   - Replaces epsilon-greedy action sampling with Dirichlet noise injected directly into the MCTS root node's prior probabilities.
   - The `--epsilon` parameter dictates the exploration fraction (the weight of the noise vs the network's prior probabilities).

In both modes, the `--epsilon` parameter decays linearly from its starting value to 0 over the first 80% of the training episodes, gradually shifting the agent from exploration to strict exploitation.

## Curriculum Learning
- A `CurriculumWrapper` intercepts the environment at the beginning of training episodes.
- **Functionality**: Automatically forces the agent to attach energy to the Active Pokémon deterministically for the first `N` turns, setting up a simplified board state where the RL policy only needs to learn to declare an attack.
- **Decay**: The number of curriculum turns decays linearly from 2 to 0 over the first 50% of the training episodes, gradually exposing the agent to standard turn-one setups.

## Action Masking Validation
- Strict action masking is applied at the root of `agent.py` before passing options to MCTS.
- `OptionType.ATTACK` is masked out if the Active Pokémon does not have enough attached energy.
- `OptionType.ATTACH` is masked out if the agent has already successfully attached an energy this turn.
- This ensures the MCTS search tree does not evaluate illegal edge branches that the environment engine might surface, maximizing node search efficiency.

## Canonical Action Ordering (Intra-Turn)
To optimize the MCTS search depth, we enforce a strict, canonical action hierarchy during root Main Phase choices (when `OptionType.END` is available). This artificially masks technically legal backward-phase actions, deduplicating the horizontal state space and forcing rollouts deeper toward the combat phase.
- **Phase 1: PLAY (7)** - Play cards from hand (Basics, Items, Supporters, Stadiums)
- **Phase 2: EVOLVE (9)** - Evolve Pokémon on the board
- **Phase 3: ATTACH (8)** - Attach Energy from hand
- **Phase 4: BOARD ACTIVATION (10, 12, 15)** - Abilities, Retreats, and Skills
- **Phase 5: COMBAT (13, 14)** - Attack or Pass (End Turn)

Backward transitions (e.g., trying to execute a Phase 1 `PLAY` after a Phase 3 `ATTACH`) are dynamically masked out by the environment wrapper in `agent.py`.
