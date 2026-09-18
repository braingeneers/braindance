# game.py

**Path:** `braindance/examples/paper_cartpole/game.py`
**Module:** `braindance.examples.paper_cartpole.game`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
One session of the GUI Cartpole-Force Train protocol; no workstation scheduler.

## Connections
- **Used by:** `braindance.examples.1_cartpole` — import consumer hint; not a proven runtime call.
- **Uses:** `Mapping` from `braindance.analysis.mapping` — imports (static evidence).
- **Uses:** `get_output_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `maxwell_params` from `braindance.core.params` — imports (static evidence).
- **Uses:** `CartPolePhase` from `braindance.core.phases2` — imports (static evidence).
- **Uses:** `PhaseManager` from `braindance.core.phases2` — imports (static evidence).
- **Uses:** `TetanusTrainer` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `generate_permutations` from `braindance.core.trainer` — imports (static evidence).

## Dependencies
- `braindance.analysis.mapping.Mapping` — intra-repo import; source import evidence.
- `braindance.config.get_output_dir` — intra-repo import; source import evidence.
- `braindance.core.maxwell_env.MaxwellEnv` — intra-repo import; source import evidence.
- `braindance.core.params.maxwell_params` — intra-repo import; source import evidence.
- `braindance.core.phases2.CartPolePhase` — intra-repo import; source import evidence.
- `braindance.core.phases2.PhaseManager` — intra-repo import; source import evidence.
- `braindance.core.trainer.TetanusTrainer` — intra-repo import; source import evidence.
- `braindance.core.trainer.generate_permutations` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `main(config, run_index=0, n_episodes=200, max_time_sec=900)`
> Function `main` performs the module operation described by its surrounding code.
> **Called by:** braindance/examples/1_cartpole.py:31 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/examples/paper_cartpole/game.py:4`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Uses BrainDance configured output directory for generated results. |

## Data Shapes
None

## Notes
None
