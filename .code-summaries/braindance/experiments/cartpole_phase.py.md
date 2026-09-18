# cartpole_phase.py

**Path:** `braindance/experiments/cartpole_phase.py`
**Module:** `braindance.experiments.cartpole_phase`
**Feature Area:** `Games and Reinforcement`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Runs a single legacy CartPole phase using fixed sensory, motor and training neuron assignments. The phase manager owns the Maxwell acquisition session.

## Connections
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `maxwell_params` from `braindance.core.params` — imports (static evidence).
- **Uses:** `CartPolePhase` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `NeuralSweepPhase` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `PhaseManager` from `braindance.core.phases` — imports (static evidence).
- **Shared data:** MaxwellEnv -> CartPolePhase -> PhaseManager.

## Dependencies
- `braindance.core.maxwell_env.MaxwellEnv` — intra-repo import; source import evidence.
- `braindance.core.params.maxwell_params` — intra-repo import; source import evidence.
- `braindance.core.phases.CartPolePhase` — intra-repo import; source import evidence.
- `braindance.core.phases.NeuralSweepPhase` — intra-repo import; source import evidence.
- `braindance.core.phases.PhaseManager` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| config-phase.cfg; read_period_ms=200; max_time_sec=1800 |

## Data Shapes
- Sensory/training indexes versus motor recording channels

## Notes
- Executes hardware experiment at import and mutates shared core.params.maxwell_params; edit site-specific config/electrode/save paths before use.
