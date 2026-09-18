# causal_connectivity.py

**Path:** `braindance/experiments/causal_connectivity.py`
**Module:** `braindance.experiments.causal_connectivity`
**Feature Area:** `Stimulation`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Runs a fixed-amplitude stimulation sweep over configured electrodes to collect causal responses. PhaseManager executes 50 replicates at 2 Hz with ordering 'ran'.

## Connections
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `maxwell_params` from `braindance.core.params` — imports (static evidence).
- **Uses:** `NeuralSweepPhase` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `PhaseManager` from `braindance.core.phases` — imports (static evidence).
- **Shared data:** MaxwellEnv -> NeuralSweepPhase -> PhaseManager.run.

## Dependencies
- `braindance.core.maxwell_env.MaxwellEnv` — intra-repo import; source import evidence.
- `braindance.core.params.maxwell_params` — intra-repo import; source import evidence.
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
| amp_bounds=(150,150,1), config.cfg, save_dir=c20185 |

## Data Shapes
- neuron_list indexes stim_electrodes

## Notes
- Executes hardware experiment at import and mutates shared core.params.maxwell_params; edit site-specific config/electrode/save paths before use.
