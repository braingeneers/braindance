# stim_pattern.py

**Path:** `braindance/experiments/stim_pattern.py`
**Module:** `braindance.experiments.stim_pattern`
**Feature Area:** `Stimulation`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Runs a five-minute Maxwell recording with an optional repeating stimulation pattern. Current stim_pattern=None leaves stimulation disabled.

## Connections
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `maxwell_params` from `braindance.core.params` — imports (static evidence).
- **Shared data:** Main loop tests env.stim_dt before MaxwellEnv.step.

## Dependencies
- `braindance.core.maxwell_env.MaxwellEnv` — intra-repo import; source import evidence.
- `braindance.core.params.maxwell_params` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| config_stim.cfg; save_dir=mo_experiment; stim_Hz=1 |

## Data Shapes
- Pattern would be a list of stim/delay commands

## Notes
- Mutates shared maxwell_params; finally invokes env._cleanup.
