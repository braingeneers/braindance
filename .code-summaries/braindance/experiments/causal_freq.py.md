# causal_freq.py

**Path:** `braindance/experiments/causal_freq.py`
**Module:** `braindance.experiments.causal_freq`
**Feature Area:** `Stimulation`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Runs timed silent, causal and tetanus phases with a manual acquisition loop. Causal phases stimulate each electrode 30 times while tetanus phases send ordered four-electrode patterns.

## Connections
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `maxwell_params` from `braindance.core.params` — imports (static evidence).
- **Shared data:** Uses MaxwellEnv.time_elapsed and stim_dt to choose env.step actions; calls env._cleanup after loop.

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
| tetanus_Hz=50, causal_Hz=1, delay_ms=5; max_time_sec=7200 |

## Data Shapes
- Simple action ([electrode_index],150,100); tetanus list of stim/delay tuples

## Notes
- Executes hardware experiment at import and mutates shared core.params.maxwell_params; edit site-specific config/electrode/save paths before use.
