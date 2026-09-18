# sample_experiment.py

**Path:** `braindance/experiments/sample_experiment.py`
**Module:** `braindance.experiments.sample_experiment`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Demonstrates raw Maxwell acquisition from a fixed dummy recording. Alternates stimulation between two electrode indexes once per second.

## Connections
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `maxwell_params` from `braindance.core.params` — imports (static evidence).
- **Shared data:** Uses MaxwellEnv.dt and stim_dt for acquisition and pulse timing.

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
| dummy is a site-specific raw.h5 path; max_time_sec=60; multiprocess=False; render=False |

## Data Shapes
- MaxwellEnv.step returns observation and done

## Notes
- Mutates shared maxwell_params; runnable body is main-guarded.
