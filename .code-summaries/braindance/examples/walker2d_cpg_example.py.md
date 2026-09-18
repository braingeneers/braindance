# walker2d_cpg_example.py

**Path:** `braindance/examples/walker2d_cpg_example.py`
**Module:** `braindance.examples.walker2d_cpg_example`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Runs Walker2d neural control with rhythmic CPG sensory stimulation and learned or direct motor decoding. Supports checkpoint resume and reusing another experiment's RT-Sort data instead of repeating recording and sorting.

## Connections
- **Uses:** `get_data_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `RecordPhaseV3` from `braindance.core.phases_v3.phases3` — imports (static evidence).
- **Uses:** `Walker2dCPGPhaseV3` from `braindance.core.phases_v3.phases3_walker2d_cpg` — imports (static evidence).
- **Uses:** `RTSortPhaseV3` from `braindance.core.phases_v3.phases_analysis_3` — imports (static evidence).
- **Shared data:** Optional RecordPhaseV3 -> RTSortPhaseV3 -> Walker2dCPGPhaseV3.
- **Shared data:** Source loader validates rt_sort_object then copies data and mapping from get_data_dir()/source_name.

## Dependencies
- `braindance.config.get_data_dir` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3.RecordPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3_walker2d_cpg.Walker2dCPGPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases_analysis_3.RTSortPhaseV3` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `_load_source_experiment_data(exp: Experiment, source_name: str)`
> Copy all DataContext keys + mapping from a previous experiment into *exp*.
> **Called by:** braindance/examples/walker2d_cpg_example.py:123 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/walker2d_cpg_example.py:30`
### `run_walker2d_cpg_experiment(params, resume=False, name='walker2d_cpg', source_experiment=None)`
> Run a Walker2d experiment with CPG-driven sensory input.
> **Called by:** braindance/examples/walker2d_cpg_example.py:315 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/walker2d_cpg_example.py:72`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| CLI --config,--verbose,--n-episodes,--learning-rate,--gamma,--render-mode |
| --name,--source-experiment,--resume,--frequencies,--neurons-per-freq,--pattern-mode fixed\|oscillating |
| --decode-mode ppo\|direct,--max-episode-steps,--read-period-ms,--trainer-type,--max-stim-hz |
| --normalize-mode warmup\|running,--warmup-duration,--no-log-rates,--ema-alpha |

## Data Shapes
- Comma-separated frequencies -> float list; final_policy_state reports CPG frequencies and pattern mode

## Notes
- Uses private Experiment._load_all_recording_data when reusing a source.
