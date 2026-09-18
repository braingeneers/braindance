# walker2d_reinforcement_example.py

**Path:** `braindance/examples/walker2d_reinforcement_example.py`
**Module:** `braindance.examples.walker2d_reinforcement_example`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Runs Walker2d neural control with fixed-tanh or learned sensory encoding and continuous torque decoding. Can reuse saved RT-Sort data from another experiment to skip baseline and sorting phases.

## Connections
- **Uses:** `get_data_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `RecordPhaseV3` from `braindance.core.phases_v3.phases3` — imports (static evidence).
- **Uses:** `Walker2dPhaseV3` from `braindance.core.phases_v3.phases3_walker2d` — imports (static evidence).
- **Uses:** `RTSortPhaseV3` from `braindance.core.phases_v3.phases_analysis_3` — imports (static evidence).
- **Shared data:** RecordPhaseV3(30 seconds) -> RTSortPhaseV3(min_spikes=100) -> Walker2dPhaseV3.
- **Shared data:** _load_source_experiment_data reads configured data root, requires rt_sort_object and copies non-None context values plus mapping.

## Dependencies
- `braindance.config.get_data_dir` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3.RecordPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3_walker2d.Walker2dPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases_analysis_3.RTSortPhaseV3` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `_load_source_experiment_data(exp: Experiment, source_name: str)`
> Copy all DataContext keys + mapping from a previous experiment into *exp*.
> **Called by:** braindance/examples/walker2d_reinforcement_example.py:125 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/walker2d_reinforcement_example.py:27`
### `run_walker2d_reinforcement_experiment(params, resume=False, name='walker2d_reinforcement', source_experiment=None)`
> Run a Walker2d experiment with neural reservoir control.
> **Called by:** braindance/examples/walker2d_reinforcement_example.py:268 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/walker2d_reinforcement_example.py:77`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| CLI --config,--verbose,--n-episodes,--learning-rate,--gamma,--render-mode |
| --name,--source-experiment,--resume,--read-period-ms,--n-features,--max-episode-steps,--trainer-type,--max-stim-hz |
| --encode-mode fixed_tanh\|policy_continuous; --decode-mode ppo\|direct |

## Data Shapes
- Experiment data includes episode_rewards,total_episodes,final_policy_state

## Notes
- Requires configured acquisition/replay environment and RT-Sort resources; reported completion does not check exp.run return.
