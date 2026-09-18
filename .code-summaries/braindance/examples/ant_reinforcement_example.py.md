# ant_reinforcement_example.py

**Path:** `braindance/examples/ant_reinforcement_example.py`
**Module:** `braindance.examples.ant_reinforcement_example`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Runs Ant neural reservoir control after baseline recording and RT-Sort. Configures projected sensory features, continuous torque decoding and optional training stimulation.

## Connections
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `RecordPhaseV3` from `braindance.core.phases_v3.phases3` — imports (static evidence).
- **Uses:** `AntPhaseV3` from `braindance.core.phases_v3.phases3_ant` — imports (static evidence).
- **Uses:** `RTSortPhaseV3` from `braindance.core.phases_v3.phases_analysis_3` — imports (static evidence).
- **Shared data:** RecordPhaseV3(30 seconds) -> RTSortPhaseV3(min_spikes=100) -> AntPhaseV3.

## Dependencies
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3.RecordPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3_ant.AntPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases_analysis_3.RTSortPhaseV3` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `run_ant_reinforcement_experiment(params, resume=False, name='ant_reinforcement')`
> Run an Ant experiment with neural reservoir control.
> **Called by:** braindance/examples/ant_reinforcement_example.py:189 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/ant_reinforcement_example.py:19`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| CLI --config,--verbose,--n-episodes,--learning-rate,--gamma,--render-mode |
| --name,--resume,--read-period-ms,--n-features,--max-episode-steps,--trainer-type |
| --encode-mode fixed_sigmoid\|policy_continuous; --decode-mode ppo\|direct; --reward-mode forward_x\|planar_speed |

## Data Shapes
- Experiment data includes episode_rewards,total_episodes,final_policy_state

## Notes
- Requires configured acquisition/replay environment and RT-Sort resources; reported completion does not check exp.run return.
