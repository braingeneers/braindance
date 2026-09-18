# mspacman_reinforcement_example.py

**Path:** `braindance/examples/mspacman_reinforcement_example.py`
**Module:** `braindance.examples.mspacman_reinforcement_example`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Runs neural reservoir control of Ms. Pac-Man after baseline recording and RT-Sort. Exposes learned/fixed sensory encoding and spatial/PPO action decoding with optional live visualization.

## Connections
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `RecordPhaseV3` from `braindance.core.phases_v3.phases3` — imports (static evidence).
- **Uses:** `MsPacManPhaseWithViz` from `braindance.core.phases_v3.phases3_pacman` — imports (static evidence).
- **Uses:** `RTSortPhaseV3` from `braindance.core.phases_v3.phases_analysis_3` — imports (static evidence).
- **Shared data:** RecordPhaseV3(30 seconds) -> RTSortPhaseV3(min_spikes=100) -> MsPacManPhaseWithViz.

## Dependencies
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3.RecordPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3_pacman.MsPacManPhaseWithViz` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases_analysis_3.RTSortPhaseV3` — intra-repo import; source import evidence.

## Classes
None

## Functions
### `run_mspacman_reinforcement_experiment(params)`
> Run a Ms. Pac-Man experiment with neural reservoir control.
> **Called by:** braindance/examples/mspacman_reinforcement_example.py:130 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/mspacman_reinforcement_example.py:13`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| CLI --config,--verbose,--n-episodes,--learning-rate,--gamma,--render-mode |
| --encode-mode fixed_sigmoid\|policy_continuous; --decode-mode ppo\|spatial |
| --n-features,--disable-viz |

## Data Shapes
- Experiment data includes episode_rewards,total_episodes,final_policy_state

## Notes
- Requires configured acquisition/replay environment and RT-Sort resources; reported completion does not check exp.run return.
