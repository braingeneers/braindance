# foodland_reinforcement_example.py

**Path:** `braindance/examples/foodland_reinforcement_example.py`
**Module:** `braindance.examples.foodland_reinforcement_example`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Runs FoodLand neural control after baseline recording and RT-Sort. Configures food/hazard rewards and PCA, PPO or direct decoding with a separate PCA calibration interval.

## Connections
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `RecordPhaseV3` from `braindance.core.phases_v3.phases3` — imports (static evidence).
- **Uses:** `FoodLandPhaseV3` from `braindance.core.phases_v3.phases3_foodland` — imports (static evidence).
- **Uses:** `RTSortPhaseV3` from `braindance.core.phases_v3.phases_analysis_3` — imports (static evidence).
- **Shared data:** RecordPhaseV3(30 seconds) -> RTSortPhaseV3(min_spikes=100) -> FoodLandPhaseV3.

## Dependencies
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3.RecordPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3_foodland.FoodLandPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases_analysis_3.RTSortPhaseV3` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `run_foodland_reinforcement_experiment(params, resume=False, name='foodland_reinforcement')`
> Run a FoodLand experiment with neural reservoir control.
> **Called by:** braindance/examples/foodland_reinforcement_example.py:228 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/foodland_reinforcement_example.py:13`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| CLI --config,--verbose,--n-episodes,--learning-rate,--gamma,--render-mode |
| --name,--resume,--read-period-ms,--n-features,--max-episode-steps,--trainer-type |
| --encode-mode fixed_sigmoid\|policy_continuous; --decode-mode pca\|ppo\|direct |
| --food-count,--spike-count,--reward-type,--hunger,--pca-calibration-s |

## Data Shapes
- Experiment data includes episode_rewards,total_episodes,final_policy_state

## Notes
- Requires configured acquisition/replay environment and RT-Sort resources; reported completion does not check exp.run return.
