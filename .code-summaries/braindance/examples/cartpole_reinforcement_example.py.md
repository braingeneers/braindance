# cartpole_reinforcement_example.py

**Path:** `braindance/examples/cartpole_reinforcement_example.py`
**Module:** `braindance.examples.cartpole_reinforcement_example`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Builds a baseline-recording and RT-Sort pipeline before neural REINFORCE control of continuous CartPole. Automatically allocates sensory, motor and training groups from detected sequences.

## Connections
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `RecordPhaseV3` from `braindance.core.phases_v3.phases3` — imports (static evidence).
- **Uses:** `CartPolePhasWithViz` from `braindance.core.phases_v3.phases3_loop` — imports (static evidence).
- **Uses:** `RTSortPhaseV3` from `braindance.core.phases_v3.phases_analysis_3` — imports (static evidence).
- **Shared data:** RecordPhaseV3 -> RTSortPhaseV3 -> CartPolePhasWithViz.

## Dependencies
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3.RecordPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3_loop.CartPolePhasWithViz` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases_analysis_3.RTSortPhaseV3` — intra-repo import; source import evidence.

## Classes
None

## Functions
### `run_cartpole_reinforcement_experiment(params)`
> Run a CartPole experiment with REINFORCE learning.
> **Called by:** braindance/examples/cartpole_reinforcement_example.py:105 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/cartpole_reinforcement_example.py:18`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| CLI --config, --verbose |
| 30 s baseline; min_spikes=100; read_period_ms=50; learning_rate=.01,gamma=.99,policy_hidden_size=64 |

## Data Shapes
- Motor firing-rate vector drives continuous action; accesses exp.data.total_episodes as results mapping

## Notes
- Requires RT-Sort resources and sufficient detected sequences; main function does not branch on exp.run failure.
