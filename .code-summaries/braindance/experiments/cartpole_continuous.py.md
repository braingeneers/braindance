# cartpole_continuous.py

**Path:** `braindance/experiments/cartpole_continuous.py`
**Module:** `braindance.experiments.cartpole_continuous`
**Feature Area:** `Games and Reinforcement`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Schedules continuous CartPole phases with artifact removal and adaptive tetanus training. Each run can power a smart plug around Maxwell acquisition.

## Connections
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `maxwell_params` from `braindance.core.params` — imports (static evidence).
- **Uses:** `CartPolePhase` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `NeuralSweepPhase` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `PhaseManager` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `TetanusTrainer` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `generate_tetanus_pattern` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `SmartPlug` from `braindance.core.utils` — imports (static evidence).
- **Shared data:** Experimenter.run_experiment creates MaxwellEnv and CartPolePhase, then calls PhaseManager.run.
- **Shared data:** schedule.run_pending dispatches jobs.

## Dependencies
- `braindance.core.maxwell_env.MaxwellEnv` — intra-repo import; source import evidence.
- `braindance.core.params.maxwell_params` — intra-repo import; source import evidence.
- `braindance.core.phases.CartPolePhase` — intra-repo import; source import evidence.
- `braindance.core.phases.NeuralSweepPhase` — intra-repo import; source import evidence.
- `braindance.core.phases.PhaseManager` — intra-repo import; source import evidence.
- `braindance.core.trainer.TetanusTrainer` — intra-repo import; source import evidence.
- `braindance.core.trainer.generate_tetanus_pattern` — intra-repo import; source import evidence.
- `braindance.core.utils.SmartPlug` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `schedule` — external or unresolved local import; source import evidence.

## Classes
### Experimenter()
> unclear — see source
**Source:** `braindance/experiments/cartpole_continuous.py:51`
**Kind:** class. **Instantiated by:** braindance/experiments/cartpole_continuous.py:122 (named-call hint)
**Constructor:** `__init__(self, params, neuron_list, sensory_neurons, motor_neurons, training_neurons, plug, n_episodes, trainer=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `motor_neurons` | inferred at runtime | `motor_neurons` |
| `neuron_list` | inferred at runtime | `neuron_list` |
| `params` | inferred at runtime | `params` |
| `plug` | inferred at runtime | `plug` |
| `sensory_neurons` | inferred at runtime | `sensory_neurons` |
| `trainer` | inferred at runtime | `trainer` |
| `training_neurons` | inferred at runtime | `training_neurons` |
**Methods:**
#### `now(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/experiments/cartpole_continuous.py:63`
#### `run_experiment(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/experiments/cartpole_continuous.py:68`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| read_period_ms=150, train_period_ms=200, wait_period_ms=1000, assistive=0 |
| 300 episodes; daily every two hours; config_st.cfg; save_dir=./fast |

## Data Shapes
- Sensory/training arrays index stim_electrodes; motor neurons are recording channel IDs

## Notes
- Module-level shared parameter mutation and SmartPlug construction.
- run_experiment reads global n_episodes and tt instead of stored trainer; main disables plug.
