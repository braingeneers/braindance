# cartpole_schedule.py

**Path:** `braindance/experiments/cartpole_schedule.py`
**Module:** `braindance.experiments.cartpole_schedule`
**Feature Area:** `Games and Reinforcement`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Schedules overnight CartPole experiments with fixed sensory, motor and training neuron groups. Each scheduled run powers the Coleman smart plug around acquisition.

## Connections
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `maxwell_params` from `braindance.core.params` — imports (static evidence).
- **Uses:** `CartPolePhase` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `NeuralSweepPhase` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `PhaseManager` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `SmartPlug` from `braindance.core.utils` — imports (static evidence).
- **Shared data:** Experimenter.run_experiment creates MaxwellEnv and CartPolePhase, then calls PhaseManager.run.
- **Shared data:** schedule.run_pending dispatches jobs.

## Dependencies
- `braindance.core.maxwell_env.MaxwellEnv` — intra-repo import; source import evidence.
- `braindance.core.params.maxwell_params` — intra-repo import; source import evidence.
- `braindance.core.phases.CartPolePhase` — intra-repo import; source import evidence.
- `braindance.core.phases.NeuralSweepPhase` — intra-repo import; source import evidence.
- `braindance.core.phases.PhaseManager` — intra-repo import; source import evidence.
- `braindance.core.utils.SmartPlug` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `schedule` — external or unresolved local import; source import evidence.

## Classes
### Experimenter()
> unclear — see source
**Source:** `braindance/experiments/cartpole_schedule.py:30`
**Kind:** class. **Instantiated by:** braindance/experiments/cartpole_schedule.py:91 (named-call hint)
**Constructor:** `__init__(self, params, neuron_list, sensory_neurons, motor_neurons, training_neurons, plug)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `motor_neurons` | inferred at runtime | `motor_neurons` |
| `neuron_list` | inferred at runtime | `neuron_list` |
| `params` | inferred at runtime | `params` |
| `plug` | inferred at runtime | `plug` |
| `sensory_neurons` | inferred at runtime | `sensory_neurons` |
| `training_neurons` | inferred at runtime | `training_neurons` |
**Methods:**
#### `now(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/experiments/cartpole_schedule.py:39`
#### `run_experiment(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/experiments/cartpole_schedule.py:43`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Daily 19:00 through 09:00 every two hours |
| config1.cfg; save_dir=./data; read_period_ms=200 |

## Data Shapes
- Sensory/training arrays index stim_electrodes; motor neurons are recording channel IDs

## Notes
- Module-level shared parameter mutation and SmartPlug construction.
- Scheduler does not run an immediate experiment; shutdown is not protected by finally.
