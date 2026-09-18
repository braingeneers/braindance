# phases_env.py

**Path:** `braindance/core/phases_env.py`
**Module:** `braindance.core.phases_env`
**Feature Area:** `Games and Reinforcement`
**Entry point:** no — library or imported component

## Overview
Implements a legacy FoodLand feedback phase alternating neural readout, game steps, training stimulation, and waits. Sensory, motor, and training mappings must be supplied through setter callbacks.

## Connections
- **Uses:** `ArtifactRemoval` from `braindance.core.artifact_removal` — imports (static evidence).
- **Uses:** `base_env` from `braindance.core` — imports (static evidence).
- **Uses:** `Phase` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `FoodLandEnv` from `braindance.games.food_land` — imports (static evidence).
- **Shared data:** FoodLandEnv game state feeds sensory rates; ArtifactRemoval supplies motor spikes; optional trainer receives phase reference.

## Dependencies
- `braindance.core.artifact_removal.ArtifactRemoval` — intra-repo import; source import evidence.
- `braindance.core.base_env` — intra-repo import; source import evidence.
- `braindance.core.phases.Phase` — intra-repo import; source import evidence.
- `braindance.games.food_land.FoodLandEnv` — intra-repo import; source import evidence.
- `gymnasium` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### FoodLandPhase(Phase)
> Phase for the FoodLand game
**Source:** `braindance/core/phases_env.py:25`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, env: base_env.BaseEnv, sensory_neurons: list, motor_neurons: list, training_neurons: list, amp_mv: int=400, phase_width: int=100, read_period_ms: int=100, train_period_ms: int=100, wait_period_ms: int=200, n_episodes: int=10, trainer=None, artifact_removal=True, spike_thresh=[-3.1, -20], verbose=False, max_time=np.inf, minibatch_size=3)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `artifact_removal` | inferred at runtime | `artifact_removal` |
| `artifact_remover` | inferred at runtime | `[]` |
| `current_food` | inferred at runtime | `0` |
| `env` | inferred at runtime | `env` |
| `episode` | inferred at runtime | `0` |
| `episode_reward` | inferred at runtime | `None` |
| `episode_reward_change` | inferred at runtime | `None` |
| `food_got` | inferred at runtime | `[]` |
| `game_env` | inferred at runtime | `FoodLandEnv(render_mode='human', reward_type='dense', max_steps=500)` |
| `game_obs` | inferred at runtime | `None` |
| `get_motor_signal` | inferred at runtime | `lambda: func(self)` |
| `get_training_signal` | inferred at runtime | `lambda: func(self)` |
| `last_action_ind` | inferred at runtime | `None` |
| `last_action_inds` | inferred at runtime | `deque(maxlen=self.minibatch_size)` |
| `max_time` | inferred at runtime | `max_time` |
| `minibatch_size` | inferred at runtime | `minibatch_size` |
| `motor_neurons` | inferred at runtime | `np.array(motor_neurons)` |
| `motor_spike_count` | inferred at runtime | `np.zeros(len(self.motor_neurons))` |
| `motor_spike_rate` | inferred at runtime | `np.zeros(len(self.motor_neurons))` |
| `n_episodes` | inferred at runtime | `n_episodes` |
| `predicted_time` | inferred at runtime | `(read_period_ms + train_period_ms) * 20 * n_episodes / 1000` |
| `read_period_ms` | inferred at runtime | `read_period_ms` |
| `rewards` | inferred at runtime | `[]` |
| `sensory_neurons` | inferred at runtime | `np.array(sensory_neurons)` |
| `sensory_stim_Hz` | inferred at runtime | `np.zeros(len(self.sensory_neurons))` |
| `set_sensory_signal` | inferred at runtime | `lambda game_env_obs: func(self, game_env_obs)` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `state` | inferred at runtime | `'read'` |
| `train_period_ms` | inferred at runtime | `train_period_ms` |
| `trainer` | inferred at runtime | `trainer` |
| `training_neurons` | inferred at runtime | `np.array(training_neurons)` |
| `verbose` | inferred at runtime | `verbose` |
| `wait_period_ms` | inferred at runtime | `wait_period_ms` |
**Methods:**
#### `set_env(self, env)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_env.py:131`
#### `set_sensory_function(self, func)`
> Set the function to use to map the game observation to the sensory neurons
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_env.py:135`
#### `set_sensory_signal(self, game_env_obs)`
> Maps the game observation to the sensory neurons. This observation is of the form: ndarray with shape (4,): - Cart Position, Cart Velocity, Pole Angle, Pole Velocity At Tip
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_env.py:141`
#### `set_motor_function(self, func)`
> Set the function to use to map the motor neurons to the action
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_env.py:164`
#### `get_motor_signal(self, moving_avg=0.2)`
> Readout from the read/motor neurons set The action returned should be an integer {0,1} corresponding to the action
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_env.py:171`
#### `set_training_function(self, func)`
> Set the function to use to map the training neurons to the action
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_env.py:206`
#### `get_training_signal(self, train_Hz=30, stim_count=3, delay_ms=5, use_change=True)`
> The stimulation on the training neurons
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_env.py:213`
#### `run(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_env.py:241`
#### `time_elapsed(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_env.py:444`
#### `info(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_env.py:447`
#### `predicted_time(self)`
> Returns the predicted time for the phase to run in seconds
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_env.py:457`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| sensory_neurons/training_neurons are stimulation indices; motor_neurons are channels |
| read_period_ms, train_period_ms, wait_period_ms, n_episodes, max_time, minibatch_size |

## Data Shapes
- Game log includes food/spike signals, agent position/direction, contacts, rewards, and movement/turning speeds.

## Notes
- Default mapping methods raise NotImplementedError; optional streaming artifact path explicitly handles at most three motor channels.
