# phases3_foodland.py

**Path:** `braindance/core/phases_v3/phases3_foodland.py`
**Module:** `braindance.core.phases_v3.phases3_foodland`
**Feature Area:** `Games and Reinforcement`
**Entry point:** no — library or imported component

## Overview
Wraps FoodLand into a compact feature/action API and runs it as a closed-loop neural phase. Sensory features drive stimulation through fixed or learned encoding, while motor activity is decoded by calibrated PCA, PPO, or a direct two-neuron mapping.

## Connections
- **Used by:** `braindance.examples.foodland_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Uses:** `convert_uint16_maxwell` from `braindance.analysis.data_loader` — imports (static evidence).
- **Uses:** `ContinuousOutputPPO` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `InputPolicyActorCritic` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `PPORolloutBuffer` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `ppo_update` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `sigmoid` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `PhaseV3` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `ContextualTrainer` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `TetanusTrainer` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `generate_tetanus_pattern` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `FoodLandEnv` from `braindance.games.food_land` — imports (static evidence).
- **Uses:** `get_rt_sort_path` from `braindance` — imports (static evidence).
- **Shared data:** `FoodLandFeatureEnv` adapts `games.food_land.FoodLandEnv`; the phase uses experiment mapping/RT-Sort, `MaxwellEnv`, sklearn PCA, PPO codecs, and optional trainers.

## Dependencies
- `braindance.analysis.data_loader.convert_uint16_maxwell` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.ContinuousOutputPPO` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.InputPolicyActorCritic` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.PPORolloutBuffer` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.ppo_update` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.sigmoid` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.PhaseV3` — intra-repo import; source import evidence.
- `braindance.core.trainer.ContextualTrainer` — intra-repo import; source import evidence.
- `braindance.core.trainer.TetanusTrainer` — intra-repo import; source import evidence.
- `braindance.core.trainer.generate_tetanus_pattern` — intra-repo import; source import evidence.
- `braindance.games.food_land.FoodLandEnv` — intra-repo import; source import evidence.
- `braindance.get_rt_sort_path` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `sklearn.decomposition.PCA` — external or unresolved local import; source import evidence.
- `spikelab.SpikeData` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.
- `torch.optim` — external or unresolved local import; source import evidence.

## Classes
### FoodLandFeatureEnv()
> Adapt legacy FoodLand reset/step semantics, normalize observations into features, and expose bounded turn/speed actions.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:32`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/phases3_foodland.py:217 (named-call hint)
**Constructor:** `__init__(self, render_mode: str | None=None, n_features: int=2, feature_gain: float=6.0, reward_type: str='dense', food_count: int=3, spike_count: int=0, hunger: float=0.5, max_episode_steps: int=100)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `env` | inferred at runtime | `FoodLandEnv(render_mode=render_mode, reward_type=reward_type, food_count=food_count, spike_count=spike_count, hunger=hunger, use_…` |
| `feature_gain` | inferred at runtime | `float(feature_gain)` |
| `n_features` | inferred at runtime | `int(n_features)` |
| `obs_dim` | inferred at runtime | `int(sample_obs.shape[0])` |
**Methods:**
#### `get_feature_dim(self)`
> Return the configured encoded feature count used to size the input policy.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:67`
#### `_normalize_observation(self, observation)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:70`
#### `_extract_features(self, observation)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:87`
#### `clip_action(self, action)`
> Validate a two-value action and clip turn and speed to their distinct ranges.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:92`
#### `reset(self)`
> Reset FoodLand and return encoded features plus the raw observation.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:101`
#### `step(self, action)`
> Translate action order, step FoodLand, classify termination versus truncation, and return encoded features.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:106`
#### `close(self)`
> Close the wrapped FoodLand environment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:118`
### FoodLandPhaseV3(PhaseV3)
> Coordinate sensory encoding, online sorting, three motor decoding choices, FoodLand episodes, and optional training stimulation.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:122`
**Kind:** class. **Instantiated by:** braindance/examples/foodland_reinforcement_example.py:25 (named-call hint)
**Constructor:** `__init__(self, sensory_neurons: list | None=None, motor_neurons: list | None=None, training_neurons: list | None=None, amp_mv: int=200, phase_width: int=100, read_period_ms: int=50, train_period_ms: int=200, wait_period_ms: int=400, n_episodes: int=100, trainer=None, trainer_type: str | None=None, use_contextual_trainer: bool=False, contextual_trainer_kwargs: dict | None=None, minibatch_size: int=5, verbose: bool=False, max_time=np.inf, rt_sort_path: str | None=None, detection_model_path: str | None=None, use_numba: bool=True, learning_rate: float=0.0003, gamma: float=0.99, gae_lambda: float=0.95, clip_epsilon: float=0.2, ppo_epochs: int=4, ppo_minibatch_size: int=32, entropy_coef: float=0.01, value_coef: float=0.5, max_grad_norm: float=0.5, policy_hidden_size: float=64, n_features: int=2, encode_mode: str='fixed_sigmoid', decode_mode: str='pca', max_stim_hz: float=10.0, input_gain: float=6.0, input_center: float=0.5, direct_scale: float=3.0, direct_offset: float=1.0, render_mode: str | None='human', reward_type: str='dense', food_count: int=3, spike_count: int=0, hunger: float=0.5, max_episode_steps: int=100, pca_calibration_s: float=30.0, pca_neurons: list[int] | None=None, recording_tag: str='foodland', suffix: str='_foodland')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `amp_mv` | inferred at runtime | `amp_mv` |
| `buffer_write_pos` | inferred at runtime | `0` |
| `clip_epsilon` | inferred at runtime | `clip_epsilon` |
| `contextual_trainer_kwargs` | inferred at runtime | `contextual_trainer_kwargs or {}` |
| `data_buffer` | inferred at runtime | `None` |
| `decode_lookup` | inferred at runtime | `{}` |
| `decode_mode` | inferred at runtime | `decode_mode` |
| `decode_neurons` | inferred at runtime | `np.asarray(pca_neurons, dtype=int) if pca_neurons is not None else None` |
| `decode_spike_count` | inferred at runtime | `np.zeros(len(self.decode_neurons) if self.decode_neurons is not None else 0, dtype=np.float32)` |
| `decode_spike_rate` | inferred at runtime | `np.zeros_like(self.decode_spike_count)` |
| `detection_model_path` | inferred at runtime | `get_rt_sort_path() if detection_model_path is None else detection_model_path` |
| `direct_offset` | inferred at runtime | `float(direct_offset)` |
| `direct_scale` | inferred at runtime | `float(direct_scale)` |
| `electrodes` | inferred at runtime | `None` |
| `encode_mode` | inferred at runtime | `encode_mode` |
| `entropy_coef` | inferred at runtime | `entropy_coef` |
| `episode` | inferred at runtime | `0` |
| `episode_length` | inferred at runtime | `0` |
| `episode_reward` | inferred at runtime | `None` |
| `episode_reward_change` | inferred at runtime | `None` |
| `episode_rewards` | inferred at runtime | `[]` |
| `food_count` | inferred at runtime | `int(food_count)` |
| `gae_lambda` | inferred at runtime | `gae_lambda` |
| `game_env` | inferred at runtime | `FoodLandFeatureEnv(render_mode=render_mode, n_features=n_features, reward_type=reward_type, food_count=food_count, spike_count=sp…` |
| `game_features` | inferred at runtime | `None` |
| `game_info` | inferred at runtime | `None` |
| `gamma` | inferred at runtime | `gamma` |
| `hunger` | inferred at runtime | `float(hunger)` |
| `input_buffer` | inferred at runtime | `PPORolloutBuffer()` |
| `input_center` | inferred at runtime | `float(input_center)` |
| `input_gain` | inferred at runtime | `float(input_gain)` |
| `input_optimizer` | inferred at runtime | `None` |
| `input_policy` | inferred at runtime | `None` |
| `last_action` | inferred at runtime | `np.zeros(2, dtype=np.float32)` |
| `last_action_ind` | inferred at runtime | `None` |
| `last_action_inds` | inferred at runtime | `deque(maxlen=self.minibatch_size)` |
| `last_input_policy_grad_magnitude` | inferred at runtime | `0.0` |
| `last_output_policy_grad_magnitude` | inferred at runtime | `0.0` |
| `last_pca_normalized` | inferred at runtime | `np.zeros(2, dtype=np.float32)` |
| `last_pca_projection` | inferred at runtime | `np.zeros(2, dtype=np.float32)` |
| `learning_rate` | inferred at runtime | `learning_rate` |
| `max_episode_steps` | inferred at runtime | `max_episode_steps` |
| `max_grad_norm` | inferred at runtime | `max_grad_norm` |
| `max_stim_hz` | inferred at runtime | `float(max_stim_hz)` |
| `max_time` | inferred at runtime | `max_time` |
| `minibatch_size` | inferred at runtime | `minibatch_size` |
| `motor_electrodes` | inferred at runtime | `electrodes_array[self.motor_neurons]` |
| `motor_lookup` | inferred at runtime | `{}` |
| `motor_neurons` | inferred at runtime | `np.asarray(motor_neurons, dtype=int) if motor_neurons is not None else None` |
| `motor_spike_count` | inferred at runtime | `np.zeros(len(self.motor_neurons) if self.motor_neurons is not None else 0, dtype=np.float32)` |
| `motor_spike_rate` | inferred at runtime | `np.zeros_like(self.motor_spike_count)` |
| `n_episodes` | inferred at runtime | `n_episodes` |
| `n_features` | inferred at runtime | `n_features` |
| `output_buffer` | inferred at runtime | `PPORolloutBuffer()` |
| `output_optimizer` | inferred at runtime | `None` |
| `output_policy` | inferred at runtime | `None` |
| `pca_calibration_s` | inferred at runtime | `float(pca_calibration_s)` |
| `pca_components` | inferred at runtime | `None` |
| `pca_explained_variance` | inferred at runtime | `None` |
| `pca_mean` | inferred at runtime | `None` |
| `pca_model` | inferred at runtime | `None` |
| `pca_projection_mean` | inferred at runtime | `None` |
| `pca_projection_std` | inferred at runtime | `None` |
| `pending_input_transition` | inferred at runtime | `None` |
| `pending_output_transition` | inferred at runtime | `None` |
| `phase_width` | inferred at runtime | `phase_width` |
| `policy_hidden_size` | inferred at runtime | `int(policy_hidden_size)` |
| `ppo_epochs` | inferred at runtime | `ppo_epochs` |
| `ppo_minibatch_size` | inferred at runtime | `ppo_minibatch_size` |
| `read_period_ms` | inferred at runtime | `read_period_ms` |
| `reward_type` | inferred at runtime | `reward_type` |
| `rt_sort` | inferred at runtime | `None` |
| `rt_sort_path` | inferred at runtime | `rt_sort_path` |
| `sensory_electrodes` | inferred at runtime | `electrodes_array[self.sensory_neurons]` |
| `sensory_neurons` | inferred at runtime | `np.asarray(sensory_neurons, dtype=int) if sensory_neurons is not None else None` |
| `sensory_stim_Hz` | inferred at runtime | `np.zeros(len(self.sensory_neurons) if self.sensory_neurons is not None else 0, dtype=np.float32)` |
| `sensory_stim_inds` | inferred at runtime | `np.arange(len(self.sensory_electrodes), dtype=int)` |
| `spike_count` | inferred at runtime | `int(spike_count)` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `state` | inferred at runtime | `'read'` |
| `stim_electrodes` | inferred at runtime | `np.concatenate([self.sensory_electrodes, self.training_electrodes]).astype(int)` |
| `train_period_ms` | inferred at runtime | `train_period_ms` |
| `trainer` | inferred at runtime | `trainer` |
| `trainer_type` | inferred at runtime | `trainer_type` |
| `training_electrodes` | inferred at runtime | `electrodes_array[self.training_neurons]` |
| `training_neurons` | inferred at runtime | `np.asarray(training_neurons, dtype=int) if training_neurons is not None else None` |
| `training_stim_inds` | inferred at runtime | `np.arange(len(self.sensory_electrodes), len(self.stim_electrodes), dtype=int)` |
| `use_contextual_trainer` | inferred at runtime | `use_contextual_trainer` |
| `use_numba` | inferred at runtime | `use_numba` |
| `value_coef` | inferred at runtime | `value_coef` |
| `verbose` | inferred at runtime | `verbose` |
| `wait_period_ms` | inferred at runtime | `wait_period_ms` |
**Methods:**
#### `allocate_neurons(n_sequences, n_sensory=2, min_motor=2, min_training=2, max_training=6)`
> Partition sequences into sensory, motor, and training groups.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:135`
#### `configure_from_experiment(self, experiment: Experiment)`
> Bind sorter/mapping, validate PCA neuron selection, derive stimulation electrodes, and initialize policies/trainers.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:347`
#### `_create_deferred_trainer(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:453`
#### `customize_environment_params(self, env_params: dict) -> dict`
> Inject concatenated sensory and training electrode IDs into acquisition-environment construction.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:517`
#### `_add_to_buffer(self, new_data, n_channels)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:521`
#### `_get_window(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:545`
#### `_fixed_sigmoid_rates(self, features)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:555`
#### `set_sensory_signal(self, game_features)`
> Convert current FoodLand features to sensory rates through fixed sigmoid or a learned input policy.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:560`
#### `_update_motor_rates(self, moving_avg=0.2)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:581`
#### `_update_decode_rates(self, moving_avg=0.2)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:588`
#### `_format_continuous_action(self, raw_action)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:595`
#### `get_motor_signal_ppo(self, moving_avg=0.2)`
> Decode motor rates through continuous PPO and convert the second output to nonnegative speed.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:606`
#### `get_motor_signal_direct(self, moving_avg=0.2)`
> Directly scale the first two motor rates into turn and speed.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:618`
#### `get_motor_signal_pca(self, moving_avg=0.2)`
> Project selected-neuron counts through calibrated PCA and standardize the two action coordinates.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:635`
#### `_step_sensory_environment(self, buffer_size)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:648`
#### `_process_observation(self, obs, n_channels, step_size_samples, samples_since_last_process)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:668`
#### `_record_policy_transitions(self, reward, done)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:702`
#### `get_training_signal(self, train_Hz=30, stim_count=3, delay_ms=5, use_change=True)`
> Update/select configured trainer stimulation or construct a random sequential training pulse.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:725`
#### `_update_policies(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:760`
#### `_run_pca_calibration(self, buffer_size, n_channels, step_size_samples, samples_since_last_process)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:800`
#### `_select_game_action(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:859`
#### `run(self, experiment)`
> Warm up RT-Sort, calibrate PCA when requested, execute the phase state machine, log, update policies, and return results.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:866`
#### `cleanup(self)`
> Close the FoodLand environment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:1178`
#### `info(self)`
> Report neuron assignments, state timings, codec modes, and PCA duration.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_foodland.py:1181`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| FoodLand wrapper parameters include dense/specified reward type, food/spike counts, hunger, episode limit, and 1..observation-dimension features. |
| Encode modes: `fixed_sigmoid`/`policy_continuous`; decode modes: `pca`/`ppo`/`direct`; PCA calibration defaults to 30 seconds. |
| Default allocation needs 6 sequences: 2 sensory, at least 2 motor, and 2-6 training. |

## Data Shapes
- Exposed game action is `[turn, speed]` with turn in [-1,1] and speed in [0,1], reordered before calling `FoodLandEnv`.
- PCA calibration matrix is time bins x selected decode neurons; two components, projection mean/std, and explained variance are retained.
- Run writes game/reward/pattern CSVs and returns episode count, optional `SpikeData`, policy plus PCA state, and rewards.

## Notes
- Observation normalization assumes positional fields at indices 2-3 and other semantic fields at indices 4-8, then encodes only the leading `n_features`.
- PCA projects the current raw decode-neuron spike-count vector even though a smoothed decode rate is also updated.
- Calibration steps the acquisition environment without sensory stimulation and raises if fewer than two bins are collected.
