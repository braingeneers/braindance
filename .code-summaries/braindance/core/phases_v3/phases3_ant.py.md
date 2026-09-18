# phases3_ant.py

**Path:** `braindance/core/phases_v3/phases3_ant.py`
**Module:** `braindance.core.phases_v3.phases3_ant`
**Feature Area:** `Games and Reinforcement`
**Entry point:** no — library or imported component

## Overview
Runs a closed-loop Ant-v5 experiment that encodes eight projected game features as sensory stimulation and decodes sorted motor-neuron activity into eight continuous actions. Optional PPO policies learn either side of the neural reservoir, while a tetanus/contextual trainer can apply reward-dependent training stimulation between episodes.

## Connections
- **Used by:** `braindance.examples.ant_reinforcement_example` — import consumer hint; not a proven runtime call.
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
- **Uses:** `AntFeatureEnv` from `braindance.games.ant` — imports (static evidence).
- **Uses:** `get_rt_sort_path` from `braindance` — imports (static evidence).
- **Shared data:** Consumes `Experiment.data.rt_sort_object` and mapping, controls `MaxwellEnv`, uses `AntFeatureEnv`, codec PPO components, and optional trainer classes.

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
- `braindance.games.ant.AntFeatureEnv` — intra-repo import; source import evidence.
- `braindance.get_rt_sort_path` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `spikelab.SpikeData` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.
- `torch.optim` — external or unresolved local import; source import evidence.

## Classes
### AntPhaseV3(PhaseV3)
> Coordinate Ant gameplay, sensory stimulation, online RT-Sort decoding, PPO learning, and optional plasticity training.
**Source:** `braindance/core/phases_v3/phases3_ant.py:34`
**Kind:** class. **Instantiated by:** braindance/examples/ant_reinforcement_example.py:35 (named-call hint)
**Constructor:** `__init__(self, sensory_neurons: list | None=None, motor_neurons: list | None=None, training_neurons: list | None=None, amp_mv: int=200, phase_width: int=100, read_period_ms: int=50, train_period_ms: int=200, wait_period_ms: int=400, n_episodes: int=100, trainer=None, trainer_type: str | None=None, use_contextual_trainer: bool=False, contextual_trainer_kwargs: dict | None=None, minibatch_size: int=5, verbose: bool=False, max_time=np.inf, rt_sort_path: str | None=None, detection_model_path: str | None=None, use_numba: bool=True, learning_rate: float=0.0003, gamma: float=0.99, gae_lambda: float=0.95, clip_epsilon: float=0.2, ppo_epochs: int=4, ppo_minibatch_size: int=32, entropy_coef: float=0.01, value_coef: float=0.5, max_grad_norm: float=0.5, policy_hidden_size: int=64, n_features: int=8, projection_seed: int=0, projection_scale: float=4.0, encode_mode: str='fixed_sigmoid', decode_mode: str='ppo', reward_mode: str='forward_x', max_stim_hz: float=10.0, input_gain: float=6.0, input_center: float=0.5, direct_scale: float=3.0, direct_offset: float=1.0, render_mode: str | None='human', max_episode_steps: int=1000, recording_tag: str='ant', suffix: str='_ant')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `amp_mv` | inferred at runtime | `amp_mv` |
| `buffer_write_pos` | inferred at runtime | `0` |
| `clip_epsilon` | inferred at runtime | `clip_epsilon` |
| `contextual_trainer_kwargs` | inferred at runtime | `contextual_trainer_kwargs or {}` |
| `data_buffer` | inferred at runtime | `None` |
| `decode_mode` | inferred at runtime | `decode_mode` |
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
| `gae_lambda` | inferred at runtime | `gae_lambda` |
| `game_env` | inferred at runtime | `AntFeatureEnv(render_mode=render_mode, n_features=n_features, projection_seed=projection_seed, projection_scale=projection_scale,…` |
| `game_features` | inferred at runtime | `None` |
| `game_info` | inferred at runtime | `None` |
| `gamma` | inferred at runtime | `gamma` |
| `input_buffer` | inferred at runtime | `PPORolloutBuffer()` |
| `input_center` | inferred at runtime | `float(input_center)` |
| `input_gain` | inferred at runtime | `float(input_gain)` |
| `input_optimizer` | inferred at runtime | `None` |
| `input_policy` | inferred at runtime | `None` |
| `last_action_ind` | inferred at runtime | `None` |
| `last_action_inds` | inferred at runtime | `deque(maxlen=self.minibatch_size)` |
| `last_input_policy_grad_magnitude` | inferred at runtime | `0.0` |
| `last_output_policy_grad_magnitude` | inferred at runtime | `0.0` |
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
| `pending_input_transition` | inferred at runtime | `None` |
| `pending_output_transition` | inferred at runtime | `None` |
| `phase_width` | inferred at runtime | `phase_width` |
| `policy_hidden_size` | inferred at runtime | `policy_hidden_size` |
| `ppo_epochs` | inferred at runtime | `ppo_epochs` |
| `ppo_minibatch_size` | inferred at runtime | `ppo_minibatch_size` |
| `read_period_ms` | inferred at runtime | `read_period_ms` |
| `reward_mode` | inferred at runtime | `reward_mode` |
| `rt_sort` | inferred at runtime | `None` |
| `rt_sort_path` | inferred at runtime | `rt_sort_path` |
| `sensory_electrodes` | inferred at runtime | `electrodes_array[self.sensory_neurons]` |
| `sensory_neurons` | inferred at runtime | `np.asarray(sensory_neurons, dtype=int) if sensory_neurons is not None else None` |
| `sensory_stim_Hz` | inferred at runtime | `np.zeros(len(self.sensory_neurons) if self.sensory_neurons is not None else 0, dtype=np.float32)` |
| `sensory_stim_inds` | inferred at runtime | `np.arange(len(self.sensory_electrodes), dtype=int)` |
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
#### `allocate_neurons(n_sequences, n_sensory=8, min_motor=8, min_training=2, max_training=6)`
> Partition detected sorter sequences into fixed sensory, remaining motor, and bounded training groups.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant.py:65`
#### `configure_from_experiment(self, experiment: Experiment)`
> Bind the sorter and mapping, allocate neuron groups, derive stimulation electrodes, and construct requested policies/trainers.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant.py:251`
#### `_create_deferred_trainer(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant.py:341`
#### `customize_environment_params(self, env_params: dict) -> dict`
> Inject phase-selected sensory and training electrodes into environment construction.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant.py:392`
#### `_add_to_buffer(self, new_data, n_channels)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant.py:400`
#### `_get_window(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant.py:424`
#### `_fixed_sigmoid_rates(self, features)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant.py:438`
#### `set_sensory_signal(self, game_features)`
> Map Ant features to stimulation rates with fixed sigmoid encoding or a learned continuous input policy.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant.py:443`
#### `_update_motor_rates(self, moving_avg=0.2)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant.py:468`
#### `get_motor_signal(self, moving_avg=0.2)`
> Decode smoothed motor activity into a sampled continuous PPO action and stage its rollout transition.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant.py:475`
#### `get_motor_signal_direct(self, moving_avg=0.2)`
> Linearly map the first eight smoothed motor rates to clipped actions.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant.py:488`
#### `_step_sensory_environment(self, buffer_size)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant.py:500`
#### `_process_observation(self, obs, n_channels, step_size_samples, samples_since_last_process)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant.py:520`
#### `_record_policy_transitions(self, reward, done)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant.py:552`
#### `get_training_signal(self, train_Hz=30, stim_count=3, delay_ms=5, use_change=True)`
> Select and update a configured trainer action, or build a random sequential training pulse.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant.py:575`
#### `_update_policies(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant.py:610`
#### `run(self, experiment)`
> Execute the read/game/train/wait state machine, logs, online sorting, policy updates, and result assembly.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant.py:648`
#### `cleanup(self)`
> Close the Ant game environment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant.py:923`
#### `info(self)`
> Report neuron assignments and key state-machine/codec settings.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant.py:926`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Encode modes: `fixed_sigmoid` or `policy_continuous`; decode modes: `ppo` or `direct`; default maximum sensory rate is 10 Hz. |
| Uses 20 ms RT-Sort windows, 4.5 ms overlap, 2 ms acquisition batches, 100 warmup batches, and read/train/wait state durations in milliseconds. |
| Requires at least 18 detected sequences by default: 8 sensory, at least 8 motor, and 2-6 training. |

## Data Shapes
- Game features and sensory-rate vectors have length 8; PPO actions have length 8 in [-1,1].
- Raw batches are samples x channels; the circular sort buffer is 400 x channels at 20 kHz.
- Run returns episode count, optional `SpikeData`, policy state dictionaries/mode metadata, and episode reward list; writes game/reward/pattern CSV logs.

## Notes
- Neuron inputs are sorter sequence indices, converted to physical electrodes during experiment configuration.
- PPO rollout buffers are updated after each game step and consumed at episode boundaries.
- Without a trainer, `get_training_signal` can generate a random training-neuron sequence, but the run loop only enables training when `self.trainer` exists.
