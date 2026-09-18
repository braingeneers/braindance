# phases3_ant_cpg.py

**Path:** `braindance/core/phases_v3/phases3_ant_cpg.py`
**Module:** `braindance.core.phases_v3.phases3_ant_cpg`
**Feature Area:** `Games and Reinforcement`
**Entry point:** no — library or imported component

## Overview
Runs Ant-v5 with sensory electrodes driven by fixed or sinusoidally varying central-pattern-generator rates instead of game observations. Motor activity is normalized from a CPG-driven warmup and decoded by continuous PPO or direct scaling, with optional tetanus, contextual, or adaptive tetanus training.

## Connections
- **Used by:** `braindance.examples.ant_cpg_example` — import consumer hint; not a proven runtime call.
- **Uses:** `convert_uint16_maxwell` from `braindance.analysis.data_loader` — imports (static evidence).
- **Uses:** `ContinuousOutputPPO` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `PPORolloutBuffer` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `ppo_update` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `PhaseV3` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `AdaptiveTetanusTrainer` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `ContextualTrainer` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `TetanusTrainer` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `generate_tetanus_pattern` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `AntFeatureEnv` from `braindance.games.ant` — imports (static evidence).
- **Uses:** `get_rt_sort_path` from `braindance` — imports (static evidence).
- **Shared data:** Uses `AntFeatureEnv` for reward/dynamics but discards its observation features; controls `MaxwellEnv`, online RT-Sort, `ContinuousOutputPPO`, and trainer implementations.

## Dependencies
- `braindance.analysis.data_loader.convert_uint16_maxwell` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.ContinuousOutputPPO` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.PPORolloutBuffer` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.ppo_update` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.PhaseV3` — intra-repo import; source import evidence.
- `braindance.core.trainer.AdaptiveTetanusTrainer` — intra-repo import; source import evidence.
- `braindance.core.trainer.ContextualTrainer` — intra-repo import; source import evidence.
- `braindance.core.trainer.TetanusTrainer` — intra-repo import; source import evidence.
- `braindance.core.trainer.generate_tetanus_pattern` — intra-repo import; source import evidence.
- `braindance.games.ant.AntFeatureEnv` — intra-repo import; source import evidence.
- `braindance.get_rt_sort_path` — intra-repo import; source import evidence.
- `matplotlib` — external or unresolved local import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `spikelab.SpikeData` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.
- `torch.optim` — external or unresolved local import; source import evidence.

## Classes
### AntCPGPhaseV3(PhaseV3)
> Coordinate rhythmic sensory drive, online neural decoding, Ant gameplay, PPO updates, and optional training stimulation.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:42`
**Kind:** class. **Instantiated by:** braindance/examples/ant_cpg_example.py:82 (named-call hint)
**Constructor:** `__init__(self, cpg_frequencies: list[float] | None=None, neurons_per_freq: int=1, pattern_mode: str='fixed', phase_offsets: list[float] | None=None, sensory_neurons: list | None=None, motor_neurons: list | None=None, training_neurons: list | None=None, amp_mv: int=200, phase_width: int=100, read_period_ms: int=50, train_period_ms: int=200, wait_period_ms: int=400, n_episodes: int=100, trainer=None, trainer_type: str | None=None, use_contextual_trainer: bool=False, contextual_trainer_kwargs: dict | None=None, adaptive_tetanus_kwargs: dict | None=None, minibatch_size: int=5, verbose: bool=False, max_time=np.inf, rt_sort_path: str | None=None, detection_model_path: str | None=None, use_numba: bool=True, learning_rate: float=0.0003, gamma: float=0.99, gae_lambda: float=0.95, clip_epsilon: float=0.2, ppo_epochs: int=4, ppo_minibatch_size: int=32, entropy_coef: float=0.01, value_coef: float=0.5, max_grad_norm: float=0.5, policy_hidden_size: int=64, decode_mode: str='ppo', reward_mode: str='forward_x', max_stim_hz: float=10.0, direct_scale: float=3.0, direct_offset: float=1.0, normalize_mode: str='warmup', warmup_duration_s: float=30.0, log_rates: bool=True, ema_alpha: float=0.95, render_mode: str | None='human', max_episode_steps: int=1000, recording_tag: str='ant_cpg', suffix: str='_ant_cpg')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_freq_offsets` | inferred at runtime | `list(phase_offsets)` |
| `_last_log_std` | inferred at runtime | `log_std` |
| `_last_raw_mean` | inferred at runtime | `raw_mean` |
| `_motor_input_scale` | inferred at runtime | `1.0` |
| `_rate_mean` | inferred at runtime | `None` |
| `_rate_norm_frozen` | inferred at runtime | `False` |
| `_rate_obs_count` | inferred at runtime | `0` |
| `_rate_var` | inferred at runtime | `None` |
| `adaptive_tetanus_kwargs` | inferred at runtime | `adaptive_tetanus_kwargs or {}` |
| `amp_mv` | inferred at runtime | `amp_mv` |
| `buffer_write_pos` | inferred at runtime | `0` |
| `clip_epsilon` | inferred at runtime | `clip_epsilon` |
| `contextual_trainer_kwargs` | inferred at runtime | `contextual_trainer_kwargs or {}` |
| `cpg_frequencies` | inferred at runtime | `list(cpg_frequencies)` |
| `data_buffer` | inferred at runtime | `None` |
| `decode_mode` | inferred at runtime | `decode_mode` |
| `detection_model_path` | inferred at runtime | `get_rt_sort_path() if detection_model_path is None else detection_model_path` |
| `direct_offset` | inferred at runtime | `float(direct_offset)` |
| `direct_scale` | inferred at runtime | `float(direct_scale)` |
| `electrodes` | inferred at runtime | `None` |
| `ema_alpha` | inferred at runtime | `float(ema_alpha)` |
| `entropy_coef` | inferred at runtime | `entropy_coef` |
| `episode` | inferred at runtime | `0` |
| `episode_length` | inferred at runtime | `0` |
| `episode_reward` | inferred at runtime | `None` |
| `episode_reward_change` | inferred at runtime | `None` |
| `episode_rewards` | inferred at runtime | `[]` |
| `gae_lambda` | inferred at runtime | `gae_lambda` |
| `game_env` | inferred at runtime | `AntFeatureEnv(render_mode=render_mode, n_features=self.n_cpg, reward_mode=reward_mode)` |
| `gamma` | inferred at runtime | `gamma` |
| `last_action_ind` | inferred at runtime | `None` |
| `last_action_inds` | inferred at runtime | `deque(maxlen=self.minibatch_size)` |
| `last_game_action` | inferred at runtime | `None` |
| `last_output_policy_grad_magnitude` | inferred at runtime | `0.0` |
| `learning_rate` | inferred at runtime | `learning_rate` |
| `log_rates` | inferred at runtime | `log_rates` |
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
| `n_cpg` | inferred at runtime | `len(self.cpg_frequencies) * self.neurons_per_freq` |
| `n_episodes` | inferred at runtime | `n_episodes` |
| `neuron_freqs` | inferred at runtime | `np.zeros(self.n_cpg, dtype=np.float32)` |
| `neuron_offsets` | inferred at runtime | `np.zeros(self.n_cpg, dtype=np.float32)` |
| `neurons_per_freq` | inferred at runtime | `int(neurons_per_freq)` |
| `normalize_mode` | inferred at runtime | `normalize_mode` |
| `output_buffer` | inferred at runtime | `PPORolloutBuffer()` |
| `output_optimizer` | inferred at runtime | `None` |
| `output_policy` | inferred at runtime | `None` |
| `pattern_mode` | inferred at runtime | `pattern_mode` |
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
| `warmup_duration_s` | inferred at runtime | `float(warmup_duration_s)` |
**Methods:**
#### `allocate_neurons(n_sequences, n_sensory=8, min_motor=8, min_training=2, max_training=6)`
> Partition sorter sequences into CPG sensory, motor, and bounded training groups.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:54`
#### `configure_from_experiment(self, experiment: Experiment)`
> Resolve neuron/electrode assignments and initialize the selected decoder and trainer.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:265`
#### `_create_deferred_trainer(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:357`
#### `customize_environment_params(self, env_params: dict) -> dict`
> Pass all sensory and training electrodes to the acquisition environment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:415`
#### `_add_to_buffer(self, new_data, n_channels)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:419`
#### `_get_window(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:443`
#### `set_sensory_signal(self, elapsed_s: float)`
> Set constant assigned frequencies or compute sinusoidal rate envelopes at an elapsed time.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:453`
#### `_update_motor_rates(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:465`
#### `_update_rate_stats(self, rates: np.ndarray)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:474`
#### `_freeze_rate_stats(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:487`
#### `_normalize_rates(self, rates: np.ndarray) -> np.ndarray`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:490`
#### `get_motor_signal(self)`
> Normalize smoothed motor rates, sample an eight-dimensional PPO action, and stage its transition.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:498`
#### `get_motor_signal_direct(self)`
> Map the first eight smoothed rate values directly to clipped actions.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:521`
#### `_step_sensory_environment(self, buffer_size)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:528`
#### `_process_observation(self, obs, n_channels, step_size_samples, samples_since_last_process)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:548`
#### `_record_policy_transitions(self, reward, done)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:576`
#### `get_training_signal(self, train_Hz=30, stim_count=3, delay_ms=5, use_change=True)`
> Update/select the configured trainer or construct a random sequential pulse.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:588`
#### `_update_policies(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:624`
#### `_save_reward_plot(self, save_path: str | Path)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:652`
#### `_training_pattern_for_log(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:670`
#### `run(self, experiment)`
> Warm up sorting/rate statistics, execute the read/game/train/wait loop, log results, update PPO, and return artifacts.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:684`
#### `cleanup(self)`
> Close the Ant game environment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:994`
#### `info(self)`
> Report CPG, neuron, normalization, decoder, and trainer configuration.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_ant_cpg.py:997`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Default CPG frequencies are 0.5, 1, 2, 4, and 8 Hz with one sensory neuron per frequency; `pattern_mode` is `fixed` or `oscillating`. |
| Decode mode is `ppo` or `direct`; normalization mode is `warmup` or `running`, default warmup is 30 seconds, log-rate preprocessing is enabled, and EMA alpha is 0.95. |
| PPO updates wait for at least `max(ppo_minibatch_size, 32)` transitions. |

## Data Shapes
- Sensory arrays have `len(cpg_frequencies) * neurons_per_freq` entries; motor PPO actions have 8 entries.
- Warmup and run process samples x channels through a 400-sample circular RT-Sort window; rates are per-motor-neuron Hz vectors.
- Writes Ant CPG game/reward/pattern CSVs and a reward PNG every 25 episodes.

## Notes
- Warmup stimulation and normalization use wall-clock `perf_counter`, while the main phase uses environment elapsed time.
- In warmup normalization mode, mean/variance are frozen before gameplay; running mode continues adapting.
- Adaptive tetanus is supported only in this Ant CPG variant among the assigned phase modules.
