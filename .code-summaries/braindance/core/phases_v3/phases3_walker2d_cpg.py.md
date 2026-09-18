# phases3_walker2d_cpg.py

**Path:** `braindance/core/phases_v3/phases3_walker2d_cpg.py`
**Module:** `braindance.core.phases_v3.phases3_walker2d_cpg`
**Feature Area:** `Games and Reinforcement`
**Entry point:** no — library or imported component

## Overview
Runs Walker2d with fixed-frequency or sinusoidally modulated central-pattern-generator sensory stimulation rather than observation feedback. It normalizes smoothed motor rates using CPG-driven warmup statistics and decodes them through the shared linear PPO actor or direct scaling.

## Connections
- **Used by:** `braindance.examples.walker2d_cpg_example` — import consumer hint; not a proven runtime call.
- **Uses:** `convert_uint16_maxwell` from `braindance.analysis.data_loader` — imports (static evidence).
- **Uses:** `PPORolloutBuffer` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `ppo_update` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `PhaseV3` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `LinearOutputPPO` from `braindance.core.phases_v3.phases3_walker2d` — imports (static evidence).
- **Uses:** `ContextualTrainer` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `TetanusTrainer` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `generate_tetanus_pattern` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `Walker2dFeatureEnv` from `braindance.games.walker2d` — imports (static evidence).
- **Uses:** `get_rt_sort_path` from `braindance` — imports (static evidence).
- **Shared data:** Reuses `LinearOutputPPO` from `phases3_walker2d`; otherwise connects the experiment sorter/mapping and `MaxwellEnv` to `Walker2dFeatureEnv`, PPO rollout utilities, and trainer APIs.

## Dependencies
- `braindance.analysis.data_loader.convert_uint16_maxwell` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.PPORolloutBuffer` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.ppo_update` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.PhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3_walker2d.LinearOutputPPO` — intra-repo import; source import evidence.
- `braindance.core.trainer.ContextualTrainer` — intra-repo import; source import evidence.
- `braindance.core.trainer.TetanusTrainer` — intra-repo import; source import evidence.
- `braindance.core.trainer.generate_tetanus_pattern` — intra-repo import; source import evidence.
- `braindance.games.walker2d.Walker2dFeatureEnv` — intra-repo import; source import evidence.
- `braindance.get_rt_sort_path` — intra-repo import; source import evidence.
- `matplotlib` — external or unresolved local import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `spikelab.SpikeData` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.
- `torch.nn.init` — external or unresolved local import; source import evidence.
- `torch.optim` — external or unresolved local import; source import evidence.

## Classes
### Walker2dCPGPhaseV3(PhaseV3)
> Coordinate rhythmic sensory input, normalized online motor decoding, Walker2d gameplay, PPO, and optional training stimulation.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:39`
**Kind:** class. **Instantiated by:** braindance/examples/walker2d_cpg_example.py:96 (named-call hint)
**Constructor:** `__init__(self, cpg_frequencies: list[float] | None=None, neurons_per_freq: int=1, pattern_mode: str='fixed', phase_offsets: list[float] | None=None, sensory_neurons: list | None=None, motor_neurons: list | None=None, training_neurons: list | None=None, amp_mv: int=200, phase_width: int=100, read_period_ms: int=20, train_period_ms: int=200, wait_period_ms: int=400, n_episodes: int=100, trainer=None, trainer_type: str | None=None, use_contextual_trainer: bool=False, contextual_trainer_kwargs: dict | None=None, minibatch_size: int=5, verbose: bool=False, max_time=np.inf, rt_sort_path: str | None=None, detection_model_path: str | None=None, use_numba: bool=True, learning_rate: float=0.0003, gamma: float=0.99, gae_lambda: float=0.95, clip_epsilon: float=0.2, ppo_epochs: int=4, ppo_minibatch_size: int=32, entropy_coef: float=0.01, value_coef: float=0.5, max_grad_norm: float=0.5, decode_mode: str='ppo', max_stim_hz: float=8.0, direct_scale: float=3.0, direct_offset: float=1.0, normalize_mode: str='warmup', warmup_duration_s: float=30.0, log_rates: bool=True, ema_alpha: float=0.95, render_mode: str | None='human', max_episode_steps: int=1000, recording_tag: str='walker2d_cpg', suffix: str='_walker2d_cpg')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_freq_offsets` | inferred at runtime | `list(phase_offsets)` |
| `_last_log_std` | inferred at runtime | `log_std` |
| `_last_raw_logits` | inferred at runtime | `raw_logits` |
| `_motor_input_scale` | inferred at runtime | `1.0 / np.sqrt(max(len(self.motor_neurons), 1))` |
| `_rate_mean` | inferred at runtime | `None` |
| `_rate_norm_frozen` | inferred at runtime | `False` |
| `_rate_obs_count` | inferred at runtime | `0` |
| `_rate_var` | inferred at runtime | `None` |
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
| `game_env` | inferred at runtime | `Walker2dFeatureEnv(render_mode=render_mode, n_features=self.n_cpg)` |
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
| `ppo_epochs` | inferred at runtime | `ppo_epochs` |
| `ppo_minibatch_size` | inferred at runtime | `ppo_minibatch_size` |
| `read_period_ms` | inferred at runtime | `read_period_ms` |
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
#### `allocate_neurons(n_sequences, n_sensory=5, min_motor=6, min_training=2, max_training=6)`
> Partition sorter sequences into CPG sensory, motor, and bounded training groups.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:75`
#### `configure_from_experiment(self, experiment: Experiment)`
> Resolve neuron/electrode assignments and initialize normalized linear PPO plus optional trainer.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:284`
#### `_create_deferred_trainer(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:380`
#### `customize_environment_params(self, env_params: dict) -> dict`
> Inject concatenated CPG sensory and training electrode IDs into acquisition-environment construction.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:431`
#### `_add_to_buffer(self, new_data, n_channels)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:439`
#### `_get_window(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:463`
#### `set_sensory_signal(self, elapsed_s: float)`
> Set constant assigned CPG rates or sinusoidal rate envelopes at an elapsed time.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:477`
#### `_update_motor_rates(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:498`
#### `_update_rate_stats(self, rates: np.ndarray)`
> Update running mean/variance estimates for motor spike rates.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:507`
#### `_freeze_rate_stats(self)`
> Freeze the current mean/var so normalization becomes stationary.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:521`
#### `_normalize_rates(self, rates: np.ndarray) -> np.ndarray`
> Normalize motor spike rates to zero-mean, unit-variance.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:525`
#### `get_motor_signal(self)`
> Log-transform and normalize smoothed motor Hz, sample a PPO action, and stage its transition.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:539`
#### `get_motor_signal_direct(self)`
> Map the first six smoothed motor rates directly to bounded actions.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:563`
#### `_step_sensory_environment(self, buffer_size)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:575`
#### `_process_observation(self, obs, n_channels, step_size_samples, samples_since_last_process)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:595`
#### `_record_policy_transitions(self, reward, done)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:627`
#### `get_training_signal(self, train_Hz=30, stim_count=3, delay_ms=5, use_change=True)`
> Update/select trainer stimulation or generate a random sequential pulse.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:639`
#### `_update_policies(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:674`
#### `_save_reward_plot(self, save_path: str | Path)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:702`
#### `run(self, experiment)`
> Collect CPG warmup statistics, run the read/game/train/wait state machine, log/update/plot, and assemble outputs.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:724`
#### `cleanup(self)`
> Close the Walker2d game environment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:1086`
#### `info(self)`
> Report CPG, neuron, decoder, normalization, and phase timing settings.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d_cpg.py:1089`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Default CPG frequencies are 0.5, 1, 2, 4, and 8 Hz; `pattern_mode` is `fixed` or `oscillating`, with optional per-frequency phase offsets. |
| Decode mode is `ppo` or `direct`; default maximum stimulation is 8 Hz, normalization is warmup/frozen for 30 seconds, log-rate transform is enabled, and EMA alpha is 0.95. |
| PPO requires at least `max(ppo_minibatch_size, 32)` buffered transitions before an update. |

## Data Shapes
- Sensory arrays have `len(cpg_frequencies) * neurons_per_freq` entries; actions have six entries.
- Normalized PPO input is a per-motor-neuron vector scaled by `1/sqrt(n_motor)`; running mean/variance are per neuron.
- Writes Walker2d CPG game/reward/pattern CSVs and reward plots; returns episode count, optional `SpikeData`, decoder/CPG state, and rewards.

## Notes
- The actor/critic receive explicit orthogonal initialization, small actor gain, zero biases, and Adam weight decay.
- Warmup uses wall time and sends CPG stimulation while collecting sorter-derived motor-rate statistics.
- The game observation returned by Walker2d is intentionally ignored because CPG drive is open-loop with respect to game state.
