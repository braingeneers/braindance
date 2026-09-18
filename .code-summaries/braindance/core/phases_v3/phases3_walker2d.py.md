# phases3_walker2d.py

**Path:** `braindance/core/phases_v3/phases3_walker2d.py`
**Module:** `braindance.core.phases_v3.phases3_walker2d`
**Feature Area:** `Games and Reinforcement`
**Entry point:** no — library or imported component

## Overview
Runs Walker2d-v5 with six projected features encoded as low-rate sensory stimulation and motor activity decoded to six continuous controls. Its PPO output decoder is deliberately a single linear actor with a linear critic, while input encoding may be fixed tanh or learned.

## Connections
- **Used by:** `braindance.core.phases_v3.phases3_walker2d_cpg` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.walker2d_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Uses:** `convert_uint16_maxwell` from `braindance.analysis.data_loader` — imports (static evidence).
- **Uses:** `InputPolicyActorCritic` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `PPORolloutBuffer` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `ppo_update` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `PhaseV3` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `ContextualTrainer` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `TetanusTrainer` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `generate_tetanus_pattern` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `Walker2dFeatureEnv` from `braindance.games.walker2d` — imports (static evidence).
- **Uses:** `get_rt_sort_path` from `braindance` — imports (static evidence).
- **Shared data:** Uses `Walker2dFeatureEnv`, experiment mapping/RT-Sort, `MaxwellEnv`, shared input PPO/rollout utilities, and optional trainer implementations.

## Dependencies
- `braindance.analysis.data_loader.convert_uint16_maxwell` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.InputPolicyActorCritic` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.PPORolloutBuffer` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.ppo_update` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.PhaseV3` — intra-repo import; source import evidence.
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
- `torch.distributions.Normal` — external or unresolved local import; source import evidence.
- `torch.nn` — external or unresolved local import; source import evidence.
- `torch.optim` — external or unresolved local import; source import evidence.

## Classes
### LinearOutputPPO(nn.Module)
> Provide a no-hidden-layer Gaussian actor and linear critic for tanh-bounded continuous PPO actions.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:43`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/phases3_walker2d.py:374 (named-call hint); braindance/core/phases_v3/phases3_walker2d_cpg.py:345 (named-call hint)
**Constructor:** `__init__(self, input_size: int, action_size: int)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `actor` | inferred at runtime | `nn.Linear(input_size, action_size)` |
| `critic` | inferred at runtime | `nn.Linear(input_size, 1)` |
| `log_std` | inferred at runtime | `nn.Parameter(torch.full((action_size,), -0.5))` |
**Methods:**
#### `act(self, state_tensor, deterministic: bool=False)`
> Sample or deterministically choose a raw Gaussian action, tanh-squash it, and return action, corrected log probability, and value.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:58`
#### `evaluate(self, states, actions)`
> Reconstruct raw actions and compute PPO log probabilities, entropy, and critic values for a batch.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:74`
### Walker2dPhaseV3(PhaseV3)
> Coordinate Walker2d gameplay, low-rate sensory encoding, online neural decoding, PPO, and optional training stimulation.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:85`
**Kind:** class. **Instantiated by:** braindance/examples/walker2d_reinforcement_example.py:101 (named-call hint)
**Constructor:** `__init__(self, sensory_neurons: list | None=None, motor_neurons: list | None=None, training_neurons: list | None=None, amp_mv: int=200, phase_width: int=100, read_period_ms: int=20, train_period_ms: int=200, wait_period_ms: int=400, n_episodes: int=100, trainer=None, trainer_type: str | None=None, use_contextual_trainer: bool=False, contextual_trainer_kwargs: dict | None=None, minibatch_size: int=5, verbose: bool=False, max_time=np.inf, rt_sort_path: str | None=None, detection_model_path: str | None=None, use_numba: bool=True, learning_rate: float=0.0003, gamma: float=0.99, gae_lambda: float=0.95, clip_epsilon: float=0.2, ppo_epochs: int=4, ppo_minibatch_size: int=32, entropy_coef: float=0.01, value_coef: float=0.5, max_grad_norm: float=0.5, policy_hidden_size: int=64, n_features: int=6, projection_seed: int=0, projection_scale: float=4.0, encode_mode: str='fixed_tanh', decode_mode: str='ppo', max_stim_hz: float=2.0, input_gain: float=6.0, input_center: float=0.5, direct_scale: float=3.0, direct_offset: float=1.0, render_mode: str | None='human', max_episode_steps: int=1000, recording_tag: str='walker2d', suffix: str='_walker2d')`
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
| `game_env` | inferred at runtime | `Walker2dFeatureEnv(render_mode=render_mode, n_features=n_features, projection_seed=projection_seed, projection_scale=projection_s…` |
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
| `last_game_action` | inferred at runtime | `None` |
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
#### `allocate_neurons(n_sequences, n_sensory=6, min_motor=6, min_training=2, max_training=6)`
> Partition sequences into six sensory, motor, and training groups.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:119`
#### `configure_from_experiment(self, experiment: Experiment)`
> Bind sorter/mapping, derive stimulation electrodes, and initialize policies/trainers.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:304`
#### `_create_deferred_trainer(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:393`
#### `customize_environment_params(self, env_params: dict) -> dict`
> Inject concatenated sensory and training electrode IDs into acquisition-environment construction.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:444`
#### `_add_to_buffer(self, new_data, n_channels)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:452`
#### `_get_window(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:476`
#### `_tanh_rates(self, logits: np.ndarray) -> np.ndarray`
> Map logits to continuous stim rates in [0, max_stim_hz] via tanh shift.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:490`
#### `set_sensory_signal(self, game_features)`
> Map game features or learned latent inputs through a shifted tanh into sensory stimulation rates.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:499`
#### `_update_motor_rates(self, ema_alpha=0.9)`
> Update motor spike rate estimate in Hz using a slow exponential moving average.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:527`
#### `get_motor_signal(self)`
> Convert smoothed Hz rates into a six-dimensional PPO action and stage its transition.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:543`
#### `get_motor_signal_direct(self)`
> Directly scale the first six motor rates to clipped actions.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:556`
#### `_step_sensory_environment(self, buffer_size)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:568`
#### `_process_observation(self, obs, n_channels, step_size_samples, samples_since_last_process)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:588`
#### `_record_policy_transitions(self, reward, done)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:620`
#### `get_training_signal(self, train_Hz=30, stim_count=3, delay_ms=5, use_change=True)`
> Update/select a trainer or create a random sequential training pulse.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:643`
#### `_update_policies(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:678`
#### `_save_reward_plot(self, save_path: str | Path)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:712`
#### `run(self, experiment)`
> Execute warmup and read/game/train/wait state transitions, logs, policy updates, plots, and result assembly.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:734`
#### `cleanup(self)`
> Close the Walker2d game environment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:1010`
#### `info(self)`
> Report neuron assignments, timings, modes, and stimulation limit.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_walker2d.py:1013`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Encode modes: `fixed_tanh` and `policy_continuous`; decode modes: `ppo` and `direct`; default maximum stimulation is 2 Hz. |
| Read windows default to 20 ms; training/wait periods are 200/400 ms and episodes cap at 1000 game steps. |
| Default allocation requires 14 sequences: 6 sensory, at least 6 motor, and 2-6 training. |

## Data Shapes
- Feature, sensory rate, and action vectors each have length 6 by default; all motor-neuron rates feed the linear PPO actor.
- Motor counts are converted to Hz using the read period and smoothed with EMA alpha 0.9.
- Writes Walker2d game/reward/pattern CSVs and reward PNGs; returns episode count, optional `SpikeData`, policy states/modes, and rewards.

## Notes
- The learned input policy's returned rates are ignored; its latent action is remapped with the phase's tanh rate function and stored as the PPO action.
- Reward plots overwrite `reward_plot.png` in the recording directory every 25 episodes.
- The direct decoder uses only the first six motor neurons even if more were allocated.
