# phases3_pacman.py

**Path:** `braindance/core/phases_v3/phases3_pacman.py`
**Module:** `braindance.core.phases_v3.phases3_pacman`
**Feature Area:** `Games and Reinforcement`
**Entry point:** no — library or imported component

## Overview
Runs Ms. Pac-Man with projected game features encoded as sensory stimulation and sorted motor activity decoded into Atari's nine discrete actions. Decoding can use discrete PPO or the spatial center of activity across physical motor-electrode positions; an optional subclass feeds the existing neural RL visualizer.

## Connections
- **Used by:** `braindance.examples.mspacman_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Uses:** `convert_uint16_maxwell` from `braindance.analysis.data_loader` — imports (static evidence).
- **Uses:** `NeuralRLVisualizer` from `braindance.core.phases_v3.cartpole_viz_helper` — imports (static evidence).
- **Uses:** `DiscreteOutputPPO` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `InputPolicyActorCritic` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `PPORolloutBuffer` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `ppo_update` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `sigmoid` from `braindance.core.phases_v3.codec` — imports (static evidence).
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `PhaseV3` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `ContextualTrainer` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `TetanusTrainer` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `generate_tetanus_pattern` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `MsPacmanFeatureEnv` from `braindance.games.mspacman` — imports (static evidence).
- **Uses:** `get_rt_sort_path` from `braindance` — imports (static evidence).
- **Shared data:** Uses `MsPacmanFeatureEnv`, experiment mapping/RT-Sort, `MaxwellEnv`, PPO codec classes, optional trainer APIs, and `NeuralRLVisualizer`.

## Dependencies
- `braindance.analysis.data_loader.convert_uint16_maxwell` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.cartpole_viz_helper.NeuralRLVisualizer` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.DiscreteOutputPPO` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.InputPolicyActorCritic` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.PPORolloutBuffer` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.ppo_update` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.codec.sigmoid` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.PhaseV3` — intra-repo import; source import evidence.
- `braindance.core.trainer.ContextualTrainer` — intra-repo import; source import evidence.
- `braindance.core.trainer.TetanusTrainer` — intra-repo import; source import evidence.
- `braindance.core.trainer.generate_tetanus_pattern` — intra-repo import; source import evidence.
- `braindance.games.mspacman.MsPacmanFeatureEnv` — intra-repo import; source import evidence.
- `braindance.get_rt_sort_path` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `spikelab.SpikeData` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.
- `torch.optim` — external or unresolved local import; source import evidence.

## Classes
### MsPacManPhase(PhaseV3)
> Coordinate Ms. Pac-Man gameplay, sensory stimulation, online sorting, discrete/spatial decoding, PPO, and optional training stimulation.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:34`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, sensory_neurons: list | None=None, motor_neurons: list | None=None, training_neurons: list | None=None, amp_mv: int=200, phase_width: int=100, read_period_ms: int=50, train_period_ms: int=200, wait_period_ms: int=400, n_episodes: int=100, trainer=None, trainer_type: str | None=None, use_contextual_trainer: bool=False, contextual_trainer_kwargs: dict | None=None, minibatch_size: int=5, verbose: bool=False, max_time=np.inf, rt_sort_path: str | None=None, detection_model_path: str | None=None, use_numba: bool=True, learning_rate: float=0.0003, gamma: float=0.99, gae_lambda: float=0.95, clip_epsilon: float=0.2, ppo_epochs: int=4, ppo_minibatch_size: int=32, entropy_coef: float=0.01, value_coef: float=0.5, max_grad_norm: float=0.5, policy_hidden_size: int=64, n_features: int=8, projection_seed: int=0, projection_scale: float=4.0, encode_mode: str='fixed_sigmoid', decode_mode: str='ppo', max_stim_hz: float=10.0, input_gain: float=6.0, input_center: float=0.5, spatial_deadzone: float=0.2, spatial_flip_y: bool=True, render_mode: str | None='human', recording_tag: str='mspacman', suffix: str='_mspacman')`
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
| `electrodes` | inferred at runtime | `None` |
| `encode_mode` | inferred at runtime | `encode_mode` |
| `entropy_coef` | inferred at runtime | `entropy_coef` |
| `episode` | inferred at runtime | `0` |
| `episode_length` | inferred at runtime | `0` |
| `episode_reward` | inferred at runtime | `None` |
| `episode_reward_change` | inferred at runtime | `None` |
| `episode_rewards` | inferred at runtime | `[]` |
| `gae_lambda` | inferred at runtime | `gae_lambda` |
| `game_env` | inferred at runtime | `MsPacmanFeatureEnv(render_mode=render_mode, n_features=n_features, projection_seed=projection_seed, projection_scale=projection_s…` |
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
| `last_spatial_magnitude` | inferred at runtime | `0.0` |
| `last_spatial_vector` | inferred at runtime | `np.zeros(2, dtype=np.float32)` |
| `learning_rate` | inferred at runtime | `learning_rate` |
| `max_grad_norm` | inferred at runtime | `max_grad_norm` |
| `max_stim_hz` | inferred at runtime | `float(max_stim_hz)` |
| `max_time` | inferred at runtime | `max_time` |
| `minibatch_size` | inferred at runtime | `minibatch_size` |
| `motor_electrodes` | inferred at runtime | `electrodes_array[self.motor_neurons]` |
| `motor_lookup` | inferred at runtime | `{}` |
| `motor_neurons` | inferred at runtime | `np.asarray(motor_neurons, dtype=int) if motor_neurons is not None else None` |
| `motor_positions` | inferred at runtime | `None` |
| `motor_positions_centered` | inferred at runtime | `None` |
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
| `spatial_deadzone` | inferred at runtime | `float(spatial_deadzone)` |
| `spatial_flip_y` | inferred at runtime | `bool(spatial_flip_y)` |
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
#### `allocate_neurons(n_sequences, n_sensory=8, min_motor=6, min_training=2, max_training=6)`
> Partition sorter sequences into sensory, motor, and training groups.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:48`
#### `configure_from_experiment(self, experiment: Experiment)`
> Bind sorter and mapping, derive physical motor geometry and stimulation electrodes, and initialize selected policies/trainers.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:229`
#### `_create_deferred_trainer(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:327`
#### `customize_environment_params(self, env_params: dict) -> dict`
> Inject concatenated sensory and training electrode IDs into acquisition-environment construction.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:391`
#### `_add_to_buffer(self, new_data, n_channels)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:395`
#### `_get_window(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:419`
#### `_fixed_sigmoid_rates(self, features)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:429`
#### `set_sensory_signal(self, game_features)`
> Convert game features to sensory rates with fixed sigmoid encoding or a learned continuous input policy.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:434`
#### `_update_motor_rates(self, moving_avg=0.2)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:455`
#### `get_motor_signal(self, moving_avg=0.2)`
> Sample a discrete PPO action from smoothed motor activity and stage its rollout transition.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:461`
#### `get_motor_signal_spatial(self, moving_avg=0.2)`
> Convert the activity-weighted physical motor centroid into no-op or one of eight directional game actions.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:473`
#### `_step_sensory_environment(self, buffer_size)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:499`
#### `_process_observation(self, obs, n_channels, step_size_samples, samples_since_last_process)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:517`
#### `_record_policy_transitions(self, reward, done)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:549`
#### `get_training_signal(self, train_Hz=30, stim_count=3, delay_ms=5, use_change=True)`
> Update/select configured trainer stimulation or generate a random sequential pulse.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:572`
#### `_update_policy(self, policy, optimizer, buffer, action_type)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:607`
#### `_update_policies(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:621`
#### `_collect_weights_for_viz(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:647`
#### `run(self, experiment)`
> Execute read/game/train/wait states, online sorting, logging, policy updates, and result creation.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:656`
#### `cleanup(self)`
> Close the game environment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:915`
#### `info(self)`
> Report neuron assignments, timings, episodes, and codec modes.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:918`
### MsPacManPhaseWithViz(MsPacManPhase)
> Publish phase activity, weights, actions, stimulation rates, and episode outcomes to the visualizer.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:932`
**Kind:** class. **Instantiated by:** braindance/examples/mspacman_reinforcement_example.py:21 (named-call hint)
**Constructor:** `__init__(self, *args, enable_viz=True, **kwargs)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `enable_viz` | inferred at runtime | `enable_viz` |
| `visualizer` | inferred at runtime | `None` |
**Methods:**
#### `configure_from_experiment(self, experiment)`
> Run normal phase configuration and create a visualizer sized to the configured sensory and motor groups.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:940`
#### `_action_to_plot_value(self, action)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:952`
#### `get_motor_signal(self, moving_avg=0.2)`
> Decode a PPO action and publish current neural/policy state to the visualizer.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:955`
#### `get_motor_signal_spatial(self, moving_avg=0.2)`
> Decode a spatial action and publish current neural/policy state to the visualizer.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:968`
#### `run(self, experiment)`
> Run the parent phase and publish the final episode/reward update.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:981`
#### `__del__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_pacman.py:993`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Encode mode is `fixed_sigmoid` or `policy_continuous`; decode mode is `ppo` or `spatial`; default maximum stimulation rate is 10 Hz. |
| Spatial decode uses a 0.2 normalized deadzone, optional y-axis flip, and an eight-direction Atari action lookup with zero for no-op. |
| Default neuron allocation requires 16 sequences: 8 sensory, at least 6 motor, and 2-6 training. |

## Data Shapes
- Feature and sensory-rate vectors have length 8; PPO emits scalar actions 0..8.
- Motor positions are N x 2, centered and normalized by maximum radial distance; activity-weighted centroids drive spatial action bins.
- Writes Ms. Pac-Man game/reward/pattern CSVs; returns episode count, optional `SpikeData`, policy state/modes, and rewards.

## Notes
- The visualizer scales an integer action with `action/4 - 1`, a display mapping rather than the game's directional geometry.
- Spatial mode does not populate output PPO transitions; learned input encoding may still update through PPO.
- Training stimulation is indexed into the concatenated sensory-plus-training electrode list.
