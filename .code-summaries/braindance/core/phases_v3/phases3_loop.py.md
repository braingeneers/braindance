# phases3_loop.py

**Path:** `braindance/core/phases_v3/phases3_loop.py`
**Module:** `braindance.core.phases_v3.phases3_loop`
**Feature Area:** `Experiment Phases`
**Entry point:** no — library or imported component

## Overview
Contains a basic spike-triggered stimulation phase and the older CartPole neural reinforcement-learning phase with optional visualization. Both stream raw Maxwell batches through RT-Sort circular windows; CartPole maps pole angle to sensory rates, decodes motor rates through a small policy network, and optionally applies trainer-selected tetanus patterns.

## Connections
- **Used by:** `braindance.examples.2_rapid_pairing` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.cartpole_ml` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.cartpole_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.closed_loop` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.closed_loop_experiment_v3` — import consumer hint; not a proven runtime call.
- **Uses:** `convert_uint16_maxwell` from `braindance.analysis.data_loader` — imports (static evidence).
- **Uses:** `NeuralRLVisualizer` from `braindance.core.phases_v3.cartpole_viz_helper` — imports (static evidence).
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `PhaseV3` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `ContextualTrainer` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `TetanusTrainer` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `generate_tetanus_pattern` from `braindance.core.trainer` — imports (static evidence).
- **Uses:** `cartpole_continuous` from `braindance.games` — imports (static evidence).
- **Uses:** `get_rt_sort_path` from `braindance` — imports (static evidence).
- **Shared data:** Both phases consume experiment RT-Sort/mapping state and control the configured acquisition environment; CartPole imports the continuous game, trainer APIs, and optional `NeuralRLVisualizer`.

## Dependencies
- `braindance.analysis.data_loader.convert_uint16_maxwell` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.cartpole_viz_helper.NeuralRLVisualizer` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.PhaseV3` — intra-repo import; source import evidence.
- `braindance.core.trainer.ContextualTrainer` — intra-repo import; source import evidence.
- `braindance.core.trainer.TetanusTrainer` — intra-repo import; source import evidence.
- `braindance.core.trainer.generate_tetanus_pattern` — intra-repo import; source import evidence.
- `braindance.games.cartpole_continuous` — intra-repo import; source import evidence.
- `braindance.get_rt_sort_path` — intra-repo import; source import evidence.
- `gymnasium` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `spikelab.SpikeData` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.
- `torch.nn` — external or unresolved local import; source import evidence.
- `torch.optim` — external or unresolved local import; source import evidence.

## Classes
### ClosedLoopPhaseV3(PhaseV3)
> Detect a selected sorter neuron online and immediately stimulate one configured electrode while counting detections and pulses.
**Source:** `braindance/core/phases_v3/phases3_loop.py:24`
**Kind:** class. **Instantiated by:** braindance/examples/2_rapid_pairing.py:82 (named-call hint); braindance/examples/closed_loop.py:81 (named-call hint); braindance/examples/closed_loop_experiment_v3.py:187 (named-call hint); braindance/examples/closed_loop_experiment_v3.py:53 (named-call hint)
**Constructor:** `__init__(self, duration: int=600, amp_mv: int=400, phase_width: int=100, refractory_ms: float=2.0, delay_ms: float=1.0, name: str=None, suffix: str='_closed_loop', rt_sort_path: str=None, detection_model_path: str | None=None, verbose: int=1, timing_debug: bool=False, timing_interval: int=100, use_artifact_removal: bool=True, art_removal_N: int=60, art_removal_min_val: float=-100, art_removal_max_val: float=100, art_removal_spike_thresh: List[float]=None, use_numba: bool=True, recording_tag: str='closed_loop')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `amp_mv` | inferred at runtime | `amp_mv` |
| `art_removal_N` | inferred at runtime | `art_removal_N` |
| `art_removal_max_val` | inferred at runtime | `art_removal_max_val` |
| `art_removal_min_val` | inferred at runtime | `art_removal_min_val` |
| `art_removal_spike_thresh` | inferred at runtime | `[-3.5, 20]` |
| `artifact_remover` | inferred at runtime | `None` |
| `buffer_write_pos` | inferred at runtime | `0` |
| `channels_per_neuron` | inferred at runtime | `None` |
| `data_buffer` | inferred at runtime | `None` |
| `delay_ms` | inferred at runtime | `delay_ms` |
| `detection_model_path` | inferred at runtime | `get_rt_sort_path()` |
| `duration` | inferred at runtime | `duration` |
| `electrodes` | inferred at runtime | `experiment.mapping.get_electrodes(channels=channels)` |
| `electrodes_per_neuron` | inferred at runtime | `None` |
| `input_neuron_id` | inferred at runtime | `self.selected_pair[0]` |
| `output_neuron_id` | inferred at runtime | `self.selected_pair[1]` |
| `phase_width` | inferred at runtime | `phase_width` |
| `refractory_ms` | inferred at runtime | `refractory_ms` |
| `rt_sort` | inferred at runtime | `experiment.data.rt_sort_object` |
| `rt_sort_path` | inferred at runtime | `rt_sort_path` |
| `selected_pair` | inferred at runtime | `None` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `stim_electrodes` | inferred at runtime | `[stim_electrode]` |
| `stim_inds` | inferred at runtime | `np.arange(len(self.stim_electrodes))` |
| `timing_debug` | inferred at runtime | `timing_debug` |
| `timing_interval` | inferred at runtime | `timing_interval` |
| `use_artifact_removal` | inferred at runtime | `use_artifact_removal` |
| `use_numba` | inferred at runtime | `use_numba` |
| `verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `configure_from_experiment(self, experiment: Experiment)`
> Bind the sorter/mapping and translate the selected neuron pair into detection and stimulation roles.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_loop.py:118`
#### `customize_environment_params(self, env_params: dict) -> dict`
> Customize environment parameters.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_loop.py:155`
#### `_add_to_buffer(self, new_data, n_channels)`
> Add new data to the sliding window buffer.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_loop.py:160`
#### `_get_window(self)`
> Get the current window from the buffer in chronological order.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_loop.py:193`
#### `run(self, experiment) -> Dict[str, Any]`
> Warm up RT-Sort, stream overlapping windows, trigger stimulation on selected-neuron events, and return counts plus sorted events.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_loop.py:206`
### PolicyNetwork(nn.Module)
> Map motor-neuron activity to a single bounded CartPole control value for the legacy REINFORCE loop.
**Source:** `braindance/core/phases_v3/phases3_loop.py:381`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/phases3_loop.py:694 (named-call hint)
**Constructor:** `__init__(self, input_size, hidden_size=32)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `fc1` | inferred at runtime | `nn.Linear(input_size, hidden_size)` |
| `fc2` | inferred at runtime | `nn.Linear(hidden_size, 1)` |
| `tanh` | inferred at runtime | `nn.Tanh()` |
**Methods:**
#### `forward(self, x)`
> Apply a ReLU hidden layer and tanh scalar output.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_loop.py:390`
### CartPolePhase(PhaseV3)
> Run the legacy closed-loop CartPole sensory-stimulation, motor-decoding, policy-learning, and training-stimulation state machine.
**Source:** `braindance/core/phases_v3/phases3_loop.py:396`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, sensory_neurons: list | None=None, motor_neurons: list | None=None, training_neurons: list | None=None, amp_mv: int=400, phase_width: int=100, read_period_ms: int=200, train_period_ms: int=200, wait_period_ms: int=400, n_episodes: int=10, trainer=None, trainer_type: str | None=None, artifact_removal=False, continuous=True, minibatch_size=5, verbose=False, max_time=np.inf, normalization=1, suffix: str='_cartpole', rt_sort_path: str=None, detection_model_path: str | None=None, use_numba: bool=True, learning_rate: float=0.01, gamma: float=0.99, policy_hidden_size: int=32, assistive: float=0.0, use_contextual_trainer: bool=False, contextual_trainer_kwargs: dict | None=None, recording_tag: str='cartpole')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `assistive` | inferred at runtime | `assistive` |
| `buffer_write_pos` | inferred at runtime | `0` |
| `contextual_trainer_kwargs` | inferred at runtime | `contextual_trainer_kwargs or {}` |
| `continuous` | inferred at runtime | `continuous` |
| `data_buffer` | inferred at runtime | `np.zeros((window_size_samples, n_channels), dtype=obs.dtype)` |
| `detection_model_path` | inferred at runtime | `get_rt_sort_path()` |
| `electrodes` | inferred at runtime | `None` |
| `episode` | inferred at runtime | `0` |
| `episode_length` | inferred at runtime | `0` |
| `episode_reward` | inferred at runtime | `None` |
| `episode_reward_change` | inferred at runtime | `None` |
| `game_env` | inferred at runtime | `cartpole_continuous.CartPoleContinuousEnv(render_mode='human')` |
| `game_obs` | inferred at runtime | `None` |
| `gamma` | inferred at runtime | `gamma` |
| `last_action_ind` | inferred at runtime | `None` |
| `last_action_inds` | inferred at runtime | `deque(maxlen=self.minibatch_size)` |
| `last_policy_grad_magnitude` | inferred at runtime | `0.0` |
| `learning_rate` | inferred at runtime | `learning_rate` |
| `max_time` | inferred at runtime | `max_time` |
| `minibatch_size` | inferred at runtime | `minibatch_size` |
| `motor_neurons` | inferred at runtime | `np.array(motor_neurons, dtype=int) if motor_neurons is not None else None` |
| `motor_spike_count` | inferred at runtime | `np.zeros(len(self.motor_neurons) if self.motor_neurons is not None else 0)` |
| `motor_spike_rate` | inferred at runtime | `np.zeros(len(self.motor_neurons) if self.motor_neurons is not None else 0)` |
| `n_episodes` | inferred at runtime | `n_episodes` |
| `normalization` | inferred at runtime | `normalization` |
| `optimizer` | inferred at runtime | `None` |
| `policy_hidden_size` | inferred at runtime | `policy_hidden_size` |
| `policy_net` | inferred at runtime | `None` |
| `predicted_time` | inferred at runtime | `(read_period_ms + train_period_ms) * 20 * n_episodes / 1000` |
| `read_period_ms` | inferred at runtime | `read_period_ms` |
| `rewards` | inferred at runtime | `[]` |
| `rt_sort` | inferred at runtime | `None` |
| `rt_sort_path` | inferred at runtime | `rt_sort_path` |
| `saved_log_probs` | inferred at runtime | `[]` |
| `sensory_neurons` | inferred at runtime | `np.array(sensory_neurons, dtype=int) if sensory_neurons is not None else None` |
| `sensory_stim_Hz` | inferred at runtime | `np.zeros(len(self.sensory_neurons) if self.sensory_neurons is not None else 0)` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `state` | inferred at runtime | `'read'` |
| `stim_electrodes` | inferred at runtime | `stim_electrodes.tolist()` |
| `stim_inds` | inferred at runtime | `np.arange(len(self.stim_electrodes))` |
| `suffix` | inferred at runtime | `suffix` |
| `train_period_ms` | inferred at runtime | `train_period_ms` |
| `trainer` | inferred at runtime | `trainer` |
| `trainer_type` | inferred at runtime | `trainer_type` |
| `training_neurons` | inferred at runtime | `np.array(training_neurons, dtype=int) if training_neurons is not None else None` |
| `use_contextual_trainer` | inferred at runtime | `use_contextual_trainer` |
| `use_numba` | inferred at runtime | `use_numba` |
| `verbose` | inferred at runtime | `verbose` |
| `wait_period_ms` | inferred at runtime | `wait_period_ms` |
**Methods:**
#### `allocate_neurons(n_sequences, n_sensory=2, min_motor=4, min_training=2, max_training=6)`
> Partition detected sequences into two sensory, motor, and bounded training groups.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_loop.py:439`
#### `configure_from_experiment(self, experiment: Experiment)`
> Bind sorter/mapping, auto-allocate groups, derive sensory stimulation electrodes, and initialize policy/trainer objects.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_loop.py:623`
#### `_create_deferred_trainer(self)`
> Create trainer after neuron allocation is known (deferred from __init__).
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_loop.py:706`
#### `customize_environment_params(self, env_params: dict) -> dict`
> Customize environment parameters.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_loop.py:761`
#### `_add_to_buffer(self, new_data, n_channels)`
> Add new data to the sliding window buffer.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_loop.py:766`
#### `_get_window(self)`
> Get the current window from the buffer in chronological order.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_loop.py:799`
#### `set_sensory_signal(self, game_env_obs)`
> Encode pole angle as asymmetric stimulation rates on two sensory neurons.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_loop.py:812`
#### `get_motor_signal(self, moving_avg=0.2)`
> Smooth motor counts and obtain a scalar policy action while retaining differentiable outputs for later REINFORCE update.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_loop.py:836`
#### `get_training_signal(self, train_Hz=30, stim_count=3, delay_ms=5, use_change=True)`
> Update/select trainer stimulation or generate a random training-neuron sequence.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_loop.py:883`
#### `run(self, experiment)`
> Execute CartPole read/game/train/wait states with online sorting, stimulation, logging, and policy updates.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_loop.py:947`
#### `update_policy(self)`
> Compute normalized discounted returns, backpropagate the simplified REINFORCE loss, and report gradient magnitude to a contextual trainer.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_loop.py:1307`
#### `_compute_gradient_magnitude(self)`
> Compute L2 norm of the policy network gradients.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_loop.py:1350`
#### `time_elapsed(self)`
> Delegate elapsed-time reporting to the phase base implementation.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_loop.py:1361`
#### `info(self)`
> Report neuron assignments and phase timing/episode configuration.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_loop.py:1364`
#### `predicted_time(self)`
> Returns the predicted time for the phase to run in seconds
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_loop.py:1374`
### CartPolePhasWithViz(CartPolePhase)
> Extend CartPolePhase by publishing motor rates, policy weights, actions, sensory rates, and episode outcomes to the visualizer.
**Source:** `braindance/core/phases_v3/phases3_loop.py:1383`
**Kind:** class. **Instantiated by:** braindance/examples/cartpole_ml.py:60 (named-call hint); braindance/examples/cartpole_reinforcement_example.py:35 (named-call hint)
**Constructor:** `__init__(self, *args, enable_viz=True, **kwargs)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `enable_viz` | inferred at runtime | `enable_viz` |
| `visualizer` | inferred at runtime | `None` |
**Methods:**
#### `configure_from_experiment(self, experiment)`
> Configure and setup visualizer.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_loop.py:1391`
#### `get_motor_signal(self, moving_avg=0.2)`
> Override to include visualization updates.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_loop.py:1406`
#### `run(self, experiment)`
> Override to update visualizer with episode data.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_loop.py:1429`
#### `__del__(self)`
> Cleanup visualizer on deletion.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_loop.py:1446`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `ClosedLoopPhaseV3` defaults to 600 seconds, 400 mV, 100 microsecond phase width, and 20 ms sort windows with 10 ms overlap. |
| `CartPolePhase` defaults to continuous control, 200 ms read/train, 400 ms wait, 10 episodes, Adam learning rate 0.01, and REINFORCE gamma 0.99. |
| CartPole default auto-allocation requires 8 sequences: 2 sensory, at least 4 motor, and 2-6 training. |

## Data Shapes
- Raw acquisition batches and circular windows are samples x channels; sorter events identify neuron ID in element zero.
- `PolicyNetwork` maps a motor-rate vector through a 32-unit hidden layer to one tanh action.
- Closed-loop returns counts/file/`SpikeData`; CartPole logs game/reward/pattern CSVs and intends to return episode count, sorted data, and policy state.

## Notes
- Closed-loop stimulation targets the electrode of `input_neuron_id` when spikes are detected from `output_neuron_id`, despite the class description saying input activity maps to output stimulation.
- Artifact-removal and refractory/delay parameters are stored but unused in `ClosedLoopPhaseV3.run`.
- `CartPolePhase.run` returns early with bare `return` on its normal episode/time limit, skipping log closure and the documented result dictionary.
- `CartPolePhase.predicted_time` method collides with the instance attribute of the same name and would be shadowed.
