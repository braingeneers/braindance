# phases.py

**Path:** `braindance/core/phases.py`
**Module:** `braindance.core.phases`
**Feature Area:** `Experiment Phases`
**Entry point:** no — library or imported component

## Overview
Implements the original phase manager, spontaneous recording, amplitude sweeps, frequency stimulation, and CartPole feedback experiments. The manager groups acquisition phases into recordings and passes an AnalysisDAO through analysis phases.

## Connections
- **Used by:** `braindance.core.phases_analysis` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_analysis_2` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_env` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.3_busybee` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.paper_cartpole.causal` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.paper_cartpole.recording` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.amplitude_sweep` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.cartpole_continuous` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.cartpole_fast` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.cartpole_phase` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.cartpole_schedule` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.causal_connectivity` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.causal_tetanus_phases` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.multi_freq_stim` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.neural_config_analysis` — import consumer hint; not a proven runtime call.
- **Uses:** `ArtifactRemoval` from `braindance.core.artifact_removal` — imports (static evidence).
- **Uses:** `base_env` from `braindance.core` — imports (static evidence).
- **Uses:** `AnalysisPhase` from `braindance.core.phases_analysis` — imports (static evidence).
- **Uses:** `HeatmapPhase` from `braindance.core.phases_analysis` — imports (static evidence).
- **Uses:** `cartpole_continuous` from `braindance.games` — imports (static evidence).
- **Shared data:** PhaseManager imports phases_analysis; CartPolePhase maps spike counts to game actions and trainer-selected pulses back to MaxwellEnv.

## Dependencies
- `braindance.core.artifact_removal.ArtifactRemoval` — intra-repo import; source import evidence.
- `braindance.core.base_env` — intra-repo import; source import evidence.
- `braindance.core.phases_analysis.AnalysisPhase` — intra-repo import; source import evidence.
- `braindance.core.phases_analysis.HeatmapPhase` — intra-repo import; source import evidence.
- `braindance.games.cartpole_continuous` — intra-repo import; source import evidence.
- `gymnasium` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### Phase()
> Base class for all phases
**Source:** `braindance/core/phases.py:25`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, env: base_env.BaseEnv)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `env` | inferred at runtime | `env` |
| `provides` | inferred at runtime | `[]` |
| `requires` | inferred at runtime | `[]` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
**Methods:**
#### `run(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:35`
#### `validate(self, experiment=None)`
> Validates the phase to make sure it can run
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:38`
#### `time_elapsed(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:42`
#### `predicted_time(self)`
> Returns the predicted time for the phase to run in seconds
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:45`
#### `info(self)`
> Returns a dictionary of information about the phase
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:49`
#### `cleanup(self)`
> Cleans up the phase after it is done
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:53`
### PhaseManager()
> Manages phases of an experiment
**Source:** `braindance/core/phases.py:58`
**Kind:** class. **Instantiated by:** braindance/examples/3_busybee.py:35 (named-call hint); braindance/examples/paper_cartpole/causal.py:67 (named-call hint); braindance/examples/paper_cartpole/recording.py:50 (named-call hint); braindance/experiments/amplitude_sweep.py:26 (named-call hint); braindance/experiments/cartpole_continuous.py:82 (named-call hint); braindance/experiments/cartpole_fast.py:56 (named-call hint); braindance/experiments/cartpole_phase.py:32 (named-call hint); braindance/experiments/cartpole_schedule.py:53 (named-call hint); braindance/experiments/causal_connectivity.py:27 (named-call hint); braindance/experiments/causal_tetanus_phases.py:62 (named-call hint); braindance/experiments/multi_freq_stim.py:37 (named-call hint); braindance/experiments/neural_config_analysis.py:32 (named-call hint)
**Constructor:** `__init__(self, env: base_env.BaseEnv, verbose=False)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `analysis_dao` | inferred at runtime | `None` |
| `env` | inferred at runtime | `env` |
| `filenames` | inferred at runtime | `[]` |
| `phases` | inferred at runtime | `[]` |
| `save_dir` | inferred at runtime | `self.env.save_dir` |
| `verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `add_phase(self, phase: Phase)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:73`
#### `add_phase_group(self, phase_group: list)`
> Adds a group of phases to the manager, each group will belong to the same save file
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:76`
#### `log_summary(self)`
> Logs the summary of the experiment to a text file
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/phases.py:83`
#### `log_phase(self, phase)`
> Appends the phase and filename to the log file
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:91`
#### `run(self)`
> Reset acquisition between phase groups, run analysis against prior recordings, and close the environment.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases.py:108`
#### `summary(self)`
> Returns a summary of the experiment as a string
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:167`
### RecordPhase(Phase)
> Phase for recording spontaneous activity
**Source:** `braindance/core/phases.py:206`
**Kind:** class. **Instantiated by:** braindance/examples/3_busybee.py:39 (named-call hint); braindance/examples/paper_cartpole/recording.py:43 (named-call hint); braindance/experiments/causal_tetanus_phases.py:32 (named-call hint); braindance/experiments/causal_tetanus_phases.py:33 (named-call hint); braindance/experiments/causal_tetanus_phases.py:34 (named-call hint); braindance/experiments/causal_tetanus_phases.py:35 (named-call hint); braindance/experiments/neural_config_analysis.py:25 (named-call hint)
**Constructor:** `__init__(self, env: base_env.BaseEnv=None, duration: int=10)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `duration` | inferred at runtime | `duration` |
| `predicted_time` | inferred at runtime | `duration` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
**Methods:**
#### `run(self, env=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases.py:222`
#### `info(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:233`
### NeuralSweepPhase(Phase)
> Sweep the amplitude of a stimulation to find the minimum amplitude that elicits a spike
**Source:** `braindance/core/phases.py:241`
**Kind:** class. **Instantiated by:** braindance/examples/3_busybee.py:40 (named-call hint); braindance/examples/paper_cartpole/causal.py:52 (named-call hint); braindance/experiments/amplitude_sweep.py:18 (named-call hint); braindance/experiments/causal_connectivity.py:19 (named-call hint); braindance/experiments/causal_tetanus_phases.py:43 (named-call hint)
**Constructor:** `__init__(self, env: base_env.BaseEnv, neuron_list: list, amp_bounds=[150, 150, 1], stim_freq: float=1, replicates=30, phase_length: int=100, order='ran', single_connect=False, verbose=False, tag='neural_sweep')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `amplitude_end` | inferred at runtime | `amp_bounds[1]` |
| `amplitude_start` | inferred at runtime | `amp_bounds[0]` |
| `last_neuron` | inferred at runtime | `stim_commands[0][0][0]` |
| `n_amplitudes` | inferred at runtime | `amp_bounds[2]` |
| `neuron_list` | inferred at runtime | `neuron_list` |
| `order` | inferred at runtime | `order` |
| `phase_length` | inferred at runtime | `phase_length` |
| `predicted_time` | inferred at runtime | `self.n_amplitudes * len(neuron_list) * 1 / stim_freq * replicates` |
| `replicates` | inferred at runtime | `replicates` |
| `single_connect` | inferred at runtime | `single_connect` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `stim_freq` | inferred at runtime | `stim_freq` |
| `tag` | inferred at runtime | `tag` |
| `verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `generate_stim_commands(self)`
> Enumerate pulse combinations across electrodes, amplitudes, and replicates.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:319`
#### `run(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases.py:368`
#### `info(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:415`
### FrequencyStimPhase(Phase)
> Phase for stimulating a command at a certain frequency
**Source:** `braindance/core/phases.py:430`
**Kind:** class. **Instantiated by:** braindance/experiments/causal_tetanus_phases.py:51 (named-call hint); braindance/experiments/multi_freq_stim.py:27 (named-call hint); braindance/experiments/multi_freq_stim.py:30 (named-call hint)
**Constructor:** `__init__(self, env: base_env.BaseEnv, stim_command: list, stim_freq: float=1, duration: int=10, tag='frequency_stim', verbose=False, connect_units=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `connect_units` | inferred at runtime | `connect_units` |
| `duration` | inferred at runtime | `duration` |
| `predicted_time` | inferred at runtime | `duration` |
| `single_command` | inferred at runtime | `False` |
| `single_tag` | inferred at runtime | `type(self.tag) == str` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `stim_command` | inferred at runtime | `stim_command` |
| `stim_freq` | inferred at runtime | `stim_freq` |
| `tag` | inferred at runtime | `tag` |
| `verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `run(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases.py:482`
#### `info(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:524`
### CartPolePhase(Phase)
> Phase for the cartpole game
**Source:** `braindance/core/phases.py:533`
**Kind:** class. **Instantiated by:** braindance/experiments/cartpole_continuous.py:75 (named-call hint); braindance/experiments/cartpole_fast.py:51 (named-call hint); braindance/experiments/cartpole_phase.py:28 (named-call hint); braindance/experiments/cartpole_schedule.py:49 (named-call hint)
**Constructor:** `__init__(self, env: base_env.BaseEnv, sensory_neurons: list, motor_neurons: list, training_neurons: list, amp_mv: int=400, phase_width: int=100, read_period_ms: int=200, train_period_ms: int=200, wait_period_ms: int=400, n_episodes: int=10, trainer=None, artifact_removal=False, continuous=True, assistive=0.0, minibatch_size=5, spike_thresh=[-3.1, -20], verbose=False, max_time=np.inf, normalization=1, force_train=False)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `artifact_remover` | inferred at runtime | `[]` |
| `assistive` | inferred at runtime | `assistive` |
| `continuous` | inferred at runtime | `continuous` |
| `env` | inferred at runtime | `env` |
| `episode` | inferred at runtime | `0` |
| `episode_reward` | inferred at runtime | `None` |
| `episode_reward_change` | inferred at runtime | `None` |
| `force_train` | inferred at runtime | `force_train` |
| `game_env` | inferred at runtime | `cartpole_continuous.CartPoleContinuousEnv(render_mode='human')` |
| `game_obs` | inferred at runtime | `None` |
| `last_action_ind` | inferred at runtime | `None` |
| `last_action_inds` | inferred at runtime | `deque(maxlen=self.minibatch_size)` |
| `max_time` | inferred at runtime | `max_time` |
| `minibatch_size` | inferred at runtime | `minibatch_size` |
| `motor_neurons` | inferred at runtime | `np.array(motor_neurons)` |
| `motor_spike_count` | inferred at runtime | `np.zeros(len(self.motor_neurons))` |
| `motor_spike_rate` | inferred at runtime | `np.zeros(len(self.motor_neurons))` |
| `n_episodes` | inferred at runtime | `n_episodes` |
| `normalization` | inferred at runtime | `normalization` |
| `predicted_time` | inferred at runtime | `(read_period_ms + train_period_ms) * 20 * n_episodes / 1000` |
| `read_period_ms` | inferred at runtime | `read_period_ms` |
| `sensory_neurons` | inferred at runtime | `np.array(sensory_neurons)` |
| `sensory_stim_Hz` | inferred at runtime | `np.zeros(len(self.sensory_neurons))` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `state` | inferred at runtime | `'read'` |
| `train_period_ms` | inferred at runtime | `train_period_ms` |
| `trainer` | inferred at runtime | `trainer` |
| `training_neurons` | inferred at runtime | `np.array(training_neurons)` |
| `verbose` | inferred at runtime | `verbose` |
| `wait_period_ms` | inferred at runtime | `wait_period_ms` |
**Methods:**
#### `set_sensory_signal(self, game_env_obs)`
> Maps the game observation to the sensory neurons. This observation is of the form: ndarray with shape (4,): - Cart Position, Cart Velocity, Pole Angle, Pole Velocity At Tip
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases.py:650`
#### `get_motor_signal(self, moving_avg=0.2)`
> Smooth two motor-channel counts, normalize their difference, and map it to discrete or continuous cart force.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases.py:673`
#### `get_training_signal(self, train_Hz=30, stim_count=3, delay_ms=5, use_change=True)`
> The stimulation on the training neurons
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:718`
#### `run(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases.py:746`
#### `time_elapsed(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:946`
#### `info(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:949`
#### `predicted_time(self)`
> Returns the predicted time for the phase to run in seconds
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases.py:959`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Recording duration; sweep amplitudes/order/replicates; pulse frequency/width; CartPole read/train/wait periods |

## Data Shapes
- Pulse tuples use stimulation-electrode indices; CartPole motor_neurons are channel indices.
- Writes summary.txt, phase_log.csv, and game/reward/pattern CSV logs.

## Notes
- Legacy CartPole.run asserts raw mode even though its no-artifact branch iterates spike events.
- FrequencyStimPhase consumes command/tag lists with pop; predicted_time is often an instance number shadowing a method.
