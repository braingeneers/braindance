# phases3.py

**Path:** `braindance/core/phases_v3/phases3.py`
**Module:** `braindance.core.phases_v3.phases3`
**Feature Area:** `Experiment Phases`
**Entry point:** no — library or imported component

## Overview
Implements timed recording, amplitude/electrode sweeps, and fixed-frequency stimulation phases with replay-aware stepping. Also declares a CartPole phase scaffold that configures neuron groups but does not implement run.

## Connections
- **Used by:** `braindance.cli.replay` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.2_rapid_pairing` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.ant_cpg_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.ant_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.cartpole_ml` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.cartpole_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.closed_loop` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.closed_loop_experiment_v3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.foodland_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.mspacman_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.replay_experiment_v3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.simple_experiment_v3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.walker2d_cpg_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.walker2d_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Uses:** `BaseEnv` from `braindance.core.base_env` — imports (static evidence).
- **Uses:** `AnalysisPhaseV3` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `PhaseV3` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Shared data:** Experiment creates MaxwellEnv; phases call env.step and optional connect_units/disconnect_all.

## Dependencies
- `braindance.core.base_env.BaseEnv` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.AnalysisPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.PhaseV3` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### RecordPhaseV3(PhaseV3)
> Phase for recording spontaneous activity.
**Source:** `braindance/core/phases_v3/phases3.py:17`
**Kind:** class. **Instantiated by:** braindance/cli/replay.py:224 (named-call hint); braindance/examples/2_rapid_pairing.py:73 (named-call hint); braindance/examples/2_rapid_pairing.py:87 (named-call hint); braindance/examples/ant_cpg_example.py:113 (named-call hint); braindance/examples/ant_reinforcement_example.py:32 (named-call hint); braindance/examples/cartpole_ml.py:44 (named-call hint); braindance/examples/cartpole_reinforcement_example.py:29 (named-call hint); braindance/examples/closed_loop.py:72 (named-call hint); braindance/examples/closed_loop.py:86 (named-call hint); braindance/examples/closed_loop_experiment_v3.py:183 (named-call hint); braindance/examples/closed_loop_experiment_v3.py:36 (named-call hint); braindance/examples/closed_loop_experiment_v3.py:62 (named-call hint); braindance/examples/foodland_reinforcement_example.py:22 (named-call hint); braindance/examples/mspacman_reinforcement_example.py:18 (named-call hint); braindance/examples/replay_experiment_v3.py:37 (named-call hint); braindance/examples/simple_experiment_v3.py:112 (named-call hint); braindance/examples/simple_experiment_v3.py:29 (named-call hint); braindance/examples/simple_experiment_v3.py:62 (named-call hint); braindance/examples/walker2d_cpg_example.py:126 (named-call hint); braindance/examples/walker2d_reinforcement_example.py:128 (named-call hint)
**Constructor:** `__init__(self, duration: int=60, name: str=None, suffix: str='_recording', recording_tag: str='rec')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `duration` | inferred at runtime | `duration` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `verbose` | inferred at runtime | `False` |
**Methods:**
#### `configure_from_experiment(self, experiment)`
> Auto-configure from experiment config.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3.py:34`
#### `run(self, experiment) -> Dict[str, Any]`
> Execute recording.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3.py:42`
#### `info(self) -> Dict[str, Any]`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3.py:71`
### NeuralSweepPhaseV3(PhaseV3)
> Sweep stimulation amplitude to find response thresholds.
**Source:** `braindance/core/phases_v3/phases3.py:79`
**Kind:** class. **Instantiated by:** braindance/examples/simple_experiment_v3.py:32 (named-call hint)
**Constructor:** `__init__(self, neuron_list: List[int]=None, amp_bounds: tuple=(150, 400, 10), stim_freq: float=1.0, replicates: int=30, phase_length: int=100, order: str='ran', single_connect: bool=False, tag: str='neural_sweep', name: str=None, suffix: str='_sweep', recording_tag: str='sweep')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `amp_bounds` | inferred at runtime | `amp_bounds` |
| `amplitude_end` | inferred at runtime | `amp_bounds` |
| `amplitude_start` | inferred at runtime | `amp_bounds` |
| `last_neuron` | inferred at runtime | `stim_commands[0][0][0]` |
| `n_amplitudes` | inferred at runtime | `1` |
| `neuron_list` | inferred at runtime | `neuron_list` |
| `order` | inferred at runtime | `order` |
| `phase_length` | inferred at runtime | `phase_length` |
| `replicates` | inferred at runtime | `replicates` |
| `single_connect` | inferred at runtime | `single_connect` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `stim_freq` | inferred at runtime | `stim_freq` |
| `tag` | inferred at runtime | `tag` |
| `verbose` | inferred at runtime | `False` |
**Methods:**
#### `configure_from_experiment(self, experiment)`
> Auto-configure from experiment.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3.py:127`
#### `customize_environment_params(self, env_params: dict) -> dict`
> Customize environment parameters.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3.py:138`
#### `generate_stim_commands(self) -> List[tuple]`
> Generate stimulation commands for the sweep.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3.py:143`
#### `run(self, experiment) -> Dict[str, Any]`
> Execute amplitude sweep.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3.py:177`
#### `info(self) -> Dict[str, Any]`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3.py:247`
### FrequencyStimPhaseV3(PhaseV3)
> Phase for stimulating at a specific frequency.
**Source:** `braindance/core/phases_v3/phases3.py:260`
**Kind:** class. **Instantiated by:** braindance/cli/replay.py:226 (named-call hint); braindance/examples/replay_experiment_v3.py:38 (named-call hint); braindance/examples/simple_experiment_v3.py:41 (named-call hint); braindance/examples/simple_experiment_v3.py:47 (named-call hint); braindance/examples/simple_experiment_v3.py:53 (named-call hint)
**Constructor:** `__init__(self, stim_command: Union[tuple, List[tuple]], stim_freq: float=1.0, duration: int=60, tag: Union[str, List[str]]='frequency_stim', name: str=None, suffix: str='_freq_stim', recording_tag: str='freq_stim')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `duration` | inferred at runtime | `duration` |
| `single_command` | inferred at runtime | `not isinstance(stim_command[0][0], list)` |
| `single_tag` | inferred at runtime | `isinstance(tag, str)` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `stim_command` | inferred at runtime | `stim_command` |
| `stim_freq` | inferred at runtime | `stim_freq` |
| `tag` | inferred at runtime | `tag` |
| `verbose` | inferred at runtime | `False` |
**Methods:**
#### `configure_from_experiment(self, experiment)`
> Auto-configure from experiment.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3.py:300`
#### `customize_environment_params(self, env_params: dict) -> dict`
> Customize environment parameters.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3.py:308`
#### `run(self, experiment) -> Dict[str, Any]`
> Execute frequency stimulation.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3.py:313`
#### `info(self) -> Dict[str, Any]`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3.py:379`
### CartPolePhaseV3(PhaseV3)
> Phase for CartPole game with neural control.
**Source:** `braindance/core/phases_v3/phases3.py:389`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, n_episodes: int=10, amp_mv: int=400, phase_width: int=100, read_period_ms: int=200, train_period_ms: int=200, wait_period_ms: int=400, trainer=None, artifact_removal: bool=False, continuous: bool=True, assistive: float=0.0, minibatch_size: int=5, spike_thresh: List[float]=None, max_time: float=np.inf, normalization: float=1.0, force_train: bool=False, name: str=None, suffix: str='_cartpole', recording_tag: str='cartpole')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `amp_mv` | inferred at runtime | `amp_mv` |
| `artifact_removal` | inferred at runtime | `artifact_removal` |
| `assistive` | inferred at runtime | `assistive` |
| `continuous` | inferred at runtime | `continuous` |
| `force_train` | inferred at runtime | `force_train` |
| `max_time` | inferred at runtime | `max_time` |
| `minibatch_size` | inferred at runtime | `minibatch_size` |
| `motor_neurons` | inferred at runtime | `None` |
| `n_episodes` | inferred at runtime | `n_episodes` |
| `normalization` | inferred at runtime | `normalization` |
| `phase_width` | inferred at runtime | `phase_width` |
| `read_period_ms` | inferred at runtime | `read_period_ms` |
| `sensory_neurons` | inferred at runtime | `None` |
| `spike_thresh` | inferred at runtime | `spike_thresh or [-3.1, -20]` |
| `train_period_ms` | inferred at runtime | `train_period_ms` |
| `trainer` | inferred at runtime | `trainer` |
| `training_neurons` | inferred at runtime | `None` |
| `verbose` | inferred at runtime | `False` |
| `wait_period_ms` | inferred at runtime | `wait_period_ms` |
**Methods:**
#### `configure_from_experiment(self, experiment)`
> Auto-configure from experiment data and config.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3.py:453`
#### `needs_environment(self) -> bool`
> CartPole needs Maxwell environment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3.py:473`
#### `close_environment_after(self) -> bool`
> Keep environment open for potential follow-up phases.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3.py:477`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Record duration=60s; sweep amp_bounds=(150,400,10),stim_freq=1,replicates=30,phase_length=100,order ran/arn/nar/random,single_connect. |
| FrequencyStim accepts single/repeated command or finite command list and matching tags; recording_tag creates numbered recording folder. |

## Data Shapes
- Record returns recording_file/duration; sweep returns list of time/neuron/amplitude/stim_count and sweep_file; frequency phase returns stim_file/stim_count.

## Notes
- Stimulation neuron values index environment stim_units; replay chunks shrink to schedule deadlines.
- CartPolePhaseV3 remains abstract because run is missing; use implemented loop/game phases instead.
