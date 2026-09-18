# phases2.py

**Path:** `braindance/core/phases2.py`
**Module:** `braindance.core.phases2`
**Feature Area:** `Experiment Phases`
**Entry point:** no — library or imported component

## Overview
Extends legacy experiment phases with experiment references and recording result dictionaries. Includes CartPole, direct eight-action Ant control, and a neural Ant actor/critic with sensory-network learning.

## Connections
- **Used by:** `braindance.core.phases2_loop` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.paper_cartpole.game` — import consumer hint; not a proven runtime call.
- **Uses:** `ArtifactRemoval` from `braindance.core.artifact_removal` — imports (static evidence).
- **Uses:** `base_env` from `braindance.core` — imports (static evidence).
- **Uses:** `AnalysisPhase` from `braindance.core.phases_analysis` — imports (static evidence).
- **Uses:** `HeatmapPhase` from `braindance.core.phases_analysis` — imports (static evidence).
- **Uses:** `cartpole_continuous` from `braindance.games` — imports (static evidence).
- **Uses:** `LinearArtifactRemoval` from `braindance.utils.rt_linear_art_removal` — imports (static evidence).
- **Shared data:** Uses Maxwell-style env.step and optional LinearArtifactRemoval; Gymnasium Ant-v5 or local CartPole runs between neural read periods.

## Dependencies
- `braindance.core.artifact_removal.ArtifactRemoval` — intra-repo import; source import evidence.
- `braindance.core.base_env` — intra-repo import; source import evidence.
- `braindance.core.phases_analysis.AnalysisPhase` — intra-repo import; source import evidence.
- `braindance.core.phases_analysis.HeatmapPhase` — intra-repo import; source import evidence.
- `braindance.games.cartpole_continuous` — intra-repo import; source import evidence.
- `braindance.utils.rt_linear_art_removal.LinearArtifactRemoval` — intra-repo import; source import evidence.
- `gymnasium` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `proj.cartpole_v2.experiment` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.
- `torch.distributions.Normal` — external or unresolved local import; source import evidence.
- `torch.nn` — external or unresolved local import; source import evidence.
- `torch.optim` — external or unresolved local import; source import evidence.

## Classes
### Phase()
> Base class for all phases
**Source:** `braindance/core/phases2.py:31`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, env: base_env.BaseEnv, experiment=None, suffix='')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `env` | inferred at runtime | `env` |
| `experiment` | inferred at runtime | `experiment` |
| `provides` | inferred at runtime | `[]` |
| `requires` | inferred at runtime | `[]` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `suffix` | inferred at runtime | `suffix` |
**Methods:**
#### `set_env(self, env)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:44`
#### `set_experiment(self, experiment)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:47`
#### `run(self, experiment=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:50`
#### `validate(self, experiment=None)`
> Validates the phase to make sure it can run
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:53`
#### `time_elapsed(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:57`
#### `predicted_time(self)`
> Returns the predicted time for the phase to run in seconds
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:60`
#### `info(self)`
> Returns a dictionary of information about the phase
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:64`
#### `cleanup(self)`
> Cleans up the phase after it is done
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:68`
### PhaseManager()
> Manages phases of an experiment
**Source:** `braindance/core/phases2.py:73`
**Kind:** class. **Instantiated by:** braindance/examples/paper_cartpole/game.py:78 (named-call hint)
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
**Source:** `braindance/core/phases2.py:88`
#### `add_phase_group(self, phase_group: list)`
> Adds a group of phases to the manager, each group will belong to the same save file
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:91`
#### `log_summary(self)`
> Logs the summary of the experiment to a text file
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/phases2.py:98`
#### `log_phase(self, phase)`
> Appends the phase and filename to the log file
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:106`
#### `run(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:118`
#### `summary(self)`
> Returns a summary of the experiment as a string
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:171`
### RecordPhase(Phase)
> Phase for recording spontaneous activity
**Source:** `braindance/core/phases2.py:219`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, env: base_env.BaseEnv=None, duration: int=10, name: str='RecordPhase', suffix: str='_recording', verbose=False)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `duration` | inferred at runtime | `duration` |
| `env` | inferred at runtime | `env` |
| `name` | inferred at runtime | `name` |
| `predicted_time` | inferred at runtime | `duration` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `suffix` | inferred at runtime | `suffix` |
| `verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `set_env(self, env)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:247`
#### `run(self, experiment=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:250`
#### `info(self)`
> Returns a dictionary of information about the phase
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:263`
### NeuralSweepPhase(Phase)
> Sweep the amplitude of a stimulation to find the minimum amplitude that elicits a spike
**Source:** `braindance/core/phases2.py:267`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, env: base_env.BaseEnv=None, neuron_list: list=[], experiment: experiment.Experiment=None, amp_bounds=[150, 150, 1], stim_freq: float=1, replicates=30, phase_length: int=100, order='ran', single_connect=False, verbose=False, tag='neural_sweep', suffix: str='_stim')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `amplitude_end` | inferred at runtime | `amp_bounds[1]` |
| `amplitude_start` | inferred at runtime | `amp_bounds[0]` |
| `env` | inferred at runtime | `experiment.env` |
| `experiment` | inferred at runtime | `experiment` |
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
#### `set_from_experiment(self, experiment: experiment.Experiment)`
> Convert experiment stimulation-electrode count into indices and attach its environment.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:379`
#### `generate_stim_commands(self)`
> Generates the stimulation commands for the amplitude sweep
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:396`
#### `run(self, experiment=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:446`
#### `info(self)`
> Returns a dictionary of information about the phase
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:500`
### FrequencyStimPhase(Phase)
> Phase for stimulating a command at a certain frequency
**Source:** `braindance/core/phases2.py:514`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, env: base_env.BaseEnv, stim_command: list, stim_freq: float=1, duration: int=10, tag='frequency_stim', verbose=False, connect_units=None, suffix: str='_freq-stim', experiment: experiment.Experiment=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `connect_units` | inferred at runtime | `connect_units` |
| `duration` | inferred at runtime | `duration` |
| `env` | inferred at runtime | `experiment.env` |
| `experiment` | inferred at runtime | `experiment` |
| `predicted_time` | inferred at runtime | `duration` |
| `single_command` | inferred at runtime | `False` |
| `single_tag` | inferred at runtime | `type(self.tag) == str` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `stim_command` | inferred at runtime | `stim_command` |
| `stim_freq` | inferred at runtime | `stim_freq` |
| `suffix` | inferred at runtime | `suffix` |
| `tag` | inferred at runtime | `tag` |
| `verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `set_from_experiment(self, experiment: experiment.Experiment)`
> Convert experiment stimulation-electrode count into indices and attach its environment.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:586`
#### `run(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:592`
#### `info(self)`
> Returns a dictionary of information about the phase
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:636`
### CartPolePhase(Phase)
> Phase for the cartpole game
**Source:** `braindance/core/phases2.py:645`
**Kind:** class. **Instantiated by:** braindance/examples/paper_cartpole/game.py:71 (named-call hint)
**Constructor:** `__init__(self, env: base_env.BaseEnv, sensory_neurons: list, motor_neurons: list, training_neurons: list, experiment: experiment.Experiment=None, amp_mv: int=400, phase_width: int=100, read_period_ms: int=200, train_period_ms: int=200, wait_period_ms: int=400, n_episodes: int=10, trainer=None, artifact_removal=False, continuous=True, assistive=0.0, minibatch_size=5, spike_thresh=[-3.1, -20], verbose=False, max_time=np.inf, normalization=1, suffix: str='_cartpole', training_type: str='punishment')`
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
| `suffix` | inferred at runtime | `suffix` |
| `train_period_ms` | inferred at runtime | `train_period_ms` |
| `trainer` | inferred at runtime | `trainer` |
| `training_neurons` | inferred at runtime | `np.array(training_neurons)` |
| `training_type` | inferred at runtime | `training_type` |
| `verbose` | inferred at runtime | `verbose` |
| `wait_period_ms` | inferred at runtime | `wait_period_ms` |
**Methods:**
#### `set_sensory_signal(self, game_env_obs)`
> Maps the game observation to the sensory neurons. This observation is of the form: ndarray with shape (4,): - Cart Position, Cart Velocity, Pole Angle, Pole Velocity At Tip
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:785`
#### `get_motor_signal(self, moving_avg=0.2)`
> Readout from the read/motor neurons set The action returned should be an integer {0,1} corresponding to the action
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:809`
#### `get_training_signal(self, train_Hz=30, stim_count=3, delay_ms=5, use_change=True)`
> The stimulation on the training neurons
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:861`
#### `run(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:899`
#### `time_elapsed(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:1162`
#### `info(self)`
> Returns a dictionary of information about the phase
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:1165`
#### `predicted_time(self)`
> Returns the predicted time for the phase to run in seconds
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:1175`
### AntPhase(Phase)
> Phase for the Ant game
**Source:** `braindance/core/phases2.py:1180`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, env: base_env.BaseEnv, motor_neurons: list, training_neurons: list, experiment: experiment.Experiment=None, amp_mv: int=400, phase_width: int=100, read_period_ms: int=200, train_period_ms: int=200, wait_period_ms: int=400, n_episodes: int=10, trainer=None, artifact_removal=False, verbose=False, max_time=np.inf, dummy_mode: str=None, suffix: str='_ant')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `artifact_remover` | inferred at runtime | `LinearArtifactRemoval(n_channels=len(self.motor_neurons), N=60, nc_start=60, min_val=-85, max_val=85, spike_thresh=[-3.1, -25])` |
| `dummy_mode` | inferred at runtime | `dummy_mode` |
| `env` | inferred at runtime | `env` |
| `episode` | inferred at runtime | `0` |
| `episode_reward` | inferred at runtime | `None` |
| `game_env` | inferred at runtime | `gym.make('Ant-v5', render_mode='human')` |
| `game_obs` | inferred at runtime | `None` |
| `last_action_ind` | inferred at runtime | `None` |
| `max_time` | inferred at runtime | `max_time` |
| `motor_neurons` | inferred at runtime | `np.array(motor_neurons)` |
| `motor_spike_count` | inferred at runtime | `np.zeros(len(self.motor_neurons))` |
| `motor_spike_rate` | inferred at runtime | `np.zeros(len(self.motor_neurons))` |
| `n_episodes` | inferred at runtime | `n_episodes` |
| `predicted_time` | inferred at runtime | `(read_period_ms + train_period_ms) * 20 * n_episodes / 1000` |
| `read_period_ms` | inferred at runtime | `read_period_ms` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `state` | inferred at runtime | `'read'` |
| `suffix` | inferred at runtime | `suffix` |
| `train_period_ms` | inferred at runtime | `train_period_ms` |
| `trainer` | inferred at runtime | `trainer` |
| `training_neurons` | inferred at runtime | `np.array(training_neurons)` |
| `verbose` | inferred at runtime | `verbose` |
| `wait_period_ms` | inferred at runtime | `wait_period_ms` |
**Methods:**
#### `get_motor_signal(self, moving_avg=0.3)`
> Readout from the read/motor neurons set The action returned should be a numpy array of shape (8,) with values between -1 and 1
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:1287`
#### `get_training_signal(self, train_Hz=30, stim_count=3, delay_ms=5)`
> The stimulation on the training neurons
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:1320`
#### `run(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:1343`
#### `time_elapsed(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:1558`
#### `info(self)`
> Returns a dictionary of information about the phase
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:1561`
#### `predicted_time(self)`
> Returns the predicted time for the phase to run in seconds
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:1570`
### NeuralAntPhase(AntPhase)
> Neural network version of AntPhase that uses a neural network for action selection. The network maps from motor spike rates to action outputs (between -1 and 1).
**Source:** `braindance/core/phases2.py:1575`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, env: base_env.BaseEnv, motor_neurons: list, training_neurons: list, network_arch: list=[16], learning_rate: float=0.001, gamma: float=0.99, gae_lambda: float=0.95, clip_ratio: float=0.1, train_iters: int=10, batch_size: int=64, dummy_mode: str=None, **kwargs)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `actor` | inferred at runtime | `self._build_actor().to(self.device)` |
| `actor_optimizer` | inferred at runtime | `optim.Adam(self.actor.parameters(), lr=learning_rate)` |
| `batch_size` | inferred at runtime | `batch_size` |
| `clip_ratio` | inferred at runtime | `clip_ratio` |
| `critic` | inferred at runtime | `self._build_critic().to(self.device)` |
| `critic_optimizer` | inferred at runtime | `optim.Adam(self.critic.parameters(), lr=learning_rate)` |
| `device` | inferred at runtime | `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |
| `dummy_mode` | inferred at runtime | `dummy_mode` |
| `episode_reward` | inferred at runtime | `np.mean(rewards[-5:])` |
| `gae_lambda` | inferred at runtime | `gae_lambda` |
| `game_norm` | inferred at runtime | `nn.LayerNorm(game_state_dim)` |
| `gamma` | inferred at runtime | `gamma` |
| `hidden` | inferred at runtime | `nn.Sequential(*layers)` |
| `input_norm` | inferred at runtime | `nn.LayerNorm(input_dim)` |
| `learning_rate` | inferred at runtime | `learning_rate` |
| `log_std_head` | inferred at runtime | `nn.Sequential(nn.Linear(prev_dim, 8), nn.Hardtanh(-2, 0))` |
| `max_memory_size` | inferred at runtime | `10000` |
| `max_sensory_memory_size` | inferred at runtime | `10000` |
| `mean_head` | inferred at runtime | `nn.Sequential(nn.Linear(prev_dim, 8), nn.Tanh())` |
| `memory` | inferred at runtime | `[]` |
| `motor_norm` | inferred at runtime | `nn.LayerNorm(motor_dim)` |
| `motor_spike_count` | inferred at runtime | `np.zeros(len(self.motor_neurons))` |
| `motor_spike_rate` | inferred at runtime | `moving_avg * self.motor_spike_rate + (1 - moving_avg) * self.motor_spike_count` |
| `network_arch` | inferred at runtime | `network_arch` |
| `output` | inferred at runtime | `nn.Sequential(nn.Linear(prev_dim, 2), nn.Softplus())` |
| `sensory_actor` | inferred at runtime | `self._build_sensory_actor().to(self.device)` |
| `sensory_memory` | inferred at runtime | `[]` |
| `sensory_optimizer` | inferred at runtime | `optim.Adam(self.sensory_actor.parameters(), lr=learning_rate)` |
| `sensory_stim_Hz` | inferred at runtime | `firing_rates.squeeze().cpu().numpy()` |
| `shared` | inferred at runtime | `nn.Sequential(*layers)` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `state` | inferred at runtime | `'game'` |
| `train_iters` | inferred at runtime | `train_iters` |
**Methods:**
#### `_build_sensory_actor(self)`
> Build a neural network for mapping game state and motor rates to sensory signals
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:1666`
#### `set_sensory_signal(self, game_env_obs)`
> Maps the game observation to the sensory neurons. This observation is of the form: ndarray with shape (4,): - Cart Position, Cart Velocity, Pole Angle, Pole Velocity At Tip
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:1729`
#### `_update_sensory_network(self)`
> Update the sensory network using the collected experience
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:1775`
#### `_build_actor(self)`
> Build a more stable actor network
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:1827`
#### `_build_critic(self)`
> Build a more stable critic network
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:1886`
#### `get_motor_signal(self, moving_avg=0.3)`
> Readout from the read/motor neurons set The action returned should be a numpy array of shape (8,) with values between -1 and 1
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:1929`
#### `_compute_gae(self, rewards, values, next_values, dones)`
> Compute Generalized Advantage Estimation
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:2030`
#### `_update_networks(self)`
> Compute GAE and clipped PPO actor/value updates from accumulated episode transitions.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:2051`
#### `run(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2.py:2191`
#### `time_elapsed(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:2419`
#### `info(self)`
> Returns a dictionary of information about the phase
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:2422`
#### `predicted_time(self)`
> Returns the predicted time for the phase to run in seconds
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2.py:2431`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Experiment params stim_electrodes; recording/sweep/frequency options |
| CartPole training_type punishment/reward/always; Ant dummy_mode zeros/random; neural PPO hyperparameters |

## Data Shapes
- Record/sweep return filename and time_elapsed; Ant actions shape (8,) in [-1,1].
- NeuralAnt memory stores state, action, log_prob, value, reward, next_value, done.

## Notes
- NeuralAnt sensory model hardcodes 111 game-state inputs and two outputs; critic forward skips its hidden layer.
- Several episode-limit returns bypass final CSV closes.
- The optional proj.cartpole_v2 import is guarded by TYPE_CHECKING and is not required at runtime.
