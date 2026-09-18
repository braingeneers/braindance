# phases2_loop.py

**Path:** `braindance/core/phases2_loop.py`
**Module:** `braindance.core.phases2_loop`
**Feature Area:** `Experiment Phases`
**Entry point:** no — library or imported component

## Overview
Contains a preliminary input-to-stimulation closed-loop phase plus legacy direct and neural Ant phases. Ant feedback alternates neural acquisition, eight-dimensional actions, training pulses, and waits with optional PPO updates.

## Connections
- **Uses:** `base_env` from `braindance.core` — imports (static evidence).
- **Uses:** `Phase` from `braindance.core.phases2` — imports (static evidence).
- **Uses:** `generate_stimulations` from `braindance.core.stim_commands` — imports (static evidence).
- **Uses:** `LinearArtifactRemoval` from `braindance.utils.rt_linear_art_removal` — imports (static evidence).
- **Shared data:** Uses phases2.Phase, generate_stimulations, and LinearArtifactRemoval; neural Ant critic includes its hidden layer.

## Dependencies
- `braindance.core.base_env` — intra-repo import; source import evidence.
- `braindance.core.phases2.Phase` — intra-repo import; source import evidence.
- `braindance.core.stim_commands.generate_stimulations` — intra-repo import; source import evidence.
- `braindance.utils.rt_linear_art_removal.LinearArtifactRemoval` — intra-repo import; source import evidence.
- `gymnasium` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `proj.cartpole_v2.experiment` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.
- `torch.distributions.Normal` — external or unresolved local import; source import evidence.
- `torch.nn` — external or unresolved local import; source import evidence.
- `torch.optim` — external or unresolved local import; source import evidence.

## Classes
### ClosedLoopPhase(Phase)
> Phase for a base closed-loop experiment
**Source:** `braindance/core/phases2_loop.py:19`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, env: base_env.BaseEnv, input_neurons: list, output_neurons: list, experiment: experiment.Experiment=None, amp_mv: int=400, phase_width: int=100, read_period_ms: int=200, train_period_ms: int=200, wait_period_ms: int=400, n_episodes: int=10, trainer=None, artifact_removal=False, minibatch_size=5, spike_thresh=[-3.1, -20], verbose=False, max_time=np.inf, suffix: str='_closed_loop')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `artifact_remover` | inferred at runtime | `LinearArtifactRemoval(n_channels=len(self.motor_neurons), N=60, nc_start=60, min_val=-85, max_val=85, spike_thresh=[-3.1, -25])` |
| `env` | inferred at runtime | `env` |
| `episode` | inferred at runtime | `0` |
| `input_neurons` | inferred at runtime | `np.array(input_neurons)` |
| `input_stim_Hz` | inferred at runtime | `np.zeros(len(self.input_neurons))` |
| `max_time` | inferred at runtime | `max_time` |
| `minibatch_size` | inferred at runtime | `minibatch_size` |
| `motor_spike_rate` | inferred at runtime | `moving_avg * self.motor_spike_rate + (1 - moving_avg) * self.motor_spike_count` |
| `n_episodes` | inferred at runtime | `n_episodes` |
| `output_neurons` | inferred at runtime | `np.array(output_neurons)` |
| `output_spike_count` | inferred at runtime | `np.zeros(len(self.output_neurons))` |
| `output_spike_rate` | inferred at runtime | `np.zeros(len(self.output_neurons))` |
| `predicted_time` | inferred at runtime | `0` |
| `read_period_ms` | inferred at runtime | `read_period_ms` |
| `spike_count` | inferred at runtime | `np.zeros(len(self.output_neurons))` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `state` | inferred at runtime | `'read'` |
| `suffix` | inferred at runtime | `suffix` |
| `train_period_ms` | inferred at runtime | `train_period_ms` |
| `verbose` | inferred at runtime | `verbose` |
| `wait_period_ms` | inferred at runtime | `wait_period_ms` |
**Methods:**
#### `get_input_signal(self, obs)`
> Maps the obs to a stimulation command We use the input_output_map to find which output electrodes to stimulate
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2_loop.py:116`
#### `get_motor_signal(self, moving_avg=0.2)`
> Readout from the read/motor neurons set The action returned should be an integer {0,1} corresponding to the action
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2_loop.py:133`
#### `run(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2_loop.py:152`
#### `time_elapsed(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2_loop.py:200`
#### `info(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2_loop.py:203`
#### `predicted_time(self)`
> Returns the predicted time for the phase to run in seconds
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2_loop.py:213`
### AntPhase(Phase)
> Phase for the Ant game
**Source:** `braindance/core/phases2_loop.py:218`
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
**Source:** `braindance/core/phases2_loop.py:315`
#### `get_training_signal(self, train_Hz=30, stim_count=3, delay_ms=5)`
> The stimulation on the training neurons
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2_loop.py:343`
#### `run(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2_loop.py:364`
#### `time_elapsed(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2_loop.py:533`
#### `info(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2_loop.py:536`
#### `predicted_time(self)`
> Returns the predicted time for the phase to run in seconds
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2_loop.py:545`
### NeuralAntPhase(AntPhase)
> Neural network version of AntPhase that uses a neural network for action selection. The network maps from motor spike rates to action outputs (between -1 and 1).
**Source:** `braindance/core/phases2_loop.py:550`
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
| `gamma` | inferred at runtime | `gamma` |
| `hidden` | inferred at runtime | `nn.Sequential(*layers)` |
| `input_norm` | inferred at runtime | `nn.LayerNorm(input_dim)` |
| `learning_rate` | inferred at runtime | `learning_rate` |
| `log_std_head` | inferred at runtime | `nn.Sequential(nn.Linear(prev_dim, 8), nn.Hardtanh(-2, 0))` |
| `max_memory_size` | inferred at runtime | `10000` |
| `mean_head` | inferred at runtime | `nn.Sequential(nn.Linear(prev_dim, 8), nn.Tanh())` |
| `memory` | inferred at runtime | `[]` |
| `motor_spike_count` | inferred at runtime | `np.zeros(len(self.motor_neurons))` |
| `motor_spike_rate` | inferred at runtime | `moving_avg * self.motor_spike_rate + (1 - moving_avg) * self.motor_spike_count` |
| `network_arch` | inferred at runtime | `network_arch` |
| `output` | inferred at runtime | `nn.Linear(prev_dim, 1)` |
| `shared` | inferred at runtime | `nn.Sequential(*layers)` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `state` | inferred at runtime | `'game'` |
| `train_iters` | inferred at runtime | `train_iters` |
**Methods:**
#### `_build_actor(self)`
> Build a more stable actor network
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2_loop.py:625`
#### `_build_critic(self)`
> Build a more stable critic network
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2_loop.py:680`
#### `get_motor_signal(self, moving_avg=0.3)`
> Readout from the read/motor neurons set using neural network The action returned should be a numpy array of shape (8,) with values between -1 and 1
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2_loop.py:724`
#### `_compute_gae(self, rewards, values, next_values, dones)`
> Compute Generalized Advantage Estimation
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2_loop.py:814`
#### `_update_networks(self)`
> Update actor and critic networks using PPO with robust safeguards
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2_loop.py:835`
#### `run(self)`
> Override run method to include neural network updates
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases2_loop.py:945`
#### `time_elapsed(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2_loop.py:1122`
#### `info(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2_loop.py:1125`
#### `predicted_time(self)`
> Returns the predicted time for the phase to run in seconds
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases2_loop.py:1134`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| input_neurons/output_neurons; pulse amplitude and width; read/train/wait periods |
| Ant dummy_mode and neural learning_rate/gamma/gae_lambda/clip_ratio/train_iters/batch_size |

## Data Shapes
- Ant actions shape (8,); input_output_map is indexed by active input positions to build stimulation tuples.

## Notes
- ClosedLoopPhase references undefined motor_neurons/process_observation and requires externally supplied input_output_map.
- Imports project-specific proj.cartpole_v2.experiment; retained Ant implementation differs from phases2 sensory-network version.
