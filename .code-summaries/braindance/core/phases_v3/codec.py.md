# codec.py

**Path:** `braindance/core/phases_v3/codec.py`
**Module:** `braindance.core.phases_v3.codec`
**Feature Area:** `Games and Reinforcement`
**Entry point:** no — library or imported component

## Overview
Provides shared input-stimulation and output-action actor-critic policies for neural game phases. Implements rollout buffering, generalized advantage estimation, and clipped PPO updates with gradient clipping.

## Connections
- **Used by:** `braindance.core.phases_v3.phases3_ant` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_ant_cpg` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_foodland` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_pacman` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_walker2d` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_walker2d_cpg` — import consumer hint; not a proven runtime call.
- **Shared data:** Game phase modules import codec policies and ppo_update.

## Dependencies
- `numpy` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.
- `torch.distributions.Categorical` — external or unresolved local import; source import evidence.
- `torch.distributions.Normal` — external or unresolved local import; source import evidence.
- `torch.nn` — external or unresolved local import; source import evidence.

## Classes
### PPORolloutBuffer()
> Simple rollout storage for PPO-style updates.
**Source:** `braindance/core/phases_v3/codec.py:25`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/phases3_ant.py:240 (named-call hint); braindance/core/phases_v3/phases3_ant.py:241 (named-call hint); braindance/core/phases_v3/phases3_ant_cpg.py:256 (named-call hint); braindance/core/phases_v3/phases3_foodland.py:336 (named-call hint); braindance/core/phases_v3/phases3_foodland.py:337 (named-call hint); braindance/core/phases_v3/phases3_pacman.py:220 (named-call hint); braindance/core/phases_v3/phases3_pacman.py:221 (named-call hint); braindance/core/phases_v3/phases3_walker2d.py:293 (named-call hint); braindance/core/phases_v3/phases3_walker2d.py:294 (named-call hint); braindance/core/phases_v3/phases3_walker2d_cpg.py:271 (named-call hint)
**Constructor:** `__init__(self)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `actions` | inferred at runtime | `[]` |
| `dones` | inferred at runtime | `[]` |
| `log_probs` | inferred at runtime | `[]` |
| `rewards` | inferred at runtime | `[]` |
| `states` | inferred at runtime | `[]` |
| `values` | inferred at runtime | `[]` |
**Methods:**
#### `add(self, state, action, log_prob, value, reward, done)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/codec.py:31`
#### `clear(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/codec.py:39`
#### `__len__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/codec.py:47`
### InputPolicyActorCritic(nn.Module)
> Map game-state vectors to stochastic per-electrode stimulation rates.
**Source:** `braindance/core/phases_v3/codec.py:55`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/phases3_ant.py:310 (named-call hint); braindance/core/phases_v3/phases3_foodland.py:422 (named-call hint); braindance/core/phases_v3/phases3_pacman.py:297 (named-call hint); braindance/core/phases_v3/phases3_walker2d.py:363 (named-call hint)
**Constructor:** `__init__(self, input_size, action_size, hidden_size=64)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `actor` | inferred at runtime | `nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_siz…` |
| `critic` | inferred at runtime | `nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_siz…` |
| `log_std` | inferred at runtime | `nn.Parameter(torch.full((action_size,), -0.5))` |
**Methods:**
#### `latent_to_rates(self, latent_action, max_stim_hz)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/codec.py:80`
#### `act(self, state_tensor, max_stim_hz, deterministic=False)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/codec.py:83`
#### `evaluate(self, states, actions)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/codec.py:100`
### DiscreteOutputPPO(nn.Module)
> Categorical PPO policy for discrete action spaces (e.g. Ms. Pac-Man).
**Source:** `braindance/core/phases_v3/codec.py:114`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/phases3_pacman.py:308 (named-call hint)
**Constructor:** `__init__(self, input_size, action_size, hidden_size=64)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `actor` | inferred at runtime | `nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_siz…` |
| `critic` | inferred at runtime | `nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_siz…` |
**Methods:**
#### `act(self, state_tensor, deterministic=False)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/codec.py:134`
#### `evaluate(self, states, actions)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/codec.py:144`
### ContinuousOutputPPO(nn.Module)
> Gaussian PPO policy for continuous action spaces (e.g. Ant).
**Source:** `braindance/core/phases_v3/codec.py:153`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/phases3_ant.py:321 (named-call hint); braindance/core/phases_v3/phases3_ant_cpg.py:326 (named-call hint); braindance/core/phases_v3/phases3_foodland.py:433 (named-call hint)
**Constructor:** `__init__(self, input_size, action_size, hidden_size=64)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `actor` | inferred at runtime | `nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_siz…` |
| `critic` | inferred at runtime | `nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_siz…` |
| `log_std` | inferred at runtime | `nn.Parameter(torch.full((action_size,), -0.5))` |
**Methods:**
#### `act(self, state_tensor, deterministic=False)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/codec.py:174`
#### `evaluate(self, states, actions)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/codec.py:192`

## Functions
### `sigmoid(x)`
> unclear — see source
> **Called by:** braindance/core/phases_v3/phases3_ant.py:441 (named-call hint); braindance/core/phases_v3/phases3_foodland.py:558 (named-call hint); braindance/core/phases_v3/phases3_foodland.py:90 (named-call hint); braindance/core/phases_v3/phases3_pacman.py:432 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/codec.py:17`
### `compute_gae(rewards, values, dones, gamma, gae_lambda)`
> Compute Generalized Advantage Estimation.
> **Called by:** braindance/core/phases_v3/codec.py:267 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/codec.py:209`
### `compute_gradient_magnitude(module)`
> Compute L2 norm of gradients across all parameters.
> **Called by:** braindance/core/phases_v3/codec.py:317 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/codec.py:232`
### `ppo_update(policy, optimizer, buffer, action_type, gamma, gae_lambda, clip_epsilon, ppo_epochs, ppo_minibatch_size, entropy_coef, value_coef, max_grad_norm)`
> Train policy/value heads from buffered transitions and clear rollout.
> **Called by:** braindance/core/phases_v3/phases3_ant.py:623 (named-call hint); braindance/core/phases_v3/phases3_ant.py:631 (named-call hint); braindance/core/phases_v3/phases3_ant_cpg.py:638 (named-call hint); braindance/core/phases_v3/phases3_foodland.py:773 (named-call hint); braindance/core/phases_v3/phases3_foodland.py:784 (named-call hint); braindance/core/phases_v3/phases3_pacman.py:608 (named-call hint); braindance/core/phases_v3/phases3_walker2d.py:691 (named-call hint); braindance/core/phases_v3/phases3_walker2d.py:699 (named-call hint); braindance/core/phases_v3/phases3_walker2d_cpg.py:688 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/codec.py:241`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Network hidden_size=64; PPO gamma/gae_lambda/clip_epsilon/epochs/minibatch/entropy/value/max_grad_norm supplied by caller. |

## Data Shapes
- Input policy Gaussian latent vector→sigmoid-scaled stimulation rates; discrete output categorical action; continuous output tanh action vector.
- PPORolloutBuffer stores state/action/log_prob/value/reward/done; ppo_update returns final gradient norm and clears buffer.

## Notes
- GAE uses zero bootstrap at rollout end; training tensors are constructed on CPU.
- Continuous output corrects log probabilities for tanh; entropy uses unsquashed Normal.
