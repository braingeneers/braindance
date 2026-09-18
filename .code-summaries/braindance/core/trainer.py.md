# trainer.py

**Path:** `braindance/core/trainer.py`
**Module:** `braindance.core.trainer`
**Feature Area:** `Games and Reinforcement`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Selects and adapts tetanus stimulation patterns using value-based eligibility traces or a contextual policy network. Includes adaptive pattern synthesis, a pole-angle rule, and pattern generation utilities.

## Connections
- **Used by:** `braindance.core.phases_v3.phases3_ant` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_ant_cpg` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_foodland` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_loop` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_pacman` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_walker2d` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_walker2d_cpg` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.paper_cartpole.game` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.cartpole_continuous` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.causal_tetanus_phases` — import consumer hint; not a proven runtime call.
- **Shared data:** Game phases call get_action/update_values and attach themselves as trainer.phase; patterns use Maxwell stim/delay command tuples.

## Dependencies
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.
- `torch.nn` — external or unresolved local import; source import evidence.
- `torch.optim` — external or unresolved local import; source import evidence.

## Classes
### ContextualTrainer()
> Contextual RL trainer that uses an MLP to select stimulation patterns based on the current state of the system.
**Source:** `braindance/core/trainer.py:11`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/phases3_ant.py:380 (named-call hint); braindance/core/phases_v3/phases3_ant_cpg.py:394 (named-call hint); braindance/core/phases_v3/phases3_foodland.py:501 (named-call hint); braindance/core/phases_v3/phases3_loop.py:751 (named-call hint); braindance/core/phases_v3/phases3_pacman.py:375 (named-call hint); braindance/core/phases_v3/phases3_walker2d.py:432 (named-call hint); braindance/core/phases_v3/phases3_walker2d_cpg.py:419 (named-call hint)
**Constructor:** `__init__(self, choices, freqs, hidden_sizes=None, learning_rate=0.01, gamma=0.95, baseline_window=20, normalize_context=True, no_stim_bias=0.0)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `action_history` | inferred at runtime | `[]` |
| `baseline_window` | inferred at runtime | `baseline_window` |
| `choices` | inferred at runtime | `choices` |
| `context_count` | inferred at runtime | `0` |
| `context_dim` | inferred at runtime | `5` |
| `context_mean` | inferred at runtime | `np.zeros(self.context_dim)` |
| `context_var` | inferred at runtime | `np.ones(self.context_dim)` |
| `eligibility_traces` | inferred at runtime | `torch.zeros(self.n_actions)` |
| `episode_rewards` | inferred at runtime | `deque(maxlen=baseline_window)` |
| `freqs` | inferred at runtime | `list(freqs)` |
| `gamma` | inferred at runtime | `gamma` |
| `hidden_sizes` | inferred at runtime | `hidden_sizes` |
| `last_action_idx` | inferred at runtime | `None` |
| `last_action_log_prob` | inferred at runtime | `None` |
| `last_context` | inferred at runtime | `None` |
| `learning_rate` | inferred at runtime | `learning_rate` |
| `n_actions` | inferred at runtime | `self.n_patterns + 1` |
| `n_patterns` | inferred at runtime | `len(choices)` |
| `no_stim_bias` | inferred at runtime | `no_stim_bias` |
| `normalize_context` | inferred at runtime | `normalize_context` |
| `optimizer` | inferred at runtime | `optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)` |
| `phase` | inferred at runtime | `None` |
| `policy_grad_magnitude` | inferred at runtime | `0.0` |
| `policy_net` | inferred at runtime | `nn.Sequential(*layers)` |
| `probs` | inferred at runtime | `np.ones(self.n_actions) / self.n_actions` |
| `reward_deltas` | inferred at runtime | `[]` |
| `reward_history` | inferred at runtime | `deque(maxlen=baseline_window)` |
| `values` | inferred at runtime | `np.ones(self.n_actions) * 10` |
**Methods:**
#### `_build_network(self)`
> Build the MLP policy network.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/trainer.py:107`
#### `set_policy_grad_magnitude(self, value)`
> Set the policy gradient magnitude from external source. Called by CartPolePhase after policy update.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/trainer.py:123`
#### `compute_context(self, motor_spike_rates, episode_reward, episode_length, max_episode_length=500)`
> Compute the context/state vector from current observations.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:130`
#### `_update_normalization_stats(self, context)`
> Update running mean and variance for context normalization.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/trainer.py:179`
#### `_normalize(self, context)`
> Normalize context using running statistics.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:187`
#### `get_action(self, context=None, ind=None)`
> Select an action based on the current context.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/trainer.py:192`
#### `_print_probs(self)`
> Print current action probabilities.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:262`
#### `update(self, reward_delta, action_index=None)`
> Apply reward-weighted policy gradient with an eligibility trace for the chosen stimulation action.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/trainer.py:271`
#### `update_values(self, reward, ind, update_probs=True)`
> Compatibility method matching TetanusTrainer interface.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:304`
#### `record_episode_reward(self, episode_reward)`
> Record episode reward for baseline calculation.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:319`
#### `get_probs(self)`
> Return current action probabilities.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:323`
#### `get_values(self)`
> Return pseudo-values for logging.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:327`
#### `get_state_dict(self)`
> Get the state dict of the policy network.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:331`
#### `load_state_dict(self, state_dict)`
> Load state dict into the policy network.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:335`
### TetanusTrainer()
> unclear — see source
**Source:** `braindance/core/trainer.py:340`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/phases3_ant.py:373 (named-call hint); braindance/core/phases_v3/phases3_ant_cpg.py:387 (named-call hint); braindance/core/phases_v3/phases3_foodland.py:491 (named-call hint); braindance/core/phases_v3/phases3_loop.py:744 (named-call hint); braindance/core/phases_v3/phases3_pacman.py:365 (named-call hint); braindance/core/phases_v3/phases3_walker2d.py:425 (named-call hint); braindance/core/phases_v3/phases3_walker2d_cpg.py:412 (named-call hint); braindance/core/trainer.py:685 (named-call hint); braindance/examples/paper_cartpole/game.py:49 (named-call hint); braindance/examples/paper_cartpole/game.py:50 (named-call hint); braindance/examples/paper_cartpole/game.py:51 (named-call hint); braindance/experiments/cartpole_continuous.py:47 (named-call hint)
**Constructor:** `__init__(self, choices, freqs, start_value=10, learning_rate=0.1, floor=10, discount_factor=0.3)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_update_count` | inferred at runtime | `0` |
| `choices` | inferred at runtime | `choices` |
| `discount_factor` | inferred at runtime | `discount_factor` |
| `floor` | inferred at runtime | `floor` |
| `freqs` | inferred at runtime | `freqs` |
| `learning_rate` | inferred at runtime | `learning_rate` |
| `n_choices` | inferred at runtime | `len(choices)` |
| `print_interval` | inferred at runtime | `20` |
| `probs` | inferred at runtime | `self.update_probs(self.values)` |
| `start_value` | inferred at runtime | `start_value` |
| `traces` | inferred at runtime | `np.zeros((self.n_choices,))` |
| `values` | inferred at runtime | `np.ones((self.n_choices,)) * start_value` |
**Methods:**
#### `update_values(self, reward, ind, update_probs=True)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/trainer.py:362`
#### `update_probs(self, values, bias=0)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/trainer.py:381`
#### `get_action(self, ind=None)`
> Return a random action based on the current probabilities, and the index of the action
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:390`
#### `get_probs(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:399`
#### `get_values(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:402`
### AdaptiveTetanusTrainer()
> Bandit trainer over synthesized tetanus parameter combinations.
**Source:** `braindance/core/trainer.py:406`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/phases3_ant_cpg.py:403 (named-call hint)
**Constructor:** `__init__(self, neurons, amp_mv, pulse_width, stim_count_choices=(2, 3, 4), delay_ms_choices=(3, 5, 8), freq_choices=(20, 30, 40), start_value=10, learning_rate=0.1, floor=10, discount_factor=0.3, max_random_orders=4, seed=0)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `amp_mv` | inferred at runtime | `int(amp_mv)` |
| `delay_ms_choices` | inferred at runtime | `sorted({max(1, int(v)) for v in delay_ms_choices})` |
| `discount_factor` | inferred at runtime | `discount_factor` |
| `floor` | inferred at runtime | `floor` |
| `freq_choices` | inferred at runtime | `sorted({max(1, int(v)) for v in freq_choices})` |
| `learning_rate` | inferred at runtime | `learning_rate` |
| `max_random_orders` | inferred at runtime | `max(0, int(max_random_orders))` |
| `n_choices` | inferred at runtime | `len(self.choices)` |
| `neurons` | inferred at runtime | `[int(n) for n in neurons]` |
| `phase` | inferred at runtime | `None` |
| `probs` | inferred at runtime | `self.update_probs(self.values)` |
| `pulse_width` | inferred at runtime | `int(pulse_width)` |
| `rng` | inferred at runtime | `np.random.default_rng(seed)` |
| `start_value` | inferred at runtime | `start_value` |
| `stim_count_choices` | inferred at runtime | `sorted({max(1, min(int(v), len(self.neurons))) for v in stim_count_choices})` |
| `traces` | inferred at runtime | `np.zeros((self.n_choices,))` |
| `values` | inferred at runtime | `np.ones((self.n_choices,)) * start_value` |
**Methods:**
#### `_canonical_orderings(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:450`
#### `_build_choice_space(self)`
> Deduplicate parameter combinations over canonical and seeded random neuron orderings.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:482`
#### `_pattern_from_spec(self, spec)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:509`
#### `update_values(self, reward, ind, update_probs=True)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/trainer.py:520`
#### `update_probs(self, values, bias=0)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/trainer.py:532`
#### `get_action(self, ind=None)`
> Return a tetanus action and action index from the learned search space.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:545`
#### `get_probs(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:552`
#### `get_values(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:555`
#### `get_choice_specs(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:558`
### HebbianTrainer()
> unclear — see source
**Source:** `braindance/core/trainer.py:619`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, choices, freqs, phase=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `choices` | inferred at runtime | `choices` |
| `freqs` | inferred at runtime | `freqs` |
| `n_choices` | inferred at runtime | `len(choices)` |
| `phase` | inferred at runtime | `phase` |
**Methods:**
#### `update_values(self, reward, ind, update_probs=True)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:631`
#### `update_probs(self, values, bias=-5)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:634`
#### `get_action(self, ind=None)`
> Returns an action based on the pole angle
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:637`

## Functions
### `generate_tetanus_pattern(neurons, stim_count=5, delay_ms=5, amp_mv=400, pulse_width=100, random=False, replace=False)`
> Generates a tetanus pattern for neuron stimulation.
> **Called by:** braindance/core/phases_v3/phases3_ant.py:349 (named-call hint); braindance/core/phases_v3/phases3_ant.py:356 (named-call hint); braindance/core/phases_v3/phases3_ant.py:364 (named-call hint); braindance/core/phases_v3/phases3_ant_cpg.py:363 (named-call hint); braindance/core/phases_v3/phases3_ant_cpg.py:370 (named-call hint); braindance/core/phases_v3/phases3_ant_cpg.py:378 (named-call hint); braindance/core/phases_v3/phases3_foodland.py:461 (named-call hint); braindance/core/phases_v3/phases3_foodland.py:470 (named-call hint); braindance/core/phases_v3/phases3_foodland.py:480 (named-call hint); braindance/core/phases_v3/phases3_loop.py:722 (named-call hint); braindance/core/phases_v3/phases3_loop.py:726 (named-call hint); braindance/core/phases_v3/phases3_loop.py:731 (named-call hint); braindance/core/phases_v3/phases3_loop.py:736 (named-call hint); braindance/core/phases_v3/phases3_pacman.py:335 (named-call hint); braindance/core/phases_v3/phases3_pacman.py:344 (named-call hint); braindance/core/phases_v3/phases3_pacman.py:354 (named-call hint); braindance/core/phases_v3/phases3_walker2d.py:401 (named-call hint); braindance/core/phases_v3/phases3_walker2d.py:408 (named-call hint); braindance/core/phases_v3/phases3_walker2d.py:416 (named-call hint); braindance/core/phases_v3/phases3_walker2d_cpg.py:388 (named-call hint); braindance/core/phases_v3/phases3_walker2d_cpg.py:395 (named-call hint); braindance/core/phases_v3/phases3_walker2d_cpg.py:403 (named-call hint); braindance/experiments/cartpole_continuous.py:39 (named-call hint); braindance/experiments/causal_tetanus_phases.py:48 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:562`
### `generate_permutations(neurons, stim_count=5, delay_ms=5, amp_mv=400, pulse_width=100)`
> Generate every requested-length neuron permutation as alternating stimulation and delay commands.
> **Called by:** braindance/examples/paper_cartpole/game.py:48 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:590`
### `get_random_reward(reward_min=-10, reward_max=10)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:651`
### `get_random_correlated_reward(previous_reward, reward_min=-10, reward_max=10)`
> unclear — see source
> **Called by:** braindance/core/trainer.py:696 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:654`
### `update_values(values, reward, ind, learning_rate=0.1)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:663`
### `update_probs(values)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:670`
### `plot_probs(probs)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/trainer.py:674`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| ContextualTrainer hidden_sizes, learning_rate, gamma, baseline_window, normalize_context, no_stim_bias |
| Tetanus trainers use start_value, floor, learning_rate, discount_factor; adaptive choices vary neuron order, count, delay, and frequency |

## Data Shapes
- get_action returns (pattern or None, frequency_Hz, action_index).
- Context vector has five float32 entries: motor mean/variance, policy gradient magnitude, reward delta, normalized episode length.

## Notes
- TetanusTrainer.__init__ assigns self.probs from update_probs, whose return is None; a later update restores probabilities.
- Contextual policy update uses the last sampled log probability; caller must maintain action/update ordering.
