# food_land.py

**Path:** `braindance/games/food_land.py`
**Module:** `braindance.games.food_land`
**Feature Area:** `Games and Reinforcement`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Simulates a moving agent collecting food and avoiding hazards using inverse-distance sensory signals. Supports sparse/dense rewards, optional history observations and a keyboard or heuristic player.

## Connections
- **Used by:** `braindance.core.phases_env` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_foodland` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.environments` — import consumer hint; not a proven runtime call.
- **Shared data:** Reinforcement examples use FoodLandEnv with sensory encoding and action callbacks.

## Dependencies
- `gym` — external or unresolved local import; source import evidence.
- `gym.spaces` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pygame` — external or unresolved local import; source import evidence.
- `pygame.gfxdraw` — external or unresolved local import; source import evidence.

## Classes
### FoodLandEnv(gym.Env)
> unclear — see source
**Source:** `braindance/games/food_land.py:14`
**Kind:** class. **Instantiated by:** braindance/core/phases_env.py:70 (named-call hint); braindance/core/phases_v3/phases3_foodland.py:48 (named-call hint); braindance/examples/streaming_workshop/environments.py:30 (named-call hint); braindance/games/food_land.py:471 (named-call hint)
**Constructor:** `__init__(self, render_mode: Optional[str]=None, reward_type='dense', food_count=3, spike_count=0, hunger=0.5, history=3, use_history_observation=False, max_steps=100, log_dir=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `action_history` | inferred at runtime | `deque(maxlen=history)` |
| `action_shape` | inferred at runtime | `(2,)` |
| `action_space` | inferred at runtime | `spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), shape=(2,), dtype=np.float32)` |
| `agent_dir` | inferred at runtime | `0` |
| `agent_pos` | inferred at runtime | `np.array([400, 300], dtype=np.float32)` |
| `clock` | inferred at runtime | `None` |
| `draw_food_ani` | inferred at runtime | `0` |
| `draw_spike_ani` | inferred at runtime | `0` |
| `food_count` | inferred at runtime | `food_count` |
| `food_got` | inferred at runtime | `0` |
| `food_positions` | inferred at runtime | `[]` |
| `food_signal` | inferred at runtime | `0` |
| `food_signal_grad` | inferred at runtime | `0` |
| `food_signal_prev` | inferred at runtime | `0` |
| `hunger` | inferred at runtime | `hunger` |
| `is_contacting_wall` | inferred at runtime | `False` |
| `isopen` | inferred at runtime | `True` |
| `log` | inferred at runtime | `self.init_log(log_dir)` |
| `max_distance` | inferred at runtime | `math.sqrt(self.screen_width ** 2 + self.screen_height ** 2)` |
| `max_steps` | inferred at runtime | `max_steps` |
| `my_font` | inferred at runtime | `pygame.font.SysFont('Comic Sans MS', 30)` |
| `observation_history` | inferred at runtime | `deque(maxlen=history)` |
| `observation_shape` | inferred at runtime | `(4 * history + 2 * (history - 1),)` |
| `observation_space` | inferred at runtime | `spaces.Box(low=-1, high=1, shape=self.observation_shape, dtype=np.float32)` |
| `render_mode` | inferred at runtime | `render_mode` |
| `reward_type` | inferred at runtime | `reward_type` |
| `run_speed` | inferred at runtime | `0.5` |
| `screen` | inferred at runtime | `None` |
| `screen_height` | inferred at runtime | `600` |
| `screen_width` | inferred at runtime | `800` |
| `signal_range` | inferred at runtime | `100` |
| `spike_count` | inferred at runtime | `spike_count` |
| `spike_hit` | inferred at runtime | `0` |
| `spike_positions` | inferred at runtime | `[]` |
| `spike_signal` | inferred at runtime | `0` |
| `step_count` | inferred at runtime | `0` |
| `surf` | inferred at runtime | `pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)` |
| `turn_speed` | inferred at runtime | `8` |
| `use_history_observation` | inferred at runtime | `use_history_observation` |
**Methods:**
#### `init_log(self, log_path, header=['agent_pos_x', 'agent_pos_y', 'agent_dir', 'food_signal', 'spike_signal', 'food_got', 'spike_hit', 'reward'])`
> Creates a csv file to log the data, returns the file object
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/games/food_land.py:117`
#### `step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/food_land.py:126`
#### `get_observation(self)`
> Concatenates history, needed for RL, but not for organoid
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/food_land.py:214`
#### `clip_action(self, action)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/food_land.py:234`
#### `calculate_reward(self, food_got, spike_hit)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/food_land.py:238`
#### `reset(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/food_land.py:246`
#### `generate_food(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/food_land.py:270`
#### `generate_spikes(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/food_land.py:276`
#### `render(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/food_land.py:282`
#### `close(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/food_land.py:411`

## Functions
### `compute_smart_action(observation)`
> Current observation is: >>> observation = np.array([self.food_signal, self.spike_signal], dtype=np.float32) Food signal acts as a "smell" signal, same with spike signal. We want to move towards food and avoid spikes (if spikes are used)
> **Called by:** braindance/games/food_land.py:482 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/games/food_land.py:419`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| food_count=3,spike_count=0,hunger=.5,history=3,max_steps=100 |
| reward_type=dense; use_history_observation=False; CLI --smart/-s |

## Data Shapes
- Action two values; implementation uses action[0] for displacement and action[1] for heading change
- Actual base observation has 9 values: food_signal,spike_signal,x,y,direction,food_got,spike_hit,wall_contact,food_signal_grad
- Legacy gym step returns (observation,reward,done,info)

## Notes
- Declared nonhistory observation shape is (8,) despite 9 returned values; history shape also needs verification.
- Uses legacy gym and pygame; reset does not clear history or previous food signal.
