# environments.py

**Path:** `braindance/examples/streaming_workshop/environments.py`
**Module:** `braindance.examples.streaming_workshop.environments`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Adapts CartPole, FoodLand and Ant to a shared 20 ms workshop game interface. Normalizes reset/step outputs and emits browser-renderable scene geometry.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.playground` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.session` — import consumer hint; not a proven runtime call.
- **Uses:** `CartPoleContinuousEnv` from `braindance.games.cartpole_continuous` — imports (static evidence).
- **Uses:** `FoodLandEnv` from `braindance.games.food_land` — imports (static evidence).

## Dependencies
- `braindance.games.cartpole_continuous.CartPoleContinuousEnv` — intra-repo import; source import evidence.
- `braindance.games.food_land.FoodLandEnv` — intra-repo import; source import evidence.
- `gymnasium` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### WorkshopGame()
> Adapter exposing CartPole, FoodLand, and Ant through the workshop game protocol.
**Source:** `braindance/examples/streaming_workshop/environments.py:11`
**Kind:** class. **Instantiated by:** braindance/examples/streaming_workshop/session.py:205 (named-call hint); braindance/examples/streaming_workshop/session.py:392 (named-call hint); braindance/examples/streaming_workshop/session.py:459 (named-call hint)
**Constructor:** `__init__(self, name='cartpole', seed=7, episode_seconds=10.0)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_food_rng` | inferred at runtime | `np.random.get_state()` |
| `action_names` | inferred at runtime | `['cart force (-1..1)']` |
| `env` | inferred at runtime | `CartPoleContinuousEnv(render_mode=None)` |
| `last_end_reason` | inferred at runtime | `''` |
| `observation_names` | inferred at runtime | `['cart position (m)', 'cart velocity (m/s)', 'pole angle (rad)', 'pole angular velocity (rad/s)']` |
| `steps` | inferred at runtime | `0` |
**Methods:**
#### `action_bounds(self)`
> Bounds in workshop action order (FoodLand is turn, then speed).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/environments.py:56`
#### `reset(self)`
> Resets the selected game and returns its numeric observation.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/examples/streaming_workshop/environments.py:62`
#### `step(self, action)`
> Applies one workshop action, normalizes game output, and updates termination reason/step count.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/examples/streaming_workshop/environments.py:82`
#### `scene(self, observation)`
> Converts the current game state into browser scene geometry.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/environments.py:101`
#### `close(self)`
> Closes the underlying game environment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/environments.py:120`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| name=cartpole,seed=7,episode_seconds=10; Ant frame_skip=2 |

## Data Shapes
- reset -> float observation; step -> observation,reward,done
- scene: kind,observation plus FoodLand agent/food/hazards or Ant capsule endpoints/radii

## Notes
- FoodLand adapter swaps action order and saves/restores global NumPy RNG state.
- Optional game dependencies imported only for selected environment.
