# mspacman.py

**Path:** `braindance/games/mspacman.py`
**Module:** `braindance.games.mspacman`
**Feature Area:** `Games and Reinforcement`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Wraps ALE Ms. Pac-Man RAM observations with frozen random-projection sensory features. Also exposes an interactive keyboard player with WASD and arrow controls.

## Connections
- **Used by:** `braindance.core.phases_v3.phases3_pacman` — import consumer hint; not a proven runtime call.
- **Shared data:** Registers ALE environments with Gymnasium; gymnasium.utils.play runs keyboard mode.

## Dependencies
- `ale_py` — external or unresolved local import; source import evidence.
- `gymnasium` — external or unresolved local import; source import evidence.
- `gymnasium.utils.play.play` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### MsPacmanFeatureEnv()
> Wrap `ALE/MsPacman-v5` and expose projected RAM features.
**Source:** `braindance/games/mspacman.py:29`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/phases3_pacman.py:125 (named-call hint)
**Constructor:** `__init__(self, render_mode: str | None=None, n_features: int=8, projection_seed: int=0, projection_scale: float=4.0)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `env` | inferred at runtime | `gym.make('ALE/MsPacman-v5', obs_type='ram', render_mode=render_mode)` |
| `last_features` | inferred at runtime | `None` |
| `last_ram_obs` | inferred at runtime | `None` |
| `n_features` | inferred at runtime | `int(n_features)` |
| `projection_matrix` | inferred at runtime | `rng.normal(size=(128, self.n_features)).astype(np.float32)` |
| `projection_scale` | inferred at runtime | `float(projection_scale)` |
**Methods:**
#### `get_feature_dim(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/mspacman.py:70`
#### `get_action_name(self, action)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/mspacman.py:73`
#### `extract_features(self, ram_obs)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/mspacman.py:79`
#### `reset(self, **kwargs)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/mspacman.py:92`
#### `step(self, action)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/mspacman.py:104`
#### `render(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/mspacman.py:118`
#### `close(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/mspacman.py:121`

## Functions
### `_sigmoid(x)`
> unclear — see source
> **Called by:** braindance/games/mspacman.py:89 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/games/mspacman.py:13`
### `_register_ale_envs()`
> unclear — see source
> **Called by:** braindance/games/mspacman.py:128 (named-call hint); braindance/games/mspacman.py:51 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/games/mspacman.py:17`
### `run_keyboard_mspacman(zoom=4)`
> unclear — see source
> **Called by:** braindance/games/mspacman.py:187 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/games/mspacman.py:125`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| n_features=8, projection_seed=0, projection_scale=4.0; keyboard CLI --zoom |

## Data Shapes
- RAM shape (128,) uint8; features float32 (n_features,); discrete actions 0â€“8
- info includes raw_ram,projected_features and action_name

## Notes
- Requires ale-py and Atari game resources.
