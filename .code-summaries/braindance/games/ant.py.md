# ant.py

**Path:** `braindance/games/ant.py`
**Module:** `braindance.games.ant`
**Feature Area:** `Games and Reinforcement`
**Entry point:** no — library or imported component

## Overview
Wraps Gymnasium Ant-v5 with frozen random projections that turn continuous observations into bounded sensory features. Optionally replaces forward reward with planar speed while retaining other reward components.

## Connections
- **Used by:** `braindance.core.phases_v3.phases3_ant` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_ant_cpg` — import consumer hint; not a proven runtime call.
- **Shared data:** AntFeatureEnv.extract_features normalizes observation then applies matrix projection and sigmoid.

## Dependencies
- `gymnasium` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### AntFeatureEnv()
> Wrap ``Ant-v5`` and expose projected observation features.
**Source:** `braindance/games/ant.py:19`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/phases3_ant.py:144 (named-call hint); braindance/core/phases_v3/phases3_ant_cpg.py:164 (named-call hint)
**Constructor:** `__init__(self, render_mode: str | None=None, n_features: int=8, projection_seed: int=0, projection_scale: float=4.0, healthy_z_range: tuple[float, float]=(0.3, 1.0), reward_mode: str='forward_x')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `dt` | inferred at runtime | `float(getattr(self.env.unwrapped, 'dt', 0.05))` |
| `env` | inferred at runtime | `gym.make('Ant-v5', render_mode=render_mode, terminate_when_unhealthy=True, healthy_z_range=healthy_z_range)` |
| `last_features` | inferred at runtime | `None` |
| `last_raw_obs` | inferred at runtime | `None` |
| `last_xy_position` | inferred at runtime | `None` |
| `n_features` | inferred at runtime | `int(n_features)` |
| `obs_dim` | inferred at runtime | `obs_dim` |
| `obs_low` | inferred at runtime | `np.where(np.isfinite(obs_low), obs_low, -5.0)` |
| `obs_range` | inferred at runtime | `obs_range` |
| `projection_matrix` | inferred at runtime | `rng.normal(size=(obs_dim, self.n_features)).astype(np.float32)` |
| `projection_scale` | inferred at runtime | `float(projection_scale)` |
| `reward_mode` | inferred at runtime | `reward_mode` |
**Methods:**
#### `get_feature_dim(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/ant.py:86`
#### `get_action_dim(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/ant.py:89`
#### `extract_features(self, obs)`
> Project a raw Ant observation to a compact feature vector in (0, 1).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/ant.py:92`
#### `_xy_from_info(self, info)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/ant.py:101`
#### `_compute_reward(self, reward, info, current_xy)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/ant.py:106`
#### `reset(self, **kwargs)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/ant.py:121`
#### `step(self, action)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/ant.py:135`
#### `render(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/ant.py:152`
#### `close(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/ant.py:155`

## Functions
### `_sigmoid(x)`
> unclear — see source
> **Called by:** braindance/games/ant.py:98 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/games/ant.py:15`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| n_features=8, projection_seed=0, projection_scale=4.0 |
| render_mode and healthy bounds configure underlying MuJoCo environment |
| reward_mode: forward_x or planar_speed |

## Data Shapes
- Action float vector length 8; features float32 vector length n_features in (0,1)
- reset -> (features,info); step -> (features,reward,terminated,truncated,info)

## Notes
- Requires Gymnasium MuJoCo environment dependencies; normalizes unbounded observation ranges with finite fallbacks.
