# walker2d.py

**Path:** `braindance/games/walker2d.py`
**Module:** `braindance.games.walker2d`
**Feature Area:** `Games and Reinforcement`
**Entry point:** no — library or imported component

## Overview
Wraps Gymnasium Walker2d-v5 with frozen random projections that turn continuous observations into bounded sensory features. Retains raw observations in info while forwarding environment rewards and termination flags.

## Connections
- **Used by:** `braindance.core.phases_v3.phases3_walker2d` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_walker2d_cpg` — import consumer hint; not a proven runtime call.
- **Shared data:** Walker2dFeatureEnv.extract_features normalizes observation then applies matrix projection and sigmoid.

## Dependencies
- `gymnasium` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### Walker2dFeatureEnv()
> Wrap ``Walker2d-v5`` and expose projected observation features.
**Source:** `braindance/games/walker2d.py:19`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/phases3_walker2d.py:198 (named-call hint); braindance/core/phases_v3/phases3_walker2d_cpg.py:180 (named-call hint)
**Constructor:** `__init__(self, render_mode: str | None=None, n_features: int=6, projection_seed: int=0, projection_scale: float=4.0, healthy_z_range: tuple[float, float]=(0.8, 2.0), healthy_angle_range: tuple[float, float]=(-1.0, 1.0))`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `env` | inferred at runtime | `gym.make('Walker2d-v5', render_mode=render_mode, terminate_when_unhealthy=True, healthy_z_range=healthy_z_range, healthy_angle_ra…` |
| `last_features` | inferred at runtime | `None` |
| `last_raw_obs` | inferred at runtime | `None` |
| `n_features` | inferred at runtime | `int(n_features)` |
| `obs_dim` | inferred at runtime | `obs_dim` |
| `obs_low` | inferred at runtime | `np.where(np.isfinite(obs_low), obs_low, -5.0)` |
| `obs_range` | inferred at runtime | `obs_range` |
| `projection_matrix` | inferred at runtime | `rng.normal(size=(obs_dim, self.n_features)).astype(np.float32)` |
| `projection_scale` | inferred at runtime | `float(projection_scale)` |
**Methods:**
#### `get_feature_dim(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/walker2d.py:82`
#### `get_action_dim(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/walker2d.py:85`
#### `extract_features(self, obs)`
> Project a raw Walker2d observation to a compact feature vector in (0, 1).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/walker2d.py:88`
#### `reset(self, **kwargs)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/walker2d.py:97`
#### `step(self, action)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/walker2d.py:109`
#### `render(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/walker2d.py:122`
#### `close(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/walker2d.py:125`

## Functions
### `_sigmoid(x)`
> unclear — see source
> **Called by:** braindance/games/walker2d.py:94 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/games/walker2d.py:15`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| n_features=6, projection_seed=0, projection_scale=4.0 |
| render_mode and healthy bounds configure underlying MuJoCo environment |

## Data Shapes
- Action float vector length 6; features float32 vector length n_features in (0,1)
- reset -> (features,info); step -> (features,reward,terminated,truncated,info)

## Notes
- Requires Gymnasium MuJoCo environment dependencies; normalizes unbounded observation ranges with finite fallbacks.
