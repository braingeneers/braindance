# cartpole_continuous.py

**Path:** `braindance/games/cartpole_continuous.py`
**Module:** `braindance.games.cartpole_continuous`
**Feature Area:** `Games and Reinforcement`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Implements continuous-force CartPole with wraparound cart position and a 16-degree pole failure threshold. Includes a simple multi-environment wrapper and keyboard/random-action runner.

## Connections
- **Used by:** `braindance.core.phases` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases2` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_loop` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.environments` — import consumer hint; not a proven runtime call.
- **Shared data:** Legacy and V3 CartPole phases use this environment for neural control.

## Dependencies
- `gymnasium` — external or unresolved local import; source import evidence.
- `gymnasium.envs.classic_control.utils` — external or unresolved local import; source import evidence.
- `gymnasium.error.DependencyNotInstalled` — external or unresolved local import; source import evidence.
- `gymnasium.experimental.vector.VectorEnv` — external or unresolved local import; source import evidence.
- `gymnasium.logger` — external or unresolved local import; source import evidence.
- `gymnasium.spaces` — external or unresolved local import; source import evidence.
- `gymnasium.vector.VectorEnv` — external or unresolved local import; source import evidence.
- `gymnasium.vector.utils.batch_space` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pygame` — external or unresolved local import; source import evidence.
- `pygame.gfxdraw` — external or unresolved local import; source import evidence.

## Classes
### CartPoleContinuousEnv(gym.Env[np.ndarray, Union[int, np.ndarray]])
> ## Description
**Source:** `braindance/games/cartpole_continuous.py:22`
**Kind:** class. **Instantiated by:** braindance/examples/streaming_workshop/environments.py:21 (named-call hint); braindance/games/cartpole_continuous.py:370 (named-call hint); braindance/games/cartpole_continuous.py:438 (named-call hint); braindance/games/cartpole_continuous.py:440 (named-call hint); braindance/games/cartpole_continuous.py:442 (named-call hint)
**Constructor:** `__init__(self, render_mode: Optional[str]=None, sb3=False)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `action_space` | inferred at runtime | `spaces.Box(np.array([-1]), np.array([1]))` |
| `clock` | inferred at runtime | `None` |
| `force_mag` | inferred at runtime | `10.0` |
| `gravity` | inferred at runtime | `9.8` |
| `isopen` | inferred at runtime | `True` |
| `kinematics_integrator` | inferred at runtime | `'euler'` |
| `length` | inferred at runtime | `0.5` |
| `masscart` | inferred at runtime | `1.0` |
| `masspole` | inferred at runtime | `0.1` |
| `my_font` | inferred at runtime | `pygame.font.SysFont('Comic Sans MS', 30)` |
| `observation_space` | inferred at runtime | `spaces.Box(-high, high, dtype=np.float32)` |
| `polemass_length` | inferred at runtime | `self.masspole * self.length` |
| `render_mode` | inferred at runtime | `render_mode` |
| `sb3` | inferred at runtime | `sb3` |
| `score` | inferred at runtime | `0` |
| `screen` | inferred at runtime | `None` |
| `screen_height` | inferred at runtime | `400` |
| `screen_width` | inferred at runtime | `600` |
| `state` | inferred at runtime | `None` |
| `steps_beyond_terminated` | inferred at runtime | `None` |
| `surf` | inferred at runtime | `pygame.Surface((self.screen_width, self.screen_height))` |
| `tau` | inferred at runtime | `0.02` |
| `theta_threshold_radians` | inferred at runtime | `16 * 2 * math.pi / 360` |
| `total_mass` | inferred at runtime | `self.masspole + self.masscart` |
| `x_threshold` | inferred at runtime | `2.4` |
**Methods:**
#### `step(self, action)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/cartpole_continuous.py:137`
#### `reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/cartpole_continuous.py:210`
#### `render(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/cartpole_continuous.py:234`
#### `close(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/cartpole_continuous.py:357`
### CartPoleVectorEnv(gym.Env)
> unclear — see source
**Source:** `braindance/games/cartpole_continuous.py:367`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, num_envs, render_mode=None, only_angle=True)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `action_space` | inferred at runtime | `self.envs[0].action_space` |
| `envs` | inferred at runtime | `[CartPoleContinuousEnv(render_mode=render_mode, sb3=False) for _ in range(num_envs)]` |
| `num_envs` | inferred at runtime | `num_envs` |
| `observation_space` | inferred at runtime | `spaces.Box(low=np.repeat(self.envs[0].observation_space.low[np.newaxis, :], num_envs, axis=0), high=np.repeat(self.envs[0].observ…` |
| `only_angle` | inferred at runtime | `only_angle` |
**Methods:**
#### `reset(self, seed=None, options=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/cartpole_continuous.py:379`
#### `step(self, actions)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/cartpole_continuous.py:388`
#### `render(self, mode='human')`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/cartpole_continuous.py:414`
#### `close(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/cartpole_continuous.py:418`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| render_mode, sb3; tau=.02, force_mag=10, x_threshold=2.4 |
| CartPoleVectorEnv num_envs,only_angle=True; CLI --render_mode, --time |

## Data Shapes
- Scalar continuous action in [-1,1]; observation [x,x_dot,theta,theta_dot] float32
- Vector observations shape (num_envs,4); sb3 changes reset return

## Notes
- Docstring describes standard discrete CartPole but implementation uses continuous forces and cart wraparound.
- Vector wrapper ignores seeds, returns final scalar truncation and first info, and passes unsupported render argument.
