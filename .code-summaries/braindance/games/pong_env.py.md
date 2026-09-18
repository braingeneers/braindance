# pong_env.py

**Path:** `braindance/games/pong_env.py`
**Module:** `braindance.games.pong_env`
**Feature Area:** `Games and Reinforcement`
**Entry point:** no — library or imported component

## Overview
Implements a paddle-and-ball Gymnasium environment with optional human opponent. Rewards paddle hits and terminates when the ball exits the left side.

## Connections
- **Used by:** `braindance.experiments.pong` — import consumer hint; not a proven runtime call.

## Dependencies
- `gymnasium` — external or unresolved local import; source import evidence.
- `gymnasium.spaces` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pygame` — external or unresolved local import; source import evidence.

## Classes
### PongEnv(gym.Env[np.ndarray, int | np.ndarray])
> Deterministic Gymnasium Pong environment in which the left paddle moves up/down to intercept a horizontally traveling ball, with optional human right paddle.
**Source:** `braindance/games/pong_env.py:12`
**Kind:** class. **Instantiated by:** braindance/experiments/pong.py:68 (named-call hint)
**Constructor:** `__init__(self, render_mode: str | None=None, play_human: bool=False, max_episode_steps: int=3600)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_pygame` | Any | `None` |
| `action_space` | inferred at runtime | `spaces.MultiDiscrete([2, 2]) if play_human else spaces.Discrete(2)` |
| `ball_radius` | inferred at runtime | `15` |
| `ball_speed` | inferred at runtime | `20 / self.screen_width` |
| `ball_velocity_x` | inferred at runtime | `self.ball_speed * self.np_random.choice((-1, 1))` |
| `ball_velocity_y` | inferred at runtime | `self.ball_speed * self.np_random.uniform(-1.0, 1.0)` |
| `ball_x` | inferred at runtime | `0.5` |
| `ball_y` | inferred at runtime | `0.5` |
| `clock` | Any | `None` |
| `elapsed_steps` | inferred at runtime | `0` |
| `human_paddle_pos` | inferred at runtime | `0.5` |
| `max_episode_steps` | inferred at runtime | `max_episode_steps` |
| `observation_space` | inferred at runtime | `spaces.Box(low=np.array([0.0, 0.0, -self.ball_speed, 50.0, -self.ball_speed, -self.ball_speed], dtype=np.float32), high=np.array(…` |
| `paddle_length` | inferred at runtime | `140 / self.screen_height` |
| `paddle_pos` | inferred at runtime | `0.5` |
| `paddle_speed` | inferred at runtime | `25 / self.screen_height` |
| `paddle_width` | inferred at runtime | `25` |
| `play_human` | inferred at runtime | `play_human` |
| `render_mode` | inferred at runtime | `render_mode` |
| `screen` | Any | `None` |
**Methods:**
#### `_observation(self) -> np.ndarray`
> Returns the current six-element float32 state vector.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/pong_env.py:58`
#### `reset(self, *, seed: int | None=None, options: dict[str, Any] | None=None)`
> Seeds Gymnasium RNG when requested, initializes paddle/ball state and returns the initial observation/info pair.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/pong_env.py:64`
#### `_move_paddle(self, position: float, action: int) -> float`
> Applies an up/down action and clips paddle center within the screen.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/pong_env.py:79`
#### `step(self, action: int | np.ndarray)`
> Updates paddle and ball physics, detects paddle hits and termination/truncation, and returns reward plus hit info.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/pong_env.py:86`
#### `_init_pygame(self) -> None`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/pong_env.py:136`
#### `render(self)`
> Draws the current game with lazy Pygame setup and returns an RGB array in rgb_array mode.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/pong_env.py:155`
#### `close(self) -> None`
> Releases Pygame display and references.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/pong_env.py:182`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `render_mode` is None, human, or rgb_array; `play_human=False`; `max_episode_steps=3600`; canvas is 800×600, paddle length 140 px, ball radius 15 px, and render FPS 60. |

## Data Shapes
- Observation is float32 `[paddle_y, ball_y, ball_y_velocity, paddle_length_pixels, ball_x, ball_x_velocity]` with six values. Action space is Discrete(2), or MultiDiscrete([2,2]) when play_human=True. `step` returns `(observation, reward, terminated, truncated, {"hit": bool})`; `reset` returns `(observation, {})`.

## Notes
- Paddle and ball positions/velocities are normalized; paddle height is reported in pixels. Pygame is imported lazily only when render is called. `PaddleBallEnv` aliases PongEnv for compatibility.
