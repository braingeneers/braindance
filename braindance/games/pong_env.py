"""A small, deterministic Pong environment using the Gymnasium API."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PongEnv(gym.Env[np.ndarray, int | np.ndarray]):
    """Move the left paddle to intercept a ball.

    Observations contain paddle y, ball y, ball y velocity, paddle height,
    ball x, and ball x velocity. Positions and velocities are normalized;
    paddle height is in pixels. With ``play_human=True``, actions are
    ``[left_paddle, right_paddle]``. Zero moves up and one moves down.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        play_human: bool = False,
        max_episode_steps: int = 3_600,
    ):
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render mode: {render_mode!r}")
        self.render_mode = render_mode
        self.play_human = play_human
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self.max_episode_steps = max_episode_steps
        self.screen_width, self.screen_height = 800, 600
        self.paddle_width = 25
        self.paddle_speed = 25 / self.screen_height
        self.paddle_length = 140 / self.screen_height
        self.ball_radius = 15
        self.ball_speed = 20 / self.screen_width
        self.action_space = spaces.MultiDiscrete([2, 2]) if play_human else spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, 0.0, -self.ball_speed, 50.0, -self.ball_speed, -self.ball_speed],
                dtype=np.float32,
            ),
            high=np.array(
                [1.0, 1.0, self.ball_speed, 600.0, 1 + self.ball_speed, self.ball_speed],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.screen: Any = None
        self.clock: Any = None
        self._pygame: Any = None

    def _observation(self) -> np.ndarray:
        return np.array([
            self.paddle_pos, self.ball_y, self.ball_velocity_y,
            self.paddle_length * self.screen_height, self.ball_x, self.ball_velocity_x,
        ], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.paddle_pos = 0.5
        self.ball_x = 0.5
        self.ball_y = 0.5
        self.ball_velocity_x = self.ball_speed * self.np_random.choice((-1, 1))
        self.ball_velocity_y = self.ball_speed * self.np_random.uniform(-1.0, 1.0)
        self.elapsed_steps = 0
        if self.play_human:
            self.human_paddle_pos = 0.5
        observation = self._observation()
        if self.render_mode == "human":
            self.render()
        return observation, {}

    def _move_paddle(self, position: float, action: int) -> float:
        direction = -1 if action == 0 else 1
        half_height = self.paddle_length / 2
        return float(np.clip(
            position + direction * self.paddle_speed, half_height, 1 - half_height
        ))

    def step(self, action: int | np.ndarray):
        if self.play_human:
            actions = np.asarray(action)
            if (not np.issubdtype(actions.dtype, np.integer)
                    or not self.action_space.contains(actions)):
                raise ValueError(f"Invalid action: {action!r}")
            paddle_action, human_action = (int(value) for value in actions)
            self.human_paddle_pos = self._move_paddle(self.human_paddle_pos, human_action)
        else:
            if not self.action_space.contains(action):
                raise ValueError(f"Invalid action: {action!r}")
            paddle_action = int(action)
        self.elapsed_steps += 1
        self.paddle_pos = self._move_paddle(self.paddle_pos, paddle_action)
        previous_ball_x = self.ball_x
        self.ball_x += self.ball_velocity_x
        self.ball_y += self.ball_velocity_y
        if self.ball_y <= 0:
            self.ball_y = -self.ball_y
            self.ball_velocity_y = abs(self.ball_velocity_y)
        elif self.ball_y >= 1:
            self.ball_y = 2 - self.ball_y
            self.ball_velocity_y = -abs(self.ball_velocity_y)

        hit = False
        paddle_edge = (10 + self.paddle_width + self.ball_radius) / self.screen_width
        if (self.ball_velocity_x < 0
                and previous_ball_x >= paddle_edge >= self.ball_x
                and abs(self.ball_y - self.paddle_pos) <= self.paddle_length / 2):
            self.ball_x = paddle_edge
            self.ball_velocity_x = abs(self.ball_velocity_x)
            hit = True
        right_edge = 1 - paddle_edge
        if self.play_human:
            if (self.ball_velocity_x > 0
                    and previous_ball_x <= right_edge <= self.ball_x
                    and abs(self.ball_y - self.human_paddle_pos) <= self.paddle_length / 2):
                self.ball_x = right_edge
                self.ball_velocity_x = -abs(self.ball_velocity_x)
        elif self.ball_x >= 1:
            self.ball_x = 2 - self.ball_x
            self.ball_velocity_x = -abs(self.ball_velocity_x)

        terminated = self.ball_x < 0 or (self.play_human and self.ball_x > 1)
        truncated = self.elapsed_steps >= self.max_episode_steps
        reward = 1.0 if hit else 0.0
        if self.render_mode == "human":
            self.render()
        return self._observation(), reward, terminated, truncated, {"hit": hit}

    def _init_pygame(self) -> None:
        if self._pygame is not None:
            return
        try:
            import pygame
        except ImportError as exc:
            raise gym.error.DependencyNotInstalled(
                'Pong rendering requires pygame. Install with `pip install "braindance[rl]"`.'
            ) from exc
        self._pygame = pygame
        pygame.init()
        if self.render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("BrainDance Pong")
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("Call render only after constructing PongEnv with render_mode.")
            return None
        self._init_pygame()
        pygame = self._pygame
        self.screen.fill((0, 0, 0))
        paddle_height = round(self.paddle_length * self.screen_height)
        paddle_y = round(self.paddle_pos * self.screen_height - paddle_height / 2)
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (10, paddle_y, self.paddle_width, paddle_height))
        if self.play_human:
            human_y = round(self.human_paddle_pos * self.screen_height - paddle_height / 2)
            pygame.draw.rect(self.screen, (255, 0, 0),
                             (self.screen_width - 10 - self.paddle_width, human_y,
                              self.paddle_width, paddle_height))
        pygame.draw.circle(self.screen, (255, 255, 255),
                           (round(self.ball_x * self.screen_width),
                            round(self.ball_y * self.screen_height)), self.ball_radius)
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        return np.transpose(np.asarray(pygame.surfarray.pixels3d(self.screen)),
                            axes=(1, 0, 2)).copy()

    def close(self) -> None:
        if self._pygame is not None:
            if self.screen is not None and self.render_mode == "human":
                self._pygame.display.quit()
            self._pygame.quit()
        self.screen = self.clock = self._pygame = None


# Historical name retained for old experiment code.
PaddleBallEnv = PongEnv
