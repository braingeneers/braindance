"""Continuous tilting labyrinth with optional Pygame rendering (Gymnasium API)."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class LabyrinthEnv(gym.Env):
    """Actions are [horizontal, vertical] tilt in [-1, 1]; y increases downward.

    Observations are [x, y, vx, vy] in board-width units and seconds.
    Each step integrates ``dt`` simulation seconds independently of rendering.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, dt=1 / 60, max_episode_steps=7200):
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("Invalid render_mode")
        if not np.isfinite(dt) or not 0 < dt <= 0.2:
            raise ValueError("dt must be in (0, 0.2]")
        if not isinstance(max_episode_steps, int) or max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be a positive integer")
        self.render_mode, self.dt = render_mode, dt
        self.max_episode_steps = max_episode_steps
        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            np.array([0, 0, -0.6, -0.6], dtype=np.float32),
            np.array([1, 1, 0.6, 0.6], dtype=np.float32), dtype=np.float32)
        self.radius = 0.018
        self.start = np.array([0.1, 0.12])
        self.goal = np.array([0.9, 0.88])
        # Alternating gates form a connected serpentine path.
        self.walls = [(0.00, 0.27, 0.74, 0.025),
                      (0.26, 0.53, 0.74, 0.025),
                      (0.00, 0.77, 0.74, 0.025)]
        self.holes = np.array([[0.48, 0.16], [0.60, 0.41], [0.40, 0.65], [0.57, 0.9]])
        self.place_centers = np.array([(x, y) for y in (0.2, 0.8)
                                       for x in np.linspace(0.1, 0.9, 5)])
        self.input_activity = np.zeros(10)
        self.output_activity = np.zeros(4)
        self.tilt = np.zeros(2)
        self._pygame = self.screen = None
        self._ended = True
        self.closed = False
        self.control_hint = "Arrows: tilt   R: reset   Esc: quit"

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position, self.velocity = self.start.copy(), np.zeros(2)
        self.steps = 0
        self._ended = False
        self.closed = False
        self.tilt = np.zeros(2)
        self.input_activity = np.zeros(10)
        self.output_activity = np.zeros(4)
        return self._observation(), {"outcome": "running"}

    def _observation(self):
        return np.concatenate((self.position, self.velocity)).astype(np.float32)

    def step(self, action):
        if self._ended:
            raise RuntimeError("Call reset before stepping or after an episode ends")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Tilt must be a finite two-vector in [-1, 1]")
        self.tilt = action.copy()
        outcome = "running"
        # Small bounded substeps prevent tunneling through walls and holes.
        substeps = int(np.ceil(self.dt / 0.004))
        h = self.dt / substeps
        for _ in range(substeps):
            self.velocity += (1.8 * action - 2.5 * self.velocity) * h
            self.velocity = np.clip(self.velocity, -0.6, 0.6)
            for axis in (0, 1):
                old = self.position[axis]
                self.position[axis] += self.velocity[axis] * h
                collision = not self.radius <= self.position[axis] <= 1 - self.radius
                for x, y, w, height in self.walls:
                    nearest = np.clip(self.position, [x, y], [x + w, y + height])
                    collision |= np.linalg.norm(self.position - nearest) < self.radius
                if collision:
                    self.position[axis] = old
                    self.velocity[axis] = 0
            if np.any(np.linalg.norm(self.holes - self.position, axis=1) < 0.032):
                outcome = "hole"
                break
            if np.linalg.norm(self.goal - self.position) < 0.045:
                outcome = "goal"
                break
        self.steps += 1
        terminated = outcome != "running"
        truncated = self.steps >= self.max_episode_steps and not terminated
        if truncated:
            outcome = "timeout"
        self._ended = terminated or truncated
        reward = 1.0 if outcome == "goal" else -1.0 if outcome == "hole" else 0.0
        return self._observation(), reward, terminated, truncated, {"outcome": outcome}

    def render(self):
        if self.render_mode is None:
            return None
        if self._pygame is None:
            import pygame
            self._pygame = pygame
            pygame.font.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((1040, 720))
                pygame.display.set_caption("BrainDance | Neural Labyrinth")
            else:
                self.screen = pygame.Surface((1040, 720))
        pg, screen = self._pygame, self.screen
        if self.render_mode == "human":
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.closed = True
        screen.fill((13, 20, 32))
        font = pg.font.Font(None, 25)
        small = pg.font.Font(None, 21)
        title = pg.font.Font(None, 38)
        def label(text, xy, color=(216, 229, 242), face=font):
            screen.blit(face.render(text, True, color), xy)
        def point(p):
            return (int(30 + p[0] * 640), int(58 + p[1] * 640))
        label("NEURAL LABYRINTH", (30, 16), face=title)
        pg.draw.rect(screen, (28, 43, 59), (30, 58, 640, 640), border_radius=12)
        for i, center in enumerate(self.place_centers):
            level = float(self.input_activity[i])
            color = (35, int(65 + 125 * level), int(85 + 140 * level))
            pg.draw.circle(screen, color, point(center), 26, 2)
            label(str(i + 1), (point(center)[0] - 6, point(center)[1] - 7), color, small)
        for x, y, w, h in self.walls:
            pg.draw.rect(screen, (154, 174, 190), (*point((x, y)), int(w * 640), int(h * 640)), border_radius=4)
        for hole in self.holes:
            pg.draw.circle(screen, (5, 9, 16), point(hole), 22)
            pg.draw.circle(screen, (81, 106, 129), point(hole), 22, 2)
        pg.draw.circle(screen, (63, 211, 153), point(self.goal), 29, 3)
        label("GOAL", (point(self.goal)[0] - 23, point(self.goal)[1] - 7), face=small)
        pg.draw.circle(screen, (249, 197, 89), point(self.position), 12)
        pg.draw.circle(screen, (255, 234, 173), (point(self.position)[0] - 3, point(self.position)[1] - 4), 4)
        label("10 SPATIAL INPUTS", (712, 68))
        for i, level in enumerate(self.input_activity):
            y = 104 + i * 27
            label(f"{i + 1:02}", (712, y), face=small)
            pg.draw.rect(screen, (35, 51, 69), (748, y, 244, 15), border_radius=4)
            pg.draw.rect(screen, (69, 196, 216), (748, y, int(244 * np.clip(level, 0, 1)), 15), border_radius=4)
        label("NEURAL OUTPUT / TILT", (712, 402))
        for i, name in enumerate(("Left", "Right", "Up", "Down")):
            y = 440 + i * 28
            label(name, (712, y), face=small)
            pg.draw.rect(screen, (35, 51, 69), (777, y, 215, 15), border_radius=4)
            pg.draw.rect(screen, (180, 151, 248), (777, y, int(215 * np.clip(self.output_activity[i], 0, 1)), 15), border_radius=4)
        origin = (850, 608)
        pg.draw.circle(screen, (62, 82, 105), origin, 33, 1)
        pg.draw.line(screen, (249, 197, 89), origin,
                     (int(origin[0] + 30 * self.tilt[0]), int(origin[1] + 30 * self.tilt[1])), 4)
        label(self.control_hint, (712, 664), face=small)
        if self.render_mode == "human":
            pg.display.flip()
            return None
        return np.transpose(pg.surfarray.array3d(screen), (1, 0, 2))

    def close(self):
        if self._pygame is not None and self.render_mode == "human":
            self._pygame.display.quit()
        self.screen = self._pygame = None
        self.closed = True
