"""
Ant-v5 environment wrapper with random-projection feature extraction.

Mirrors the pattern of ``mspacman.py``: raw observations are compressed via a
frozen random projection into a compact feature vector suitable for driving
sensory stimulation in the reservoir pipeline.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class AntFeatureEnv:
    """Wrap ``Ant-v5`` and expose projected observation features.

    The Ant observation is a 27-dimensional continuous vector (positions,
    velocities, joint angles, etc.).  We normalise by the observation-space
    bounds, center, apply a frozen random projection down to ``n_features``
    dimensions, and squash through a sigmoid so features live in (0, 1).

    Parameters
    ----------
    render_mode : str or None
        Passed to ``gymnasium.make``.
    n_features : int
        Dimensionality of the projected feature vector (default 8).
    projection_seed : int
        Seed for the random projection matrix (frozen after init).
    projection_scale : float
        Gain applied before the sigmoid squash.
    """

    N_ACTIONS = 8

    def __init__(
        self,
        render_mode: str | None = None,
        n_features: int = 8,
        projection_seed: int = 0,
        projection_scale: float = 4.0,
        healthy_z_range: tuple[float, float] = (0.3, 1.0),
        reward_mode: str = "forward_x",
    ):
        if reward_mode not in {"forward_x", "planar_speed"}:
            raise ValueError(f"Unknown reward_mode: {reward_mode}")

        self.env = gym.make(
            "Ant-v5",
            render_mode=render_mode,
            terminate_when_unhealthy=True,
            healthy_z_range=healthy_z_range,
        )
        self.n_features = int(n_features)
        self.projection_scale = float(projection_scale)
        self.reward_mode = reward_mode
        self.dt = float(getattr(self.env.unwrapped, "dt", 0.05))

        obs_dim = int(np.prod(self.env.observation_space.shape))
        self.obs_dim = obs_dim

        obs_high = np.asarray(self.env.observation_space.high, dtype=np.float32).ravel()
        obs_low = np.asarray(self.env.observation_space.low, dtype=np.float32).ravel()
        obs_range = obs_high - obs_low
        # Clamp infinite ranges to a large but finite value
        obs_range = np.where(np.isfinite(obs_range), obs_range, 10.0)
        obs_range = np.where(obs_range > 0, obs_range, 1.0)
        self.obs_range = obs_range
        self.obs_low = np.where(np.isfinite(obs_low), obs_low, -5.0)

        rng = np.random.default_rng(projection_seed)
        self.projection_matrix = rng.normal(
            size=(obs_dim, self.n_features),
        ).astype(np.float32)
        self.projection_matrix /= np.sqrt(obs_dim)

        self.last_raw_obs = None
        self.last_features = None
        self.last_xy_position = None

    def get_feature_dim(self):
        return self.n_features

    def get_action_dim(self):
        return self.N_ACTIONS

    def extract_features(self, obs):
        """Project a raw Ant observation to a compact feature vector in (0, 1)."""
        obs = np.asarray(obs, dtype=np.float32).ravel()
        normalized = (obs - self.obs_low) / self.obs_range
        centered = normalized - 0.5
        projected = centered @ self.projection_matrix
        features = _sigmoid(self.projection_scale * projected)
        return features.astype(np.float32)

    def _xy_from_info(self, info):
        x_pos = float(info.get("x_position", 0.0))
        y_pos = float(info.get("y_position", 0.0))
        return np.asarray([x_pos, y_pos], dtype=np.float32)

    def _compute_reward(self, reward, info, current_xy):
        if self.reward_mode == "forward_x":
            return float(reward)

        prev_xy = current_xy if self.last_xy_position is None else self.last_xy_position
        delta_xy = current_xy - prev_xy
        planar_speed = float(np.linalg.norm(delta_xy) / max(self.dt, 1e-8))
        forward_reward = float(info.get("reward_forward", 0.0))
        custom_reward = float(reward) - forward_reward + planar_speed

        info["planar_displacement"] = float(np.linalg.norm(delta_xy))
        info["planar_speed"] = planar_speed
        info["custom_reward"] = custom_reward
        return custom_reward

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        features = self.extract_features(raw_obs)

        self.last_raw_obs = np.asarray(raw_obs, dtype=np.float32)
        self.last_features = features

        info = dict(info)
        self.last_xy_position = self._xy_from_info(info)
        info["raw_obs"] = self.last_raw_obs.copy()
        info["projected_features"] = features.copy()
        info["reward_mode"] = self.reward_mode
        return features.copy(), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).ravel()
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        features = self.extract_features(raw_obs)

        self.last_raw_obs = np.asarray(raw_obs, dtype=np.float32)
        self.last_features = features

        info = dict(info)
        current_xy = self._xy_from_info(info)
        reward = self._compute_reward(reward, info, current_xy)
        self.last_xy_position = current_xy
        info["raw_obs"] = self.last_raw_obs.copy()
        info["projected_features"] = features.copy()
        info["reward_mode"] = self.reward_mode
        return features.copy(), reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
