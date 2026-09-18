"""Thin adapters around BrainDance games; optional imports stay local."""
from threading import RLock

import numpy as np


# Legacy FoodLand uses the process-global RNG; serialize temporary state swaps.
_FOOD_RNG_LOCK = RLock()


class WorkshopGame:
    def __init__(self, name='cartpole', seed=7, episode_seconds=10.):
        self.name, self.seed, self.episode = name, seed, 0
        if not np.isfinite(episode_seconds) or episode_seconds < .02:
            raise ValueError('Episode length must be at least 20 ms')
        self.max_steps, self.steps = round(episode_seconds / .02), 0
        self.last_end_reason = ''
        try:
            if name == 'cartpole':
                from braindance.games.cartpole_continuous import CartPoleContinuousEnv
                self.env = CartPoleContinuousEnv(render_mode=None)
                self.observation_names = ['cart position (m)', 'cart velocity (m/s)',
                                          'pole angle (rad)', 'pole angular velocity (rad/s)']
                self.action_names = ['cart force (-1..1)']
            elif name == 'foodland':
                from braindance.games.food_land import FoodLandEnv
                with _FOOD_RNG_LOCK:
                    outside = np.random.get_state()
                    try:
                        self.env = FoodLandEnv(render_mode=None, reward_type='dense', food_count=3,
                                               spike_count=2, hunger=.5, use_history_observation=False,
                                               max_steps=1000)
                    finally:
                        np.random.set_state(outside)
                self.observation_names = ['food signal', 'hazard signal', 'x (px)', 'y (px)',
                                          'direction (rad)', 'food captured', 'hazard hits',
                                          'wall contact', 'food gradient']
                self.action_names = ['turn (-1..1)', 'forward speed (0..1)']
            elif name == 'ant':
                import gymnasium as gym
                self.env = gym.make('Ant-v5', render_mode=None, frame_skip=2,
                                    healthy_z_range=(.3, 1.), max_episode_steps=self.max_steps)
                if not np.isclose(self.env.unwrapped.dt, .02):
                    self.env.close()
                    raise ValueError('Ant physics dt must be 20 ms')
                self.observation_names = [f'Ant observation {i}' for i in range(self.env.observation_space.shape[0])]
                model = self.env.unwrapped.model
                self.action_names = [f'{model.joint(int(j)).name} torque (-1..1)'
                                     for j in model.actuator_trnid[:, 0]]
            else:
                raise ValueError(f'Unknown environment: {name}')
        except ImportError as exc:
            raise RuntimeError(f'{name} dependency missing: {exc}. See workshop README optional dependencies.') from exc

    @property
    def action_bounds(self):
        """Bounds in workshop action order (FoodLand is turn, then speed)."""
        if self.name == 'foodland':
            return [[-1., 1.], [0., 1.]]
        return [[-1., 1.] for _ in self.action_names]

    def reset(self):
        self.steps = 0
        self.last_end_reason = ''
        if self.name == 'foodland':
            # Legacy game uses global numpy RNG. Retain its state locally.
            with _FOOD_RNG_LOCK:
                outside = np.random.get_state()
                try:
                    np.random.seed(self.seed + self.episode)
                    self.env.food_signal_prev = self.env.food_signal_grad = 0
                    self.env.is_contacting_wall = False
                    obs = self.env.reset()
                    self._food_rng = np.random.get_state()
                finally:
                    np.random.set_state(outside)
        else:
            obs, _ = self.env.reset(seed=self.seed + self.episode)
        self.episode += 1
        return np.asarray(obs, dtype=float)

    def step(self, action):
        if self.name == 'foodland':
            with _FOOD_RNG_LOCK:
                outside = np.random.get_state()
                try:
                    np.random.set_state(self._food_rng)
                    obs, reward, done, _ = self.env.step(np.array([action[1], action[0]]))
                    self._food_rng = np.random.get_state()
                finally:
                    np.random.set_state(outside)
        else:
            obs, reward, terminated, truncated, _ = self.env.step(float(action[0]) if self.name == 'cartpole' else action)
            done = terminated or truncated
        self.steps += 1
        self.last_end_reason = 'environment termination' if done else ''
        if self.steps >= self.max_steps:
            done, self.last_end_reason = True, 'episode time limit'
        return np.asarray(obs, dtype=float), float(reward), bool(done)

    def scene(self, observation):
        scene = {'kind': self.name, 'observation': observation.tolist()}
        if self.name == 'foodland':
            scene.update(agent=np.asarray(self.env.agent_pos).tolist(), direction=float(self.env.agent_dir),
                         food=np.asarray(self.env.food_positions).tolist(), hazards=np.asarray(self.env.spike_positions).tolist())
        elif self.name == 'ant':
            model, data = self.env.unwrapped.model, self.env.unwrapped.data
            geometry = []
            for i in range(model.ngeom):
                kind = int(model.geom_type[i])
                if kind not in (2, 3):  # MuJoCo sphere/capsule; skip the floor plane.
                    continue
                center, radius = data.geom_xpos[i], float(model.geom_size[i, 0])
                axis = data.geom_xmat[i].reshape(3, 3)[:, 2]
                half = float(model.geom_size[i, 1]) if kind == 3 else 0.
                geometry.append(dict(a=(center - half * axis).tolist(), b=(center + half * axis).tolist(), radius=radius))
            scene.update(geometry=geometry, root=data.xpos[1].tolist())
        return scene

    def close(self):
        self.env.close()
