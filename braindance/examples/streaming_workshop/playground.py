"""Independent, manually stepped game sandbox (never opens acquisition hardware)."""
from threading import RLock

import numpy as np

from .environments import WorkshopGame


class Playground:
    def __init__(self, game_factory=WorkshopGame):
        self.game_factory = game_factory
        self.game = None
        self.lock = RLock()
        self.observation = None
        self.done = False
        self.reward = 0.
        self.time = 0.

    def control(self, command):
        with self.lock:
            kind = command.get('kind')
            if kind == 'close':
                self.close()
                return {'status': 'closed'}
            if kind == 'reset':
                name = command.get('environment', 'cartpole')
                if name not in {'cartpole', 'foodland', 'ant'}:
                    raise ValueError('Choose CartPole, FoodLand or Ant')
                seed = command.get('seed', 7)
                if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
                    raise ValueError('Seed must be an integer from 0 to 4294967295')
                game = self.game_factory(name=name, seed=seed, episode_seconds=120.)
                try:
                    observation = game.reset()
                except Exception:
                    game.close()
                    raise
                self.close()
                self.game, self.observation = game, observation
                self.time = self.reward = 0.
                self.done = False
            elif kind == 'step':
                if self.game is None:
                    raise ValueError('Load an environment first')
                if self.done:
                    raise ValueError('Episode ended; reset before stepping again')
                steps = command.get('steps', 1)
                if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 5:
                    raise ValueError('Steps must be an integer from 1 to 5')
                action = np.asarray(command.get('action'), dtype=float)
                bounds = self.game.action_bounds
                if (action.shape != (len(bounds),) or not np.all(np.isfinite(action))
                        or np.any(action < np.asarray(bounds)[:, 0])
                        or np.any(action > np.asarray(bounds)[:, 1])):
                    raise ValueError('Action must contain one finite, in-range value per control')
                samples = []
                for _ in range(steps):
                    self.observation, reward, self.done = self.game.step(action)
                    self.reward += reward
                    self.time += .02
                    samples.append({'time': self.time, 'observation': self.observation.tolist()})
                    if self.done:
                        break
                result = self.snapshot()
                result['samples'] = samples
                result['action'] = action.tolist()
                return result
            else:
                raise ValueError('Unknown playground command')
            return self.snapshot()

    def snapshot(self):
        return dict(environment=self.game.name, observation=self.observation.tolist(),
                    observation_names=self.game.observation_names,
                    action_names=self.game.action_names, action_bounds=self.game.action_bounds,
                    scene=self.game.scene(self.observation), time=self.time,
                    reward=self.reward, done=self.done, end_reason=self.game.last_end_reason)

    def close(self):
        with self.lock:
            if self.game is not None:
                self.game.close()
                self.game = None
