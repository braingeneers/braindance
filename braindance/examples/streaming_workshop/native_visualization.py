"""Browser observation of native game environments, outside scientific phases."""
import json
import time

import numpy as np


def game_scene(kind, environment, observation):
    """Describe a game using geometry; no desktop renderer or video dependency."""
    env = environment
    while hasattr(env, 'env'):
        env = env.env
    scene = dict(kind=kind, observation=np.asarray(observation).tolist())
    if kind == 'foodland':
        scene.update(agent=np.asarray(env.agent_pos).tolist(), direction=float(env.agent_dir),
                     food=np.asarray(env.food_positions).tolist(), hazards=np.asarray(env.spike_positions).tolist())
    elif kind == 'ant':
        model, data = env.model, env.data
        geometry = []
        for i in range(model.ngeom):
            shape = int(model.geom_type[i])
            if shape not in (2, 3):
                continue
            center = data.geom_xpos[i]
            axis = data.geom_xmat[i].reshape(3, 3)[:, 2]
            half = float(model.geom_size[i, 1]) if shape == 3 else 0.
            geometry.append(dict(a=(center-half*axis).tolist(), b=(center+half*axis).tolist(),
                                 radius=float(model.geom_size[i, 0])))
        scene.update(geometry=geometry, root=data.xpos[1].tolist())
    return scene


class ObservedGame:
    """Transparent game proxy that samples browser state after reset/step."""
    def __init__(self, environment, kind, progress, phase_id):
        self.environment, self.kind = environment, kind
        self.progress, self.phase_id = progress, phase_id
        self.last_publish = 0.
        self.reward = 0.
        self.episode_reward = 0.
        self.episodes = 0

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def publish(self, observation, force=False):
        now = time.monotonic()
        if not force and now-self.last_publish < .05:
            return
        state = dict(phase=self.phase_id, phase_kind='environment',
                     scene=game_scene(self.kind, self.environment, observation),
                     reward=self.reward, episode_reward=self.episode_reward, episodes=self.episodes)
        temporary = self.progress.with_suffix('.tmp')
        temporary.write_text(json.dumps(state), encoding='utf-8')
        temporary.replace(self.progress)
        self.last_publish = now

    def reset(self, *args, **kwargs):
        result = self.environment.reset(*args, **kwargs)
        self.episode_reward = 0.
        self.publish(result[0], force=True)
        return result

    def step(self, *args, **kwargs):
        result = self.environment.step(*args, **kwargs)
        self.reward += float(result[1])
        self.episode_reward += float(result[1])
        ended = bool(result[2]) or (len(result) == 5 and bool(result[3]))
        self.episodes += int(ended)
        self.publish(result[0], force=ended)
        return result
