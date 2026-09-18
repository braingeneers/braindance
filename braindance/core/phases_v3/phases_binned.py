"""V3 phases for binned neural acquisition and participant-defined controllers.

``experiment.phase_runtime`` supplies acquisition, mapping callbacks and a game
adapter. It may observe scientific events; persistence, pacing, visualization,
interactive controls and transport belong to that runtime, never to a phase.
"""
from typing import Protocol
from numbers import Real
import math

import numpy as np

from .phase_base_v3 import PhaseV3


class BinnedRuntime(Protocol):
    """Runner-provided services; counts are units/channel counts per sample bin.

    read_counts returns (counts, acquired_frames); sampling_hz converts those
    frames to seconds. Stimuli use local input indices, mV and microseconds.
    A game exposes reset() and step(action) -> (observation, reward, done).
    """
    dt: float
    sampling_hz: float
    n: int
    game: object

    def read_counts(self, action=None, tag=None, pace=True): ...
    def map(self, name, value, dt): ...
    def stimulate(self, action, tag): ...
    def reset_mapping(self): ...
    def emit(self, event, **values): ...
    def finish_step(self): ...


class BinnedPhaseV3(PhaseV3):
    """Acquisition phases that can share a continuously running environment."""

    @staticmethod
    def _bin_count(duration, dt):
        """Reject invalid durations rather than rounding partial acquisition bins."""
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(value) or value <= 0 for value in (duration, dt)):
            raise ValueError('Duration and bin width must be finite positive numbers')
        bins = duration / dt
        if not math.isfinite(bins) or bins < 1 or not math.isclose(bins, round(bins), rel_tol=0, abs_tol=1e-9):
            raise ValueError('Duration must contain a positive whole number of bins')
        return round(bins)

    def close_environment_after(self):
        return False

    def run(self, experiment):
        raise NotImplementedError('Choose a concrete binned phase')


class BinnedRecordingPhaseV3(BinnedPhaseV3):
    """Measure baseline rates from exact acquisition frame counts."""
    inputs = []
    outputs = ['recording_baseline_hz']

    def __init__(self, duration=3., name=None):
        super().__init__(name=name)
        self.duration = duration

    def run(self, experiment):
        runtime = experiment.phase_runtime
        bins = self._bin_count(self.duration, runtime.dt)
        total, frames = np.zeros(runtime.n), 0
        for _ in range(bins):
            counts, count = runtime.read_counts()
            total += counts
            frames += count
        if frames == 0:
            raise ValueError('Recording must acquire at least one bin')
        baseline = (total / (frames / runtime.sampling_hz)).tolist()
        runtime.emit('baseline', baseline_hz=baseline)
        return {'recording_baseline_hz': baseline}


class ResponseProbePhaseV3(BinnedPhaseV3):
    """Randomized stimulus/sham trials with matched pre/post windows."""
    inputs = ['recording_baseline_hz']
    outputs = ['response_probe_hz', 'response_probe_trials']

    def __init__(self, repeats=4, seed=7, amplitude_mv=100., phase_width_us=100, name=None):
        super().__init__(name=name)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ValueError('Response probe repeats must be a positive integer')
        self.repeats, self.seed = repeats, seed
        self.amplitude_mv, self.phase_width_us = amplitude_mv, phase_width_us

    def run(self, experiment):
        runtime = experiment.phase_runtime
        trials = [(ch, sham) for _ in range(self.repeats)
                  for ch in range(2) for sham in (False, True)]
        np.random.default_rng(self.seed).shuffle(trials)
        values = {(ch, sham): [] for ch in range(2) for sham in (False, True)}
        responses, completed = np.zeros((2, runtime.n)), [[0, 0], [0, 0]]
        for trial, (ch, sham) in enumerate(trials):
            runtime.emit('trial', trial=trial + 1, total=len(trials), channel=ch, sham=sham)
            pre, post = np.zeros(runtime.n), np.zeros(runtime.n)
            for i in range(5):
                pulse = ([ch], self.amplitude_mv, self.phase_width_us) if i == 4 and not sham else None
                counts, _ = runtime.read_counts(pulse, tag='causal_sham' if sham else f'causal_input_{ch}')
                pre += counts
            for _ in range(5):
                counts, _ = runtime.read_counts()
                post += counts
            values[ch, sham].append((post - pre) / (5 * runtime.dt))
            for _ in range(5):
                runtime.read_counts()
            for channel in range(2):
                stimulated, control = values[channel, False], values[channel, True]
                if stimulated and control:
                    responses[channel] = np.mean(stimulated, axis=0) - np.mean(control, axis=0)
                    completed[channel] = [len(stimulated), len(control)]
            runtime.emit('response', responses=responses.copy(), trials=[row[:] for row in completed])
        runtime.emit('trial_end')
        return {'response_probe_hz': responses.tolist(), 'response_probe_trials': completed}


class MappedEnvironmentPhaseV3(BinnedPhaseV3):
    """Closed-loop game controlled by decode, train and encode callbacks.

    The runner selects CartPole, FoodLand or Ant and supplies the same normalized
    game API. Scientific state stays local; observers receive each completed
    action and the corresponding observation before an episode reset.
    """
    inputs = ['recording_baseline_hz']
    outputs = ['environment_episodes', 'environment_reward']

    def __init__(self, duration=120., amplitude_mv=100., phase_width_us=100, name=None):
        super().__init__(name=name)
        self.duration = duration
        self.amplitude_mv, self.phase_width_us = amplitude_mv, phase_width_us

    def run(self, experiment):
        runtime = experiment.phase_runtime
        bins = self._bin_count(self.duration, runtime.dt)
        runtime.emit('baseline', baseline_hz=experiment.data.recording_baseline_hz)
        runtime.reset_mapping()
        observation = runtime.game.reset()
        runtime.emit('game_reset', observation=observation)
        credit, reward_total, episodes = np.zeros(2), 0., 0
        for _ in range(bins):
            counts, frames = runtime.read_counts(pace=False)
            dt = frames / runtime.sampling_hz
            action = runtime.map('decode', counts, dt)
            previous_observation = observation.copy()
            observation, reward, done = runtime.game.step(action)
            runtime.map('train', dict(observation=previous_observation.tolist(),
                next_observation=observation.tolist(), action=action.tolist(),
                spike_counts=counts.tolist(), reward=float(reward), done=bool(done)), dt)
            rates = runtime.map('encode', observation, dt)
            reward_total += reward
            credit += rates * dt
            due = np.flatnonzero(credit >= 1 - 1e-10)
            credit[due] -= 1
            if len(due):
                runtime.stimulate((due.tolist(), self.amplitude_mv, self.phase_width_us), 'encoded_sensory')
            if done:
                episodes += 1
            runtime.emit('game_step', observation=observation, action=action, rates=rates,
                         delivered=due.tolist(), reward=reward_total, episodes=episodes, done=done)
            if done:
                observation = runtime.game.reset()
                runtime.reset_mapping()
                credit[:] = 0
                runtime.emit('game_reset', observation=observation)
            runtime.finish_step()
        return {'environment_episodes': episodes, 'environment_reward': float(reward_total)}
