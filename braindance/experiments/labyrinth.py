"""Spatial sensory encoding and four-population control for a PhaseV3 labyrinth."""
from __future__ import annotations

import numpy as np

from braindance.core.phases_v3.phase_base_v3 import PhaseV3
from braindance.games.labyrinth import LabyrinthEnv


def encode_position(position, centers, sigma=0.22):
    """Gaussian place fields; nearest field wins the single sensory pulse."""
    activity = np.exp(-np.sum((centers - np.asarray(position)) ** 2, axis=1) / (2 * sigma ** 2))
    return activity, int(np.argmax(activity))


def decode_counts(counts, population_sizes):
    """Mean spikes per channel; opposing populations cancel, silence is neutral."""
    means = np.asarray(counts, dtype=float) / np.asarray(population_sizes)
    tilt = np.array([means[1] - means[0], means[3] - means[2]])
    totals = np.array([means[1] + means[0], means[3] + means[2]])
    return np.divide(tilt, totals, out=np.zeros(2), where=totals > 0).astype(np.float32)


class LabyrinthPhase(PhaseV3):
    """Ten physical stimulation electrodes and disjoint L/R/U/D channel groups.

    One nearest-place-field pulse precedes each acquisition window. Spike
    windows include the opening and boundary packets, matching Maxwell packet
    semantics. They are motor windows, not estimates of evoked latency.
    A decision advances the game by ``decision_ms`` simulation milliseconds.
    Recording channels are electrode events, not identified/sorted neurons.
    """

    provides = ["labyrinth_results"]

    def __init__(self, input_electrodes, output_channels, *, amplitude_mv,
                 phase_width_us, decision_ms=50, sampling_hz=20_000, episodes=5,
                 max_decisions=6000, episode_seconds=120, render=False, seed=0,
                 name="Labyrinth", recording_tag=None):
        super().__init__(name=name, recording_tag=recording_tag)
        self.input_electrodes = list(input_electrodes)
        directions = ("left", "right", "up", "down")
        if set(output_channels) != set(directions):
            raise ValueError("output_channels must specify left, right, up, down")
        self.output_channels = {key: list(output_channels[key]) for key in directions}
        ids = self.input_electrodes + [c for group in self.output_channels.values() for c in group]
        if any(not isinstance(c, (int, np.integer)) or isinstance(c, bool) or c < 0 for c in ids):
            raise ValueError("Electrode/channel IDs must be nonnegative integers")
        if len(self.input_electrodes) != 10 or len(set(self.input_electrodes)) != 10:
            raise ValueError("Exactly ten distinct input electrodes are required")
        flat = [c for group in self.output_channels.values() for c in group]
        if any(not group for group in self.output_channels.values()) or len(set(flat)) != len(flat):
            raise ValueError("Output populations must be nonempty, unique and disjoint")
        for key, value in dict(amplitude_mv=amplitude_mv, phase_width_us=phase_width_us,
                               decision_ms=decision_ms, sampling_hz=sampling_hz,
                               episode_seconds=episode_seconds).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if decision_ms > 200:
            raise ValueError("decision_ms must be at most 200")
        for value in (episodes, max_decisions):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("episodes and max_decisions must be positive integers")
        self.amplitude_mv, self.phase_width_us = amplitude_mv, phase_width_us
        self.decision_ms, self.sampling_hz = decision_ms, sampling_hz
        self.episodes, self.max_decisions = episodes, max_decisions
        self.episode_seconds, self.render, self.seed = episode_seconds, render, seed
        self.game = None

    def customize_environment_params(self, env_params):
        params = super().customize_environment_params(env_params).copy()
        params.update(observation_type="spikes", stim_electrodes=self.input_electrodes.copy())
        return params

    def run(self, experiment):
        if self.env is None:
            raise RuntimeError("Experiment must attach the acquisition environment")
        configured = list(self.env.stim_electrodes)
        if any(configured.count(e) != 1 for e in self.input_electrodes):
            raise ValueError("Each input electrode must occur exactly once in env.stim_electrodes")
        local_indices = [configured.index(e) for e in self.input_electrodes]
        groups = list(self.output_channels.values())
        if any(c >= self.env.num_channels for group in groups for c in group):
            raise ValueError("Output channel outside configured acquisition channel range")
        sampling_hz = (self.env.replay_source.sampling_hz
                       if getattr(self.env, "is_replay", False) else self.sampling_hz)
        period_frames = round(sampling_hz * self.decision_ms / 1000)
        if period_frames < 1:
            raise ValueError("Decision window is shorter than one acquisition frame")
        results = dict(input_electrodes=self.input_electrodes,
                       output_channels=self.output_channels, local_indices=local_indices,
                       seed=self.seed, decision_ms=self.decision_ms, sampling_hz=sampling_hz,
                       amplitude_mv=self.amplitude_mv, phase_width_us=self.phase_width_us,
                       windows=[], episodes=[], stop_reason="decision_limit")
        self.game = LabyrinthEnv(render_mode="human" if self.render else None,
                                 dt=self.decision_ms / 1000,
                                 max_episode_steps=max(1, int(np.ceil(self.episode_seconds * 1000 / self.decision_ms))))
        self.game.control_hint = "Neural control | Close window to stop"
        try:
            observation, _ = self.game.reset(seed=self.seed)
            episode_reward = 0.0
            for _ in range(self.max_decisions):
                activity, sensory_index = encode_position(observation[:2], self.game.place_centers)
                self.game.input_activity = activity
                if self.render:
                    self.game.render()
                    if self.game.closed:
                        results["stop_reason"] = "window_closed"
                        break
                events, done = self.env.step()
                if done:
                    results["stop_reason"] = "acquisition_done"
                    break
                events = list(events or [])
                first_frame = self.env.latest_frame
                if first_frame is None:
                    raise RuntimeError("Acquisition did not provide a frame number")
                self.env.stimulate([("stim", [local_indices[sensory_index]],
                                     self.amplitude_mv, self.phase_width_us)])
                while self.env.latest_frame < first_frame + period_frames:
                    previous_frame = self.env.latest_frame
                    more, done = self.env.step()
                    if done:
                        break
                    if self.env.latest_frame is None or self.env.latest_frame <= previous_frame:
                        raise RuntimeError("Acquisition frame clock did not advance")
                    events.extend(more or [])
                if done:
                    results["stop_reason"] = "acquisition_done"
                    break
                counts = [sum(event.channel in group for event in events) for group in groups]
                tilt = decode_counts(counts, [len(group) for group in groups])
                means = np.array(counts) / np.array([len(group) for group in groups])
                self.game.output_activity = means / max(1, means.max())
                before = observation.copy()
                observation, reward, terminated, truncated, info = self.game.step(tilt)
                episode_reward += reward
                results["windows"].append(dict(first_frame=int(first_frame),
                    last_frame=int(self.env.latest_frame), counts=counts, tilt=tilt.tolist(),
                    sensory_index=sensory_index, observation=before.tolist(),
                    next_observation=observation.tolist(), reward=reward,
                    terminated=terminated, truncated=truncated))
                if terminated or truncated:
                    results["episodes"].append(dict(reward=episode_reward, **info,
                                                    terminated=terminated, truncated=truncated))
                    print(f"Labyrinth episode {len(results['episodes'])}: {info['outcome']}")
                    if len(results["episodes"]) == self.episodes:
                        results["stop_reason"] = "episodes_complete"
                        break
                    observation, _ = self.game.reset(seed=self.seed + len(results["episodes"]))
                    episode_reward = 0.0
            return {"labyrinth_results": results}
        finally:
            self.cleanup()

    def cleanup(self):
        if self.game is not None:
            self.game.close()
