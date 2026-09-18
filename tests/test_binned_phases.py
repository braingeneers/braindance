"""Scientific phases execute without a workshop session or visualization."""
from types import SimpleNamespace

import numpy as np

from braindance.core.phases_v3.phases_binned import (
    BinnedRecordingPhaseV3, MappedEnvironmentPhaseV3, ResponseProbePhaseV3,
)


class Runtime:
    dt, sampling_hz, n = .02, 20000, 2

    def __init__(self):
        self.samples, self.events, self.stimuli, self.transitions = [], [], [], []
        self.resets, self.steps = 0, 0
        self.game = self

    def read_counts(self, action=None, tag=None, pace=True):
        self.samples.append((action, tag, pace))
        return np.array([1, 2]), 400

    def emit(self, event, **values):
        self.events.append((event, values))

    def reset_mapping(self):
        self.resets += 1

    def reset(self):
        return np.array([0.])

    def step(self, action):
        self.steps += 1
        return np.array([self.steps]), 1., True

    def map(self, name, value, dt):
        if name == 'decode':
            return np.array([.25])
        if name == 'encode':
            return np.array([40., 0.])
        self.transitions.append(value)
        return None

    def stimulate(self, action, tag):
        self.stimuli.append((action, tag))

    def finish_step(self):
        pass


def test_record_uses_frames_and_returns_only_declared_data():
    runtime = Runtime()
    result = BinnedRecordingPhaseV3(duration=.04).run(SimpleNamespace(phase_runtime=runtime))
    assert result == {'recording_baseline_hz': [50., 100.]}
    assert len(runtime.samples) == 2


def test_probes_keep_matched_shams_and_exact_pulse_timing():
    runtime = Runtime()
    result = ResponseProbePhaseV3(repeats=2, amplitude_mv=80., phase_width_us=120).run(
        SimpleNamespace(phase_runtime=runtime))
    assert result == {'response_probe_hz': [[0., 0.], [0., 0.]],
                      'response_probe_trials': [[2, 2], [2, 2]]}
    pulses = [(i, sample[0]) for i, sample in enumerate(runtime.samples) if sample[0]]
    assert len(runtime.samples) == 120
    assert len(pulses) == 4
    assert all(i % 15 == 4 and pulse[1:] == (80., 120) for i, pulse in pulses)
    assert sorted(pulse[0][0] for _, pulse in pulses) == [0, 0, 1, 1]


def test_episode_reset_resets_observation_and_fractional_pulse_credit():
    runtime = Runtime()
    exp = SimpleNamespace(phase_runtime=runtime, data=SimpleNamespace(recording_baseline_hz=[0., 0.]))
    result = MappedEnvironmentPhaseV3(duration=.06).run(exp)
    assert result == {'environment_episodes': 3, 'environment_reward': 3.}
    # Every episode is one bin: 40 Hz * .02 s never reaches a whole pulse.
    assert runtime.stimuli == []
    assert runtime.resets == 4
    assert [t['observation'] for t in runtime.transitions] == [[0.], [0.], [0.]]
    assert all(sample[2] is False for sample in runtime.samples)
    assert len([e for e, _ in runtime.events if e == 'game_step']) == 3



def test_invalid_durations_fail_before_acquisition_or_game_reset():
    import pytest

    for phase_class in (BinnedRecordingPhaseV3, MappedEnvironmentPhaseV3):
        for duration in (0, -.02, .01, .03, float('nan'), float('inf'), True, '1'):
            runtime = Runtime()
            exp = SimpleNamespace(phase_runtime=runtime,
                                  data=SimpleNamespace(recording_baseline_hz=[0., 0.]))
            with pytest.raises(ValueError):
                phase_class(duration=duration).run(exp)
            assert runtime.samples == runtime.events == []
            assert runtime.resets == runtime.steps == 0


def test_invalid_runtime_bin_width_rejected():
    import pytest

    for dt in (0, -.02, float('nan'), float('inf'), True, '.02'):
        runtime = Runtime()
        runtime.dt = dt
        with pytest.raises(ValueError):
            BinnedRecordingPhaseV3(duration=.04).run(SimpleNamespace(phase_runtime=runtime))
        assert runtime.samples == []


def test_probe_repeats_requires_positive_integer():
    import pytest

    for repeats in (0, -1, 1.5, True, '2', float('nan')):
        with pytest.raises(ValueError, match='positive integer'):
            ResponseProbePhaseV3(repeats=repeats)


def test_invalid_probe_spec_returns_errors_before_construction():
    from braindance.examples.streaming_workshop.experiment_spec import verify_spec
    report = verify_spec({'phases': [
        dict(id='record', type='recording', params={}),
        dict(id='probe', type='causal', params={'causal_repeats': 0}),
    ]})
    assert not report['ok']
    assert any('causal_repeats' in message for message in report['errors'])
