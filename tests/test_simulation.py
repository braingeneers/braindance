import numpy as np
import pytest

from braindance.core.simulation import NeuralSimulationSource
from braindance.core.maxwell_env import MaxwellEnv


def test_simulation_split_reads_preserve_events_and_waveform():
    whole = NeuralSimulationSource(seed=12, speed=0)
    split = NeuralSimulationSource(seed=12, speed=0)
    a = whole.read(2000)
    parts = [split.read(n) for n in [13, 387, 1, 799, 800]]
    np.testing.assert_array_equal(a['raw_uint16'], np.concatenate([p['raw_uint16'] for p in parts]))
    assert a['events'] == [row for p in parts for row in p['events']]
    assert whole.elapsed_s == split.elapsed_s == .1
    for i, row in enumerate(a['events']):
        for event in row:
            if i + 12 < len(a['raw_uint16']):
                assert a['raw_uint16'][i:i + 12, event.channel].min() < 470


def test_stimulation_future_feedback_and_adjacency(tmp_path):
    matrix = np.zeros((2, 2))
    matrix[1, 0] = 1.2
    connected = NeuralSimulationSource(num_channels=2, background=0, speed=0, adjacency=matrix)
    isolated = NeuralSimulationSource(num_channels=2, background=0, speed=0, adjacency=np.zeros((2, 2)))
    for source in [connected, isolated]:
        source.read(400)
        source.on_stimulation(([0], 200, 100), frame=399, stim_electrodes=[0, 1])
    a, b = connected.read(400), isolated.read(400)
    ae = [e for row in a['events'] for e in row]
    be = [e for row in b['events'] for e in row]
    assert all(e.frame >= 400 for e in ae)
    assert any(e.channel == 1 for e in ae)
    assert not any(e.channel == 1 for e in be)


def test_local_mode_never_uses_real_maxlab_or_manual_send(monkeypatch, tmp_path):
    import braindance.core.maxwell_env as module

    class Forbidden:
        def __getattr__(self, key):
            raise AssertionError(f'Hardware API accessed: {key}')

    monkeypatch.setattr(module, 'maxlab', Forbidden())
    source = NeuralSimulationSource(num_channels=2, speed=0)
    env = MaxwellEnv(name='local', save_dir=tmp_path, stim_electrodes=[0, 1], verbose=0,
                     replay={'source': source, 'write_output': False})
    env.step(buffer_size=400)
    old_frame = env.latest_frame
    env.stimulate(([0, 1], 100, 100), tag='local')
    assert env.latest_frame == old_frame  # Sending never acquires an extra bin.
    assert source._pulses
    with pytest.raises(ValueError, match='manual'):
        env.step(action=('manual', Forbidden(), [0]), buffer_size=400)
    env.close()
    env.close()


def test_simulator_writer_replay_round_trip(tmp_path):
    from braindance.core.replay import H5ReplaySource
    source = NeuralSimulationSource(num_channels=2, speed=0)
    env = MaxwellEnv(name='roundtrip', save_dir=tmp_path, observation_type='raw', verbose=0,
                     replay={'source': source, 'write_output': True})
    expected, _ = env.step(buffer_size=2000)
    events = env.latest_batch['events']
    env.close()
    replay = H5ReplaySource(tmp_path / 'roundtrip.raw.h5', speed=0)
    actual = replay.read(2000)
    np.testing.assert_allclose(np.asarray(expected), actual['raw_float32'], rtol=1e-6, atol=1e-8)
    expected_events = [e for row in events for e in row]
    actual_events = [e for row in actual['events'] for e in row]
    assert expected_events
    assert [(e.frame, e.channel) for e in expected_events] == [(e.frame, e.channel) for e in actual_events]
    np.testing.assert_allclose([e.amplitude for e in expected_events], [e.amplitude for e in actual_events])
    replay.close()


def test_source_validation_and_clock_pacing():
    class Clock:
        now = 0.
        def __call__(self):
            return self.now
        def sleep(self, seconds):
            self.now += seconds
    clock = Clock()
    source = NeuralSimulationSource(speed=2, clock=clock, sleep=clock.sleep)
    source.read(400)
    assert clock.now == .01
    assert source.elapsed_s == .02
    with pytest.raises(ValueError):
        source.set_adjacency([[float('nan')]])


def test_spatial_geometry_gaussian_waveforms_and_neuron_count():
    source = NeuralSimulationSource(num_channels=400, grid_shape=(20, 20),
                                    num_neurons=1, neuron_positions=[[175., 175.]],
                                    background=0, noise_uv=0, speed=0)
    assert source.adjacency.shape == (1, 1)
    assert source.mapping[210]['electrode'] == 2210
    np.testing.assert_array_equal(source.mapping[210][['x', 'y']].tolist(), (175., 175.))
    assert source.spatial_weights[0, 210] == 1
    assert source.spatial_weights[0, 211] == pytest.approx(np.exp(-17.5**2 / (2 * 20**2)))
    source._voltage[:] = 1.1
    batch = source.read(80)
    assert batch['events'][0][0].channel == 210
    # Subtract the known slow baseline and compare the raw ADC waveform peak.
    adc = batch['raw_uint16'].astype(float) - 512
    expected = source.templates[0, 8] * source.spatial_weights[0]
    baseline = 3 * np.sin(2 * np.pi * 8 / 20000 * .7 + np.arange(400))
    np.testing.assert_allclose(adc[8] - baseline, expected, atol=.51)
    np.testing.assert_allclose(batch['raw_float32'], adc * .001, atol=1e-8)


def test_spatial_reset_and_partitioned_stream_are_identical():
    source = NeuralSimulationSource(num_channels=16, grid_shape=(4, 4), num_neurons=5, speed=0)
    first = source.read(2000)
    source.on_stimulation(([0], 100, 100), 1999, [0])
    source.reset()
    assert not source._pulses
    pieces = [source.read(count) for count in (1, 19, 397, 1583)]
    np.testing.assert_array_equal(first['raw_uint16'], np.concatenate([p['raw_uint16'] for p in pieces]))
    assert first['events'] == [row for part in pieces for row in part['events']]
    assert first['events'] != [[] for _ in range(2000)]
    positions = source.neuron_positions.copy()
    source.reset(seed=77)
    np.testing.assert_array_equal(source.neuron_positions, positions)
    with pytest.raises(ValueError, match='shape'):
        source.set_adjacency(np.zeros((16, 16)))
    with pytest.raises(ValueError, match='workspace'):
        source.set_neuron_positions(np.full((5, 2), -1))


def test_spatial_maxwell_raw_routing_and_roundtrip(tmp_path):
    from braindance.core.replay import H5ReplaySource
    source = NeuralSimulationSource(num_channels=16, grid_shape=(4, 4), num_neurons=3,
                                    neuron_positions=[[0, 0], [17.5, 17.5], [52.5, 52.5]], speed=0)
    env = MaxwellEnv(name='spatial', save_dir=tmp_path, observation_type='raw', verbose=0,
                     stim_electrodes=[0, 221], replay={'source': source, 'write_output': True})
    raw, _ = env.step(buffer_size=400)
    assert np.asarray(raw).shape == (400, 16)
    env.stimulate(([1], 100, 100))
    assert source._pulses[0][2] == 5  # physical electrode 221 routes to channel index 5
    env.close()
    replay = H5ReplaySource(tmp_path / 'spatial.raw.h5', speed=0)
    np.testing.assert_array_equal(replay.mapping, source.mapping)
    np.testing.assert_allclose(replay.read(400)['raw_float32'], raw, atol=1e-8)
    replay.close()


@pytest.mark.parametrize('kwargs', [
    {'num_channels': 2.5}, {'num_channels': float('inf')}, {'num_neurons': 1.1},
    {'num_neurons': True}, {'duration_s': float('nan')}, {'duration_s': float('inf')},
    {'duration_s': None}, {'grid_shape': [2.1, 4]}, {'grid_shape': 20},
    {'grid_shape': []}, {'seed': -1}, {'seed': .5}, {'noise_uv': None},
    {'spatial_sigma_um': 0}, {'electrode_pitch_um': float('nan')},
])
def test_simulator_rejects_malformed_numeric_configuration(kwargs):
    with pytest.raises(ValueError):
        NeuralSimulationSource(**kwargs)


def test_delayed_stimulation_and_template_survive_single_sample_boundaries():
    whole = NeuralSimulationSource(num_channels=2, num_neurons=1, neuron_positions=[[0, 0]],
                                   background=0, speed=0)
    split = NeuralSimulationSource(num_channels=2, num_neurons=1, neuron_positions=[[0, 0]],
                                   background=0, speed=0)
    for source in (whole, split):
        source.on_stimulation([('delay', .95), ('stim', [0], 200, 100)], -1, [0])
    expected = whole.read(80)
    pieces = [split.read(n) for n in (19, 1, 1, 2, 57)]
    np.testing.assert_array_equal(expected['raw_uint16'], np.concatenate([p['raw_uint16'] for p in pieces]))
    assert expected['events'] == [row for part in pieces for row in part['events']]
    events = [event for row in expected['events'] for event in row]
    assert events[0].frame == 20  # frame-19 pulse takes effect at the next 1ms network step
    assert not any(expected['events'][:20])
    with pytest.raises(ValueError):
        whole.read(1.5)
    with pytest.raises(ValueError):
        whole.on_stimulation(([.5], 100, 100), 79, [0])
    with pytest.raises(ValueError):
        whole.on_stimulation([('delay', 1), ('delay', -.5), ('stim', [0], 100, 100)], 79, [0])
    assert not whole._pulses
    whole.close()
    with pytest.raises(RuntimeError, match='closed'):
        whole.reset()
