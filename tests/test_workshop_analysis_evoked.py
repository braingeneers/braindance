import numpy as np
import pytest

from braindance.examples.streaming_workshop.analysis_evoked import analyze_evoked


def test_pattern_normalization_blanking_and_latency_windows():
    result = analyze_evoked([100, 400, 700], ['A', 'A', 'B'],
                           spike_times_ms=[101, 105, 120, 121, 405, 750],
                           unit_ids=[2] * 6,
                           params={'pre_ms': 20, 'post_ms': 100, 'blank_ms': 3, 'first_order_ms': 10})
    assert result['pattern_counts'] == {'A': 2, 'B': 1}
    metrics = result['response_metrics']
    assert metrics['short_probability'] == [[1.], [0.]]
    assert metrics['late_mean_spikes'] == [[1.], [1.]]
    assert metrics['total_probability'] == [[1.], [1.]]
    raster = result['plots'][0]
    assert raster['x'] == [5., 20., 21., 5.]
    assert raster['y'] == [1, 1, 1, 2]
    assert any(value is None for value in result['plots'][1]['y'])


def test_raw_cubic_cleanup_detects_known_evoked_negative_spikes():
    fs = 20000
    rng = np.random.default_rng(5)
    trace = rng.normal(0, .3, 20000)
    events = [200, 500]
    for event in events:
        center = round((event + 7) * fs / 1000)
        trace[center-2:center+3] -= [5, 15, 30, 15, 5]
    updates = []
    result = analyze_evoked(events, ['[1]', '[1]'],
                           raw_reader=lambda channel, start, stop: trace[start:stop],
                           sampling_frequency=fs, params={'post_ms': 100},
                           progress=lambda *update: updates.append(update))
    assert result['response_metrics']['short_probability'] == [[1.]]
    assert result['plots'][0]['title'].startswith('Raw stimulation overlap')
    assert result['plots'][1]['title'].startswith('Cubic artifact-removed overlap')
    waveform = result['plots'][0]['series'][0]
    assert len(waveform['x']) == len(waveform['y'])
    assert any(abs(x - 7) < .2 for x in result['plots'][2]['x'])
    assert updates[-1][0] == 1


def test_empty_units_preserved_in_response_matrix():
    result = analyze_evoked([100], ['A'], spike_times_ms=[105], unit_ids=[2],
                           selected_units=[2, 9], params={'post_ms': 100})
    assert result['response_metrics']['short_probability'] == [[1., 0.]]


@pytest.mark.parametrize('params', [{'blank_ms': 10}, {'bin_ms': .0001}, {'threshold_sigma': float('nan')}])
def test_invalid_parameters(params):
    with pytest.raises(ValueError):
        analyze_evoked([100], ['A'], spike_times_ms=[105], unit_ids=[1], params=params)


def test_incomplete_raw_window_is_rejected():
    with pytest.raises(ValueError, match='Incomplete raw'):
        analyze_evoked([100], ['A'], raw_reader=lambda *args: np.zeros(4))
