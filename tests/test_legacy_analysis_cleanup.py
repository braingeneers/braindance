"""Contracts retained while deprecated analysis APIs remain available."""
import numpy as np
import pandas as pd
import pytest

from braindance.analysis import data_loader


def test_dao_warning_preserves_selection_and_filtering(tmp_path):
    mapping = pd.DataFrame({'channel': [0, 1], 'electrode': [101, 205],
                            'x': [0., 17.5], 'y': [0., 0.]})
    spikes = pd.DataFrame({'channel': [0, 1, 0], 'frame': [2, 5, 9],
                           'amplitude': [-10., -20., -30.]})
    with pytest.warns(FutureWarning, match='AnalysisDAO is deprecated'):
        dao = data_loader.AnalysisDAO(spikes=spikes, mapping=mapping)
    dao.select_electrodes([205, 101])
    assert dao.selected_channels == [1, 0]
    dao.select_channels([0])
    assert dao.selected_electrodes == [101]
    pd.testing.assert_frame_equal(dao.get_spikes(channels=[0], frame_bounds=(0, 5)), spikes.iloc[[0]])

    saved = tmp_path / 'legacy.pkl'
    dao.save(str(saved))
    with pytest.warns(FutureWarning, match='AnalysisDAO is deprecated'):
        restored = data_loader.AnalysisDAO.load(str(saved))
    pd.testing.assert_frame_equal(restored.spikes, spikes)
    assert restored.selected_electrodes == [101]


def test_retained_maxwell_loader_values_and_channel_order(maxwell_h5):
    raw = (512 + np.arange(800, dtype=np.uint16).reshape(200, 4) % 100)
    values = data_loader.load_data_maxwell(str(maxwell_h5), channels=[3, 1], start=7, length=12)
    expected = (raw[7:19, [3, 1]].T.astype(np.float32) - 512) * np.float32(3.147125e-6) * 1024 * 1000
    np.testing.assert_allclose(values, expected)
    assert values.dtype == np.float32
    windows = data_loader.load_windows_maxwell(str(maxwell_h5), [7, 20], 12, channels=[3, 1])
    np.testing.assert_allclose(windows[0], expected)
    assert not hasattr(data_loader, 'load_windows')


def test_deprecated_threshold_preserves_times_and_amplitudes():
    from braindance.analysis import causal_connectivity as causal
    trace = np.array([[0., -12., 0., 0., -15., 0.]])
    with pytest.warns(FutureWarning, match='causal_connectivity is deprecated'):
        spikes, amps = causal.threshold_data(trace, -10, fs_ms=1)
    np.testing.assert_array_equal(spikes[0], [1, 4])
    np.testing.assert_array_equal(amps[0], [-12., -15.])


@pytest.mark.parametrize('entrypoint', ['clean_stim_response', 'clean_stim_responses', 'clean_stim_responses_all'])
def test_deprecated_cleaning_entrypoints_warn(entrypoint, monkeypatch):
    from braindance.analysis import causal_connectivity as causal
    data = np.arange(4.)
    monkeypatch.setattr(causal, '_' + entrypoint, lambda values: values + 1)
    with pytest.warns(FutureWarning, match='causal_connectivity is deprecated'):
        result = getattr(causal, entrypoint)(data)
    np.testing.assert_array_equal(result, data + 1)


@pytest.mark.parametrize('argument', ['react_inds', 'stim'])
def test_implicit_project_json_removed(maxwell_h5, tmp_path, argument):
    from braindance.analysis import causal_connectivity as causal
    with pytest.warns(FutureWarning), pytest.raises(ValueError, match='explicit JSON'):
        causal.main(str(maxwell_h5), save_dir=str(tmp_path), **{argument: 'json'})


def test_compiled_cleaning_still_matches_batched_result():
    from braindance.analysis import causal_connectivity as causal
    data = np.sin(np.arange(240, dtype=np.float64) / 17)
    with pytest.warns(FutureWarning):
        single = causal.clean_stim_response(data)
    with pytest.warns(FutureWarning):
        batched = causal.clean_stim_responses(np.stack([data, data]))
    with pytest.warns(FutureWarning):
        shaped = causal.clean_stim_responses_all(np.stack([data, data])[None, ...])
    np.testing.assert_allclose(batched, np.stack([single, single]))
    np.testing.assert_allclose(shaped[0], batched)
