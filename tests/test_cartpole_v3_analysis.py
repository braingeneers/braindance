"""Numerical fixtures for native paper CartPole analysis phases."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from braindance.core.phases_v3.phases3_cartpole_analysis import (
    CartPoleFootprintPhaseV3, CartPoleCausalAnalysisPhaseV3,
)


def test_footprint_selection_uses_channel_ids_and_filters_routing(tmp_path, monkeypatch):
    from braindance.analysis import data_loader

    electrodes = np.arange(7) * 6 + 1000
    # Non-default DataFrame index ensures channel IDs, not row labels, drive scoring.
    mapping = pd.DataFrame(dict(channel=np.arange(7), electrode=electrodes,
                                orig_channel=np.arange(7) + 40), index=np.arange(7) + 70)
    spikes = pd.DataFrame([dict(channel=ch, frame=1000 + k * 500, amplitude=-100)
                           for ch in range(7) for k in range(20)])
    monkeypatch.setattr(data_loader, 'load_mapping_maxwell', lambda _: mapping)
    monkeypatch.setattr(data_loader, 'load_info_maxwell', lambda _: {'shape': (7, 1200000)})

    def load(filename, spikes=False, start=0, length=None, channels=None):
        if spikes:
            return spike_fixture.copy()
        if start == 1000000:
            return np.broadcast_to(np.tile([-1., 1.], 100000), (7, 200000))
        # Each candidate has a unique set of spikes so the loader knows its channel.
        channel = (int(start) + 100 - 1000) // 20000
        wave = np.zeros((7, length))
        wave[channel, 100] = -20
        if channel == 0:
            wave[1:3, 100] = -20  # Exceeds permitted footprint size.
        return wave

    spike_fixture = spikes.copy()
    spike_fixture['frame'] += spike_fixture.channel * 20000
    monkeypatch.setattr(data_loader, 'load_data_maxwell', load)
    configured = electrodes[::-1].tolist() + [9999]
    exp = SimpleNamespace(data=SimpleNamespace(recording_file=str(tmp_path / 'rec'),
                          recording_duration=60, stim_electrodes=configured))
    result = CartPoleFootprintPhaseV3(num_channel_thresh=2).run(exp)
    assert result['stim_electrodes'] == electrodes[:0:-1].tolist()
    assert set(result['selected_channels']) == set(range(1, 7))
    assert len(result['footprint_waves']) == 6
    assert pd.read_csv(result['mapping_file_path']).orig_channel.tolist() == list(range(40, 47))
    exp.data.stim_electrodes = electrodes[:5].tolist()
    with pytest.raises(ValueError, match='at least six'):
        CartPoleFootprintPhaseV3(num_channel_thresh=2).run(exp)


def test_causal_native_artifact_removal_counts_and_axis_order(tmp_path, monkeypatch):
    from braindance.analysis import data_loader

    mapping = pd.DataFrame(dict(channel=[0, 1], electrode=[17, 80]))
    log = pd.DataFrame(dict(stim_electrodes=[[80], [17], [80]], time_mod=[1., 2., 3.]))
    monkeypatch.setattr(data_loader, 'load_mapping_maxwell', lambda _: mapping)
    monkeypatch.setattr(data_loader, 'load_info_maxwell', lambda _: {'shape': (2, 100000)})
    monkeypatch.setattr(data_loader, 'adjust_stim_times2', lambda *a, **k: log)
    calls = []

    def load(filename, start, length, channels):
        calls.append((start, length, channels))
        result = np.zeros((2, length))
        # Physical electrode 80 has two repeats; all peaks are relative to blanking.
        result[:, np.array([100, 800, 2100]) + 40] = -50
        return result

    monkeypatch.setattr(data_loader, 'load_data_maxwell', load)
    exp = SimpleNamespace(data=SimpleNamespace(sweep_file=str(tmp_path / 'causal')))
    result = CartPoleCausalAnalysisPhaseV3().run(exp)
    assert result['valid_stim_electrodes'] == [80, 17]
    assert result['causal_channels'] == [1, 0]
    np.testing.assert_array_equal(np.load(tmp_path / 'derived/causal_connectivity_first.npy'),
                                  [[0, 2], [1, 0]])
    np.testing.assert_array_equal(np.load(tmp_path / 'derived/causal_connectivity_multi.npy'),
                                  [[0, 4], [2, 0]])
    np.testing.assert_array_equal(result['causal_reactivity_times'][0, 1][0], [100, 800, 2100])
    assert result['causal_reactivity'][0, 1].sum() == 6
    assert all(width == 6100 and channels == [1, 0] for _, width, channels in calls)


def test_causal_rejects_truncated_final_window(tmp_path, monkeypatch):
    from braindance.analysis import data_loader

    monkeypatch.setattr(data_loader, 'load_mapping_maxwell',
                        lambda _: pd.DataFrame(dict(channel=[0], electrode=[17])))
    monkeypatch.setattr(data_loader, 'load_info_maxwell', lambda _: {'shape': (1, 26000)})
    monkeypatch.setattr(data_loader, 'adjust_stim_times2',
                        lambda *a, **k: pd.DataFrame(dict(stim_electrodes=[[17]], time_mod=[1.])))
    exp = SimpleNamespace(data=SimpleNamespace(sweep_file=str(tmp_path / 'causal')))
    with pytest.raises(ValueError, match='full causal response window'):
        CartPoleCausalAnalysisPhaseV3().run(exp)
