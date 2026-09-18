"""Loading uncorrected logs must not rewrite downloaded source data."""
import pickle

import numpy as np
import pandas as pd
from spikelab import SpikeData

from braindance.utils.data_manager import load_catalog


def _recording_fixture(tmp_path, artifact_times=None):
    folder = tmp_path / 'demo' / 'chip' / 'rec'
    folder.mkdir(parents=True)
    metadata = {} if artifact_times is None else {'artifact_times': artifact_times}
    spikes = SpikeData([[10., 20.]], length=2000., metadata=metadata)
    with (folder / 'rec_spike_data.pkl').open('wb') as handle:
        pickle.dump(spikes, handle)
    log_path = folder / 'rec_log.csv'
    times = np.arange(128) / 100
    pd.DataFrame({'time': times, 'stim_electrodes': ['[42]'] * len(times),
                  'tag': ['fixture_stimulation'] * len(times)}).to_csv(log_path, index=False)
    catalog = tmp_path / 'catalog.csv'
    pd.DataFrame([{'proj': 'demo', 'chip': 'chip', 'experiment': 'rec'}]).to_csv(catalog, index=False)
    return catalog, log_path, times


def test_loading_without_artifacts_preserves_original_csv_bytes(tmp_path):
    catalog, log_path, times = _recording_fixture(tmp_path)
    original = log_path.read_bytes()
    rec = load_catalog(catalog, base_path=tmp_path)[0]
    try:
        np.testing.assert_allclose(rec.stim_log['time'], times)
        assert 'time_mod' not in rec.stim_log
        assert log_path.read_bytes() == original
        rec.clear_cache()
        np.testing.assert_allclose(rec.stim_log['time'], times)
        assert log_path.read_bytes() == original
    finally:
        rec.clear_cache()


def test_actual_artifact_correction_is_still_saved_and_reloaded(tmp_path):
    artifacts = np.arange(128) * 10. + 2.
    catalog, log_path, times = _recording_fixture(tmp_path, artifacts)
    original = log_path.read_bytes()
    rec = load_catalog(catalog, base_path=tmp_path)[0]
    try:
        np.testing.assert_allclose(rec.stim_log['time'], times)
        np.testing.assert_allclose(rec.stim_log['time_mod'], artifacts / 1000)
        corrected = log_path.read_bytes()
        assert corrected != original
        np.testing.assert_allclose(pd.read_csv(log_path)['time_mod'], artifacts / 1000)
        rec.clear_cache()
        np.testing.assert_allclose(rec.stim_log['time_mod'], artifacts / 1000)
        assert log_path.read_bytes() == corrected
    finally:
        rec.clear_cache()
