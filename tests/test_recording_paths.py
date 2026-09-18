from pathlib import Path

import pandas as pd
import pytest

from braindance.utils.data_manager import Recording


@pytest.mark.parametrize('layout', ['nested', 'standard', 'flat'])
def test_catalog_paths_keep_nested_and_legacy_layouts(tmp_path, layout):
    chip = tmp_path / 'project/chip'
    if layout == 'nested':
        folder = chip / 'session/experiment/recording'
    elif layout == 'standard':
        folder = chip / 'recording'
    else:
        folder = chip
    folder.mkdir(parents=True)
    (folder / 'recording_spike_data.pkl').touch()
    rec = Recording(pd.Series(dict(proj='project', chip='chip', experiment='session/experiment/recording')), base_path=tmp_path)
    assert rec._spikes_path == folder / 'recording_spike_data.pkl'
    assert rec._raw_data_path == folder / 'recording.raw.h5'
    if layout == 'nested':
        assert rec._mapping_path == folder.parent / 'mapping.csv'


def test_explicit_catalog_paths_are_portable_and_absolute_overrides_preserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / 'bundle'
    absolute = tmp_path / 'elsewhere/custom.log.csv'
    rec = Recording(pd.Series(dict(proj='p', chip='c', experiment='exp/rec',
                                   data_path='p/c/exp/rec/spikes.pkl',
                                   raw_data_path='p/c/exp/rec/002.raw.h5',
                                   log_path=str(absolute), mapping_path='mapping.csv',
                                   results_path='p/c/exp/rec/results')), base_path=base)
    assert rec._spikes_path == base / 'p/c/exp/rec/spikes.pkl'
    assert rec._raw_data_path == base / 'p/c/exp/rec/002.raw.h5'
    assert rec._stim_log_path == absolute
    assert rec._mapping_path == base / 'mapping.csv'
    assert rec._resolved_paths['results'] == base / 'p/c/exp/rec/results'
