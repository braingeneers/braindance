"""Batch workspace validates uploads and preserves the selected baseline/targets."""
import threading
import time
from pathlib import Path

import h5py
import pytest

from braindance.examples.streaming_workshop import sorting_workspace as module


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'sorter_choices', lambda: [dict(id='rt-sort', label='RT-Sort', available=True, reason='')])
    service = module.SortingWorkspace(tmp_path)
    yield service
    service.close()


def upload(workspace, tmp_path, name):
    path = tmp_path / name
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('sig', shape=(4, 100), dtype='int16')
    return workspace.add_upload(path, name)


def test_upload_rejects_non_raw_files(workspace, tmp_path):
    path = tmp_path / 'bad.h5'
    path.write_bytes(b'not an hdf5 file')
    with pytest.raises(ValueError, match='raw Maxwell'):
        workspace.add_upload(path, path.name)
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('spikes', data=[1, 2])
    with pytest.raises(ValueError, match='no supported Maxwell raw'):
        workspace.add_upload(path, path.name)
    assert workspace.control({})['files'] == []


def test_selected_batch_and_download(workspace, tmp_path, monkeypatch):
    baseline, first, second = [upload(workspace, tmp_path, name) for name in ('baseline.h5', 'first.h5', 'second.h5')]
    entered, release = threading.Event(), threading.Event()
    calls = []

    def run(baseline_path, paths, output, **kwargs):
        calls.append((baseline_path, paths, kwargs))
        entered.set()
        assert release.wait(5)
        artifacts = []
        for path in paths:
            artifact = Path(path).with_suffix('.npz')
            artifact.write_bytes(b'result')
            artifacts.append(dict(recording_path=path, spikes_path=str(artifact), unit_count=1, spike_count=2))
        return dict(recordings=artifacts, output_dir=str(output))

    monkeypatch.setattr(module, 'run_batch_sorting', run)
    command = dict(action='run', sorter='rt-sort', baseline_id=baseline['id'], target_ids=[second['id'], first['id'], second['id']], params={'device': 'cpu'})
    job = workspace.control(command)
    try:
        assert entered.wait(5)
        with pytest.raises(ValueError, match='already running'):
            workspace.control(command)
    finally:
        release.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = workspace.control({})['jobs'][0]
        if current['status'] in ('completed', 'failed'):
            break
        time.sleep(.01)
    assert current['status'] == 'completed', current
    assert calls[0][:2] == (baseline['path'], [second['path'], first['path']])
    assert calls[0][2]['params'] == {'device': 'cpu'}
    assert [r['name'] for r in current['result']['recordings']] == ['second.h5', 'first.h5']
    assert workspace.artifact(job['id'], 0).read_bytes() == b'result'
    with pytest.raises(ValueError):
        workspace.artifact(job['id'], -1)
    with pytest.raises(ValueError):
        workspace.artifact('missing', 0)


def test_invalid_selection_and_failed_sorter(workspace, tmp_path, monkeypatch):
    baseline = upload(workspace, tmp_path, 'baseline.h5')
    command = dict(action='run', sorter='rt-sort', baseline_id=baseline['id'], target_ids=[])
    with pytest.raises(ValueError, match='at least one'):
        workspace.control(command)
    command['target_ids'] = [baseline['id']]
    command['baseline_id'] = 'missing'
    with pytest.raises(ValueError, match='baseline'):
        workspace.control(command)
    command['baseline_id'] = baseline['id']

    def fail(*args, **kwargs):
        raise RuntimeError('Routing differs from baseline')

    monkeypatch.setattr(module, 'run_batch_sorting', fail)
    workspace.control(command)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = workspace.control({})['jobs'][0]
        if job['status'] == 'failed':
            break
        time.sleep(.01)
    assert job['status'] == 'failed'
    assert 'Routing differs' in job['error']


def test_nwb_requires_raw_electrical_series(workspace, tmp_path):
    path = tmp_path / 'recording.nwb'
    with h5py.File(path, 'w') as handle:
        handle.create_group('acquisition')
    with pytest.raises(ValueError, match='ElectricalSeries'):
        workspace.add_upload(path, path.name)
    with h5py.File(path, 'a') as handle:
        series = handle['acquisition'].create_group('ElectricalSeries')
        series.attrs['neurodata_type'] = 'ElectricalSeries'
        series.create_dataset('data', shape=(100, 4), dtype='float32')
    assert workspace.add_upload(path, path.name)['name'] == 'recording.nwb'
