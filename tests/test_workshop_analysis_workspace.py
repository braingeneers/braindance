"""Real file and scientific API checks for the offline analysis workspace."""
import json
import time

import h5py
import numpy as np
import pytest

from braindance.examples.streaming_workshop.analysis_workspace import AnalysisWorkspace


def recording(tmp_path):
    path = tmp_path / 'recording.raw.h5'
    with h5py.File(path, 'w') as h:
        h['data_store/data0000/groups/routed/raw'] = np.arange(2000).reshape(2,1000)
        h['data_store/data0000/groups/routed/frame_nos'] = np.arange(10000,11000)
        h['data_store/data0000/spikes'] = np.array([(10010,0,2),(10012,1,3),(10210,0,2),(10212,1,3),(10610,0,2),(10612,1,3)],dtype=[('frameno','i8'),('channel','i4'),('amplitude','f4')])
    return path


def wait(workspace, job):
    for _ in range(1000):
        state=workspace.control(dict(action='status',job_id=job['id']))
        if state['status'] in ('completed','failed'):
            return state
        time.sleep(.01)
    pytest.fail('Analysis did not finish')


def test_phase_bounds_raw_spikes_and_metadata(tmp_path):
    recording(tmp_path)
    attempt=tmp_path/'attempts'/'baseline'/'0001'; attempt.mkdir(parents=True)
    (attempt/'attempt.json').write_text(json.dumps(dict(phase='baseline',attempt=1,start_frame=10199,end_frame=10499,status='completed')))
    workspace=AnalysisWorkspace(tmp_path/'out')
    try:
        state=workspace.control(dict(action='load',path=str(tmp_path)))['workspace']
        row=state['files'][0]
        job=workspace.control(dict(action='run',kind='raw',file_id=row['id'],params=dict(sample_rate=1000,duration_ms=1000)))
        result=wait(workspace,job)
        assert result['status']=='completed',result
        assert result['kind']=='raw'
        assert result['result']['plots'][0]['y']==list(range(200,500))
        result=wait(workspace,workspace.control(dict(action='run',kind='spikes',file_id=row['id'],params=dict(sample_rate=1000))))
        assert result['result']['plots'][0]['x']==[10.,12.]
    finally:
        workspace.close()


def test_real_spikelab_sttc_and_signed_latency(tmp_path):
    pytest.importorskip('spikelab')
    path=tmp_path/'sorted.npz'
    np.savez(path,times_ms=[100,300,500,102,302,502],unit_ids=[7,7,7,9,9,9],all_unit_ids=[7,9],duration_ms=1000)
    workspace=AnalysisWorkspace(tmp_path/'out')
    try:
        row=workspace.control(dict(action='load',path=str(path)))['workspace']['files'][0]
        result=workspace._analyze(row,'sttc',dict(delta_ms=5),lambda *args:None)
        assert np.allclose(result['plots'][0]['z'],1)
        result=workspace._analyze(row,'latency',dict(unit_id=7,window_ms=10,bin_ms=2),lambda *args:None)
        plot=result['plots'][0]
        assert sum(plot['z'][0])==3
        assert plot['x'][int(np.argmax(plot['z'][0]))]>0
    finally:
        workspace.close()


def test_overlap_respects_phase_and_log_time(tmp_path):
    path=recording(tmp_path)
    path.with_name('recording_log.csv').write_text('time,time_mod\n0.01,0.01\n0.22,0.22\n0.9,0.9\n')
    workspace=AnalysisWorkspace(tmp_path/'out')
    try:
        row=workspace._file(path,'phase',[10199,10499])
        result=workspace._analyze(row,'overlap',dict(sample_rate=1000,pre_ms=10,post_ms=20),lambda *args:None)
        series=result['plots'][0]['series']
        assert len(series)==1
        assert series[0]['y']==list(range(210,240))
    finally:
        workspace.close()


def test_json_load_excludes_configured_analysis_phases(tmp_path):
    (tmp_path/'experiment.json').write_text(json.dumps(dict(phases=[dict(id='record',type='record'),dict(id='analysis',type='custom_analysis')])))
    workspace=AnalysisWorkspace(tmp_path/'out')
    try:
        state=workspace.control(dict(action='load',path=str(tmp_path/'experiment.json')))['workspace']
        assert [p['name'] for p in state['phases']]==['record']
        assert state['warnings']
    finally:
        workspace.close()


def test_job_errors_are_visible(tmp_path):
    path=recording(tmp_path)
    workspace=AnalysisWorkspace(tmp_path/'out')
    try:
        row=workspace.control(dict(action='load',path=str(path)))['workspace']['files'][0]
        job=workspace.control(dict(action='run',kind='raw',file_id=row['id'],params=dict(channel=999)))
        assert wait(workspace,job)['error']=='Channel is outside this recording'
    finally:
        workspace.close()


def test_replay_overlap_uses_frames_instead_of_wall_clock(tmp_path):
    path=recording(tmp_path)
    path.with_name('recording_log.csv').write_text('time,replay_frame\n99,10220\n')
    workspace=AnalysisWorkspace(tmp_path/'out')
    try:
        row=workspace._file(path,'phase')
        result=workspace._analyze(row,'overlap',dict(sample_rate=1000,pre_ms=10,post_ms=20),lambda *args:None)
        assert result['plots'][0]['series'][0]['y']==list(range(210,240))
    finally:
        workspace.close()


def test_native_analysis_and_groups_not_presented_as_recording_phases(tmp_path):
    path=recording(tmp_path)
    folder=tmp_path/'0001'; folder.mkdir(); path.rename(folder/path.name)
    (tmp_path/'experiment_log.json').write_text(json.dumps(dict(phase_log=[
        dict(phase_idx=0,phase_name='acquire',phase_class='RecordPhaseV3',recording_dir='0001'),
        dict(phase_idx=1,phase_name='response',phase_class='CausalAnalysisV3',recording_dir='0001'),
        dict(phase_idx=2,phase_name='group',phase_class='PhaseGroup',recording_dir='0001') ])))
    workspace=AnalysisWorkspace(tmp_path/'out')
    try:
        state=workspace.control(dict(action='load',path=str(tmp_path)))['workspace']
        assert [p['name'] for p in state['phases']]==['acquire']
    finally:
        workspace.close()


def test_sorted_results_preserve_each_phase_interval(tmp_path, monkeypatch):
    from braindance.examples.streaming_workshop import analysis_sorting
    path=recording(tmp_path)
    workspace=AnalysisWorkspace(tmp_path/'out')
    try:
        row=workspace._file(path,'baseline',[10199,10499])
        workspace.workspace.update(files=[row],phases=[dict(id='baseline',files=[row['id']])])
        spikes=tmp_path/'sorted.npz'
        np.savez(spikes,times_ms=[10,210,610],unit_ids=[0,0,0],duration_ms=1000)
        monkeypatch.setattr(analysis_sorting,'run_sorting',lambda *args,**kwargs:dict(recordings=[dict(recording_path=str(path),spikes_path=str(spikes))]))
        workspace._analyze(row,'sorting',dict(baseline_path=str(path)),lambda *args:None)
        added=workspace.workspace['files'][-1]
        result=workspace._analyze(added,'spikes',dict(sample_rate=1000),lambda *args:None)
        assert result['plots'][0]['x']==[10.]
    finally:
        workspace.close()


def test_sorted_evoked_response_uses_logged_patterns(tmp_path):
    path = tmp_path / 'sorted.npz'
    np.savez(path, times_ms=[105., 125., 305.], unit_ids=[7, 7, 7],
             all_unit_ids=[7], duration_ms=1000.)
    path.with_name('sorted_log.csv').write_text('time,stim_pattern\n0.1,A\n0.3,A\n')
    workspace = AnalysisWorkspace(tmp_path / 'out')
    try:
        row = workspace.control(dict(action='load', path=str(path)))['workspace']['files'][0]
        assert row['capabilities']['overlap']
        result = wait(workspace, workspace.control(dict(action='run', kind='overlap',
                      file_id=row['id'], params=dict(pre_ms=20, post_ms=100, blank_ms=3))))
        assert result['status'] == 'completed', result
        assert result['result']['pattern_counts'] == {'A': 2}
        metrics = result['result']['response_metrics']
        assert metrics['short_probability'] == [[1.0]]
        assert metrics['late_mean_spikes'] == [[0.5]]
    finally:
        workspace.close()


def test_browse_defaults_to_configured_data_folder(tmp_path, monkeypatch):
    data = tmp_path / 'configured_data'
    experiment = data / 'experiment_one'
    experiment.mkdir(parents=True)
    (experiment / 'experiment.json').write_text('{}')
    (data / 'raw.h5').touch()
    (data / 'unrelated.txt').touch()
    monkeypatch.setenv('BRAINDANCE_DATA_DIR', str(data))
    workspace = AnalysisWorkspace(tmp_path / 'output')
    try:
        listing = workspace.control({'action': 'browse'})
        assert listing['path'] == str(data)
        assert [(entry['name'], entry['kind']) for entry in listing['entries']] == [
            ('experiment_one', 'experiment'), ('raw.h5', 'recording')]
        nested = workspace.control({'action': 'browse', 'path': str(experiment)})
        assert nested['parent'] == str(data)
        assert nested['entries'][0]['name'] == 'experiment.json'
        assert workspace.control({'action': 'browse', 'path': str(data / 'missing')})['warning']
        assert workspace.workspace['files'] == []
    finally:
        workspace.close()
