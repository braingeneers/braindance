from pathlib import Path
import time

import numpy as np
import pytest

from braindance.examples.streaming_workshop.session import WorkshopSession
from braindance.examples.streaming_workshop.environments import WorkshopGame


@pytest.mark.parametrize('style', ['config', 'phases'])
def test_code_export_runs_and_preserves_hand_edits(tmp_path, workshop_config, style):
    import subprocess
    import sys
    from braindance.examples.streaming_workshop.code_export import code_bundle, export_bundle
    code = Path(workshop_config['functions_file']).read_text(encoding='utf-8')
    preview = code_bundle(workshop_config, code, style=style)
    assert not preview['errors']
    assert list(tmp_path.iterdir()) == []
    first = Path(export_bundle(tmp_path, workshop_config, code, style=style)['directory'])
    assert (first / 'participant_functions.py').read_text(encoding='utf-8') == code
    for args in (['--verify'], []):
        result = subprocess.run([sys.executable, str(first / 'experiment.py'), '--output-dir',
                                 str(tmp_path / 'runs'), *args], capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
        assert ('verified' if args else 'completed') in result.stdout
    (first / 'experiment.py').write_text('# hand edited\n', encoding='utf-8')
    second = Path(export_bundle(tmp_path, workshop_config, code, style=style)['directory'])
    assert second != first
    assert (first / 'experiment.py').read_text() == '# hand edited\n'
    assert code_bundle(workshop_config, 'def broken(', style=style)['errors']
    with pytest.raises(ValueError):
        export_bundle(tmp_path, workshop_config, 'def broken(', style=style)
    workshop_config['phases'] = [dict(id='record', type='native:phases3.RecordPhaseV3', params={'duration': .04})]
    native = Path(export_bundle(tmp_path, workshop_config, code, style=style)['directory'])
    result = subprocess.run([sys.executable, str(native / 'experiment.py'), '--output-dir',
                             str(tmp_path / 'native_runs')], cwd=native,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_train_hook_updates_decoder_and_encoder(tmp_path, workshop_config):
    target = tmp_path / 'learning.py'
    target.write_text('''
def decode(counts, dt_s, params, state):
    return [state.get('learned', 0.)], {}
def encode(observation, dt_s, params, state):
    return [state.get('rate', 0.), 0.], {}
def train(transition, dt_s, params, state):
    state['decode']['learned'] = state['decode'].get('learned', 0.) + .01
    state['encode']['rate'] = 10.
    return {'reward': transition['reward']}
''')
    workshop_config.update(functions_file=str(target), environment_seconds=.1)
    session = WorkshopSession(tmp_path / 'run', workshop_config)
    session.run(skip=True)
    assert session.status == 'completed', session.error
    np.testing.assert_allclose([h['action'][0] for h in session.history], np.arange(5)*.01)
    assert all(h['rates'] == [10., 0.] for h in session.history)
    assert 'reward' in session.snapshot['training']
    target.write_text(target.read_text().replace("return {'reward': transition['reward']}", 'return []'))
    with pytest.raises(ValueError, match='train must return'):
        session.reload_functions()


def test_phase_local_stimulation_routes_physical_electrodes(tmp_path, workshop_config, monkeypatch):
    from braindance.core.simulation import NeuralSimulationSource
    routed = []
    original = NeuralSimulationSource.on_stimulation
    def observe(self, action, frame, stim_electrodes):
        routed.append([stim_electrodes[i] for i in action[0]])
        return original(self, action, frame, stim_electrodes)
    monkeypatch.setattr(NeuralSimulationSource, 'on_stimulation', observe)
    code = tmp_path / 'mapping.py'
    code.write_text('def decode(counts,dt,params,state): return [0.]*params["action_size"], {}\ndef encode(*args): return [40.,0.], {}')
    workshop_config.update(functions_file=str(code), baseline_hz=[0.]*8, phases=[
        dict(id='first', type='environment', params={'environment_seconds': .1, 'stim_electrodes': [2,3], 'left_channels':[0], 'right_channels':[1]}),
        dict(id='second', type='environment', params={'environment_seconds': .1, 'stim_electrodes': [6,7], 'left_channels':[4], 'right_channels':[5]})])
    session = WorkshopSession(tmp_path / 'run', workshop_config)
    session.run()
    assert session.status == 'completed', session.error
    assert routed == [[2]]*4 + [[6]]*4


def test_phase_restart_retains_attempts_and_clock(tmp_path, workshop_config):
    import json
    workshop_config.update(speed=1, environment_seconds=.3, headless=False)
    session = WorkshopSession(tmp_path, workshop_config)
    session.start(skip=True)
    try:
        deadline = time.monotonic()+5
        while getattr(session, 'attempt', 0) < 1 and time.monotonic()<deadline:
            time.sleep(.01)
        session.commands.put({'kind':'restart_phase'})
        session.thread.join(timeout=5)
        assert session.status == 'completed', session.error
        records=[json.loads(p.read_text()) for p in sorted((session.run_dir/'attempts/environment').glob('*/attempt.json'))]
        assert [r['status'] for r in records] == ['restarted','completed']
        assert records[1]['start_frame'] >= records[0]['end_frame']
        assert len(list((session.run_dir/'attempts/environment').glob('*/functions.py'))) == 2
    finally:
        session.stop_event.set()
        session.thread.join(timeout=5)


def test_native_catalog_and_actual_record_phase(tmp_path, workshop_config):
    from braindance.examples.streaming_workshop.native_catalog import native_catalog
    from braindance.examples.streaming_workshop.native_runner import NativeSession
    catalog=native_catalog()
    assert 'native:phases3_ant.AntPhaseV3' in catalog
    assert catalog['native:phases3_ant.AntPhaseV3']['mapping_contract']['motor_min'] == 8
    workshop_config.update(experiment_name='native_test', channels=400, num_neurons=12,
                          grid_shape=[20, 20], phases=[dict(id='native_record', type='native:phases3.RecordPhaseV3', params={'duration': .04})])
    session=NativeSession(tmp_path, workshop_config)
    session.run()
    assert session.status == 'completed', session.error
    assert list(tmp_path.glob('native_*/native_test_summary.json'))
    assert list(tmp_path.glob('native_*/recordings/**/*.raw.h5')) or list(tmp_path.glob('native_*/**/*.raw.h5'))
    from braindance.core.replay import H5ReplaySource
    source = H5ReplaySource(next(tmp_path.glob('native_*/**/*.raw.h5')), speed=0)
    try:
        assert source.num_channels == 400
        assert source.mapping['electrode'][20] == 220
        assert source.read(400)['raw_float32'].shape == (400, 400)
    finally:
        source.close()


def test_native_phase_settings_configure_without_leaking(tmp_path, workshop_config):
    from braindance.examples.streaming_workshop.native_runner import NativeSession, verify_native
    phase = dict(id='stim', type='native:phases3.FrequencyStimPhaseV3',
                 params={'stim_command': [[0],100,100], 'stim_freq': 5., 'duration': .04},
                 settings={'stim_electrodes':[0]})
    workshop_config.update(experiment_name='native_stim_test', phases=[phase])
    assert verify_native(workshop_config)['ok']
    session=NativeSession(tmp_path, workshop_config)
    session.run()
    assert session.status == 'completed', session.error
    workshop_config['phases']=[phase, dict(id='missing_mapping', type=phase['type'], params=phase['params'])]
    report=verify_native(workshop_config)
    assert not report['ok']
    assert any('missing_mapping: missing stim_electrodes' in error for error in report['errors'])


def test_different_games_and_channel_contracts_per_phase(tmp_path, workshop_config):
    pytest.importorskip('mujoco')
    from braindance.examples.streaming_workshop.experiment_spec import verify_spec
    workshop_config.update(baseline_hz=[0.]*8, phases=[
        dict(id='cart', type='environment', params={'environment':'cartpole', 'environment_seconds':.04}),
        dict(id='ant', type='environment', params={'environment':'ant', 'environment_seconds':.04})])
    session=WorkshopSession(tmp_path, workshop_config)
    session.run()
    assert session.status == 'completed', session.error
    assert [len(h['action']) for h in session.history] == [1,1,8,8]
    workshop_config['phases'][1]['params']['stim_electrodes']=[0]
    report=verify_spec(workshop_config)
    assert not report['ok']
    assert any('exactly 2' in message for message in report['errors'])


def test_phase_builder_dependency_order_and_explicit_baseline(workshop_config):
    from braindance.examples.streaming_workshop.experiment_spec import verify_spec
    workshop_config['phases'] = [dict(id='play', type='environment', params={}),
                                  dict(id='baseline', type='recording', params={})]
    report = verify_spec(workshop_config)
    assert not report['ok']
    assert report['phases'][0]['inputs']['recording_baseline_hz'] is None
    workshop_config['phases'].reverse()
    report = verify_spec(workshop_config)
    assert report['ok']
    assert report['phases'][1]['inputs']['recording_baseline_hz'] == 'phase baseline'
    workshop_config['phases'] = [dict(id='play', type='environment', params={})]
    workshop_config['baseline_hz'] = [0.] * 8
    assert verify_spec(workshop_config)['ok']
    workshop_config['phases'][0]['params']['environment_seconds'] = .015
    assert not verify_spec(workshop_config)['ok']


def test_configured_repeated_phases_run_and_preflight_has_no_output(tmp_path, workshop_config, monkeypatch):
    workshop_config.update(experiment_name='class_experiment', phases=[
        dict(id='record_a', type='recording', params={'record_seconds': .04}),
        dict(id='play_a', type='environment', params={'environment_seconds': .04}),
        dict(id='record_b', type='recording', params={'record_seconds': .06}),
        dict(id='play_b', type='environment', params={'environment_seconds': .06})])
    probe = WorkshopSession(tmp_path / 'verify', dict(workshop_config))
    probe.run(verify_only=True)
    assert probe.status == 'verified', probe.error
    assert probe.env is None
    assert probe.snapshot['scene'] is None
    assert not (tmp_path / 'verify').exists()
    session = WorkshopSession(tmp_path / 'run', dict(workshop_config))
    scenes = {}
    publish = session.publish

    def capture_scene(*args, **kwargs):
        publish(*args, **kwargs)
        snapshot = session.snapshot
        scenes.setdefault(snapshot['phase'], []).append(snapshot.get('scene'))

    monkeypatch.setattr(session, 'publish', capture_scene)
    session.run()
    assert session.status == 'completed', session.error
    assert session.snapshot['phases'] == ['record_a', 'play_a', 'record_b', 'play_b']
    assert len(session.history) == 10
    for phase in ('record_a', 'record_b'):
        assert scenes[phase] and all(scene is None for scene in scenes[phase])
    for phase in ('play_a', 'play_b'):
        assert any(scene and scene['kind'] == 'cartpole' for scene in scenes[phase])
    assert (session.run_dir / 'class_experiment_summary.json').exists()
    assert (session.run_dir / 'experiment.json').exists()


def test_verify_python_without_execution(tmp_path):
    from braindance.examples.streaming_workshop.profiles import verify_python
    sentinel = tmp_path / 'should_not_exist'
    code = f'open({str(sentinel)!r}, "w").write("executed")\ndef encode(a,b,c,d): return [], {{}}\ndef decode(*args): return [], {{}}'
    assert verify_python(code)['ok']
    assert not sentinel.exists()
    broken = verify_python('def encode(:\n    pass')
    assert not broken['ok'] and broken['line'] == 1
    assert not verify_python('def encode(a): pass\ndef decode(*args): pass')['ok']
    assert not verify_python('def encode(*args): pass')['ok']
    assert not verify_python('def encode(*args): pass\ndef decode(*args, required): pass')['ok']


def test_profile_roundtrip_revisions_and_validation(tmp_path):
    from braindance.examples.streaming_workshop.profiles import ProfileStore
    from braindance.examples.streaming_workshop import functions
    store = ProfileStore(tmp_path / 'profiles', functions.__file__)
    code = store.template() + '\nraise RuntimeError("must not execute on save or load")\n'
    first = store.save('class_1', code, {'environment': 'ant', 'adjacency': [[0, .5], [-.2, 0]]})
    assert first['settings']['adjacency'][1][0] == -.2
    second = store.save('class_1', first['code'], {'environment': 'cartpole'})
    assert second['code'].count('SETTINGS =') == 1
    assert store.list() == ['class_1']
    assert len(list((store.directory / 'revisions/class_1').glob('*.py'))) == 3
    with pytest.raises(SyntaxError):
        store.save('class_1', 'def broken(', {})
    assert store.load('class_1') == second
    with pytest.raises(ValueError):
        store.save('../outside', code, {})


def test_ant_actions_geometry_and_episode_reset(tmp_path, workshop_config):
    pytest.importorskip('mujoco')
    workshop_config.update(environment='ant', episode_seconds=.1, environment_seconds=.6)
    session = WorkshopSession(tmp_path, workshop_config)
    session.run(skip=True)
    assert session.status == 'completed', session.error
    assert session.episodes >= 5
    assert session.snapshot['last_episode_end'] == 'episode time limit'
    assert all(len(h['action']) == 8 for h in session.history)
    assert any(np.ptp(h['action']) > 0 for h in session.history)
    assert len(session.scene['geometry']) >= 9
    assert all(np.isfinite(g['a']).all() and np.isfinite(g['b']).all() for g in session.scene['geometry'])


@pytest.fixture
def workshop_config():
    from braindance.examples.streaming_workshop import functions
    return dict(environment='cartpole', channels=8, seed=7, source=None, live_config=None,
                record_seconds=.1, environment_seconds=.2, causal_repeats=1,
                speed=0, detection='events', threshold_uv=-30., write_output=False,
                functions_file=functions.__file__, amplitude_mv=100., phase_width_us=100,
                headless=True)


def test_three_phases_share_clock_and_causal_trials(tmp_path, workshop_config):
    session = WorkshopSession(tmp_path, workshop_config)
    session.run()
    assert session.status == 'completed', session.error
    assert session.causal_trials == [[1, 1], [1, 1]]
    assert session.snapshot['phases'] == ['recording', 'causal', 'environment']
    assert len(session.history) == 5 + 4 * 15 + 10
    np.testing.assert_allclose([h['t'] for h in session.history], np.arange(1, 76) * .02)
    expected = np.sum([h['counts'] for h in list(session.history)[:5]], axis=0) / .1
    np.testing.assert_allclose(session.params['baseline_hz'], expected)
    assert (session.run_dir / 'calibration.pkl').exists()


def test_episode_reward_resets_without_losing_total(tmp_path, workshop_config, monkeypatch):
    workshop_config.update(episode_seconds=.04, environment_seconds=.10)
    session = WorkshopSession(tmp_path, workshop_config)
    rewards = []
    publish = session.publish

    def capture_reward(*args, **kwargs):
        publish(*args, **kwargs)
        if session.snapshot.get('scene'):
            rewards.append((session.reward, session.episode_reward))

    monkeypatch.setattr(session, 'publish', capture_reward)
    session.run(skip=True)
    assert session.status == 'completed', session.error
    assert session.snapshot['episodes'] == 2
    assert session.snapshot['reward'] == 5.
    assert session.snapshot['episode_reward'] == 1.
    assert (2., 0.) in rewards
    assert (4., 0.) in rewards


def test_custom_mappings_actual_action_and_stimulus(tmp_path, workshop_config):
    functions = tmp_path / 'my_functions.py'
    functions.write_text('''
def decode(counts, dt_s, params, state):
    return [0.375], {"counts": counts.tolist()}
def encode(observation, dt_s, params, state):
    return [0., 40.], {"angle": float(observation[2])}
''')
    workshop_config['functions_file'] = str(functions)
    session = WorkshopSession(tmp_path / 'out', workshop_config)
    session.run(skip=True)
    assert session.status == 'completed', session.error
    assert len(session.history) == 10
    assert session.snapshot['phases'] == ['environment']
    assert all(h['action'] == [.375] and h['rates'] == [0., 40.] for h in session.history)
    assert sum(1 in h['delivered'] for h in session.history) == 8
    assert all(0 not in h['delivered'] for h in session.history)
    assert session.scene['observation'] == list(session.history)[-1]['observation']


def test_invalid_quick_start_pools_rejected(tmp_path, workshop_config):
    workshop_config.update(left_channels=[0], right_channels=[0])
    session = WorkshopSession(tmp_path, workshop_config)
    session.run(skip=True)
    assert session.status == 'error'
    assert 'disjoint' in session.error


def test_restart_invalid_setup_shows_error_without_stale_history(tmp_path, workshop_config):
    session = WorkshopSession(tmp_path, workshop_config)
    session.run(skip=True)
    assert session.status == 'completed'
    session.config['environment'] = 'missing'
    session.run(skip=True)
    assert session.snapshot['status'] == 'error'
    assert 'Unknown environment' in session.snapshot['error']
    assert 'history' not in session.snapshot


@pytest.mark.parametrize('environment', ['cartpole', 'foodland', 'ant'])
def test_actual_environment_adapter(environment):
    if environment == 'foodland':
        pytest.importorskip('gym')
        pytest.importorskip('pygame')
    if environment == 'ant':
        pytest.importorskip('mujoco')
    game = WorkshopGame(environment)
    try:
        obs = game.reset()
        assert len(obs) == len(game.observation_names)
        action = np.zeros(len(game.action_names))
        if environment == 'foodland':
            action[1] = .5
            before = np.asarray(game.env.agent_pos).copy()
        after, reward, done = game.step(action)
        assert after.shape == obs.shape
        assert np.all(np.isfinite(after))
        assert game.scene(after)['kind'] == environment
        if environment == 'foodland':
            assert np.linalg.norm(np.asarray(game.env.agent_pos) - before) > 0
    finally:
        game.close()


def test_threshold_spikes_detected_from_waveforms(tmp_path, workshop_config):
    workshop_config.update(detection='threshold')
    session = WorkshopSession(tmp_path, workshop_config)
    session.run(skip=True)
    assert session.status == 'completed', session.error
    assert sum(sum(h['counts']) for h in session.history) > 0


def test_spatial_recording_uses_maxwell_raw_pipeline(tmp_path, workshop_config):
    from braindance.core.replay import H5ReplaySource
    workshop_config.update(channels=400, num_neurons=12, grid_shape=[20, 20],
        detection='threshold', write_output=True, record_seconds=.2,
        phases=[dict(id='record', type='recording', params={})])
    session = WorkshopSession(tmp_path, workshop_config)
    session.run()
    assert session.status == 'completed', session.error
    spatial = session.snapshot['spatial']
    assert np.asarray(spatial['channel_positions']).shape == (400, 2)
    assert np.asarray(spatial['neuron_positions']).shape == (12, 2)
    assert np.asarray(session.snapshot['adjacency']).shape == (12, 12)
    assert len(spatial['peak_uv']) == len(spatial['rms_uv']) == 400
    assert sum(sum(row['counts']) for row in session.history) > 0
    recording, = session.run_dir.rglob('*.raw.h5')
    source = H5ReplaySource(recording, speed=0)
    try:
        assert source.num_channels == 400
        batch = source.read(4000)
        assert batch['raw_float32'].shape == (4000, 400)
        np.testing.assert_array_equal(source.mapping, session.source.mapping)
        uv = batch['raw_float32'][-400:] * 1000 / source.gain
        np.testing.assert_allclose(spatial['peak_uv'], np.maximum(0, -uv.min(axis=0)), atol=.001)
        np.testing.assert_allclose(spatial['rms_uv'], np.sqrt(np.mean(uv * uv, axis=0)), atol=.001)
    finally:
        source.close()


def test_spatial_stimulation_uses_physical_ids_in_phase_validation(workshop_config):
    from braindance.examples.streaming_workshop.experiment_spec import verify_spec
    workshop_config.update(channels=400, num_neurons=12, grid_shape=[20, 20],
        stim_electrodes=[220, 4180])
    assert verify_spec(workshop_config)['ok']
    workshop_config['stim_electrodes'] = [20, 21]  # These channel numbers are not mapped electrode IDs.
    assert not verify_spec(workshop_config)['ok']


def test_reload_preserves_last_valid_module_on_error(tmp_path, workshop_config):
    target = tmp_path / 'functions.py'
    target.write_text(Path(workshop_config['functions_file']).read_text())
    workshop_config['functions_file'] = str(target)
    session = WorkshopSession(tmp_path / 'run', workshop_config)
    session.run(skip=True)
    previous = session.module
    target.write_text('invalid syntax here!')
    with pytest.raises(SyntaxError):
        session.reload_functions()
    assert session.module is previous
    target.write_text('def decode(*args): return [0.], {}\ndef encode(*args): return [10., 20.], {}')
    session.reload_functions()
    assert session.module is not previous
    assert session.module.encode(None, None, None, None)[0] == [10., 20.]


@pytest.mark.parametrize('channels', [8, 400])
def test_rt_sort_uses_internal_chunks_and_unit_counts(tmp_path, workshop_config, monkeypatch, channels):
    import sys
    import types
    calls = []

    class Sorter:
        num_seqs, samp_freq, buffer_size = 2, 20, 100
        @classmethod
        def load_from_file(cls, path, model=None):
            return cls()
        def reset(self):
            pass
        def running_sort(self, raw, latest_frame, use_numba=False):
            calls.append((raw.copy(), latest_frame))
            return [(0, latest_frame / 20)]

    module = types.ModuleType('braindance.core.spikesorter.rt_sort')
    module.RTSort = Sorter
    monkeypatch.setitem(sys.modules, module.__name__, module)
    workshop_config.update(detection='rt-sort', sorter_path='mock-sorter.pkl', channels=channels)
    if channels == 400:
        workshop_config.update(num_neurons=12, grid_shape=[20, 20])
    session = WorkshopSession(tmp_path, workshop_config)
    session.run(skip=True)
    assert session.status == 'completed', session.error
    assert len(calls) == 40
    assert all(raw.shape == (100, channels) for raw, _ in calls)
    assert [frame for _, frame in calls] == list(range(99, 4000, 100))
    assert all(h['counts'] == [4, 0] for h in session.history)
    assert max(np.max(np.abs(raw)) for raw, _ in calls) < 1  # No second ADC conversion.


def test_runtime_function_error_pause_reload_resume(tmp_path, workshop_config):
    target = tmp_path / 'editable.py'
    target.write_text('''
def decode(counts, dt, params, state):
    if sum(counts): raise ValueError("exercise failure")
    return [0.], {}
def encode(*args): return [0., 0.], {}
''')
    workshop_config.update(functions_file=str(target), headless=False, environment_seconds=.1)
    session = WorkshopSession(tmp_path / 'run', workshop_config)
    session.start(skip=True)
    deadline = time.monotonic() + 5
    try:
        while not session.paused and time.monotonic() < deadline:
            time.sleep(.01)
        assert session.paused
        assert 'exercise failure' in session.error
        frame = session.env.latest_frame
        target.write_text('def decode(*args): return [.25], {}\ndef encode(*args): return [0., 0.], {}')
        session.commands.put({'kind': 'reload'})
        session.commands.put({'kind': 'pause'})
        session.thread.join(timeout=5)
        assert session.status == 'completed', session.error
        assert frame < session.env.latest_frame
        assert len(session.history) == 5
    finally:
        session.stop_event.set()
        session.thread.join(timeout=5)


@pytest.mark.parametrize('loop', [False, True])
def test_h5_replay_through_workshop(tmp_path, workshop_config, loop):
    from braindance.core.maxwell_env import MaxwellEnv
    from braindance.core.simulation import NeuralSimulationSource
    source = NeuralSimulationSource(speed=0)
    env = MaxwellEnv(name='fixture', save_dir=tmp_path, verbose=0,
                     replay={'source': source, 'write_output': True})
    env.step(buffer_size=750 if loop else 10000)
    env.close()
    workshop_config['source'] = str(tmp_path / 'fixture.raw.h5')
    workshop_config['loop'] = loop
    session = WorkshopSession(tmp_path / 'replayed', workshop_config)
    session.run(skip=True)
    assert session.status == 'completed', session.error
    assert 'open-loop' in session.snapshot['source']
    assert session.snapshot['adjacency'] is None


def test_performance_rates_exclude_pacing_from_capacity(tmp_path):
    from collections import deque
    session = WorkshopSession(tmp_path, {'speed': 1})
    session.sampling_hz, session.bin_frames, session.raw_channels = 20000, 400, 8
    session._performance_ticks = deque([(0.02, 0.004), (0.02, 0.006)])
    session._total_frames, session._total_bins, session._deadline_misses = 800, 2, 0
    rates = session.performance()
    assert rates['sampling_hz'] == rates['acquired_hz'] == 20000
    assert rates['processing_capacity_hz'] == 80000
    assert rates['bin_hz'] == 50
    assert rates['realtime_factor'] == 1
    assert rates['p95_ms'] == pytest.approx(5.9)


def test_threaded_raw_run_reports_complete_sample_counts(tmp_path, workshop_config):
    import json
    workshop_config.update(detection='threshold', environment_seconds=.1)
    session = WorkshopSession(tmp_path, workshop_config)
    session.start(skip=True)
    session.thread.join(timeout=10)
    assert not session.thread.is_alive()
    assert session.status == 'completed', session.error
    perf = session.snapshot['performance']
    assert perf['worker'] == 'thread'
    assert perf['sampling_hz'] == 20000
    assert perf['frames'] == 2000
    assert perf['bins'] == 5
    assert perf['acquired_hz'] > 0
    assert perf['deadline_misses'] == 0  # unpaced has no wall-clock deadline
    saved = json.loads((session.run_dir / 'performance.json').read_text())
    assert saved['frames'] == perf['frames']


def test_vectorized_threshold_matches_samplewise_reference(tmp_path, workshop_config):
    from braindance.core.replay import H5ReplaySource
    workshop_config.update(detection='threshold', write_output=True, record_seconds=.2,
                           phases=[dict(id='record', type='recording', params={})])
    session = WorkshopSession(tmp_path, workshop_config)
    session.run()
    assert session.status == 'completed', session.error
    source = H5ReplaySource(next(session.run_dir.rglob('*.raw.h5')), speed=0)
    previous = np.zeros(session.raw_channels, dtype=bool)
    last = np.full(session.raw_channels, -100000, dtype=np.int64)
    try:
        for history in session.history:
            batch = source.read(400)
            counts = np.zeros(session.raw_channels, dtype=int)
            # Reconstruct simulator conversion exactly; H5's unit conversion has
            # different float32 rounding right at the detection threshold.
            raw_uv = (batch['raw_uint16'].astype(np.float32) - 512) * .001 * 1000
            for frame, values in zip(batch['frame_numbers'], raw_uv):
                below = values < workshop_config['threshold_uv']
                hit = below & ~previous & ((int(frame) - last) >= 40)
                counts += hit
                last[hit] = int(frame)
                previous = below
            np.testing.assert_array_equal(counts, history['counts'])
    finally:
        source.close()


def test_pacing_recovers_scheduler_oversleep(tmp_path, monkeypatch):
    from collections import deque
    from types import SimpleNamespace
    import braindance.examples.streaming_workshop.session as module
    session = WorkshopSession(tmp_path, {'speed': 1})
    session.source = object()
    session.dt, session.bin_frames = .02, 400
    session.processing = [0.]
    session._performance_ticks = deque()
    session._total_frames = session._total_bins = session._deadline_misses = 0
    session._pacing_debt = 0.
    waits = []
    session.stop_event = SimpleNamespace(wait=waits.append)
    clock = iter([.004, .023, .028, .040])
    monkeypatch.setattr(module.time, 'perf_counter', lambda: next(clock))
    session._tick_start = 0.
    session.finish_tick()
    assert session._pacing_debt == pytest.approx(.003)
    session._tick_start = .023
    session.finish_tick()
    assert waits == pytest.approx([.016, .012])
    assert session._pacing_debt == pytest.approx(0.)
    assert session._total_frames == 800
