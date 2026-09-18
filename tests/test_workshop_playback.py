import csv
import json

import numpy as np
import pytest

from braindance.examples.streaming_workshop.playback import PlaybackSession, inspect_playback
from test_streaming_workshop import workshop_config  # noqa: F401


def test_recording_playback_preserves_raw_events_without_simulated_game(tmp_path, maxwell_h5):
    session = PlaybackSession(tmp_path, {'playback_source': str(maxwell_h5), 'speed': 0})
    session.run()
    assert session.status == 'completed', session.error
    assert session.snapshot['scene'] is None
    assert session.snapshot['history'][-1]['counts'] == [0, 1, 0, 1]
    assert len(session.snapshot['raw']) == 4
    assert session.snapshot['playback_time'] == .01
    assert not (tmp_path / 'playback.jsonl').exists()


def test_cartpole_legacy_misaligned_header_and_numpy_state(tmp_path):
    log = tmp_path / 'cartpole_game_log.csv'
    with log.open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time', 'pole_angle', 'reward', 'action', 'spike_count_l', 'spike_count_r', 'state'])
        writer.writerow([.02, .3, 1, .5, '[1, 2]', '[ 1.2 0.1 0.3 -0.2 ]'])
        writer.writerow([.04, .4, 1, -.5, '[2, 3]', '[ 1.3 0.2 0.4 -0.1 ]'])
    assert inspect_playback(tmp_path)['games'] == ['cartpole']
    session = PlaybackSession(tmp_path, {'playback_source': str(log), 'speed': 0})
    session.run()
    assert session.status == 'completed', session.error
    assert session.snapshot['scene']['observation'] == [1.3, .2, .4, -.1]
    assert session.snapshot['history'][0]['action'] == [.5]
    assert session.snapshot['reward'] == 2
    assert not session.snapshot['playback_notice']


def test_cartpole_angle_only_is_explicit(tmp_path):
    log = tmp_path / 'run_game_log.csv'
    log.write_text('time,pole_angle,reward,action,spike_rates,state\n0.02,0.3,1,0.5,"[3,4]",game\n')
    session = PlaybackSession(tmp_path, {'playback_source': str(log), 'speed': 0})
    session.run()
    assert session.status == 'completed', session.error
    assert session.snapshot['scene']['observation'] == [0., 0., .3, 0.]
    assert 'cart position is unavailable' in session.snapshot['playback_notice']


def test_raw_and_matching_game_log_replay_together(tmp_path, maxwell_h5):
    log = maxwell_h5.with_name('fixture_game_log.csv')
    log.write_text('time,pole_angle,reward,action,state\n0.005,0.3,1,0.5,"[1,0,0.3,0]"\n')
    session = PlaybackSession(tmp_path, {'playback_source': str(maxwell_h5), 'speed': 0})
    session.run()
    assert session.status == 'completed', session.error
    assert session.snapshot['scene']['observation'] == [1., 0., .3, 0.]
    assert session.snapshot['history'][-1]['counts'] == [0, 1, 0, 1]


def test_workshop_records_and_replays_original_game_and_signals(tmp_path, workshop_config):
    from braindance.examples.streaming_workshop.session import WorkshopSession
    workshop_config.update(environment_seconds=.1)
    original = WorkshopSession(tmp_path, workshop_config)
    original.run(skip=True)
    assert original.status == 'completed', original.error
    rows = [json.loads(line) for line in (original.run_dir / 'playback.jsonl').read_text().splitlines()]
    assert len(rows) == 5
    replay = PlaybackSession(tmp_path, {'playback_source': str(original.run_dir), 'speed': 0})
    replay.run()
    assert replay.status == 'completed', replay.error
    assert replay.snapshot['scene'] == original.snapshot['scene']
    assert replay.snapshot['raw'] == original.snapshot['raw']
    assert replay.snapshot['history'] == list(original.history)
    assert replay.snapshot['reward'] == original.snapshot['reward']
    assert replay.snapshot['episode_reward'] == original.snapshot['episode_reward']
    np.testing.assert_allclose([r['playback_time'] for r in rows], [.02, .04, .06, .08, .1])


def test_unsupported_game_log_is_not_claimed_supported(tmp_path):
    log = tmp_path / 'ant_game_log.csv'
    log.write_text('time,reward,action,state,z_pos\n.02,1,0,game,.5\n')
    with pytest.raises(ValueError, match='No saved'):
        inspect_playback(log)


def test_pause_step_and_stop_remain_responsive_during_game_log_gaps(tmp_path):
    import time
    log = tmp_path / 'slow_game_log.csv'
    log.write_text('time,pole_angle,reward,action,state\n0.01,0.1,1,0,game\n10,0.2,1,1,game\n20,0.3,1,0,game\n')
    session = PlaybackSession(tmp_path, {'playback_source': str(log), 'speed': 1})
    session.start()
    try:
        deadline = time.monotonic() + 2
        while not session.snapshot.get('history') and time.monotonic() < deadline:
            time.sleep(.01)
        assert session.snapshot['playback_time'] == .01
        session.commands.put({'kind': 'pause'})
        while not session.paused and time.monotonic() < deadline:
            time.sleep(.01)
        assert session.paused
        assert session.snapshot['playback_time'] == .01
        session.commands.put({'kind': 'step'})
        while session.snapshot['playback_time'] == .01 and time.monotonic() < deadline:
            time.sleep(.01)
        assert session.snapshot['playback_time'] == 10
        assert session.paused
    finally:
        session.stop_event.set()
        session.thread.join(timeout=2)
    assert session.status == 'stopped'


def test_experiment_manifest_empty_paths_and_nonfinite_logs(tmp_path):
    manifest = tmp_path / 'experiment.json'
    manifest.write_text('{}')
    log = tmp_path / 'cartpole_game_log.csv'
    log.write_text('time,pole_angle,reward,action,state\n.02,.1,nan,0,game\n')
    assert inspect_playback(manifest)['path'] == str(tmp_path)
    with pytest.raises(ValueError, match='Choose a recording'):
        inspect_playback('')
    session = PlaybackSession(tmp_path, {'playback_source': str(manifest), 'speed': 0})
    session.run()
    assert session.status == 'error'
    assert 'Nonfinite' in session.error
    json.dumps(session.snapshot, allow_nan=False)


def test_saved_non_cartpole_scenes_are_preserved(tmp_path):
    frames = [dict(playback_time=.02, phase='foodland', scene={'kind': 'foodland', 'agent': [2, 3], 'food': [[5, 6]]}, history=[]),
              dict(playback_time=.04, phase='ant', scene={'kind': 'ant', 'geometry': [{'a': [0, 1, 2], 'b': [1, 2, 3], 'radius': .2}]}, history=[])]
    (tmp_path / 'playback.jsonl').write_text(''.join(json.dumps(frame) + '\n' for frame in frames))
    session = PlaybackSession(tmp_path, {'playback_source': str(tmp_path), 'speed': 0})
    session.run()
    assert session.status == 'completed', session.error
    assert session.snapshot['scene'] == frames[-1]['scene']
