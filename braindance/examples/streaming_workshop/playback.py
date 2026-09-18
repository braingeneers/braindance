"""Read-only playback of saved workshop telemetry and Maxwell experiments."""
import ast
import csv
import json
import math
import os
from collections import deque
from pathlib import Path
import queue
import re
import threading
import time

import numpy as np

from braindance.core.replay import H5ReplaySource
from .discovery import is_experiment


def _files(path):
    if path.is_file():
        return [path]
    result = []
    for directory, children, names in os.walk(path, followlinks=False):
        folder = Path(directory)
        children[:] = sorted(name for name in children if not name.startswith('.')
                             and not (folder / name).is_symlink() and not is_experiment(folder / name))
        result.extend(folder / name for name in sorted(names) if not (folder / name).is_symlink())
    return sorted(result, key=lambda p: [int(v) if v.isdigit() else v.lower()
                                       for v in re.split(r'(\d+)', str(p))])


def _vector(value):
    if value is None or value == '':
        return []
    try:
        result = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError):
        try:
            result = [float(v) for v in value.strip('[] ').split()]
        except (ValueError, AttributeError):
            return []
    try:
        values = np.asarray(result, dtype=float).reshape(-1)
        return values.tolist() if np.all(np.isfinite(values)) else []
    except (TypeError, ValueError):
        return []


def _cartpole_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            try:
                stamp, angle = float(row['time']), float(row['pole_angle'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'Invalid CartPole time/angle in {path.name}') from exc
            if not math.isfinite(stamp) or not math.isfinite(angle):
                raise ValueError(f'Nonfinite CartPole time/angle in {path.name}')
            state = _vector(row.get('state'))
            # Historical writers put game_obs in column six under a seven-column header.
            if len(state) != 4:
                state = _vector(row.get('spike_count_r'))
            partial = len(state) != 4
            if partial:
                state = [0., 0., angle, 0.]
            action = _vector(row.get('action'))
            reward = float(row.get('reward') or 0)
            if not math.isfinite(reward):
                raise ValueError(f'Nonfinite CartPole reward in {path.name}')
            yield dict(t=stamp, observation=state, action=action, rates=[], delivered=[], counts=[],
                       reward=reward, partial=partial)


def inspect_playback(path):
    """Describe locally available saved playback without running experiment code."""
    if not str(path or '').strip():
        raise ValueError('Choose a recording or experiment path for playback')
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.name in ('experiment.json', 'experiment_log.json'):
        path = path.parent
    if not path.exists():
        raise ValueError(f'Playback path does not exist: {path}')
    files = _files(path)
    telemetry = [p for p in files if p.name == 'playback.jsonl']
    raw = [p for p in files if p.suffix.lower() in ('.h5', '.hdf5')
           or (p == path and p.suffix.lower() == '.npy')]
    logs = [p for p in files if p.name.endswith('_game_log.csv')]
    if path.is_file() and path.suffix.lower() in ('.h5', '.hdf5'):
        prefix = path.name.removesuffix('.h5').removesuffix('.hdf5').removesuffix('.raw')
        logs = [p for p in path.parent.glob(prefix + '_game_log.csv') if p.is_file()]
    supported_logs, unsupported = [], []
    for log in logs:
        with log.open(newline='', encoding='utf-8') as stream:
            fields = csv.DictReader(stream).fieldnames or []
        (supported_logs if {'time', 'pole_angle'} <= set(fields) else unsupported).append(log)
    if not telemetry and not raw and not supported_logs:
        raise ValueError('No saved workshop playback, Maxwell H5 recording, or supported CartPole game log found')
    games = []
    if telemetry:
        with telemetry[0].open(encoding='utf-8') as stream:
            first = next((json.loads(line) for line in stream if line.strip()), {})
        kind = (first.get('scene') or {}).get('kind')
        if kind:
            games.append(kind)
    elif supported_logs:
        games.append('cartpole')
    return dict(path=str(path), supported=True, mode='telemetry' if telemetry else 'recordings',
                files=[str(p) for p in raw], telemetry=[str(p) for p in telemetry],
                game_logs=[str(p) for p in supported_logs], games=games,
                unsupported_game_logs=[str(p) for p in unsupported],
                message=('Saved scenes and signals replay together.' if telemetry else
                         'Recorded CartPole state replays alongside available signals; angle-only logs show a fixed cart.' if supported_logs else
                         'Recorded signals available; no supported saved game state was found.'))


class PlaybackSession:
    """Expose saved frames through the workshop monitor's existing session protocol."""
    def __init__(self, output_dir, config):
        self.output_dir, self.config = Path(output_dir), dict(config)
        self.commands = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = None
        self.status, self.error, self.phase, self.paused = 'ready', '', 'playback', False
        self.snapshot = dict(status=self.status, phase=self.phase, error='', restart_supported=False)

    def start(self, overrides=None, skip=False):
        if self.thread and self.thread.is_alive():
            raise ValueError('Stop the current playback before starting another')
        self.config.update(overrides or {})
        self.stop_event.clear()
        self.commands = queue.Queue()
        self.paused = False
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def _controls(self):
        step = False
        while not self.stop_event.is_set():
            while not self.commands.empty():
                command = self.commands.get_nowait()['kind']
                if command == 'pause':
                    self.paused = not self.paused
                elif command == 'step':
                    self.paused, step = True, True
                self.snapshot = dict(self.snapshot, paused=self.paused)
            if not self.paused or step:
                return True
            self.stop_event.wait(.02)
        return False

    def _frames(self, info):
        if info['telemetry']:
            offset, previous = 0., 0.
            for file in info['telemetry']:
                with Path(file).open(encoding='utf-8') as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        frame = json.loads(line)
                        json.dumps(frame, allow_nan=False)
                        stamp = float(frame.get('playback_time', (frame.get('history') or [{}])[-1].get('t', 0)))
                        yield offset + stamp, frame
                        previous = max(previous, stamp)
                offset += previous
            return
        unused_logs = set(info['game_logs'])
        offset = 0.
        for file in info['files']:
            path = Path(file)
            prefix = path.name.removesuffix('.h5').removesuffix('.hdf5').removesuffix('.raw')
            log = next((p for p in info['game_logs'] if Path(p).parent == path.parent
                        and Path(p).name == prefix + '_game_log.csv'), None)
            rows = iter(_cartpole_rows(Path(log))) if log else iter(())
            upcoming, current = next(rows, None), None
            if log:
                unused_logs.discard(log)
            source = H5ReplaySource(file, speed=0)
            elapsed, reward = 0., 0.
            try:
                while True:
                    batch = source.read(max(1, round(source.sampling_hz * .02)))
                    if batch is None:
                        break
                    elapsed += len(batch['frame_numbers']) / source.sampling_hz
                    while upcoming is not None and upcoming['t'] <= elapsed:
                        current = upcoming
                        reward += current['reward']
                        upcoming = next(rows, None)
                    counts = np.bincount([e.channel for events in batch['events'] for e in events], minlength=source.num_channels).tolist()
                    uv = batch['raw_float32'][:, :16] * 1000 / source.gain
                    width = max(1, len(uv) // 80)
                    grouped = uv[:len(uv) // width * width].reshape(-1, width, uv.shape[1])
                    raw = np.stack([grouped.min(axis=1), grouped.max(axis=1)], axis=1).reshape(-1, uv.shape[1]).T.tolist()
                    history = dict(t=offset + elapsed, counts=counts, observation=[], action=[], rates=[], delivered=[])
                    frame = dict(phase=path.stem, raw=raw, scene=None, reward=reward)
                    if current:
                        history.update({k: current[k] for k in ('observation', 'action')})
                        frame.update(scene=dict(kind='cartpole', observation=current['observation']),
                                     playback_notice='Only pole angle was recorded; cart position is unavailable.' if current['partial'] else '')
                    frame['history'] = [history]
                    yield offset + elapsed, frame
                # Retain game rows beyond a truncated/missing tail of the acquisition.
                while upcoming is not None:
                    current, upcoming = upcoming, next(rows, None)
                    reward += current['reward']
                    yield offset + current['t'], self._game_frame(current, path.stem, offset, reward)
                    elapsed = max(elapsed, current['t'])
            finally:
                source.close()
            offset += elapsed
        for log in sorted(unused_logs):
            elapsed, reward = 0., 0.
            for row in _cartpole_rows(Path(log)):
                reward += row['reward']
                yield offset + row['t'], self._game_frame(row, Path(log).stem, offset, reward)
                elapsed = max(elapsed, row['t'])
            offset += elapsed

    def _game_frame(self, row, phase, offset, reward):
        return dict(phase=phase, history=[dict(row, t=offset + row['t'])], reward=reward,
                    raw=[], scene=dict(kind='cartpole', observation=row['observation']),
                    playback_notice='Only pole angle was recorded; cart position is unavailable.' if row['partial'] else '')

    def run(self, skip=False, verify_only=False):
        frames = None
        try:
            info = inspect_playback(self.config['playback_source'])
            speed = float(self.config.get('speed', 1))
            if not math.isfinite(speed) or speed < 0:
                raise ValueError('Playback speed must be finite and nonnegative')
            self.status, self.error = 'verified' if verify_only else 'running', ''
            self.snapshot = dict(status=self.status, phase='playback', error='', paused=False,
                                 restart_supported=False, playback=info, source='Saved experiment playback',
                                 output=info['path'], history=[], raw=[], scene=None,
                                 observation_names=[], action_names=[], phases=[], detection='recorded events')
            if verify_only:
                return
            history, previous = deque(maxlen=250), 0.
            frames = self._frames(info)
            for stamp, frame in frames:
                if not math.isfinite(stamp) or stamp < previous:
                    raise ValueError('Saved playback timestamps must be finite and chronological')
                if not self._controls():
                    break
                remaining = (stamp - previous) / speed if speed and not self.paused else 0.
                while remaining > 0 and not self.stop_event.is_set():
                    if not self._controls():
                        break
                    if self.paused:  # One explicit step skips pacing.
                        break
                    duration = min(.02, remaining)
                    self.stop_event.wait(duration)
                    remaining -= duration
                if self.stop_event.is_set():
                    break
                previous = stamp
                history.extend(frame.get('history', []))
                self.phase = frame.get('phase', 'playback')
                scene = frame.get('scene') or {}
                names = ['cart position (m)', 'cart velocity (m/s)', 'pole angle (rad)', 'pole angular velocity (rad/s)'] if scene.get('kind') == 'cartpole' else []
                self.snapshot = dict(self.snapshot, **{k: v for k, v in frame.items() if k not in ('status', 'history', 'paused', 'source', 'output', 'restart_supported')})
                self.snapshot.update(status=self.status, paused=self.paused, error='', history=list(history), playback_time=stamp,
                                     observation_names=frame.get('observation_names', names),
                                     action_names=frame.get('action_names', ['recorded action'] if names else []))
                json.dumps(self.snapshot, allow_nan=False)
            self.status = 'stopped' if self.stop_event.is_set() else 'completed'
        except Exception as exc:
            self.status, self.error = 'error', f'{type(exc).__name__}: {exc}'
        finally:
            if frames is not None:
                frames.close()
            self.snapshot = dict(self.snapshot, status=self.status, error=self.error, paused=self.paused)
