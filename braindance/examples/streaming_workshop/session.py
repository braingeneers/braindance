"""Worker-owned V3 experiment and bounded telemetry for the local watcher."""
import copy
import json
import pickle
import queue
import threading
import time
import types
from collections import deque
from pathlib import Path

import numpy as np

from braindance.core.phases_v3.experiment_v3 import Experiment
from braindance.core.phases_v3.phase_base_v3 import PhaseGroup
from braindance.core.replay import H5ReplaySource
from braindance.core.simulation import NeuralSimulationSource
from .environments import WorkshopGame


class RestartPhase(Exception):
    """A participant requested a new attempt at the next bin boundary."""


LEGACY_OUTPUTS = {
    'recording_baseline_hz': 'workshop_baseline_hz',
    'response_probe_hz': 'workshop_causal_hz',
    'response_probe_trials': 'workshop_causal_trials',
    'environment_episodes': 'workshop_episodes',
    'environment_reward': 'workshop_reward',
}


def build_phase(kind, identifier=None, params=None):
    """Build reusable V3 scientific phases from a saved builder specification."""
    from braindance.core.phases_v3.phases_binned import (
        BinnedRecordingPhaseV3, ResponseProbePhaseV3, MappedEnvironmentPhaseV3)
    params = dict(params or {})
    common = dict(name=identifier or kind)
    if kind in ('cartpole', 'foodland', 'ant'):
        params['environment'] = kind
        kind = 'environment'
    if kind == 'recording':
        phase = BinnedRecordingPhaseV3(duration=params.get('record_seconds', 3.), **common)
    elif kind == 'causal':
        phase = ResponseProbePhaseV3(repeats=params.get('causal_repeats', 4),
            seed=params.get('seed', 7), amplitude_mv=params.get('amplitude_mv', 100.),
            phase_width_us=params.get('phase_width_us', 100), **common)
    elif kind == 'environment':
        phase = MappedEnvironmentPhaseV3(duration=params.get('environment_seconds', 120.),
            amplitude_mv=params.get('amplitude_mv', 100.),
            phase_width_us=params.get('phase_width_us', 100), **common)
    else:
        raise ValueError(f'Unknown phase type: {kind}')
    phase.kind, phase.phase_params = kind, params
    return phase


class WorkshopRuntime:
    """Acquisition/mapping services and external observers for binned V3 phases."""
    def __init__(self, session):
        self.session = session
        self.entry_reward, self.entry_episodes = session.reward, session.episodes

    @property
    def dt(self):
        return self.session.dt

    @property
    def sampling_hz(self):
        return self.session.sampling_hz

    @property
    def n(self):
        return self.session.n

    @property
    def game(self):
        return self.session.game

    def read_counts(self, action=None, tag=None, pace=True):
        return self.session.tick(action, tag, pace)

    def map(self, name, value, dt):
        result, diagnostics = self.session.call_mapping(name, value, dt)
        setattr(self.session, name + '_diagnostics', diagnostics)
        return result

    def stimulate(self, action, tag):
        s = self.session
        s.env.stimulate(s.route_stimulus(action), tag=tag)

    def reset_mapping(self):
        s = self.session
        s.encode_state.clear()
        s.decode_state.clear()
        s.train_state.clear()
        s.pending_action = None
        s.pulse_credit[:] = 0

    def emit(self, event, **values):
        s = self.session
        if event == 'baseline':
            s.params['baseline_hz'] = list(values['baseline_hz'])
        elif event == 'trial':
            s.trial = f"{values['trial']}/{values['total']} · input {values['channel']} · {'sham' if values['sham'] else 'stim'}"
        elif event == 'trial_end':
            s.trial = ''
        elif event == 'response':
            s.causal, s.causal_trials = values['responses'], values['trials']
            s.publish(force=True)
        elif event == 'game_reset':
            s.observation = values['observation']
            self.episode_start_reward = s.reward
            s.episode_reward = 0.
            s.scene = s.game.scene(s.observation)
            s.publish(force=True)
        elif event == 'game_step':
            s.observation, s.action, s.rates = values['observation'], values['action'], values['rates']
            s.reward = self.entry_reward + values['reward']
            s.episode_reward = s.reward - self.episode_start_reward
            s.episodes = self.entry_episodes + values['episodes']
            s.scene = s.game.scene(s.observation)
            s.history[-1].update(observation=s.observation.tolist(), action=s.action.tolist(),
                                rates=s.rates.tolist(), delivered=values['delivered'])
            if values['done']:
                s.last_episode_end = s.game.last_end_reason
            s.publish(force=s.paused)

    def finish_step(self):
        self.session.finish_tick()


class WorkshopExperiment(Experiment):
    """V3 runner owning interactive retries and acquisition configuration."""
    def __init__(self, session, *args, **kwargs):
        self.session = session
        super().__init__(*args, **kwargs)

    def _create_environment_for_phase(self, phase):
        s = self.session
        self.params['stim_electrodes'] = s.acquisition_electrodes
        self.params['maxwell_env'] = dict(observation_type='raw', verbose=0)
        if s.config.get('live_config'):
            self.params['config'] = s.config['live_config']
        else:
            self.params['maxwell_env']['replay'] = dict(source=s.source, write_output=s.config['write_output'])
        super()._create_environment_for_phase(phase)

    def _execute_phase(self, phase):
        s = self.session
        if not hasattr(phase, "phase_params"):
            return super()._execute_phase(phase)
        self.phase_runtime = WorkshopRuntime(s)
        entry_baseline = copy.deepcopy(s.params['baseline_hz'])
        entry_causal = s.causal.copy()
        entry_trials = copy.deepcopy(s.causal_trials)
        entry_reward, entry_episodes = s.reward, s.episodes
        attempt = 0
        while True:
            attempt += 1
            s.attempt = attempt
            folder = s.run_dir / 'attempts' / phase.name / f'{attempt:04d}'
            folder.mkdir(parents=True, exist_ok=False)
            (folder / 'functions.py').write_text(Path(s.config['functions_file']).read_text(encoding='utf-8'), encoding='utf-8')
            record = dict(phase=phase.name, attempt=attempt, start_frame=getattr(phase.env, 'latest_frame', None) if getattr(phase.env, 'latest_frame', None) is not None else -1,
                          params=phase.phase_params, baseline_hz=entry_baseline, status='running')
            (folder / 'attempt.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            try:
                self._prepare_phase(phase)
                result = super()._execute_phase(phase)
                # Old participant functions may still read the previous data keys.
                for key, value in result.items():
                    alias = LEGACY_OUTPUTS.get(key)
                    if alias:
                        self.data.update({alias: value})
                record['status'] = 'completed'
                return result
            except RestartPhase:
                record['status'] = 'restarted'
                record.update(reward_total=s.reward, episodes_total=s.episodes)
                s.error = ''
                s.params['baseline_hz'] = copy.deepcopy(entry_baseline)
                s.causal = entry_causal.copy()
                s.reward, s.episodes = entry_reward, entry_episodes
                s.causal_trials = copy.deepcopy(entry_trials)
                s.pulse_credit[:] = 0
                s.pending_action = None
                s.encode_state.clear()
                s.decode_state.clear()
                s.train_state.clear()
                s.paused = False
            except BaseException:
                record['status'] = 'interrupted'
                raise
            finally:
                record['end_frame'] = getattr(phase.env, 'latest_frame', None) if getattr(phase.env, 'latest_frame', None) is not None else -1
                record.setdefault('reward_total', s.reward)
                record.setdefault('episodes_total', s.episodes)
                (folder / 'attempt.json').write_text(json.dumps(record, indent=2), encoding='utf-8')

    def _prepare_phase(self, phase):
        s = self.session
        s.phase, s.env = phase.name, phase.env
        s.phase_kind = phase.kind
        s.scene = None
        s.config.update(phase.phase_params)
        for key in ('sensory_index', 'encoder_gain', 'decoder_gain', 'left_channels', 'right_channels', 'stim_electrodes', 'environment'):
            if key in phase.phase_params:
                s.params[key] = phase.phase_params[key]
        if phase.kind == 'environment':
            if s.game.name != s.config['environment']:
                s.game.close()
                s.game = WorkshopGame(s.config['environment'], s.config['seed'], s.config.get('episode_seconds', 10.))
                s.observation = s.game.reset()
            s.params['action_size'] = len(s.game.action_names)
            s.action = np.zeros(s.params['action_size'])
            s.game.max_steps = round(s.config.get('episode_seconds', 10.) / s.dt)
            if hasattr(s.game.env, '_max_episode_steps'):
                s.game.env._max_episode_steps = s.game.max_steps
            s.encode_state.clear()
            s.decode_state.clear()
            s.train_state.clear()
            s.reload_functions()
        if phase.env.num_channels != s.raw_channels:
            raise ValueError('Configured raw channel count does not match Maxwell acquisition')
        s.publish(force=True)


class WorkshopSession:
    def route_stimulus(self, action):
        """Map phase-local encoder outputs to the acquisition's routed union."""
        if action is None:
            return None
        indices, amplitude, width = action
        return ([self.acquisition_electrodes.index(self.params['stim_electrodes'][i]) for i in indices], amplitude, width)

    def __init__(self, output_dir, config, phases=None):
        self.output_dir, self.config = Path(output_dir), dict(config)
        self.phase_objects = None if phases is None else list(phases)
        self.commands = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = None
        self.playback_log = None
        self.paused, self.steps = False, 0
        self.status, self.error, self.phase = 'ready', '', 'setup'
        self.snapshot = {'status': 'ready', 'phase': 'setup', 'error': '', 'config': self.config}
        self._last_publish = 0

    def start(self, overrides=None, skip=False):
        if self.thread and self.thread.is_alive():
            raise ValueError('Stop the current run before starting another')
        self.config.update(overrides or {})
        self.stop_event.clear()
        self.paused, self.steps, self.error = False, 0, ''
        self.commands = queue.Queue()
        self.status = 'starting'
        self.snapshot = {'status': self.status, 'phase': 'setup', 'error': ''}
        self.thread = threading.Thread(target=self.run, args=(skip,), daemon=True)
        self.thread.start()

    def validate_action(self, value):
        a = np.asarray(value, dtype=float)
        if a.shape != (len(self.game.action_names),) or not np.all(np.isfinite(a)) or np.any(np.abs(a) > 1):
            raise ValueError(f'decode must return {len(self.game.action_names)} finite values in [-1, 1]')
        if self.game.name == 'foodland' and a[1] < 0:
            raise ValueError('Foodland forward speed must be in [0, 1]')
        return a

    def validate_rates(self, value):
        a = np.asarray(value, dtype=float)
        if a.shape != (2,) or not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(a > self.params['max_stim_hz']):
            raise ValueError('encode must return two finite stimulation rates within configured limits')
        return a

    def reload_functions(self):
        path = Path(self.config['functions_file'])
        module = types.ModuleType('workshop_participant')
        exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), module.__dict__)
        # Fresh namespace avoids stale bytecode when edits occur in the same second.
        self.validate_action(module.decode(np.zeros(self.n), self.dt, copy.deepcopy(self.params), {})[0])
        self.validate_rates(module.encode(self.observation.copy(), self.dt, copy.deepcopy(self.params), {})[0])
        if hasattr(module, 'train'):
            transition = dict(observation=self.observation.tolist(), next_observation=self.observation.tolist(),
                              action=[0.] * len(self.game.action_names), spike_counts=[0] * self.n, reward=0., done=False)
            diagnostics = module.train(transition, self.dt, copy.deepcopy(self.params), {'encode': {}, 'decode': {}})
            if not isinstance(diagnostics, dict):
                raise ValueError('train must return a diagnostics dictionary')
            json.dumps(diagnostics, allow_nan=False)
        from .custom_analysis import resolve_analysis_config
        resolved = resolve_analysis_config(self.config, path.read_text(encoding='utf-8'))
        if any(p.get('type') == 'custom_analysis' for p in resolved.get('phases', [])):
            from .experiment_spec import verify_spec
            if self.phase_objects is not None:
                from .custom_analysis import CustomAnalysisPhase, analysis_definition
                for phase, spec in zip(self.phase_objects, resolved['phases']):
                    if isinstance(phase, CustomAnalysisPhase):
                        definition = analysis_definition(spec['params'])
                        phase.inputs, phase.outputs = definition['inputs'], definition['outputs']
                        phase.options, phase.module = definition['params'], module
            report = verify_spec(resolved, phase_objects=self.phase_objects)
            if not report['ok']:
                raise ValueError('; '.join(report['errors']))
            self.config['phases'] = resolved['phases']
        self.module = module
        self.encode_state, self.decode_state, self.train_state = {}, {}, {}
        self.pending_action = None
        self.pulse_credit[:] = 0
        self.error = ''
        if hasattr(self, 'run_dir') and self.run_dir.exists():
            (self.run_dir / f'functions_{time.time_ns()}.py').write_text(path.read_text(encoding='utf-8'), encoding='utf-8')

    def call_mapping(self, name, values, dt):
        while True:
            try:
                if name == 'train':
                    if not hasattr(self.module, 'train'):
                        return None, {}
                    self.train_state.update(encode=self.encode_state, decode=self.decode_state)
                    diagnostics = self.module.train(copy.deepcopy(values), dt, copy.deepcopy(self.params), self.train_state)
                    if not isinstance(diagnostics, dict):
                        raise ValueError('train must return a diagnostics dictionary')
                    json.dumps(diagnostics, allow_nan=False)
                    return None, diagnostics
                function_state = self.decode_state if name == 'decode' else self.encode_state
                result, diagnostics = getattr(self.module, name)(values.copy(), dt, self.params, function_state)
                result = self.validate_action(result) if name == 'decode' else self.validate_rates(result)
                json.dumps(diagnostics, allow_nan=False)
                return result, diagnostics
            except Exception as exc:
                self.error = f'{name}: {type(exc).__name__}: {exc}. Fix {self.config["functions_file"]}, reload, then resume.'
                if self.config.get('headless'):
                    raise
                self.paused = True
                self.pending_action = None
                self.publish(force=True)
                wait_start = time.perf_counter()
                self.controls()
                self._tick_start += time.perf_counter() - wait_start

    def run(self, skip=False, verify_only=False):
        self.phase_plan = []
        for name in ('history', 'run_dir'):
            if hasattr(self, name):
                delattr(self, name)
        self.source = self.env = self.game = None
        self.playback_log = None
        try:
            c = self.config
            original_config = copy.deepcopy(c)
            from .experiment_spec import verify_spec
            checked = verify_spec(c, skip, phase_objects=self.phase_objects)
            if not checked['ok']:
                raise ValueError('; '.join(checked['errors']))
            if self.phase_objects is not None:
                if [p.name for p in self.phase_objects] != [p['id'] for p in checked['phases']]:
                    raise ValueError('Phase objects must match the validated phase order')
                from .custom_analysis import CustomAnalysisPhase
                for phase, spec in zip(self.phase_objects, checked['phases']):
                    if not isinstance(phase, CustomAnalysisPhase):
                        phase.kind = 'environment' if spec['type'] in ('cartpole', 'foodland', 'ant') else spec['type']
                        phase.phase_params = spec['params']
            self.phase_plan = [p['id'] for p in checked['phases']]
            self.acquisition_electrodes = list(dict.fromkeys(e for p in checked['phases'] for e in p['params'].get('stim_electrodes', [])))
            first_environment = next((p for p in checked['phases'] if p['type'] in ('environment', 'cartpole', 'foodland', 'ant')), None)
            if first_environment:
                c.update(first_environment['params'])
            if c['detection'] not in {'events', 'threshold', 'rt-sort'}:
                raise ValueError('Choose events, threshold, or rt-sort detection')
            if not np.isfinite(c['speed']) or c['speed'] < 0:
                raise ValueError('speed must be finite and nonnegative')
            if not np.isfinite(c['threshold_uv']) or c['threshold_uv'] >= 0:
                raise ValueError('threshold_uv must be a negative finite value')
            if c.get('live_config'):
                import importlib.util
                if importlib.util.find_spec('maxlab') is None:
                    raise RuntimeError('Live acquisition requires installed Maxwell maxlab software')
                if c['detection'] == 'events':
                    raise ValueError('Live workshop uses raw input: select threshold or rt-sort detection')
            if c.get('live_config') and c.get('source'):
                raise ValueError('Choose live_config or replay source, not both')
            self.dt = .02
            self.sampling_hz = 20000
            if c.get('source'):
                self.source = H5ReplaySource(c['source'], speed=0, loop=c.get('loop', False))
            elif not c.get('live_config'):
                self.source = NeuralSimulationSource(num_channels=c['channels'], seed=c['seed'], speed=0,
                    adjacency=c.get('adjacency'), **{key: c[key] for key in (
                        'num_neurons', 'grid_shape', 'neuron_positions', 'electrode_pitch_um',
                        'spatial_sigma_um', 'waveform_amplitude_uv') if key in c})
            self.n = self.source.num_channels if self.source is not None else c['channels']
            self.raw_channels = self.n
            self.sorter = None
            if c['detection'] == 'rt-sort':
                if not c.get('sorter_path'):
                    raise ValueError('RT-sort requires --sorter-path with a compatible prebuilt sorter')
                from braindance import get_rt_sort_path
                from braindance.core.spikesorter.rt_sort import RTSort
                self.sorter = RTSort.load_from_file(c['sorter_path'], model=get_rt_sort_path())
                self.sorter.reset()
                self.n = self.sorter.num_seqs
                for phase in checked['phases']:
                    if phase['type'] in ('environment', 'cartpole', 'foodland', 'ant'):
                        explicit = next((p.get('params', {}) for p in original_config.get('phases', []) if p.get('id') == phase['id']), {})
                        for key, value in [('left_channels', list(range(self.n // 2))), ('right_channels', list(range(self.n // 2, self.n)))]:
                            if key not in original_config and key not in explicit:
                                phase['params'][key] = value
                if first_environment:
                    c.update(first_environment['params'])
            if self.source is not None:
                self.sampling_hz = self.source.sampling_hz
            if not np.isclose(self.dt * self.sampling_hz, round(self.dt * self.sampling_hz)):
                raise ValueError('Sampling rate must represent an exact 20 ms bin')
            self.bin_frames = round(self.dt * self.sampling_hz)
            if self.sorter is not None and not np.isclose(self.sorter.samp_freq * 1000, self.sampling_hz):
                raise ValueError('RT-sort sampling rate does not match acquisition')
            self.game = WorkshopGame(c['environment'], c['seed'], c.get('episode_seconds', 10.))
            self.observation = self.game.reset()
            self.params = dict(environment=c['environment'], action_size=len(self.game.action_names),
                               left_channels=c.get('left_channels', list(range(self.n // 2))),
                               right_channels=c.get('right_channels', list(range(self.n // 2, self.n))),
                               stim_electrodes=c.get('stim_electrodes', [0, 1]),
                               baseline_hz=c.get('baseline_hz') or [0.] * self.n,
                               decoder_gain=c.get('decoder_gain', .025), encoder_gain=c.get('encoder_gain', 100.),
                               max_stim_hz=40., sensory_index=c.get('sensory_index', 2 if c['environment'] == 'cartpole' else 0))
            self.signature = dict(source=str(c.get('source') or c.get('live_config') or 'simulation'),
                                  channels=self.n, raw_channels=self.raw_channels, detection=c['detection'],
                                  sampling_hz=self.sampling_hz, threshold_uv=c['threshold_uv'],
                                  electrodes=self.source.mapping['electrode'].tolist() if self.source is not None else None,
                                  sorter_path=c.get('sorter_path'))
            if isinstance(self.source, NeuralSimulationSource):
                self.signature.update(seed=c['seed'], adjacency=self.source.adjacency.tolist(),
                    num_neurons=self.source.num_neurons, neuron_positions=self.source.neuron_positions.tolist(),
                    spatial_sigma_um=self.source.spatial_sigma_um,
                    waveform_amplitude_uv=self.source.waveform_amplitude_uv,
                    channel_positions=np.column_stack((self.source.mapping['x'], self.source.mapping['y'])).tolist())
            if c.get('calibration'):
                with Path(c['calibration']).open('rb') as f:
                    calibration = pickle.load(f)  # Only load your own trusted local cache.
                if calibration.get('signature') != self.signature:
                    raise ValueError('Calibration source, channel mapping, or detector does not match this run')
                self.params['baseline_hz'] = calibration['baseline_hz']
            left, right = self.params['left_channels'], self.params['right_channels']
            if not left or not right or set(left) & set(right) or any(not isinstance(i, int) or not 0 <= i < self.n for i in left + right):
                raise ValueError('Select nonempty, disjoint decoder pools using valid channel indices')
            if len(self.params['stim_electrodes']) != 2 or len(set(self.params['stim_electrodes'])) != 2:
                raise ValueError('Select two distinct stimulation electrodes')
            if self.source is not None and not set(self.params['stim_electrodes']) <= set(self.source.mapping['electrode']):
                raise ValueError('Stimulation electrodes must exist in source mapping')
            baseline = np.asarray(self.params['baseline_hz'], dtype=float)
            if baseline.shape != (self.n,) or not np.all(np.isfinite(baseline)) or np.any(baseline < 0):
                raise ValueError('Baseline must contain one nonnegative finite firing rate per channel')
            if not 0 <= self.params['sensory_index'] < len(self.observation):
                raise ValueError('Select a valid sensory observation index')
            for name in ('record_seconds', 'environment_seconds'):
                if not np.isfinite(c[name]) or c[name] < self.dt:
                    raise ValueError(f'{name} must be at least 20 ms')
            if not isinstance(c['causal_repeats'], int) or c['causal_repeats'] < 1:
                raise ValueError('causal_repeats must be a positive integer')
            self.history, self.processing = deque(maxlen=250), deque(maxlen=10000)
            self._performance_ticks = deque(maxlen=100)
            self._total_frames = self._total_bins = self._deadline_misses = 0
            self._pacing_debt = 0.
            self.causal, self.causal_trials = np.zeros((2, self.n)), [[0, 0], [0, 0]]
            self.action, self.rates = np.zeros(len(self.game.action_names)), np.zeros(2)
            self.pulse_credit = np.zeros(2)
            self.pending_action = None
            self.trial, self.reward, self.episodes = '', 0., 0
            self.episode_reward = 0.
            self.last_episode_end = ''
            self.encode_diagnostics, self.decode_diagnostics, self.train_diagnostics = {}, {}, {}
            self.raw, self.scene, self.events = [], None, []
            self.spatial = None
            self._last_detected = np.full(self.raw_channels, -100000, dtype=np.int64)
            self._below = np.zeros(self.raw_channels, dtype=bool)
            self.reload_functions()
            # Preflight uses local game/source objects only; never constructs MaxwellEnv.
            # Verify each configured mapping on a sample bin without writing a run.
            original_params = copy.deepcopy(self.params)
            for phase in checked['phases']:
                if phase['type'] not in ('environment', 'cartpole', 'foodland', 'ant'):
                    continue
                if self.game.name != phase['params']['environment']:
                    self.game.close()
                    self.game = WorkshopGame(phase['params']['environment'], c['seed'], phase['params']['episode_seconds'])
                    self.observation = self.game.reset()
                self.params.update({key: phase['params'][key] for key in ('sensory_index', 'encoder_gain', 'decoder_gain', 'left_channels', 'right_channels', 'stim_electrodes', 'environment')})
                self.params['action_size'] = len(self.game.action_names)
                if self.source is not None and not set(self.params['stim_electrodes']) <= set(self.source.mapping['electrode']):
                    raise ValueError(f'{phase["id"]}: stimulation electrode is not in source mapping')
                if any(i >= self.n for i in self.params['left_channels'] + self.params['right_channels']):
                    raise ValueError(f'{phase["id"]}: decoder channel outside detected channel/unit count')
                if not 0 <= self.params['sensory_index'] < len(self.observation):
                    raise ValueError(f'{phase["id"]}: invalid sensory_index')
                action = self.validate_action(self.module.decode(np.ones(self.n) if verify_only else np.zeros(self.n), self.dt, copy.deepcopy(self.params), {})[0])
                self.validate_rates(self.module.encode(self.observation.copy(), self.dt, copy.deepcopy(self.params), {})[0])
                if verify_only:
                    obs, _, _ = self.game.step(action)
                    if not np.all(np.isfinite(obs)):
                        raise ValueError('Environment returned nonfinite observations')
            self.params = original_params
            if verify_only:
                self.status = 'verified'
                return
            self.status = 'running'
            run_dir = self.output_dir / (time.strftime('%Y%m%d_%H%M%S') + f'_{time.time_ns() % 1000000000:09d}')
            run_dir.mkdir(parents=True, exist_ok=False)
            self.run_dir = run_dir
            self.playback_log = (run_dir / 'playback.jsonl').open('w', encoding='utf-8', buffering=1)
            (run_dir / 'functions_initial.py').write_text(Path(c['functions_file']).read_text(encoding='utf-8'), encoding='utf-8')
            (run_dir / 'setup.json').write_text(json.dumps(self.params, indent=2), encoding='utf-8')
            exp = WorkshopExperiment(self, c.get('experiment_name', 'streaming_workshop'), params=c,
                             save_dir=str(run_dir), auto_load_data=False, overwrite_existing=True)
            if skip or c.get('baseline_hz') or c.get('calibration'):
                exp.data.recording_baseline_hz = self.params['baseline_hz']
                exp.data.workshop_baseline_hz = self.params['baseline_hz']
            (run_dir / 'experiment.json').write_text(json.dumps({'name': exp.name, 'phases': checked['phases'], 'settings': c}, indent=2), encoding='utf-8')
            from .custom_analysis import CustomAnalysisPhase
            exp.add_phase(PhaseGroup(self.phase_objects if self.phase_objects is not None else [CustomAnalysisPhase(p, c, lambda: self.module) if p['type'] == 'custom_analysis' else
                build_phase(p['type'], identifier=p['id'], params={**{key: c[key] for key in ('seed', 'amplitude_mv', 'phase_width_us')}, **p['params']}) for p in checked['phases']], name='workshop'))
            # Preflight has validated canonical and legacy input aliases in order.
            ok = exp.run(validate=False)
            if not ok and not self.stop_event.is_set():
                raise RuntimeError(self.error or 'Experiment stopped; inspect phase logs in output directory')
            self.status = 'stopped' if self.stop_event.is_set() else 'completed'
            with (run_dir / 'calibration.pkl').open('wb') as f:
                pickle.dump(dict(signature=self.signature, channels=self.n, sampling_hz=self.sampling_hz,
                                 baseline_hz=self.params['baseline_hz'], causal_hz=self.causal), f)
            metrics = dict(bins=self._total_bins, median_ms=float(np.median(self.processing)) if self.processing else 0,
                           p95_ms=float(np.percentile(self.processing, 95)) if self.processing else 0,
                           max_ms=max(self.processing, default=0), status=self.status)
            metrics.update(self.performance())
            (run_dir / 'performance.json').write_text(json.dumps(metrics, indent=2))
        except Exception as exc:
            self.error, self.status = f'{type(exc).__name__}: {exc}', 'error'
        finally:
            if self.playback_log is not None:
                self.playback_log.close()
                self.playback_log = None
            if self.env is not None:
                self.env.close()
            if self.source is not None:
                self.source.close()
            if self.game is not None:
                self.game.close()
            self.publish(force=True)

    def controls(self):
        while True:
            while not self.commands.empty():
                command = self.commands.get_nowait()
                kind = command['kind']
                if kind == 'pause':
                    self.paused = not self.paused
                elif kind == 'step':
                    self.paused, self.steps = True, self.steps + 1
                elif kind == 'restart_phase':
                    raise RestartPhase()
                elif kind == 'reload':
                    self.paused = True
                    try:
                        previous_path = self.config['functions_file']
                        self.config['functions_file'] = command.get('functions_file', previous_path)
                        self.reload_functions()
                    except Exception as exc:
                        self.config['functions_file'] = previous_path
                        self.error = f'{type(exc).__name__}: {exc}'
                elif kind == 'adjacency':
                    try:
                        if not isinstance(self.source, NeuralSimulationSource):
                            raise ValueError('Adjacency is only available for neural simulation')
                        self.source.set_adjacency(command['value'])
                        self.config['adjacency'] = self.source.adjacency.tolist()
                        with (self.run_dir / 'adjacency_changes.jsonl').open('a', encoding='utf-8') as f:
                            f.write(json.dumps({'time_s': self.source.elapsed_s, 'adjacency': self.source.adjacency.tolist()}) + '\n')
                    except Exception as exc:
                        self.error = str(exc)
                self.publish(force=True)
            if self.stop_event.is_set():
                raise InterruptedError('Stopped by participant')
            if not self.paused or self.steps:
                if self.steps:
                    self.steps -= 1
                break
            time.sleep(.02)

    def finish_tick(self):
        elapsed = time.perf_counter() - self._tick_start
        self.processing[-1] = elapsed * 1000
        speed = self.config['speed']
        deadline = self.dt / speed if speed else None
        if deadline is not None and elapsed > deadline:
            self._deadline_misses += 1
        if speed and self.source is not None:
            # Recover scheduler oversleep on subsequent bins rather than letting
            # each small delay permanently reduce the acquisition sample rate.
            self.stop_event.wait(max(0, deadline - elapsed - self._pacing_debt))
        duration = time.perf_counter() - self._tick_start
        if speed and self.source is not None:
            self._pacing_debt = max(0., self._pacing_debt + duration - deadline)
        self._total_frames += self.bin_frames
        self._total_bins += 1
        self._performance_ticks.append((duration, elapsed))
        if self.playback_log is not None:
            self.publish(force=True)
            # One original bin/scene per row; bounded monitor history is rebuilt on playback.
            saved = dict(self.snapshot, history=[self.history[-1]],
                         playback_time=self._total_frames / self.sampling_hz)
            self.playback_log.write(json.dumps(saved, allow_nan=False) + '\n')

    def tick(self, action=None, tag=None, pace=True):
        self.controls()
        self._tick_start = start = time.perf_counter()
        try:
            raw, done = self.env.step(buffer_size=self.bin_frames)
            batch = self.env.latest_batch
            if self.source is not None:
                if raw is None or batch is None:
                    raise EOFError('Replay reached end of file')
                raw = np.asarray(raw)
                frames = batch['frame_numbers']
                events = [event for row in batch['events'] for event in row]
                gain = self.source.gain
                # Looping H5 sources can return short reads at file boundaries.
                while len(raw) < self.bin_frames and self.config.get('loop') and not done:
                    more, done = self.env.step(buffer_size=self.bin_frames - len(raw))
                    if more is None or not len(more):
                        break
                    batch = self.env.latest_batch
                    raw = np.concatenate([raw, np.asarray(more)])
                    frames = np.concatenate([frames, batch['frame_numbers']])
                    events.extend(event for row in batch['events'] for event in row)
            else:
                raw = np.asarray(raw)
                frames = np.arange(self.env.latest_frame - len(raw) + 1, self.env.latest_frame + 1)
                events, gain = [], self.config.get('live_gain', 512.)
            if len(raw) != self.bin_frames:
                raise EOFError('Replay ended with a partial 20 ms bin; loop or use a longer source')
            if action is not None:
                self.env.stimulate(self.route_stimulus(action), tag=tag)
            uv = raw * 1000 / gain
            if self.source is not None:
                self.spatial = dict(
                    channel_positions=np.column_stack((self.source.mapping['x'], self.source.mapping['y'])).tolist(),
                    electrodes=self.source.mapping['electrode'].tolist(),
                    peak_uv=np.maximum(0., -uv.min(axis=0)).tolist(),
                    rms_uv=np.sqrt(np.mean(uv * uv, axis=0)).tolist())
                if isinstance(self.source, NeuralSimulationSource):
                    self.spatial.update(neuron_positions=self.source.neuron_positions.tolist(),
                                        neuron_channels=self.source.spatial_weights.argmax(axis=1).tolist())
            if self.config['detection'] == 'threshold':
                events = []
                from braindance.core.replay import SpikeEvent
                # Vectorize edge detection; enforce refractory only at actual
                # crossings. Preserve time/channel ordering and cross-bin state.
                below = uv < self.config['threshold_uv']
                previous = np.concatenate((self._below[None, :], below[:-1]), axis=0)
                rows, channels = np.nonzero(below & ~previous)
                for i, channel in zip(rows, channels):
                    frame = int(frames[i])
                    if frame - self._last_detected[channel] >= 40:
                        events.append(SpikeEvent(frame, int(channel), float(raw[i, channel])))
                        self._last_detected[channel] = frame
                self._below = below[-1].copy()
            elif self.sorter is not None:
                from braindance.core.replay import SpikeEvent
                events = []
                for offset in range(0, len(raw), self.sorter.buffer_size):
                    chunk = raw[offset:offset + self.sorter.buffer_size]
                    detections = self.sorter.running_sort(chunk, latest_frame=int(frames[offset + len(chunk) - 1]), use_numba=False)
                    events.extend(SpikeEvent(round(t * self.sampling_hz / 1000), int(unit), 0.) for unit, t in detections)
            counts = np.bincount([e.channel for e in events], minlength=self.n)
            # Keep sample order and spacing: connected min/max envelopes distort
            # spike shapes. Bound telemetry to the 16 displayed channels instead.
            self.raw = uv[:, :16].round(3).T.tolist()
            self.events = [[(e.frame - int(frames[0])) / self.sampling_hz, e.channel] for e in events]
            self.history.append(dict(t=self.env.time_elapsed(), counts=counts.tolist(),
                                     observation=self.observation.tolist(), action=self.action.tolist(),
                                     rates=self.rates.tolist(), delivered=[] if action is None else action[0]))
            self.processing.append((time.perf_counter() - start) * 1000)
            if getattr(self, 'phase_kind', self.phase) != 'environment':
                self.publish(force=self.paused)
            if pace:
                self.finish_tick()
            if done:
                self.stop_event.set()
            return counts, len(frames)
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            raise

    def performance(self):
        """Rolling active-worker rates; pauses and setup are excluded.

        Capacity includes raw generation, detection, mapping and telemetry. It
        excludes intentional pacing; acquired_hz includes pacing. Neither rate
        represents browser rendering FPS or an individual channel's byte rate.
        """
        ticks = list(self._performance_ticks)
        wall = sum(duration for duration, _ in ticks)
        compute = sum(elapsed for _, elapsed in ticks)
        frames = len(ticks) * self.bin_frames
        acquired = frames / wall if wall else 0.
        return dict(sampling_hz=self.sampling_hz, acquired_hz=acquired,
                    processing_capacity_hz=frames / compute if compute else 0.,
                    realtime_factor=acquired / self.sampling_hz,
                    bin_hz=len(ticks) / wall if wall else 0.,
                    p95_ms=float(np.percentile([elapsed * 1000 for _, elapsed in ticks], 95)) if ticks else 0.,
                    deadline_misses=self._deadline_misses, frames=self._total_frames,
                    bins=self._total_bins, raw_channels=self.raw_channels,
                    worker='thread' if self.thread is not None else 'calling thread',
                    raw_preview='full samples, first 16 channels')

    def publish(self, force=False):
        now = time.perf_counter()
        if not force and now - self._last_publish < 1 / 30:
            return
        self._last_publish = now
        snap = dict(status=self.status, phase=self.phase, phase_kind=getattr(self, 'phase_kind', None),
                    error=self.error, paused=self.paused,
                    phases=getattr(self, 'phase_plan', []), attempt=getattr(self, 'attempt', 0), restart_supported=True)
        if hasattr(self, 'history'):
            snap.update(history=list(self.history), raw=self.raw, events=self.events, scene=self.scene,
                        spatial=self.spatial,
                        observation_names=self.game.observation_names, action_names=self.game.action_names,
                        baseline=self.params['baseline_hz'], causal=self.causal.tolist(), causal_trials=self.causal_trials,
                        adjacency=self.source.adjacency.tolist() if isinstance(self.source, NeuralSimulationSource) else None,
                        trial=self.trial, reward=self.reward, episode_reward=self.episode_reward, episodes=self.episodes,
                        episode_seconds=self.game.steps * .02,
                        episode_limit=self.config.get('episode_seconds', 10.), last_episode_end=self.last_episode_end,
                        functions_file=self.config['functions_file'],
                        encoder=self.encode_diagnostics, decoder=self.decode_diagnostics, training=self.train_diagnostics,
                        source='live Maxwell' if self.source is None else ('neural simulator' if isinstance(self.source, NeuralSimulationSource) else 'H5 replay · open-loop'),
                        detection=self.config['detection'], output=str(getattr(self, 'run_dir', '')),
                        p95_ms=float(np.percentile(self.processing, 95)) if self.processing else 0)
            snap['performance'] = self.performance()
            if isinstance(self.source, NeuralSimulationSource):
                snap['simulator'] = self.source.snapshot()
        # Round trip gives the HTTP threads an immutable, finite JSON snapshot.
        try:
            self.snapshot = json.loads(json.dumps(snap, allow_nan=False))
        except (TypeError, ValueError) as exc:
            self.error, self.status = f'Invalid telemetry: {exc}', 'error'
            self.snapshot = dict(status='error', error=self.error, phase=self.phase)
