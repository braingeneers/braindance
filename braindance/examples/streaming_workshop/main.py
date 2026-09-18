"""Run with python -m braindance.examples.streaming_workshop.main."""
import argparse
import json
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

from braindance.config import get_output_dir
from .session import WorkshopSession
from .profiles import ProfileStore, verify_python
from .experiment_spec import phase_catalog, verify_spec, uses_native_runner
from .native_catalog import native_catalog
from .native_runner import NativeSession
from .code_export import code_bundle, export_bundle
from .playground import Playground
from .analysis_workspace import AnalysisWorkspace
from .sorting_workspace import SortingWorkspace
from .catalog_workspace import CatalogWorkspace
from .global_settings import settings_payload
from .playback import PlaybackSession, inspect_playback


def main(environment='cartpole', source=None, live_config=None, channels=400,
         seed=7, record_seconds=3., environment_seconds=120., causal_repeats=4,
         speed=1., detection='threshold', threshold_uv=-30., write_output=True,
         output_dir=None, port=8765, headless=False, skip=False,
         functions_file=None, calibration=None, loop=False, open_browser=True,
         stim_electrodes=None, left_channels=None, right_channels=None, sorter_path=None,
         episode_seconds=10., profiles_dir=None, profile=None, num_neurons=40):
    # Change one line to select another BrainDance game:
    # environment = 'foodland'  # Optional gym + pygame dependencies.
    # environment = 'ant'       # Optional gymnasium[mujoco] dependency.

    # Local neural simulation is the default. For recorded data instead:
    # source = 'path/to/your/recording.raw.h5'

    # Live Maxwell: set your acquisition config and mapped stimulation electrodes.
    # live_config, detection, source = 'path/to/your/config.cfg', 'threshold', None
    # stim_electrodes = [YOUR_LEFT_ELECTRODE, YOUR_RIGHT_ELECTRODE]
    # channels = YOUR_ROUTED_CHANNEL_COUNT
    # Live input needs installed maxlab and a configured Maxwell workstation.
    config = dict(environment=environment, source=source, live_config=live_config,
                  channels=channels, seed=seed, record_seconds=record_seconds,
                  environment_seconds=environment_seconds, causal_repeats=causal_repeats,
                  speed=0. if str(speed) == 'max' else float(speed), detection=detection,
                  threshold_uv=threshold_uv, write_output=write_output, calibration=calibration,
                  functions_file=str(functions_file or Path(__file__).with_name('functions.py')),
                  loop=loop, amplitude_mv=100., phase_width_us=100, headless=headless, sorter_path=sorter_path,
                  episode_seconds=episode_seconds, num_neurons=num_neurons,
                  grid_shape=[20, 20] if channels == 400 else None)
    for key, value in [('stim_electrodes', stim_electrodes), ('left_channels', left_channels), ('right_channels', right_channels)]:
        if value is not None:
            config[key] = value
    output_dir = Path(output_dir or get_output_dir() / 'streaming_workshop')
    profiles = ProfileStore(profiles_dir or output_dir / 'profiles', config['functions_file'])
    selected = {'path': config['functions_file'], 'name': '', 'settings': {}}
    if profile:
        loaded = profiles.load(profile)
        config.update(loaded['settings'])
        if 'grid_shape' not in loaded['settings']:
            config['grid_shape'] = [20, 20] if config['channels'] == 400 else None
        if 'num_neurons' not in loaded['settings']:
            config['num_neurons'] = len(config['adjacency']) if config.get('adjacency') else config['channels']
        config['functions_file'] = loaded['path']
        selected.update(path=loaded['path'], name=profile, settings=loaded['settings'])
    config['headless'] = headless
    session_class = NativeSession if uses_native_runner(config) else WorkshopSession
    session = session_class(output_dir, config)
    if headless:
        session.start(skip=skip)
        session.thread.join()
        print(json.dumps({key: session.snapshot.get(key) for key in
                          ('status', 'error', 'phase', 'episodes', 'reward', 'p95_ms', 'output')}, indent=2))
        return 0 if session.status == 'completed' else 1
    token = secrets.token_urlsafe(32)
    playground = Playground()
    analysis = AnalysisWorkspace(output_dir)
    sorting = SortingWorkspace(output_dir)
    catalog = CatalogWorkspace(output_dir)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, payload, content_type='application/json'):
            body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.headers.get('Host') not in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}:
                return self.reply(403, {'error': 'Use the localhost workshop address'})
            url = urlparse(self.path)
            if url.path == '/api/sorting/download':
                query = parse_qs(url.query)
                if query.get('token', [''])[0] != token:
                    return self.reply(403, {'error': 'Invalid local control token'})
                try:
                    artifact = sorting.artifact(query.get('job', [''])[0], int(query.get('index', ['-1'])[0]))
                    with artifact.open('rb') as stream:
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/octet-stream')
                        self.send_header('Content-Length', str(artifact.stat().st_size))
                        self.send_header('Content-Disposition', f'attachment; filename="{artifact.name}"')
                        self.end_headers()
                        while chunk := stream.read(1024**2):
                            self.wfile.write(chunk)
                except (OSError, ValueError) as exc:
                    self.reply(400, {'error': str(exc)})
            elif url.path == '/api/settings':
                try:
                    self.reply(200, settings_payload())
                except (OSError, ValueError) as exc:
                    self.reply(400, {'error': str(exc)})
            elif url.path == '/api/phase-catalog':
                self.reply(200, {**phase_catalog(), **native_catalog()})
            elif url.path == '/api/profiles':
                self.reply(200, dict(names=profiles.list(), directory=str(profiles.directory),
                                     template=profiles.template(), selected=selected['name'],
                                     path=selected['path'], settings=session.config))
            elif url.path == '/api/profile':
                try:
                    loaded = profiles.load(parse_qs(url.query).get('name', [''])[0])
                    self.reply(200, loaded)
                except (OSError, ValueError, SyntaxError) as exc:
                    self.reply(400, {'error': str(exc)})
            elif self.path == '/api/state':
                current = session.config
                native = isinstance(session, NativeSession)
                source_label = 'Saved experiment playback' if isinstance(session, PlaybackSession) else 'Maxwell hardware' if current.get('live_config') else 'Dummy Maxwell (sine)' if current.get('source') == 'sine' else 'File replay' if current.get('source') else 'Simulation'
                speed = current.get('speed', 1.)
                timing = 'Live acquisition' if current.get('live_config') else 'Unpaced' if speed == 0 else f'{speed:g}× pacing'
                self.reply(200, {**session.snapshot,
                    'output': str(getattr(session, 'run_dir', '') or session.snapshot.get('output', '')),
                    'execution': dict(source=source_label, timing=timing,
                    engine='Playback' if isinstance(session, PlaybackSession) else 'Native V3' if native else 'Streaming adapter', bin_ms=None if native else 20,
                    hard_realtime=False)})
            elif self.path == '/':
                html = Path(__file__).with_name('watcher.html').read_text(encoding='utf-8')
                self.reply(200, html.replace('__TOKEN__', token), 'text/html; charset=utf-8')
            elif self.path in {'/watcher.js', '/ui.js', '/settings.js', '/builder.js', '/phase_flow.js', '/code_workspace.js', '/playground.js', '/simulator.js', '/analysis_workspace.js', '/catalog_workspace.js', '/sorting_workspace.js'}:
                self.reply(200, Path(__file__).with_name(self.path[1:]).read_text(encoding='utf-8'), 'text/javascript; charset=utf-8')
            elif self.path in {'/workshop_theme.css', '/phase_flow.css', '/analysis_workspace.css'}:
                self.reply(200, Path(__file__).with_name(self.path[1:]).read_text(encoding='utf-8'), 'text/css; charset=utf-8')
            else:
                self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            nonlocal session
            if self.headers.get('Host') not in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}:
                return self.reply(403, {'error': 'Use the localhost workshop address'})
            if self.path not in {'/api/playback', '/api/settings', '/api/upload/replay', '/api/upload/config', '/api/upload/sorting', '/api/sorting', '/api/control', '/api/playground', '/api/profile', '/api/verify', '/api/verify-experiment', '/api/code-preview', '/api/code-export', '/api/analysis-contracts', '/api/analysis', '/api/catalog'} or self.headers.get('X-Workshop-Token') != token:
                return self.reply(403, {'error': 'Invalid local control token'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if self.path.startswith('/api/upload/'):
                    sorting_upload = self.path.endswith('/sorting')
                    replay = self.path.endswith('/replay') or sorting_upload
                    filename = Path(unquote(self.headers.get('X-File-Name', '')).replace('\\', '/')).name
                    suffix = Path(filename).suffix.lower()
                    if sorting_upload and suffix not in {'.h5', '.hdf5', '.nwb'}:
                        raise ValueError('Choose a raw Maxwell H5/HDF5 or NWB recording.')
                    if suffix not in ({'.h5', '.hdf5', '.nwb'} if sorting_upload else {'.h5', '.hdf5', '.npy'} if replay else {'.cfg'}):
                        raise ValueError('Choose an H5 or NPY recording.' if replay else 'Choose a Maxwell .cfg file.')
                    if not 0 < size <= (64 * 1024**3 if replay else 10 * 1024**2):
                        raise ValueError('File must be nonempty and at most 64 GiB (replay) or 10 MiB (config).')
                    # Stream recordings to disk; never buffer a full H5 in memory.
                    uploads = output_dir / 'uploads'
                    uploads.mkdir(parents=True, exist_ok=True)
                    target = uploads / (secrets.token_hex(16) + ('.h5' if sorting_upload and suffix == '.hdf5' else suffix))
                    try:
                        with target.open('xb') as uploaded:
                            remaining = size
                            while remaining:
                                chunk = self.rfile.read(min(1024**2, remaining))
                                if not chunk:
                                    raise ValueError('File upload was interrupted. Please try again.')
                                uploaded.write(chunk)
                                remaining -= len(chunk)
                        if sorting_upload:
                            result = sorting.add_upload(target, filename)
                    except Exception:
                        target.unlink(missing_ok=True)
                        raise
                    return self.reply(200, result if sorting_upload else {'path': str(target.resolve())})
                if not 0 < size <= 200000:
                    raise ValueError('Invalid command size')
                command = json.loads(self.rfile.read(size))
                if self.path == '/api/settings':
                    if not isinstance(command, dict) or not isinstance(command.get('settings'), dict):
                        raise ValueError('Expected a settings object')
                    return self.reply(200, settings_payload(command['settings']))
                if self.path == '/api/sorting':
                    if command.get('action') == 'install':
                        if session.thread and session.thread.is_alive():
                            raise ValueError('Stop acquisition before installing sorter dependencies')
                        with analysis.lock:
                            if any(job['status'] in ('queued', 'running') for job in analysis.jobs.values()):
                                raise ValueError('Wait for analysis to finish before installing sorter dependencies')
                    return self.reply(200, sorting.control(command))
                if self.path == '/api/catalog':
                    return self.reply(200, catalog.control(command))
                if self.path == '/api/playback':
                    return self.reply(200, inspect_playback(command.get('path', '')))
                if self.path == '/api/analysis':
                    if command.get('action') == 'run' and sorting.installation:
                        raise ValueError('Restart the workshop after installing sorter dependencies')
                    if command.get('action') == 'load' and command.get('source') == 'current':
                        command['path'] = str(getattr(session, 'run_dir', '') or session.snapshot.get('output', ''))
                        if not command['path']:
                            raise ValueError('No current experiment yet. Run an experiment or load a saved recording.')
                    return self.reply(200, analysis.control(command))
                if self.path == '/api/analysis-contracts':
                    from .custom_analysis import analysis_contracts
                    return self.reply(200, analysis_contracts(command.get('code')))
                if self.path == '/api/playground':
                    return self.reply(200, playground.control(command))
                if self.path in {'/api/code-preview', '/api/code-export'}:
                    candidate = {**config, **command.get('config', {})}
                    if self.path == '/api/code-preview':
                        return self.reply(200, code_bundle(candidate, command.get('code'), style=command.get('style', 'config')))
                    return self.reply(200, export_bundle(output_dir, candidate, command.get('code'), style=command.get('style', 'config')))
                if self.path == '/api/verify-experiment':
                    candidate = {**config, **command.get('config', {})}
                    report = verify_spec(candidate, code=command.get('code'))
                    if command.get('code') is not None:
                        from .custom_analysis import verify_analysis_code
                        analysis_errors = verify_analysis_code(command['code'], candidate.get('phases', []))
                        if analysis_errors:
                            report.update(ok=False, errors=report.get('errors', []) + analysis_errors)
                    if report['ok'] and command.get('preflight'):
                        if session.thread and session.thread.is_alive():
                            raise ValueError('Stop the current run before running preflight')
                        code = command.get('code')
                        python_report = {'ok': True} if report.get('native') else verify_python(code)
                        if not python_report['ok']:
                            report.update(ok=False, errors=[python_report['message']])
                        else:
                            # Temporary source file permits testing unsaved editor code.
                            # It is removed immediately; no experiment/output is created.
                            import tempfile
                            with tempfile.TemporaryDirectory(prefix='braindance-verify-') as temporary:
                                path = Path(temporary) / 'functions.py'
                                path.write_text(code or '', encoding='utf-8')
                                candidate.update(functions_file=str(path), headless=True)
                                probe = (NativeSession if report.get('native') else WorkshopSession)(Path(temporary), candidate)
                                probe.run(verify_only=True)
                                report.update(ok=probe.status == 'verified',
                                              errors=[] if probe.status == 'verified' else [probe.error])
                            report['preflight'] = ('Native imports, constructors and V3 dependencies checked; scientific execution, future data shapes and hardware remain runtime checks.' if report.get('native') else 'Dependencies, source setup, local environment step and sample function outputs checked. No Maxwell connection or stimulation. Full-run behavior and live hardware still require a run.')
                    return self.reply(200, report)
                if self.path == '/api/verify':
                    return self.reply(200, verify_python(command.get('code')))
                if self.path == '/api/profile':
                    loaded = profiles.save(command['name'], command['code'], command['settings'])
                    return self.reply(200, loaded)
                kind = command['kind']
                if kind == 'playback':
                    if session.thread and session.thread.is_alive():
                        raise ValueError('Stop the current run before starting playback')
                    inspect_playback(command.get('path', ''))
                    session = PlaybackSession(output_dir, {'playback_source': command['path'],
                                                          'speed': command.get('speed', 1.),
                                                          'loop': False})
                    session.start()
                elif kind == 'start':
                    if sorting.installation:
                        raise ValueError('Restart the workshop after installing sorter dependencies')
                    if session.thread and session.thread.is_alive():
                        raise ValueError('Stop the current run before starting another')
                    allowed = {'environment', 'channels', 'record_seconds', 'environment_seconds',
                               'causal_repeats', 'seed', 'detection', 'left_channels', 'right_channels',
                               'stim_electrodes', 'baseline_hz', 'sensory_index', 'encoder_gain', 'decoder_gain',
                               'episode_seconds', 'adjacency', 'experiment_name', 'experiment_notes', 'phases', 'phase_plan', 'native_settings', 'initial_data',
                               'num_neurons', 'grid_shape', 'neuron_positions', 'electrode_pitch_um',
                               'spatial_sigma_um', 'waveform_amplitude_uv', 'threshold_uv', 'sorter_path', 'write_output',
                               'source', 'live_config', 'speed', 'loop'}
                    updates = command.get('config', {})
                    if not set(updates) <= allowed:
                        raise ValueError('Unsupported setup field')
                    if command.get('profile'):
                        loaded = profiles.load(command['profile'])
                        selected.update(path=loaded['path'], name=loaded['name'], settings=loaded['settings'])
                    combined = {**config, **selected['settings'], **updates}
                    session_class = NativeSession if uses_native_runner(combined) else WorkshopSession
                    session = session_class(output_dir, combined)
                    session.start({**selected['settings'], **updates, 'functions_file': selected['path'], 'headless': False}, skip=command.get('skip', False))
                elif kind == 'stop':
                    session.stop_event.set()
                elif kind in {'pause', 'step', 'reload', 'adjacency', 'restart_phase'}:
                    if isinstance(session, PlaybackSession) and kind not in {'pause', 'step'}:
                        raise ValueError('Saved playback supports Pause, Step and Stop. Load it again to restart.')
                    if isinstance(session, NativeSession):
                        raise ValueError('Native phases support Stop; pause/restart require phase-specific cooperative support. Start a new run to retain the previous attempt.')
                    if kind == 'reload' and command.get('profile'):
                        command['functions_file'] = str(profiles.path(command['profile']))
                    session.commands.put(command)
                else:
                    raise ValueError('Unknown command')
                self.reply(200, {'ok': True})
            except (ValueError, KeyError, TypeError, OSError, SyntaxError, RuntimeError, ImportError) as exc:
                self.reply(400, {'error': str(exc)})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'BrainDance workshop: http://127.0.0.1:{server.server_port}')
    if open_browser:
        webbrowser.open(f'http://127.0.0.1:{server.server_port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        analysis.close()
        sorting.close()
        playground.close()
        session.stop_event.set()
        if session.thread:
            session.thread.join(timeout=10)
        server.server_close()
    return 0


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=['cartpole', 'foodland', 'ant'], default='cartpole')
    parser.add_argument('--source', help='Maxwell H5 path; omit for neural simulation')
    parser.add_argument('--live-config', help='Use live Maxwell hardware with this acquisition configuration')
    parser.add_argument('--channels', type=int, default=400)
    parser.add_argument('--num-neurons', type=int, default=40, help='Number of simulated spiking neurons (independent of electrodes)')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--record-seconds', type=float, default=3.)
    parser.add_argument('--environment-seconds', type=float, default=120.)
    parser.add_argument('--episode-seconds', type=float, default=10.)
    parser.add_argument('--profiles-dir', help='Directory for named Python experiment profiles')
    parser.add_argument('--profile', help='Saved profile name to load at startup')
    parser.add_argument('--causal-repeats', type=int, default=4)
    parser.add_argument('--speed', default='1', help='Wall pacing multiplier, or max')
    parser.add_argument('--detection', choices=['events', 'threshold', 'rt-sort'], default='threshold')
    parser.add_argument('--sorter-path', help='Trusted prebuilt RT-sort pickle, matching raw channel order and runtime')
    parser.add_argument('--threshold-uv', type=float, default=-30.)
    parser.add_argument('--write-output', action='store_true', default=True)
    parser.add_argument('--no-write-output', dest='write_output', action='store_false', help='Disable raw H5 recording')
    parser.add_argument('--output-dir')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--skip', action='store_true', help='Start environment using supplied pools/baseline or simulator presets')
    parser.add_argument('--functions-file')
    parser.add_argument('--calibration', help='Trusted local calibration.pkl from a previous workshop run')
    parser.add_argument('--loop', action='store_true', help='Loop H5 replay at EOF')
    parser.add_argument('--no-browser', dest='open_browser', action='store_false')
    parser.add_argument('--stim-electrodes', type=int, nargs=2)
    parser.add_argument('--left-channels', type=int, nargs='+')
    parser.add_argument('--right-channels', type=int, nargs='+')
    return main(**vars(parser.parse_args(argv)))


if __name__ == '__main__':
    raise SystemExit(cli())
