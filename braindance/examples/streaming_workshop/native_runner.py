"""Run native V3 phases in an isolated process, retaining V3 output/checkpoints."""
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from .native_catalog import instantiate_native, native_catalog, validate_native_parameters


def verify_native(config, code=None):
    from .custom_analysis import resolve_analysis_config
    try:
        config = resolve_analysis_config(config, code)
    except (ValueError, OSError) as exc:
        return dict(ok=False, phases=[], errors=[str(exc)], caveats=[], native=True)
    catalog = native_catalog(include_legacy=True)
    available = dict(config.get('initial_data', {}))
    available.update({key: value for key, value in config.get('native_settings', {}).items() if value is not None})
    origins = {key: 'experiment input' for key in available}
    errors, rows, caveats = [], [], []
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', str(config.get('experiment_name', 'native_experiment'))):
        errors.append('Experiment name must use letters, numbers, underscores or hyphens')
    ids = set()
    for spec in config.get('phases', []):
        identifier = spec.get('id', '')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', identifier) or identifier in ids:
            errors.append('Native phases need unique nonempty IDs')
        ids.add(identifier)
        custom = spec.get('type') == 'custom_analysis'
        if custom:
            from .custom_analysis import analysis_definition
            try:
                entry = analysis_definition(spec.get('params', {}))
            except ValueError as exc:
                errors.append(f'{identifier}: {exc}')
                continue
        else:
            entry = catalog.get(spec.get('type'))
        if not entry:
            errors.append('Use either a native V3 sequence or a streaming sequence; do not mix execution modes')
            continue
        if not custom:
            errors.extend(f'{identifier}: {error}' for error in validate_native_parameters(spec))
        local_origins = {**origins, **{key: f'{identifier} settings' for key, value in spec.get('settings', {}).items() if value is not None}}
        contract = entry.get('mapping_contract') or {}
        constructor = spec.get('params', {})
        for key, count in [('sensory_neurons', constructor.get(contract.get('sensory_parameter'), contract.get('sensory'))),
                           ('motor_neurons', contract.get('motor_min')), ('training_neurons', contract.get('training_min'))]:
            values = constructor.get(key)
            if values is not None and count is not None:
                if not isinstance(values, list) or any(type(v) is not int or v < 0 for v in values) or len(set(values)) != len(values) or (len(values) != count if key == 'sensory_neurons' else len(values) < count):
                    errors.append(f'{identifier}: {key} does not meet native mapping count {count}')
        requirements = {key: local_origins.get(key) for key in entry['inputs']}
        errors.extend(f'{identifier}: missing {key}' for key, source in requirements.items() if source is None)
        rows.append(dict(id=identifier, type=spec['type'], params=spec.get('params', {}),
                         requirements=requirements, inputs=requirements, outputs=entry['outputs']))
        origins.update({key: f'phase {identifier}' for key in entry['outputs']})
        caveats.extend(f'{identifier}: {warning}' for warning in entry.get('caveats', []))
    if not rows:
        errors.append('Add at least one native phase')
    return dict(ok=not errors, phases=rows, errors=errors, caveats=caveats, native=True)


class NativeSession:
    def __init__(self, output_dir, config):
        self.output_dir, self.config = Path(output_dir), dict(config)
        self.thread = None
        self.stop_event = threading.Event()
        self.status, self.error = 'ready', ''
        self.snapshot = dict(status='ready', native=True, restart_supported=False)

    def start(self, overrides=None, skip=False):
        if self.thread and self.thread.is_alive():
            raise ValueError('Stop the current native experiment first')
        self.config.update(overrides or {})
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self, verify_only=False):
        try:
            self._run(verify_only)
        except Exception as exc:
            process = getattr(self, 'process', None)
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            self.status, self.error = 'error', f'{type(exc).__name__}: {exc}'
            self.snapshot.update(status='error', error=self.error, native=True)
            if hasattr(self, 'run_dir'):
                (self.run_dir / 'attempt.json').write_text(json.dumps(dict(status='error', error=self.error)), encoding='utf-8')

    def _run(self, verify_only=False):
        report = verify_native(self.config)
        if not report['ok']:
            self.status, self.error = 'error', '; '.join(report['errors'])
            self.snapshot = dict(status=self.status, error=self.error, native=True)
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = self.output_dir / ('native_' + str(time.time_ns()))
        run_dir.mkdir()
        self.run_dir = run_dir
        specification = run_dir / 'experiment.json'
        specification.write_text(json.dumps(self.config, indent=2), encoding='utf-8')
        self.status = 'running'
        progress = run_dir / 'progress.json'
        args = [sys.executable, '-u', '-m', __name__, str(specification)]
        if verify_only:
            args.append('--verify')
        with (run_dir / 'console.log').open('w', encoding='utf-8') as log:
            worker_env = dict(os.environ)
            worker_env['PYTHONPATH'] = os.pathsep.join([str(Path(__file__).resolve().parents[3]), worker_env.get('PYTHONPATH', '')])
            process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                       env=worker_env,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.process = process
            deadline = time.monotonic() + 45 if verify_only else float('inf')
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError('Native constructor preflight exceeded 45 seconds; no experiment was launched')
                if self.stop_event.wait(.15):
                    process.terminate()
                    process.wait(timeout=10)
                    break
                try:
                    current = json.loads(progress.read_text(encoding='utf-8')) if progress.exists() else {}
                except (OSError, ValueError):
                    current = {}
                self.snapshot = dict(status='running', native=True, restart_supported=False,
                                     phases=[p['id'] for p in self.config['phases']], output=str(run_dir), **current)
        self.status = 'stopped' if self.stop_event.is_set() else ('verified' if verify_only else 'completed') if process.returncode == 0 else 'error'
        tail = (run_dir / 'console.log').read_text(encoding='utf-8')[-5000:]
        self.error = tail if self.status == 'error' else ''
        self.snapshot.update(status=self.status, error=self.error, native=True, output=str(run_dir), log=tail)
        (run_dir / 'attempt.json').write_text(json.dumps(dict(status=self.status, phases=self.config['phases']), indent=2), encoding='utf-8')


def main(spec_path, verify=False):
    config = json.loads(Path(spec_path).read_text(encoding='utf-8'))
    report = verify_native(config)
    if not report['ok']:
        raise ValueError('; '.join(report['errors']))
    from .custom_analysis import CustomAnalysisPhase, verify_analysis_code
    if any(p['type'] == 'custom_analysis' for p in config['phases']):
        errors = verify_analysis_code(Path(config['functions_file']).read_text(encoding='utf-8'), config['phases'])
        if errors:
            raise ValueError('; '.join(errors))
    # Browser runs observe geometry externally and never open desktop windows.
    for spec in config['phases']:
        if spec['type'] != 'custom_analysis':
            entry = native_catalog(include_legacy=True).get(spec['type'], {})
            if 'render_mode' in entry.get('parameters', {}):
                spec.setdefault('params', {})['render_mode'] = None
    phases = [CustomAnalysisPhase(spec, {**config, 'settings': spec.get('settings', {})})
              if spec['type'] == 'custom_analysis' else instantiate_native(spec) for spec in config['phases']]
    return run_phases(config, phases, Path(spec_path).parent, verify=verify)


def run_phases(config, phases, output, verify=False):
    """Validate and execute the supplied V3 objects with native acquisition.

    ``config['phases']`` supplies matching IDs and optional scoped settings.
    Constructor arguments and contracts come from the objects themselves.
    Returns 0 on successful verification/execution, or 1 on execution failure.
    """
    from braindance.core.phases_v3.experiment_v3 import Experiment
    from braindance.core.phases_v3.phase_base_v3 import PhaseValidator
    from braindance.core.phases_v3.data_context import DataContext
    from braindance.core.simulation import NeuralSimulationSource
    phases = list(phases)
    if len(phases) != len(config.get('phases', [])):
        raise ValueError('Each phase object needs matching ID/settings metadata')
    output = Path(output)
    data = DataContext(overwrite_existing=True)
    data.update(config.get('native_settings', {}))
    data.update(config.get('initial_data', {}))
    for phase, spec in zip(phases, config['phases']):
        local = DataContext(overwrite_existing=True)
        local.update({key: data.get(key) for key in data.keys()})
        local.update(spec.get('settings', {}))
        PhaseValidator.validate_pipeline([phase], existing_data=local)
        data.update({key: True for key in phase.outputs})
    if verify:
        for phase in phases:
            for name in ('game_env', 'game'):
                value = getattr(phase, name, None)
                if hasattr(value, 'close'):
                    value.close()
        print('Native imports, constructors and V3 dependencies passed. Scientific execution and hardware were not run.')
        return 0
    output.mkdir(parents=True, exist_ok=True)

    class NativeExperiment(Experiment):
        def _run_single_phase(self, phase, phase_idx, checkpoint_boundary=True):
            original_params, original_data = self.params, self.data
            local_settings = config['phases'][phase_idx].get('settings', {})
            self.params = {**original_params, **local_settings}
            self.data = DataContext(overwrite_existing=True)
            self.data.update({key: original_data.get(key) for key in original_data.keys()})
            self.data.update(local_settings)
            progress = output / 'progress.json'
            temporary = progress.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(phase=config['phases'][phase_idx]['id'])), encoding='utf-8')
            temporary.replace(progress)
            game = getattr(phase, 'game_env', None)
            names = {'CartPolePhase': 'cartpole', 'CartPolePhasWithViz': 'cartpole',
                     'FoodLandPhaseV3': 'foodland', 'AntPhaseV3': 'ant'}
            kind = names.get(type(phase).__name__)
            if game is not None and kind:
                from .native_visualization import ObservedGame
                phase.game_env = ObservedGame(game, kind, progress, config['phases'][phase_idx]['id'])
            try:
                return super()._run_single_phase(phase, phase_idx, checkpoint_boundary)
            finally:
                if game is not None and kind:
                    phase.game_env = game
                original_data.update({key: self.data.get(key) for key in self.data.keys()
                                      if key not in local_settings or key in phase.outputs})
                self.params, self.data = original_params, original_data

        def _create_environment_for_phase(self, phase):
            # Each core environment owns its source and output file. Do not reuse a
            # source closed by a preceding native phase's cleanup.
            if not config.get('live_config'):
                source = config.get('source') or NeuralSimulationSource(num_channels=config.get('channels', 8),
                    seed=config.get('seed', 7), speed=config.get('speed', 1.), adjacency=config.get('adjacency'),
                    **{key: config[key] for key in ('num_neurons', 'grid_shape', 'neuron_positions',
                        'electrode_pitch_um', 'spatial_sigma_um', 'waveform_amplitude_uv') if key in config})
                replay = dict(source=source, write_output=True)
                if config.get('source'):
                    replay.update(speed=config.get('speed', 1.), loop=config.get('loop', False))
                else:
                    source.chunk_frames = round(source.sampling_hz * .02)
                self.params['maxwell_env'] = {'replay': replay}
            return super()._create_environment_for_phase(phase)

    params = dict(config.get('native_settings', {}))
    if config.get('live_config'):
        params['config'] = config['live_config']
    experiment = NativeExperiment(config.get('experiment_name', 'native_experiment'), params=params,
                                   save_dir=str(output), overwrite_existing=True, auto_load_data=False)
    experiment.data.update(config.get('initial_data', {}))
    for phase in phases:
        experiment.add_phase(phase)
    # The same V3 validator ran above with each phase's scoped input overlay.
    # Keep original requires intact for runtime configuration and checks.
    return 0 if experiment.run(validate=False) else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], '--verify' in sys.argv[2:]))
