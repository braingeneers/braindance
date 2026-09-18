"""Editable workshop phase definitions and preflight dependency checks."""
import copy
import math
import re

from braindance.core.phases_v3.data_context import DataContext
from braindance.core.phases_v3.phase_base_v3 import PhaseValidator
from .source_links import source_url


def phase_catalog():
    catalog = {
        'recording': dict(label='Recording', inputs=[], outputs=['recording_baseline_hz'],
                          params={'record_seconds': 3.}),
        'causal': dict(label='Response probes', inputs=['recording_baseline_hz'],
                       outputs=['response_probe_hz', 'response_probe_trials'], params={'causal_repeats': 4, 'stim_electrodes': [0, 1]}),
        'environment': dict(label='Mapped environment (CartPole / FoodLand / Ant)', inputs=['recording_baseline_hz'],
                            outputs=['environment_episodes', 'environment_reward'],
                            params={'environment_seconds': 120., 'episode_seconds': 10.,
                                    'sensory_index': 2, 'encoder_gain': 100., 'decoder_gain': .025,
                                    'environment': 'cartpole', 'left_channels': [0, 1, 2, 3],
                                    'right_channels': [4, 5, 6, 7], 'stim_electrodes': [0, 1]}),
    }
    for kind, definition in catalog.items():
        # These streaming adapters acquire data or stimulate while running;
        # even response probes are experimental work, not offline analysis.
        definition['category'] = 'experiment'
        definition['source_url'] = source_url(
            'braindance.core.phases_v3.phases_binned', {'recording': 'BinnedRecordingPhaseV3', 'causal': 'ResponseProbePhaseV3', 'environment': 'MappedEnvironmentPhaseV3'}[kind])
        definition['required_stim_electrodes'] = 0 if kind == 'recording' else 2
        definition['description'] = 'V3 phase with binned acquisition and participant encode/decode callbacks; two sensory outputs.' if kind == 'environment' else 'V3 phase with binned acquisition.'
    catalog['environment']['hidden'] = True  # Compatibility with saved parameterized phases.
    for game, label in [('cartpole', 'CartPole'), ('foodland', 'FoodLand'), ('ant', 'Ant')]:
        definition = copy.deepcopy(catalog['environment'])
        definition.update(label=label, environment=game, hidden=False)
        definition['params'].pop('environment')
        definition['params']['sensory_index'] = 2 if game == 'cartpole' else 0
        catalog[game] = definition
    catalog['custom_analysis'] = dict(label='Custom analysis', category='analysis',
        description='Decorated participant function; runs once using previous phase data.',
        inputs=[], outputs=['analysis_result'],
        params=dict(function_name='custom_analysis', inputs=[], outputs=['analysis_result'], parameter_keys=[]))
    from .data_contracts import data_metadata
    for definition in catalog.values():
        for key in ('inputs', 'outputs'):
            definition[key + '_metadata'] = data_metadata(definition[key])
        definition['data_contracts'] = data_metadata(definition['inputs'] + definition['outputs'])
        definition['requires'], definition['provides'] = definition['inputs'], definition['outputs']
    return catalog


def normalize_phases(config, skip=False):
    if skip and 'phases' not in config:
        return [dict(id='environment', type='environment', params={})]
    if 'phases' not in config:
        return [dict(id=kind, type=kind, params={}) for kind in ('recording', 'causal', 'environment')]
    phases = config['phases']
    if not isinstance(phases, list) or not 1 <= len(phases) <= 32:
        raise ValueError('Add between 1 and 32 phases')
    result, ids = [], set()
    for phase in phases:
        if not isinstance(phase, dict) or phase.get('type') not in phase_catalog():
            raise ValueError('Choose a supported phase type')
        identifier = phase.get('id', '')
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', identifier) or identifier in ids:
            raise ValueError('Phase IDs must be unique, using letters, numbers, underscores or hyphens')
        ids.add(identifier)
        params = phase.get('params', {})
        if not isinstance(params, dict) or not set(params) <= (set(phase_catalog()[phase['type']]['params']) | ({'requires', 'provides'} if phase['type'] == 'custom_analysis' else set())):
            raise ValueError(f'{identifier}: unsupported phase parameter')
        if phase['type'] == 'custom_analysis':
            from .custom_analysis import analysis_definition
            analysis_definition(params)
        result.append(dict(id=identifier, type=phase['type'], params=dict(params)))
    return result


def uses_native_runner(config):
    phases = config.get('phases', [])
    return any(str(p.get('type', '')).startswith('native:') for p in phases) or bool(phases) and all(p.get('type') == 'custom_analysis' for p in phases)


def verify_spec(config, skip=False, code=None, phase_objects=None):
    """Validate ordering with the same PhaseValidator used by Experiment.run."""
    from .session import build_phase, LEGACY_OUTPUTS
    from .custom_analysis import resolve_analysis_config
    try:
        config = resolve_analysis_config(config, code)
    except (ValueError, OSError) as exc:
        return dict(ok=False, errors=[str(exc)], phases=[])
    if uses_native_runner(config):
        from .native_runner import verify_native
        return verify_native(config, code=code)
    errors, rows = [], []
    try:
        name = config.get('experiment_name', 'streaming_workshop')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
            raise ValueError('Experiment name: use 1–64 letters, numbers, underscores or hyphens')
        phases = normalize_phases(config, skip)
    except ValueError as exc:
        return dict(ok=False, errors=[str(exc)], phases=[])
    existing = DataContext(overwrite_existing=True)
    origins = {}
    baseline = config.get('baseline_hz')
    if baseline:
        try:
            if not isinstance(baseline, list) or not all(math.isfinite(v) and v >= 0 for v in baseline):
                raise ValueError()
            if config.get('detection') != 'rt-sort' and not config.get('source') and len(baseline) != config.get('channels', 8):
                raise ValueError()
            existing.recording_baseline_hz = baseline
            origins['recording_baseline_hz'] = 'explicit baseline settings'
        except (ValueError, TypeError):
            errors.append('Baseline must contain one nonnegative finite rate per channel')
    elif config.get('calibration'):
        existing.recording_baseline_hz = 'validated during source preflight'
        origins['recording_baseline_hz'] = 'saved calibration (checked during preflight)'
    elif skip and 'phases' not in config:
        existing.recording_baseline_hz = []  # Legacy CLI quick-start zero preset.
        origins['recording_baseline_hz'] = 'legacy zero baseline preset'
    # Resolve legacy participant input keys without changing saved Python code.
    for key, alias in LEGACY_OUTPUTS.items():
        if key in origins:
            origins[alias] = origins[key]
            existing.update({alias: existing.get(key)})
    objects = []
    if phase_objects is not None and [p.name for p in phase_objects] != [p['id'] for p in phases]:
        return dict(ok=False, errors=['Phase objects must match the phase order'], phases=[])
    for index, phase in enumerate(phases):
        kind, identifier = phase['type'], phase['id']
        definition = phase_catalog()[kind]
        if kind == 'custom_analysis':
            from .custom_analysis import analysis_definition
            definition = analysis_definition(phase['params'])
        resolved = {key: phase['params'].get(key, config.get(key, default)) for key, default in definition['params'].items()}
        if definition.get('environment'):
            resolved['environment'] = definition['environment']
        if kind in ('environment', 'cartpole', 'foodland', 'ant') and 'sensory_index' not in phase['params'] and 'sensory_index' not in config:
            resolved['sensory_index'] = 2 if resolved.get('environment') == 'cartpole' else 0
        if kind in ('environment', 'cartpole', 'foodland', 'ant'):
            n = config.get('channels', 8)
            for key, default in [('left_channels', list(range(n // 2))), ('right_channels', list(range(n // 2, n)))]:
                if key not in phase['params'] and key not in config:
                    resolved[key] = default
        for key, value in (resolved.items() if kind != 'custom_analysis' else []):
            if key == 'environment':
                if value not in ('cartpole', 'foodland', 'ant'):
                    errors.append(f'{identifier}: Unknown environment; choose CartPole, Foodland or Ant')
                continue
            if key in ('left_channels', 'right_channels', 'stim_electrodes'):
                if not isinstance(value, list) or not value or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in value) or len(set(value)) != len(value):
                    errors.append(f'{identifier}: {key} needs distinct nonnegative integer IDs')
                elif key == 'stim_electrodes' and len(value) != definition['required_stim_electrodes']:
                    errors.append(f'{identifier}: requires exactly {definition["required_stim_electrodes"]} stimulation electrodes')
                elif not config.get('source') and not config.get('live_config'):
                    if key == 'stim_electrodes' and config.get('grid_shape'):
                        grid = config['grid_shape']
                        if not isinstance(grid, (list, tuple)) or len(grid) != 2 or not all(isinstance(v, int) and v > 0 for v in grid):
                            errors.append(f'{identifier}: invalid electrode grid')
                        else:
                            mapped = {i // grid[1] * 220 + i % grid[1] for i in range(config.get('channels', 8))}
                            if not set(value) <= mapped:
                                errors.append(f'{identifier}: {key} contains an unmapped physical electrode')
                    elif (key == 'stim_electrodes' or config.get('detection') != 'rt-sort') and any(v >= config.get('channels', 8) for v in value):
                        errors.append(f'{identifier}: {key} contains an unmapped channel/electrode')
                continue
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            if valid and key.endswith('_seconds'):
                valid = value >= .02 and math.isclose(value / .02, round(value / .02), abs_tol=1e-6)
            if valid and key in ('causal_repeats', 'sensory_index'):
                valid = isinstance(value, int) and value >= (1 if key == 'causal_repeats' else 0)
            if not valid:
                errors.append(f'{identifier}: invalid {key}; durations must use positive 20 ms increments')
        if kind in ('environment', 'cartpole', 'foodland', 'ant') and isinstance(resolved['left_channels'], list) and isinstance(resolved['right_channels'], list) and set(resolved['left_channels']) & set(resolved['right_channels']):
            errors.append(f'{identifier}: decoder pools must be disjoint')
        requirements = {key: origins.get(key) for key in definition['inputs']}
        for key, origin in requirements.items():
            if origin is None:
                errors.append(f'{identifier}: missing {key}; supply it in settings or add an earlier phase that provides it')
        rows.append(dict(id=identifier, type=kind, inputs=requirements, outputs=definition['outputs'],
                         requirements=requirements, provides=definition['outputs'], params=resolved))
        if phase_objects is not None:
            objects.append(phase_objects[index])
        elif kind == 'custom_analysis':
            from .custom_analysis import CustomAnalysisPhase
            objects.append(CustomAnalysisPhase(phase, config, code=code))
        elif not errors:
            objects.append(build_phase(kind, identifier=identifier, params=resolved))
        for key in definition['outputs']:
            origins[key] = f'phase {identifier}'
            if key in LEGACY_OUTPUTS:
                origins[LEGACY_OUTPUTS[key]] = origins[key]
    if not errors:
        try:
            for obj in objects:
                PhaseValidator.validate_pipeline([obj], existing_data=existing)
                existing.update({key: True for key in obj.outputs})
                existing.update({LEGACY_OUTPUTS[key]: True for key in obj.outputs if key in LEGACY_OUTPUTS})
        except Exception as exc:
            errors.append(str(exc))
    return dict(ok=not errors, errors=errors, phases=rows)
