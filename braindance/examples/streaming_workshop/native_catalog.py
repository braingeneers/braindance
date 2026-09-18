"""Describe installed V3 phases without importing their optional dependencies.

Discovery and parameter checks are static. Construction imports the selected
phase and can allocate resources, so it belongs to launch rather than preflight.
"""
import ast
import copy
from functools import lru_cache
import importlib
import inspect
import json
from pathlib import Path

from .source_links import source_url


@lru_cache(maxsize=1)
def _catalog():
    root = Path(__file__).resolve().parents[2] / 'core' / 'phases_v3'
    classes = {}
    for path in sorted(root.glob('phases*.py')):
        if path.stem == 'phases_binned':
            continue  # Registered by phase_catalog with its acquisition runtime.
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = (path.stem, node)

    def inherits(name, ancestors, seen=()):
        if name in ancestors:
            return True
        if name not in classes or name in seen:
            return False
        return any(inherits(ast.unparse(base).split('.')[-1], ancestors, (*seen, name))
                   for base in classes[name][1].bases)

    def is_phase(name):
        return inherits(name, ('PhaseV3', 'AnalysisPhaseV3'))

    def describe(name):
        module, node = classes[name]
        parents = [ast.unparse(base).split('.')[-1] for base in node.bases]
        parent = next((describe(base) for base in parents if base in classes and is_phase(base)), None)
        entry = copy.deepcopy(parent) if parent else dict(
            params={}, required_params=[], parameters={}, inputs=[], outputs=[], caveats=[])
        entry.update(label=name, module=f'braindance.core.phases_v3.{module}',
                     source_url=source_url(f'braindance.core.phases_v3.{module}', name),
                     class_name=name, native=True, description=ast.get_docstring(node) or '',
                     category='analysis' if inherits(name, ('AnalysisPhaseV3',)) else 'experiment')
        for statement in node.body:
            if isinstance(statement, ast.Assign):
                targets, value = statement.targets, statement.value
            elif isinstance(statement, ast.AnnAssign):
                targets, value = [statement.target], statement.value
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id in ('inputs', 'outputs', 'requires', 'provides') and value:
                    try:
                        entry[{'requires': 'inputs', 'provides': 'outputs'}.get(target.id, target.id)] = list(ast.literal_eval(value))
                    except (ValueError, TypeError):
                        entry['caveats'].append(f'{target.id} is computed at runtime.')
        constructor = next((item for item in node.body
                            if isinstance(item, ast.FunctionDef) and item.name == '__init__'), None)
        if constructor:
            args = constructor.args
            # Wrappers accepting **kwargs retain their parent's options.
            if not args.kwarg:
                entry.update(params={}, required_params=[], parameters={})
            positional = args.posonlyargs + args.args
            defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
            pairs = list(zip(positional, defaults)) + list(zip(args.kwonlyargs, args.kw_defaults))
            for argument, default in pairs:
                if argument.arg == 'self':
                    continue
                metadata = dict(annotation=ast.unparse(argument.annotation) if argument.annotation else '',
                                required=default is None, supported=argument not in args.posonlyargs)
                if default is None:
                    if argument.arg not in entry['required_params']:
                        entry['required_params'].append(argument.arg)
                else:
                    metadata['default_python'] = ast.unparse(default)
                    try:
                        literal = json.loads(json.dumps(ast.literal_eval(default), allow_nan=False))
                        metadata['default'] = literal
                        entry['params'][argument.arg] = literal
                    except (ValueError, TypeError):
                        metadata['nonliteral_default'] = True
                        metadata['note'] = 'Omit to retain the Python constructor default.'
                        entry['params'].pop(argument.arg, None)
                if not metadata['supported']:
                    metadata['note'] = 'Positional-only parameter requires a Python experiment.'
                entry['parameters'][argument.arg] = metadata
        entry['parameter_metadata'] = entry['parameters']
        entry['mapping_contract'] = None
        if name in ('CartPolePhase', 'CartPolePhasWithViz'):
            entry['mapping_contract'] = dict(sensory=2, motor_min=4, training_min=2)
        elif name == 'FoodLandPhaseV3':
            entry['mapping_contract'] = dict(sensory=2, sensory_parameter='n_features', motor_min=2, training_min=2)
        elif name == 'AntPhaseV3':
            entry['mapping_contract'] = dict(sensory=8, sensory_parameter='n_features', motor_min=8, training_min=2)
        elif name == 'AntCPGPhaseV3':
            entry['mapping_contract'] = dict(sensory=5, sensory_expression='len(cpg_frequencies) * neurons_per_freq',
                                             motor_min=8, training_min=2)
        if name in ('CartPolePhase', 'CartPolePhasWithViz'):
            entry['inputs'] = list(dict.fromkeys(entry['inputs'] + ['rt_sort_object', 'recording_file']))
            entry['caveats'].append('Requires an RT-sort object and recording mapping; browser runs disable the optional desktop game window.')
        caveats = {
            'ActivityPhaseV3': 'Native activity rates are random placeholders, not measured firing rates.',
            'CausalAnalysisV3': 'Native causal directions are randomized placeholders, not causal estimates.',
            'FootprintPhaseV3': 'Native footprint positions use electrode numbers, not physical coordinates.',
            'SelectionPhaseV3': 'Native spatial-distance filtering is unfinished.',
            'SpatialSelectionPhaseV3': 'Spatial selection inherits placeholder footprint coordinates.',
            'RTSortPhaseV3': 'Requires RT-sort/model dependencies. Does not declare templates or spike_trains as outputs.',
        }
        if name in caveats:
            entry['caveats'].append(caveats[name])
        entry['caveats'] = list(dict.fromkeys(entry['caveats']))
        return entry

    return {f'native:{module}.{name}': describe(name)
            for name, (module, node) in classes.items()
            if is_phase(name) and not any(
                isinstance(item, ast.FunctionDef) and any(
                    ast.unparse(decorator).endswith('abstractmethod') for decorator in item.decorator_list)
                for item in node.body)}


def native_catalog(include_legacy=False):
    """Return JSON-safe phase metadata; no native phase module is imported."""
    from .data_contracts import data_metadata
    catalog = copy.deepcopy(_catalog())
    supported = {'RecordPhaseV3', 'FrequencyStimPhaseV3', 'NeuralSweepPhaseV3',
                 'RTSortPhaseV3', 'CartPolePhase', 'FoodLandPhaseV3', 'AntPhaseV3'}
    if not include_legacy:
        catalog = {key: entry for key, entry in catalog.items() if entry['class_name'] in supported}
    for entry in catalog.values():
        for key in ('inputs', 'outputs'):
            entry[key + '_metadata'] = data_metadata(entry[key])
        entry['data_contracts'] = data_metadata(entry['inputs'] + entry['outputs'])
    return catalog


def validate_native_parameters(spec):
    """Return schema errors for a phase spec without importing native code.

This checks parameter names, JSON values, required arguments and simple types.
Dependencies, array dimensions and resource availability need separate checks.
"""
    if not isinstance(spec, dict) or spec.get('type') not in _catalog():
        return ['Choose a registered native phase type.']
    entry = _catalog()[spec['type']]
    params = spec.get('params', {})
    if not isinstance(params, dict):
        return ['Native phase parameters must be an object.']
    errors = []
    for name in entry['required_params']:
        if name not in params:
            errors.append(f'{name}: required parameter is missing.')
    for name, value in params.items():
        metadata = entry['parameters'].get(name)
        if metadata is None:
            errors.append(f'{name}: unknown native phase parameter.')
            continue
        if not metadata['supported']:
            errors.append(f'{name}: requires a Python experiment (positional-only argument).')
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            errors.append(f'{name}: use finite JSON values.')
            continue
        annotation = metadata['annotation']
        default = metadata.get('default')
        expected = annotation if annotation in ('str', 'int', 'float', 'bool', 'list', 'dict', 'tuple') else None
        if expected is None and default is not None and type(default) in (str, bool, int, float):
            expected = type(default).__name__
        if value is None and (('default' in metadata and default is None) or 'None' in annotation):
            continue
        # Native recording/stimulation phases accept fractional seconds despite
        # their older integer duration annotations (see replay_experiment_v3).
        if name in ('duration', 'max_time') and expected == 'int':
            expected = 'float'
        valid = {'str': lambda: isinstance(value, str),
                 'bool': lambda: isinstance(value, bool),
                 'int': lambda: type(value) is int,
                 'float': lambda: type(value) in (int, float),
                 'list': lambda: isinstance(value, list),
                 'tuple': lambda: isinstance(value, (list, tuple)),
                 'dict': lambda: isinstance(value, dict)}
        if expected in valid and not valid[expected]():
            errors.append(f'{name}: expected {expected}.')
    return errors


def instantiate_native(spec):
    """Construct one allowlisted phase. Call only when launching native work."""
    errors = validate_native_parameters(spec)
    if errors:
        raise ValueError('; '.join(errors))
    entry = _catalog()[spec['type']]
    phase_class = getattr(importlib.import_module(entry['module']), entry['class_name'])
    params = dict(spec.get('params', {}))
    if 'name' in entry['parameters'] and params.get('name') is None and spec.get('id'):
        params['name'] = spec['id']
    inspect.signature(phase_class).bind(**params)
    return phase_class(**params)
