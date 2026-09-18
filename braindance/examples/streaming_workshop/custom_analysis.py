"""Participant analysis decorators and their builder/runtime contract."""
import ast
import copy
import inspect
import keyword
from pathlib import Path
import types

from braindance.core.phases_v3.phase_base_v3 import AnalysisPhaseV3


def analysis_phase(*, inputs=None, outputs=None, requires=(), provides=()):
    """Mark a synchronous ``function(exp) -> dict`` as a workshop analysis phase.

    ``exp.data`` holds earlier phase outputs (attribute access or .get(key)).
    ``exp.params`` contains all experiment parameters automatically.
    Returned dictionary entries become data available to subsequent phases.
    """
    inputs = requires if inputs is None else inputs
    outputs = provides if outputs is None else outputs
    def decorate(func):
        inspect.signature(func).bind(object())
        if inspect.iscoroutinefunction(func):
            raise ValueError('Analysis functions must be synchronous')
        func.__analysis_contract__ = dict(inputs=list(inputs), outputs=list(outputs))
        return func
    return decorate


def analysis_definition(params):
    params = dict(params) if isinstance(params, dict) else params
    if isinstance(params, dict):
        for old, new in (("requires", "inputs"), ("provides", "outputs")):
            if old in params:
                params.setdefault(new, params.pop(old))
    defaults = dict(function_name='custom_analysis', inputs=[], outputs=['analysis_result'], parameter_keys=[])
    if not isinstance(params, dict) or set(params) - set(defaults):
        raise ValueError('Unsupported custom analysis parameter')
    values = {**defaults, **params}
    name = values['function_name']
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name) or name in ('encode', 'decode', 'train'):
        raise ValueError('Analysis function name must be a Python identifier other than encode, decode or train')
    for key in ('inputs', 'outputs', 'parameter_keys'):
        items = values[key]
        if not isinstance(items, list) or any(not isinstance(item, str) or not item for item in items) or len(set(items)) != len(items):
            raise ValueError(f'{key} must contain distinct nonempty strings')
    return dict(label='Custom analysis', category='analysis', params=values,
                inputs=values['inputs'], outputs=values['outputs'])


def analysis_contracts(code):
    """Read literal decorator contracts without importing or executing code."""
    contracts, errors = {}, []
    try:
        if not isinstance(code, str) or len(code) > 100000:
            raise ValueError('Python code must contain at most 100000 characters')
        tree = ast.parse(code)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = [d for d in node.decorator_list if isinstance(d, ast.Call) and
                          ((isinstance(d.func, ast.Name) and d.func.id == 'analysis_phase') or
                           (isinstance(d.func, ast.Attribute) and d.func.attr == 'analysis_phase'))]
            if not decorators:
                continue
            try:
                args = node.args
                count = len(args.posonlyargs) + len(args.args)
                if isinstance(node, ast.AsyncFunctionDef) or count - len(args.defaults) > 1 or (count < 1 and args.vararg is None) or any(d is None for d in args.kw_defaults):
                    raise ValueError('must accept one experiment argument synchronously')
                decorator = decorators[0]
                if len(decorators) != 1 or decorator.args or any(k.arg not in ('inputs', 'outputs', 'requires', 'provides') for k in decorator.keywords):
                    raise ValueError('use @analysis_phase(inputs=[...], outputs=[...])')
                declared = {kw.arg: ast.literal_eval(kw.value) for kw in decorator.keywords}
                for old, new in (('requires', 'inputs'), ('provides', 'outputs')):
                    if old in declared:
                        if new in declared:
                            raise ValueError(f'Use {new} or {old}, not both')
                        declared[new] = declared.pop(old)
                contract = {key: list(declared.get(key, [])) for key in ('inputs', 'outputs')}
                if any(not isinstance(value, (list, tuple)) for value in declared.values()):
                    raise ValueError('inputs and outputs must be literal lists or tuples')
                analysis_definition(dict(function_name=node.name, **contract))
                contracts[node.name] = contract
            except (ValueError, TypeError) as exc:
                errors.append(f'{node.name}: {exc}')
    except (SyntaxError, ValueError, TypeError) as exc:
        errors.append(f'Analysis code: {exc}')
    return dict(contracts=contracts, errors=errors)


def verify_analysis_code(code, phases):
    """Validate selected functions; persisted builder contracts are only caches."""
    report = analysis_contracts(code)
    errors = list(report['errors'])
    for spec in phases:
        if spec.get('type') != 'custom_analysis':
            continue
        try:
            name = analysis_definition(dict(function_name=spec.get('params', {}).get('function_name', 'custom_analysis')))['params']['function_name']
            if name not in report['contracts']:
                errors.append(f'{spec["id"]}: missing valid @analysis_phase function {name}')
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def resolve_analysis_config(config, code=None):
    """Replace legacy builder contracts with the declarations in participant code."""
    phases = config.get('phases', []) + config.get('phase_plan', [])
    if not any(p.get('type') == 'custom_analysis' for p in phases):
        return config
    if code is None and config.get('functions_file'):
        code = Path(config['functions_file']).read_text(encoding='utf-8')
    if code is None:
        raise ValueError('Custom analysis needs participant code to read @analysis_phase declarations')
    errors = verify_analysis_code(code, phases)
    if errors:
        raise ValueError('; '.join(errors))
    contracts = analysis_contracts(code)['contracts']
    result = copy.deepcopy(config)
    for phase in result.get('phases', []) + result.get('phase_plan', []):
        if phase.get('type') == 'custom_analysis':
            name = phase.get('params', {}).get('function_name', 'custom_analysis')
            phase['params'] = dict(function_name=name, **contracts[name])
    return result


class CustomAnalysisPhase(AnalysisPhaseV3):
    def __init__(self, spec, config, module=None, code=None):
        super().__init__(name=spec['id'])
        if module is not None and not callable(module):
            name = spec.get('params', {}).get('function_name', 'custom_analysis')
            contract = getattr(getattr(module, name, None), '__analysis_contract__', None)
            if contract is None:
                raise ValueError('Analysis function needs @analysis_phase')
            spec = {**spec, 'params': dict(function_name=name, **contract)}
        else:
            spec = resolve_analysis_config({**config, 'phases': [spec]}, code)['phases'][0]
        definition = analysis_definition(spec.get('params', {}))
        self.inputs, self.outputs = definition['inputs'], definition['outputs']
        self.options, self.config, self.module = definition['params'], config, module

    def run(self, experiment):
        module = self.module() if callable(self.module) else self.module
        if module is None:
            path = Path(self.config['functions_file'])
            module = types.ModuleType('workshop_analysis')
            exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), module.__dict__)
        func = getattr(module, self.options['function_name'], None)
        contract = getattr(func, '__analysis_contract__', None)
        if not callable(func) or contract is None:
            raise ValueError('Analysis function needs @analysis_phase')
        # Hot reload reads the active function's declaration, never old UI selections.
        self.inputs, self.outputs = contract['inputs'], contract['outputs']
        missing = [key for key in self.inputs if key not in experiment.data.keys()]
        if missing:
            raise ValueError(f'Analysis requires unavailable data: {missing}')
        original = experiment.params
        available = {**self.config, **self.config.get('native_settings', {}),
                     **self.config.get('settings', {}), **original}
        experiment.params = copy.deepcopy(available)
        try:
            result = func(experiment)
        finally:
            experiment.params = original
        if not isinstance(result, dict) or not set(self.outputs) <= set(result):
            raise ValueError(f'Analysis must return a dictionary containing {self.outputs}')
        return result
