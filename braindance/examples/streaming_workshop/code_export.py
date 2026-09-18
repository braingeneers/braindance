"""Readable Python exports. Preview is read-only; exports never overwrite drafts."""
import json
import re
import time
from pathlib import Path
from pprint import pformat
from textwrap import indent


def code_bundle(config, participant_code, style='config'):
    if style not in ('config', 'phases'):
        raise ValueError('Choose phases or config export style')
    if not isinstance(participant_code, str) or len(participant_code) > 100000:
        raise ValueError('Participant code must contain at most 100000 characters')
    from .custom_analysis import resolve_analysis_config
    resolution_errors = []
    try:
        config = resolve_analysis_config(config, participant_code)
    except ValueError as exc:
        resolution_errors.append(str(exc))
    settings = json.loads(json.dumps(config, allow_nan=False))
    phases = settings.pop('phases', None)
    if phases is None:
        from .experiment_spec import normalize_phases
        phases = normalize_phases(settings)
    settings.pop('functions_file', None)
    settings.pop('headless', None)
    settings.pop('phase_plan', None)
    setup = '''"""Source, acquisition, simulator and shared experiment settings.

Edit SETTINGS or build_settings(). Phase-specific settings live in experiment.py.
Live Maxwell requires configured hardware, maxlab and valid electrode mappings.
"""
from copy import deepcopy

SETTINGS = ''' + pformat(settings, sort_dicts=False, width=95) + '''


def build_settings():
    settings = deepcopy(SETTINGS)
    # Replay a recording instead of generating neural activity:
    # settings.update(source="path/to/recording.raw.h5", live_config=None)
    # Use configured live Maxwell acquisition instead of replay/simulation:
    # settings.update(source=None, live_config="path/to/config.cfg", detection="threshold")
    # Set native phase settings / streaming phase electrode IDs in experiment.py.
    return settings
'''
    experiment = '''"""Run this experiment from Python, independently of the browser builder.

python experiment.py --verify
python experiment.py --output-dir ./results

Inputs: the BrainDance repository and the selected phases' dependencies.
"""
import argparse
import json
import sys
from pathlib import Path

# Supports running from the repository root or an export nested within it.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for root in (Path.cwd(), *HERE.parents):
    if (root / "braindance" / "core").is_dir():
        sys.path.insert(0, str(root))
        break

from environment_setup import build_settings
from braindance.config import get_output_dir
from braindance.examples.streaming_workshop.session import WorkshopSession
from braindance.examples.streaming_workshop.native_runner import NativeSession
from braindance.examples.streaming_workshop.experiment_spec import uses_native_runner

# Order, per-phase game/mappings, constructor arguments and data inputs.
PHASES = ''' + pformat(phases, sort_dicts=False, width=95) + '''


def run(output_dir=None, verify=False):
    config = build_settings()
    config.update(phases=PHASES, functions_file=str(HERE / "participant_functions.py"), headless=True)
    native = uses_native_runner(config)
    runner = NativeSession if native else WorkshopSession
    session = runner(Path(output_dir) if output_dir else get_output_dir() / "streaming_workshop", config)
    session.run(verify_only=verify)
    print(json.dumps({key: session.snapshot.get(key) for key in ("status", "error", "output", "phases")}, indent=2))
    return 0 if session.status == ("verified" if verify else "completed") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output-dir")
    raise SystemExit(run(**vars(parser.parse_args())))
'''
    if style == 'phases':
        from .experiment_spec import uses_native_runner
        from .native_catalog import native_catalog
        native = uses_native_runner({'phases': phases})
        native_definitions = native_catalog(include_legacy=True) if native else {}
        imports = {'from braindance.examples.streaming_workshop.python_experiment import PythonExperiment'}
        native_names = {}
        calls = []
        for index, phase in enumerate(phases):
            kind, params = phase['type'], dict(phase.get('params', {}))
            local = phase.get('settings', {})
            set_name = False
            if kind == 'custom_analysis':
                imports.add('from braindance.examples.streaming_workshop.custom_analysis import CustomAnalysisPhase')
                constructor = 'CustomAnalysisPhase(' + pformat(phase, sort_dicts=False, width=80) + ', settings)'
            elif native:
                entry = native_definitions[kind]
                class_name = entry['class_name']
                alias = class_name if native_names.get(class_name, entry['module']) == entry['module'] else class_name + '_' + str(index + 1)
                native_names[alias] = entry['module']
                imports.add('from ' + entry['module'] + ' import ' + class_name + (' as ' + alias if alias != class_name else ''))
                if 'name' in entry['parameters']:
                    params.setdefault('name', phase['id'])
                set_name = params.get('name') != phase['id']
                if 'render_mode' in entry['parameters']:
                    params['render_mode'] = None
                arguments = [key + '=' + pformat(value, sort_dicts=False, width=80) for key, value in params.items()]
                constructor = alias + '(\n' + indent(',\n'.join(arguments), '    ') + '\n)'
            else:
                if kind == 'recording':
                    class_name, source_key, target_key, default = 'RecordingPhase', 'record_seconds', 'duration', 3.
                elif kind == 'causal':
                    class_name, source_key, target_key, default = 'ResponseProbePhase', 'causal_repeats', 'repeats', 4
                else:
                    game = kind if kind != 'environment' else params.pop('environment', settings.get('environment', 'cartpole'))
                    class_name = {'cartpole': 'CartPolePhase', 'foodland': 'FoodLandPhase', 'ant': 'AntPhase'}[game]
                    source_key, target_key, default = 'environment_seconds', 'duration', 120.
                imports.add('from braindance.examples.streaming_workshop.python_experiment import ' + class_name)
                value = repr(params.pop(source_key)) if source_key in params else 'settings.get(' + repr(source_key) + ', ' + repr(default) + ')'
                arguments = [target_key + '=' + value, 'name=' + repr(phase['id'])]
                if kind != 'recording':
                    for key, fallback in ([('seed', 7)] if kind == 'causal' else []) + [('amplitude_mv', 100.), ('phase_width_us', 100)]:
                        arguments.append(key + '=settings.get(' + repr(key) + ', ' + repr(fallback) + ')')
                constructor = class_name + '(\n' + indent(',\n'.join(arguments), '    ') + '\n)'
                local = params
            arguments = [constructor]
            if set_name:
                arguments.append('name=' + repr(phase['id']))
            if local:
                arguments.append('settings=' + pformat(local, sort_dicts=False, width=80))
            calls.append('    exp.add_phase(\n' + indent(',\n'.join(arguments), '        ') + '\n    )')
        start = experiment.index('from braindance.config import get_output_dir')
        end = experiment.index('\n\nif __name__ == "__main__":')
        experiment = experiment[:start] + '\n'.join(sorted(imports)) + '\n\n\ndef build_experiment():\n' + '''    settings = build_settings()
    settings.update(functions_file=str(HERE / "participant_functions.py"), headless=True)
''' + '    exp = PythonExperiment(settings, native=' + repr(native) + ')\n' + '''    # These are actual V3 objects: edit constructors, reorder calls, or use loops.
''' + '\n'.join(calls) + '''
    return exp


def run(output_dir=None, verify=False):
    session = build_experiment().run(output_dir=output_dir, verify=verify)
    print(json.dumps({key: session.snapshot.get(key) for key in ("status", "error", "output", "phases")}, indent=2))
    return 0 if session.status == ("verified" if verify else "completed") else 1
''' + experiment[end:]
    readme = '''# Exported BrainDance experiment

- `experiment.py`: ordered phases and the executable entry point.
- `environment_setup.py`: source/acquisition/simulator and shared settings.
- `participant_functions.py`: your exact participant code, including helper functions.

Activate your Python environment, then run from the BrainDance repository root:

    python PATH_TO_THIS_FOLDER/experiment.py --verify
    python PATH_TO_THIS_FOLDER/experiment.py --output-dir ./results

The repository must remain available (or installed). Native phases need their
scientific dependencies; local simulation does not need Maxwell software.
Verification checks setup/sample behavior, not every future input or live hardware.

Edit these files freely. Browser changes create a NEW export folder and do not
overwrite these files. Arbitrary Python edits are not imported back into the builder.
To continue using the UI, keep its saved profile; the Python export is a separate draft.

Streaming adapters call encode/decode and optional train hooks in participant code.
Native V3 phases use their own controllers; this participant file is retained for
reference but is not automatically connected to native training code.
Custom analysis phases call the selected @analysis_phase function in either mode.
Inside that function, exp.data holds earlier phase outputs and exp.params contains
all experiment parameters automatically. Return a dictionary with
the declared outputs keys to make those values available to later phases.
'''
    if style == 'phases':
        readme += '''
This export constructs actual V3 objects and passes them to exp.add_phase(phase).
The same instances are validated and executed with acquisition-aware setup.
You can use normal Python loops and conditions; give each added phase a unique name.
Constructor arguments and object attributes are authoritative.
Per-phase settings hold streaming mappings or scoped scientific inputs.
Native objects run in this Python process; browser-native runs retain their worker.
Choose Config (PHASES) in the browser to export the dictionary-based version.
'''
    files = {'experiment.py': experiment, 'environment_setup.py': setup,
             'participant_functions.py': participant_code, 'README.md': readme}
    from .custom_analysis import verify_analysis_code
    errors = resolution_errors + verify_analysis_code(participant_code, phases)
    for name, code in files.items():
        if name.endswith('.py'):
            try:
                compile(code, name, 'exec')
            except SyntaxError as exc:
                errors.append(f'{name}:{exc.lineno}: {exc.msg}')
    return dict(files=files, errors=errors)


def export_bundle(output_dir, config, participant_code, style='config'):
    from .experiment_spec import verify_spec
    bundle = code_bundle(config, participant_code, style=style)
    if bundle['errors']:
        raise ValueError('; '.join(bundle['errors']))
    from .custom_analysis import resolve_analysis_config
    config = resolve_analysis_config(config, participant_code)
    report = verify_spec(config, code=participant_code)
    if not report['ok']:
        raise ValueError('Fix experiment validation before export: ' + '; '.join(report['errors']))
    name = re.sub(r'[^A-Za-z0-9_-]', '_', config.get('experiment_name', 'experiment'))[:64]
    folder = Path(output_dir).resolve() / 'exports' / f'{name}_{time.time_ns()}'
    folder.mkdir(parents=True, exist_ok=False)
    for filename, content in bundle['files'].items():
        (folder / filename).write_text(content, encoding='utf-8')
    (folder / 'builder_snapshot.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    return dict(directory=str(folder), files={name: str(folder / name) for name in bundle['files']})
