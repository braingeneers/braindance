"""Custom decorated analysis contracts, execution, and exports."""
from types import SimpleNamespace
from test_streaming_workshop import workshop_config

import pytest

from braindance.core.phases_v3.data_context import DataContext
from braindance.examples.streaming_workshop.custom_analysis import (
    CustomAnalysisPhase, analysis_phase, verify_analysis_code,
)
from braindance.examples.streaming_workshop.experiment_spec import normalize_phases, verify_spec
from braindance.examples.streaming_workshop.native_runner import verify_native
from braindance.examples.streaming_workshop.code_export import code_bundle


CODE = '''from braindance.examples.streaming_workshop.custom_analysis import analysis_phase
@analysis_phase(requires=['workshop_baseline_hz'], provides=['analysis_result'])
def custom_analysis(exp):
    return {'analysis_result': sum(exp.data.workshop_baseline_hz) * exp.params['gain']}
'''


def spec():
    return dict(id='analysis', type='custom_analysis', params=dict(function_name='custom_analysis',
        inputs=['workshop_baseline_hz'], outputs=['analysis_result'], parameter_keys=['gain']))


def test_streaming_order_parameters_and_default_sequence():
    config = dict(channels=2, gain=2, phases=[dict(id='record', type='recording', params={}), spec()])
    assert verify_spec(config, code=CODE)['ok']
    config['phases'].reverse()
    assert any('missing workshop_baseline_hz' in e for e in verify_spec(config, code=CODE)['errors'])
    config['phases'].reverse()
    del config['gain']
    assert verify_spec(config, code=CODE)['ok']
    assert [p['type'] for p in normalize_phases({})] == ['recording', 'causal', 'environment']


def test_decorated_execution_and_parameter_scope():
    namespace = {}
    exec(CODE, namespace)
    data = DataContext()
    data.workshop_baseline_hz = [2, 3]
    original = {'original': 7}
    exp = SimpleNamespace(data=data, params=original)
    phase = CustomAnalysisPhase(spec(), dict(gain=2, unselected=100), SimpleNamespace(**namespace))
    assert not phase.needs_environment()
    assert phase.run(exp) == {'analysis_result': 10}
    assert exp.params is original
    namespace['custom_analysis'] = analysis_phase(requires=['workshop_baseline_hz'], provides=['analysis_result'])(lambda exp: {})
    phase.module = SimpleNamespace(**namespace)
    with pytest.raises(ValueError, match='dictionary containing'):
        phase.run(exp)
    assert exp.params is original


def test_native_order_and_file_execution(tmp_path):
    path = tmp_path / 'participant.py'
    path.write_text(CODE)
    config = dict(gain=3, functions_file=str(path), initial_data={'workshop_baseline_hz': [1, 2]}, phases=[spec()])
    assert verify_native(config)['ok']
    data = DataContext()
    data.update(config['initial_data'])
    assert CustomAnalysisPhase(spec(), config).run(SimpleNamespace(data=data, params={})) == {'analysis_result': 9}


@pytest.mark.parametrize('style', ['config', 'phases'])
def test_static_contract_and_export(style):
    assert not verify_analysis_code(CODE, [spec()])
    assert not verify_analysis_code(CODE.replace("provides=['analysis_result']", 'provides=[]'), [spec()])
    assert verify_analysis_code(CODE.replace('def custom_analysis(exp)', 'async def custom_analysis(exp)'), [spec()])
    assert verify_analysis_code('', [spec()])
    config = dict(gain=2, phases=[spec()])
    bundle = code_bundle(config, CODE, style=style)
    assert not bundle['errors']
    assert 'custom_analysis' in bundle['files']['experiment.py']
    assert bundle['files']['participant_functions.py'] == CODE
    assert code_bundle(config, '')['errors']


def test_streaming_analysis_runs_in_v3_pipeline(tmp_path, workshop_config):
    from pathlib import Path
    from braindance.examples.streaming_workshop.session import WorkshopSession
    marker = tmp_path / 'analysis-ran.txt'
    code = Path(workshop_config['functions_file']).read_text() + '\n' + CODE.replace(
        "    return {'analysis_result':", f"    __import__('pathlib').Path({str(marker)!r}).write_text(str(len(exp.data.workshop_baseline_hz)))\n    return {{'analysis_result':")
    path = tmp_path / 'custom.py'
    path.write_text(code)
    workshop_config.update(functions_file=str(path), gain=2,
        phases=[dict(id='record', type='recording', params={'record_seconds': .04}), spec()])
    session = WorkshopSession(tmp_path / 'run', workshop_config)
    session.run()
    assert session.status == 'completed', session.error
    assert int(marker.read_text()) == workshop_config['channels']


def test_native_custom_only_needs_no_acquisition(tmp_path):
    from braindance.examples.streaming_workshop.native_runner import NativeSession
    marker = tmp_path / 'native-analysis.txt'
    code = CODE.replace("    return {'analysis_result':", f"    __import__('pathlib').Path({str(marker)!r}).write_text(str(exp.params['gain']))\n    return {{'analysis_result':")
    path = tmp_path / 'custom.py'
    path.write_text(code)
    config = dict(functions_file=str(path), gain=4, initial_data={'workshop_baseline_hz': [1, 2]}, phases=[spec()])
    assert verify_spec(config)['native']
    session = NativeSession(tmp_path / 'run', config)
    session.run()
    assert session.status == 'completed', session.error
    assert marker.read_text() == '4'


def test_native_recording_then_custom_analysis(tmp_path, workshop_config):
    from braindance.examples.streaming_workshop.native_runner import NativeSession
    marker = tmp_path / 'recording-analysis.txt'
    code = CODE.replace('workshop_baseline_hz', 'recording_file').replace(
        "    return {'analysis_result': sum(exp.data.recording_file) * exp.params['gain']}",
        f"    __import__('pathlib').Path({str(marker)!r}).write_text(exp.data.recording_file)\n    return {{'analysis_result': exp.params['gain']}}")
    path = tmp_path / 'custom.py'
    path.write_text(code)
    analysis = spec()
    analysis['params']['inputs'] = ['recording_file']
    workshop_config.update(functions_file=str(path), gain=5, phases=[
        dict(id='record', type='native:phases3.RecordPhaseV3', params={'duration': .04}), analysis])
    assert verify_spec(workshop_config)['ok']
    session = NativeSession(tmp_path / 'run', workshop_config)
    session.run()
    assert session.status == 'completed', session.error
    assert marker.read_text()


def test_all_parameters_available_without_selection():
    @analysis_phase(provides=['analysis_result'])
    def custom_analysis(exp):
        return {'analysis_result': (exp.params['gain'], exp.params['unselected'], exp.params['native_value'], exp.params['original'])}
    phase = CustomAnalysisPhase(spec(), dict(gain=2, unselected=100, native_settings={'native_value': 5}),
                                SimpleNamespace(custom_analysis=custom_analysis))
    exp = SimpleNamespace(data=DataContext(), params={'original': 7})
    assert phase.run(exp) == {'analysis_result': (2, 100, 5, 7)}
    assert exp.params == {'original': 7}


def test_decorator_overrides_stale_builder_contract_everywhere(tmp_path):
    from braindance.examples.streaming_workshop.custom_analysis import analysis_contracts, resolve_analysis_config
    from braindance.examples.streaming_workshop.profiles import ProfileStore
    stale = spec()
    stale['params'].update(inputs=['outdated'], outputs=['outdated'], parameter_keys=['missing'])
    config = dict(phases=[dict(id='record', type='recording', params={}), stale], gain=2)
    assert verify_spec(config, code=CODE)['ok']
    assert analysis_contracts(CODE)['contracts']['custom_analysis']['inputs'] == ['workshop_baseline_hz']
    resolved = resolve_analysis_config(config, CODE)
    assert resolved['phases'][1]['params']['outputs'] == ['analysis_result']
    assert config['phases'][1]['params']['outputs'] == ['outdated']
    assert 'outdated' not in code_bundle(config, CODE)['files']['experiment.py']
    path = tmp_path / 'functions.py'
    path.write_text(CODE)
    config['functions_file'] = str(path)
    assert verify_spec(config)['ok']
    assert CustomAnalysisPhase(stale, config).inputs == ['workshop_baseline_hz']
    store = ProfileStore(tmp_path / 'profiles', path)
    code = 'def encode(*args): pass\ndef decode(*args): pass\n' + CODE
    loaded = store.save('analysis', code, config)
    assert loaded['settings']['phases'][1]['params']['inputs'] == ['workshop_baseline_hz']


def test_contract_parser_does_not_execute_code():
    from braindance.examples.streaming_workshop.custom_analysis import analysis_contracts
    report = analysis_contracts("raise RuntimeError('must not execute')\n" + CODE)
    assert not report['errors']
    assert report['contracts']['custom_analysis']['outputs'] == ['analysis_result']
    assert analysis_contracts(CODE.replace("requires=['workshop_baseline_hz']", 'requires=calculate_inputs()'))['errors']


def test_reload_contract_uses_active_function_and_restores_params():
    @analysis_phase(provides=['new_output'])
    def custom_analysis(exp):
        return {'new_output': exp.params['unselected']}
    namespace = {}
    exec(CODE, namespace)
    phase = CustomAnalysisPhase(spec(), dict(gain=2, unselected=42), SimpleNamespace(**namespace))
    phase.module = SimpleNamespace(custom_analysis=custom_analysis)
    exp = SimpleNamespace(data=DataContext(), params={})
    assert phase.run(exp) == {'new_output': 42}
    assert phase.inputs == []
    assert phase.outputs == ['new_output']
    assert exp.params == {}


def test_resolved_phase_plan_preserves_loop_groups(tmp_path):
    from copy import deepcopy
    from braindance.examples.streaming_workshop.custom_analysis import resolve_analysis_config
    from braindance.examples.streaming_workshop.profiles import ProfileStore
    analysis = spec()
    analysis['params'].update(inputs=['old_input'], outputs=['old_output'])
    analysis['loop'] = {'id': 'analysis_loop', 'count': 2}
    record = dict(id='record', type='recording', params={'record_seconds': 1})
    config = dict(phase_plan=[record, analysis], phases=[record, {**analysis, 'id': 'analysis_1'}, {**analysis, 'id': 'analysis_2'}])
    original = deepcopy(config)
    resolved = resolve_analysis_config(config, CODE)
    assert resolved['phase_plan'][0] == record
    assert resolved['phase_plan'][1]['loop'] == analysis['loop']
    assert resolved['phase_plan'][1]['params'] == dict(function_name='custom_analysis',
                                                       inputs=['workshop_baseline_hz'], outputs=['analysis_result'])
    assert all(p['params'] == resolved['phase_plan'][1]['params'] for p in resolved['phases'][1:])
    assert config == original
    store = ProfileStore(tmp_path / 'profiles', tmp_path / 'unused.py')
    loaded = store.save('loop_analysis', 'def encode(*args): pass\ndef decode(*args): pass\n' + CODE, config)
    assert loaded['settings']['phase_plan'] == resolved['phase_plan']
    assert store.load('loop_analysis')['settings']['phase_plan'] == resolved['phase_plan']


def test_canonical_analysis_code_runs_with_recording_output():
    code = CODE.replace('requires=', 'inputs=').replace('provides=', 'outputs=').replace('workshop_baseline_hz', 'recording_baseline_hz')
    config = dict(gain=2, phases=[dict(id='record', type='recording', params={}), spec()])
    assert verify_spec(config, code=code)['ok']
    namespace = {}
    exec(code, namespace)
    data = DataContext()
    data.recording_baseline_hz = [2, 3]
    phase = CustomAnalysisPhase(spec(), config, SimpleNamespace(**namespace))
    assert phase.inputs == ['recording_baseline_hz']
    assert phase.outputs == ['analysis_result']
    assert phase.run(SimpleNamespace(data=data, params={})) == {'analysis_result': 10}


def test_analysis_selected_electrodes_drive_native_sweep():
    from braindance.core.phases_v3.phases3 import NeuralSweepPhaseV3

    code = '''from braindance.examples.streaming_workshop.custom_analysis import analysis_phase
@analysis_phase(inputs=[], outputs=['stim_electrodes'])
def custom_analysis(exp):
    return {'stim_electrodes': [101, 205, 309]}
'''
    config = dict(phases=[spec(), dict(id='sweep', type='native:phases3.NeuralSweepPhaseV3', params={})])
    report = verify_native(config, code=code)
    assert report['ok'], report
    namespace = {}
    exec(code, namespace)
    data = DataContext()
    exp = SimpleNamespace(data=data, params={}, get_param=lambda key, default=None: default)
    data.update(CustomAnalysisPhase(spec(), config, SimpleNamespace(**namespace)).run(exp))
    sweep = NeuralSweepPhaseV3(amp_bounds=100, replicates=1)
    sweep.configure_from_experiment(exp)
    assert sweep.customize_environment_params({})['stim_electrodes'] == [101, 205, 309]
    assert [command[0] for command in sweep.generate_stim_commands()] == [[0], [1], [2]]
    # Repeated analyses may choose a different set; explicit subsets stay explicit.
    data.update({'stim_electrodes': [401, 505]}, overwrite=True)
    sweep.configure_from_experiment(exp)
    assert [command[0] for command in sweep.generate_stim_commands()] == [[0], [1]]
    subset = NeuralSweepPhaseV3(neuron_list=[1], amp_bounds=100, replicates=1)
    subset.configure_from_experiment(exp)
    assert [command[0] for command in subset.generate_stim_commands()] == [[1]]
