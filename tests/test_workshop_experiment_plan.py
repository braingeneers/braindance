"""Editable phase-object exports execute the instances users construct."""
import ast

import pytest
from test_streaming_workshop import workshop_config

from braindance.examples.streaming_workshop.code_export import code_bundle, export_bundle
from braindance.examples.streaming_workshop.experiment_plan import ExperimentPlan


def test_plan_preserves_order_settings_and_isolates_mutations():
    settings = {'initial_data': {'value': [1]}, 'phases': ['old'], 'phase_plan': ['old']}
    params, local = {'duration': .04}, {'channels': [2, 3]}
    plan = ExperimentPlan(settings)
    for index in range(2):
        assert plan.add_phase('native:phases3.RecordPhaseV3', name=f'record_{index}',
                              params=params, settings=local) is plan
    expected = {'initial_data': {'value': [1]}, 'phases': [
        {'id': f'record_{i}', 'type': 'native:phases3.RecordPhaseV3',
         'params': {'duration': .04}, 'settings': {'channels': [2, 3]}} for i in range(2)]}
    assert plan.to_config() == expected
    settings['initial_data']['value'].append(9)
    params['duration'] = 99
    local['channels'].clear()
    result = plan.to_config()
    result['phases'].reverse()
    result['phases'][0]['settings']['channels'].clear()
    result['initial_data']['value'].clear()
    assert plan.to_config() == expected


def test_generated_calls_construct_expanded_phase_objects():
    phases = [{'id': f'record_{i}', 'type': 'native:phases3.RecordPhaseV3',
               'params': {'duration': .04}, 'settings': {'custom': [i]}} for i in range(2)]
    config = {'experiment_name': 'roundtrip', 'phases': phases, 'phase_plan': [{'id': 'old', 'type': 'recording', 'params': {}}]}
    bundle = code_bundle(config, '# retained participant code\n', style='phases')
    assert not bundle['errors']
    tree = ast.parse(bundle['files']['experiment.py'])
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_phase']
    assert len(calls) == 2
    assert all(isinstance(call.args[0], ast.Call) for call in calls)
    assert 'PHASES =' not in bundle['files']['experiment.py']
    assert 'phase_plan' not in bundle['files']['environment_setup.py']
    # Execute the generated builder itself without invoking its CLI or acquisition.
    build = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'build_experiment')
    from pathlib import Path
    from braindance.examples.streaming_workshop.python_experiment import PythonExperiment
    from braindance.core.phases_v3.phases3 import RecordPhaseV3
    namespace = {'PythonExperiment': PythonExperiment, 'RecordPhaseV3': RecordPhaseV3,
                 'build_settings': lambda: {'experiment_name': 'roundtrip'}, 'HERE': Path('/export')}
    exec(compile(ast.Module(body=[build], type_ignores=[]), 'experiment.py', 'exec'), namespace)
    plan = namespace['build_experiment']()
    assert all(isinstance(phase, RecordPhaseV3) for phase in plan.phases)
    assert [phase.name for phase in plan.phases] == ['record_0', 'record_1']
    assert [phase.duration for phase in plan.phases] == [.04, .04]
    assert plan.phase_settings == [{'custom': [0]}, {'custom': [1]}]
    assert plan.settings['functions_file'] == '/export/participant_functions.py'


def test_invalid_export_style_writes_nothing(tmp_path):
    with pytest.raises(ValueError, match='export style'):
        code_bundle({}, '', style='unknown')
    with pytest.raises(ValueError, match='export style'):
        export_bundle(tmp_path, {}, '', style='unknown')
    assert not list(tmp_path.iterdir())


def test_native_runner_executes_supplied_objects_and_scopes_settings(tmp_path):
    from braindance.core.phases_v3.phase_base_v3 import AnalysisPhaseV3
    from braindance.examples.streaming_workshop.native_runner import run_phases

    seen = []

    class CapturePhase(AnalysisPhaseV3):
        inputs = ['gain']
        outputs = ['captured']

        def run(self, experiment):
            seen.append((self, self.gain, experiment.params['gain'], experiment.data.gain))
            return {'captured': self.gain}

    first, second = CapturePhase('first'), CapturePhase('second')
    config = dict(native_settings={'gain': 3}, phases=[
        {'id': 'first', 'settings': {'gain': 7}}, {'id': 'second'}])
    assert run_phases(config, [first, second], tmp_path / 'run') == 0
    assert seen == [(first, 7, 7, 7), (second, 3, 3, 3)]
    assert first.experiment.data.gain == 3
    assert first.experiment.data.captured == 3


def test_streaming_object_edits_control_execution(tmp_path, monkeypatch, workshop_config):
    from braindance.examples.streaming_workshop.python_experiment import PythonExperiment, RecordingPhase

    exp = PythonExperiment(workshop_config)
    phase = RecordingPhase(duration=.04, name='baseline')
    assert exp.add_phase(phase, settings={'record_seconds': 10.}) is exp
    phase.duration = .08
    assert exp.to_config()['phases'][0]['params']['record_seconds'] == .08
    seen = []
    original = RecordingPhase.run

    def capture(self, experiment):
        seen.append(self)
        return original(self, experiment)

    monkeypatch.setattr(RecordingPhase, 'run', capture)
    session = exp.run(tmp_path / 'run')
    assert session.status == 'completed', session.error
    assert seen == [phase]
    assert len(session.history) == 4
    assert session.history[-1]['t'] == pytest.approx(.08)


def test_streaming_analysis_uses_objects_without_reconstructing(tmp_path, monkeypatch, workshop_config):
    from pathlib import Path
    from braindance.examples.streaming_workshop import session as session_module
    from braindance.examples.streaming_workshop.custom_analysis import CustomAnalysisPhase
    from braindance.examples.streaming_workshop.python_experiment import PythonExperiment, RecordingPhase

    code = Path(workshop_config['functions_file']).read_text() + '''
from braindance.examples.streaming_workshop.custom_analysis import analysis_phase
@analysis_phase(inputs=['recording_baseline_hz'], outputs=['object_channel_count'])
def count_channels(exp):
    return {'object_channel_count': len(exp.data.recording_baseline_hz)}
'''
    functions_path = tmp_path / 'participant.py'
    functions_path.write_text(code)
    workshop_config['functions_file'] = str(functions_path)
    recording = RecordingPhase(duration=.04, name='record')
    analysis = CustomAnalysisPhase(dict(id='analysis', type='custom_analysis',
        params={'function_name': 'count_channels'}), workshop_config)
    exp = PythonExperiment(workshop_config)
    exp.add_phase(recording).add_phase(analysis)

    def forbid_reconstruction(*args, **kwargs):
        raise AssertionError('Supplied phase objects must not be reconstructed')

    monkeypatch.setattr(session_module, 'build_phase', forbid_reconstruction)
    session = exp.run(tmp_path / 'run')
    assert session.status == 'completed', session.error
    assert session.phase_objects == [recording, analysis]
    assert recording.experiment is analysis.experiment
    assert analysis.experiment.data.object_channel_count == workshop_config['channels']
    assert analysis.module is session.module
    assert len(session.history) == 2
    session.reload_functions()
    assert session.phase_objects == [recording, analysis]
    assert analysis.module is session.module
