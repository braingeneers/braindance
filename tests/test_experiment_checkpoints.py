"""Hardware-free checks for persisted phase boundaries and fail-closed groups."""
import hashlib
import json

import numpy as np
import pytest

from braindance.core.phases_v3.data_context import DataContext
from braindance.core.phases_v3.experiment_v3 import Experiment
from braindance.core.phases_v3.phase_base_v3 import AnalysisPhaseV3, PhaseGroup


class Produce(AnalysisPhaseV3):
    provides = ['value', 'array']

    def __init__(self, calls=None):
        super().__init__()
        self.calls = calls if calls is not None else []

    def run(self, experiment):
        self.calls.append('produce')
        return {'value': 7, 'array': np.arange(4)}


class Consume(AnalysisPhaseV3):
    requires = ['value', 'array']
    provides = ['derived']

    def run(self, experiment):
        return {'derived': experiment.data.value + int(experiment.data.array.sum())}


class Fail(AnalysisPhaseV3):
    def run(self, experiment):
        raise RuntimeError('controlled phase failure')


def experiment(tmp_path, *phases, auto_load_data=True):
    exp = Experiment('checkpoint_test', save_dir=tmp_path,
                     auto_load_data=auto_load_data)
    exp.verbose = False
    return exp.add_phases(*phases)


def test_analysis_only_resume_preserves_completed_output(tmp_path):
    calls = []
    first = experiment(tmp_path, Produce(calls), Consume())
    assert first.run(stop_at=1)
    path = tmp_path / 'results' / 'array.npy'
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    resumed = experiment(tmp_path, Produce(calls), Consume(), auto_load_data=False)
    assert resumed.run(resume=True)
    assert calls == ['produce']
    assert resumed.data.derived == 13
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert resumed._get_checkpoint() == 2


def test_cleanup_and_persistence_precede_success(tmp_path, monkeypatch):
    cleaned = []
    phase = Produce()
    monkeypatch.setattr(phase, 'cleanup', lambda: cleaned.append(True))
    original_save = DataContext.save

    def checked_save(self, *args, **kwargs):
        assert cleaned == [True]
        assert not (tmp_path / 'experiment_log.json').exists()
        assert kwargs['strict'] is True
        return original_save(self, *args, **kwargs)

    monkeypatch.setattr(DataContext, 'save', checked_save)
    exp = experiment(tmp_path, phase)
    assert exp.run()
    log = json.loads((tmp_path / 'experiment_log.json').read_text())
    assert log['phase_log'][0]['success'] is True
    assert len(log['phase_log'][0]['persisted_outputs']) == 2


def test_serialization_failure_never_creates_success_checkpoint(tmp_path):
    class Unserializable(AnalysisPhaseV3):
        provides = ['callback']

        def run(self, experiment):
            return {'callback': lambda: None}

    exp = experiment(tmp_path, Unserializable())
    assert not exp.run()
    assert not any(entry['success'] for entry in exp.results)
    assert 'Failed to persist data key' in exp.results[-1]['error']
    assert exp._get_checkpoint() == 0


def test_cleanup_failure_never_creates_success_checkpoint(tmp_path, monkeypatch):
    phase = Produce()

    def bad_cleanup():
        raise OSError('controlled close failure')

    monkeypatch.setattr(phase, 'cleanup', bad_cleanup)
    exp = experiment(tmp_path, phase)
    assert not exp.run()
    assert not any(entry['success'] for entry in exp.results)
    assert exp._get_checkpoint() == 0


@pytest.mark.parametrize('damage', ['missing_binary', 'corrupt_binary', 'missing_json_key', 'corrupt_json'])
def test_resume_rejects_missing_or_changed_outputs(tmp_path, damage):
    first = experiment(tmp_path, Produce())
    assert first.run()
    array_path = tmp_path / 'results' / 'array.npy'
    json_path = tmp_path / 'results' / 'results.json'
    if damage == 'missing_binary':
        array_path.unlink()
    elif damage == 'corrupt_binary':
        array_path.write_bytes(b'truncated')
    elif damage == 'missing_json_key':
        json_path.write_text('{}')
    else:
        json_path.write_text('{"value":')
    calls = []
    resumed = experiment(tmp_path, Produce(calls), auto_load_data=False)
    assert not resumed.run(resume=True)
    assert calls == []


def test_partial_group_never_skips_failed_child_on_resume(tmp_path):
    calls = []
    first = experiment(tmp_path, PhaseGroup([Produce(calls), Fail()]))
    assert not first.run()
    child_success = [row for row in first.results if row['success']]
    assert len(child_success) == 1
    assert child_success[0]['checkpoint_boundary'] is False
    resumed = experiment(tmp_path, PhaseGroup([Produce(calls), Fail()]), auto_load_data=False)
    with pytest.raises(ValueError, match='interrupted PhaseGroup'):
        resumed._get_checkpoint()
    assert not resumed.run(resume=True)
    assert calls == ['produce']


def test_completed_group_has_one_boundary_and_resumes_next_phase(tmp_path):
    calls = []
    first = experiment(tmp_path, PhaseGroup([Produce(calls)]), Consume())
    assert first.run(stop_at=1)
    boundaries = [row for row in first.results if row['success'] and row['checkpoint_boundary']]
    assert len(boundaries) == 1
    assert boundaries[0]['phase_class'] == 'PhaseGroup'
    resumed = experiment(tmp_path, PhaseGroup([Produce(calls)]), Consume(), auto_load_data=False)
    assert resumed.run(resume=True)
    assert calls == ['produce']
    assert resumed.data.derived == 13


@pytest.mark.parametrize('content', ['{"phase_log": [', '{"phase_log": []}'])
def test_corrupt_and_legacy_logs_are_preserved_and_not_restarted(tmp_path, content):
    path = tmp_path / 'experiment_log.json'
    path.write_text(content)
    calls = []
    exp = experiment(tmp_path, Produce(calls), auto_load_data=False)
    assert not exp.run(resume=True)
    assert calls == []
    assert path.read_text() == content


def test_latest_failure_invalidates_earlier_success(tmp_path):
    exp = experiment(tmp_path, Produce())
    assert exp.run()
    exp._handle_phase_failure(exp.phases[0], 0, RuntimeError('later attempt failed'))
    assert exp._get_checkpoint() == 0


def test_atomic_log_replace_failure_preserves_previous_log(tmp_path, monkeypatch):
    exp = experiment(tmp_path, Produce())
    assert exp.run()
    path = tmp_path / 'experiment_log.json'
    before = path.read_bytes()

    def bad_replace(source, destination):
        raise OSError('controlled replace failure')

    monkeypatch.setattr('braindance.core.phases_v3.experiment_v3.os.replace', bad_replace)
    with pytest.raises(OSError, match='controlled replace failure'):
        exp._save_experiment_log()
    assert path.read_bytes() == before
    assert not list(tmp_path.glob('.experiment_log.*.tmp'))


def test_phase_identity_change_is_rejected(tmp_path):
    assert experiment(tmp_path, Produce()).run()
    renamed = Produce()
    renamed.name = 'different_phase'
    resumed = experiment(tmp_path, renamed, auto_load_data=False)
    assert not resumed.run(resume=True)


def test_default_repeated_key_policy_is_unchanged():
    context = DataContext()
    context.update({'recording_file': 'first.raw.h5'})
    context.update({'recording_file': 'second.raw.h5'})
    assert context.recording_file == 'first.raw.h5'


def test_interrupted_single_phase_reruns_from_beginning(tmp_path):
    class InterruptOnce(AnalysisPhaseV3):
        requires = ['value']
        provides = ['finished']

        def __init__(self, interrupt):
            super().__init__()
            self.interrupt = interrupt
            self.cleaned = False

        def run(self, experiment):
            if self.interrupt:
                raise KeyboardInterrupt()
            return {'finished': experiment.data.value + 1}

        def cleanup(self):
            self.cleaned = True

    interrupted = InterruptOnce(True)
    first = experiment(tmp_path, Produce(), interrupted)
    with pytest.raises(KeyboardInterrupt):
        first.run()
    assert interrupted.cleaned
    assert first._get_checkpoint() == 1
    resumed = experiment(tmp_path, Produce(), InterruptOnce(False))
    assert resumed.run(resume=True)
    assert resumed.data.finished == 8


def test_resume_does_not_restore_uncommitted_partial_outputs(tmp_path):
    class Finish(AnalysisPhaseV3):
        provides = ['partial']

        def run(self, experiment):
            assert 'partial' not in experiment.data
            return {'partial': 123}

    first = experiment(tmp_path, Produce(), Finish())
    assert first.run(stop_at=1)
    # Simulate a crash after output save and before checkpoint publication.
    first.data.update({'partial': 99})
    first._persist_phase_outputs({'partial': 99})
    resumed = experiment(tmp_path, Produce(), Finish())
    assert resumed.data.partial == 99  # Existing constructor behavior.
    assert resumed.run(resume=True)
    assert resumed.data.partial == 123


def test_completed_group_with_changed_children_is_rejected(tmp_path):
    assert experiment(tmp_path, PhaseGroup([Produce()])).run()
    resumed = experiment(tmp_path, PhaseGroup([Produce(), Fail()]))
    assert not resumed.run(resume=True)


def test_resume_before_first_checkpoint_drops_uncommitted_auto_loaded_data(tmp_path):
    first = experiment(tmp_path, Produce())
    first.data.update({'value': 99, 'array': np.ones(4)})
    first._persist_phase_outputs({'value': 99, 'array': np.ones(4)})
    assert not (tmp_path / 'experiment_log.json').exists()
    resumed = experiment(tmp_path, Produce())
    assert resumed.data.value == 99
    assert resumed.run(resume=True)
    assert resumed.data.value == 7
    np.testing.assert_array_equal(resumed.data.array, np.arange(4))


@pytest.mark.parametrize('complete', [False, True])
@pytest.mark.parametrize('index_change', ['json_to_numpy', 'json_to_missing_numpy',
                                         'binary_to_json', 'corrupt_index'])
def test_restore_uses_manifest_not_mutable_index(tmp_path, complete, index_change):
    phases = [Produce()] if complete else [Produce(), Consume()]
    first = experiment(tmp_path, *phases)
    assert first.run(stop_at=1)
    index_path = tmp_path / 'results' / 'data_index.json'
    index = json.loads(index_path.read_text())
    if index_change in ('json_to_numpy', 'json_to_missing_numpy'):
        index['saved_files']['value'] = 'numpy'
        if index_change == 'json_to_numpy':
            np.save(tmp_path / 'results' / 'value.npy', np.asarray([99]))
    elif index_change == 'binary_to_json':
        index['saved_files']['array'] = 'json'
        json_path = tmp_path / 'results' / 'results.json'
        content = json.loads(json_path.read_text())
        content['array'] = 999
        json_path.write_text(json.dumps(content))
    index_path.write_text('{' if index_change == 'corrupt_index' else json.dumps(index))
    phases = [Produce()] if complete else [Produce(), Consume()]
    resumed = experiment(tmp_path, *phases, auto_load_data=False)
    assert resumed.run(resume=True)
    assert resumed.data.value == 7
    np.testing.assert_array_equal(resumed.data.array, np.arange(4))
    if not complete:
        assert resumed.data.derived == 13


@pytest.mark.parametrize('complete', [False, True])
def test_restore_failure_after_checkpoint_selection_prevents_success(tmp_path, monkeypatch, complete):
    phases = [Produce()] if complete else [Produce(), Consume()]
    assert experiment(tmp_path, *phases).run(stop_at=1)
    resumed = experiment(tmp_path, *phases, auto_load_data=False)
    choose_checkpoint = resumed._get_checkpoint

    def remove_after_verification():
        checkpoint = choose_checkpoint()
        (tmp_path / 'results' / 'array.npy').unlink()
        return checkpoint

    monkeypatch.setattr(resumed, '_get_checkpoint', remove_after_verification)
    assert not resumed.run(resume=True)
    assert 'derived' not in resumed.data


@pytest.mark.parametrize('change', ['missing', 'empty', 'omit_one'])
def test_success_requires_complete_output_descriptor_coverage(tmp_path, change):
    assert experiment(tmp_path, Produce()).run()
    log_path = tmp_path / 'experiment_log.json'
    log = json.loads(log_path.read_text())
    entry = log['phase_log'][0]
    if change == 'missing':
        del entry['persisted_outputs']
    elif change == 'empty':
        entry['persisted_outputs'] = []
    else:
        entry['persisted_outputs'].pop()
    log_path.write_text(json.dumps(log))
    assert not experiment(tmp_path, Produce(), auto_load_data=False).run(resume=True)


def test_direct_restore_supports_numpy_dict_and_pickle_formats(tmp_path):
    import pandas as pd

    class MixedOutputs(AnalysisPhaseV3):
        provides = ['arrays', 'frame', 'set_value']

        def run(self, experiment):
            return {'arrays': {'trace': np.arange(3)},
                    'frame': pd.DataFrame({'sample': [1, 2]}),
                    'set_value': {1, 2}}

    assert experiment(tmp_path, MixedOutputs()).run()
    resumed = experiment(tmp_path, MixedOutputs(), auto_load_data=False)
    assert resumed.run(resume=True)
    np.testing.assert_array_equal(resumed.data.arrays['trace'], np.arange(3))
    pd.testing.assert_frame_equal(resumed.data.frame, pd.DataFrame({'sample': [1, 2]}))
    assert resumed.data.set_value == {1, 2}


def test_direct_restore_uses_latest_committed_value_for_replaced_key(tmp_path):
    class Value(AnalysisPhaseV3):
        provides = ['value']

        def __init__(self, value):
            super().__init__()
            self.value = value

        def run(self, experiment):
            return {'value': self.value}

    first = experiment(tmp_path, Value(1), Value(2))
    first.data.set_overwrite_policy(True)  # Existing explicit opt-in behavior.
    assert first.run()
    resumed = experiment(tmp_path, Value(1), Value(2), auto_load_data=False)
    assert resumed.run(resume=True)
    assert resumed.data.value == 2
    assert resumed.data.get_overwrite_policy() is False
