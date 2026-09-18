"""Canonical V3 contracts coexist with historical phase declarations."""
from types import SimpleNamespace

import pytest

from braindance.core.phases_v3.data_context import DataContext
from braindance.core.phases_v3.phase_base_v3 import (
    AnalysisPhaseV3, PhaseValidator, QuickPhaseV3, ValidationError, phase, quick_phase,
)


class LegacyProducer(AnalysisPhaseV3):
    provides = ['recording_file']

    def run(self, experiment):
        return {'recording_file': 'recording.raw.h5'}


class CanonicalConsumer(AnalysisPhaseV3):
    inputs = ['recording_file']
    outputs = ['sorting_result']

    def run(self, experiment):
        return {'sorting_result': self.recording_file}


def test_mixed_pipeline_and_configuration():
    producer, consumer = LegacyProducer(), CanonicalConsumer()
    assert PhaseValidator.validate_pipeline([producer, consumer])
    with pytest.raises(ValidationError):
        PhaseValidator.validate_pipeline([consumer, producer])
    data = DataContext()
    data.update(producer.run(None))
    consumer.configure_from_experiment(SimpleNamespace(data=data))
    assert consumer.run(None) == {'sorting_result': 'recording.raw.h5'}
    assert consumer.info()['inputs'] == ['recording_file']
    assert consumer.info()['outputs'] == ['sorting_result']


def test_aliases_follow_class_inheritance_and_instance_changes():
    class Child(CanonicalConsumer):
        requires = ['other_file']

    assert Child.inputs is Child.requires
    first, second = Child(), Child()
    first.requires.append('mapping')
    assert first.inputs == ['other_file', 'mapping']
    assert second.inputs == Child.inputs == ['other_file']
    first.outputs = ['result']
    assert first.provides is first.outputs
    first.requires = ['new_file']
    assert first.inputs == ['new_file']
    Child.provides = ['new_result']
    assert Child.outputs == ['new_result']
    Child.inputs = ['new_input']
    assert Child.requires == ['new_input']


def test_conflicting_declarations_rejected():
    with pytest.raises(ValueError, match='conflicting inputs'):
        class Conflicting(CanonicalConsumer):
            inputs = ['one']
            requires = ['two']
    with pytest.raises(ValueError, match='Conflicting inputs'):
        QuickPhaseV3(lambda exp: {}, inputs=['one'], requires=['two'])


@pytest.mark.parametrize('factory', [QuickPhaseV3, quick_phase])
def test_function_contracts_support_both_names_and_explicit_empty(factory):
    func = lambda exp: {'value': 1}
    canonical = factory(func, inputs=[], outputs=['value'])
    legacy = factory(func, [], ['value'], 'legacy')
    assert canonical.inputs == legacy.requires == []
    assert canonical.outputs == legacy.provides == ['value']
    empty = factory(func, inputs=[], outputs=[])
    assert empty.inputs == empty.outputs == []
    decorated = phase('decorated', inputs=[], outputs=['value'])(func)
    assert decorated.outputs == ['value']
    assert decorated.run(None) == {'value': 1}
