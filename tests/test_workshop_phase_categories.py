"""Builder categories describe phase execution, independently of data contracts."""
import importlib

from braindance.examples.streaming_workshop.experiment_spec import phase_catalog


def test_native_categories_follow_inheritance_without_importing(tmp_path, monkeypatch):
    module = importlib.import_module('braindance.examples.streaming_workshop.native_catalog')
    phase_dir = tmp_path / 'core' / 'phases_v3'
    phase_dir.mkdir(parents=True)
    (phase_dir / 'phases_fixture.py').write_text('''
raise RuntimeError("Catalog discovery must never import native modules")
class Processing(AnalysisPhaseV3):
    requires = ['recording_file']
    provides = ['next_parameters']
class Derived(Processing):
    pass
class AnalysisNamedAcquisition(PhaseV3):
    provides = ['recording_file']
class Mixed(AnalysisNamedAcquisition, Processing):
    pass
''')
    monkeypatch.setattr(module, '__file__', str(tmp_path / 'examples' / 'streaming_workshop' / 'native_catalog.py'))
    module._catalog.cache_clear()
    try:
        catalog = module.native_catalog(include_legacy=True)
        for name in ('Processing', 'Derived', 'Mixed'):
            assert catalog[f'native:phases_fixture.{name}']['category'] == 'analysis'
        assert catalog['native:phases_fixture.AnalysisNamedAcquisition']['category'] == 'experiment'
        derived = catalog['native:phases_fixture.Derived']
        assert derived['inputs'] == ['recording_file']
        assert derived['outputs'] == ['next_parameters']
    finally:
        module._catalog.cache_clear()


def test_installed_categories_and_streaming_adapters():
    module = importlib.import_module('braindance.examples.streaming_workshop.native_catalog')
    native = module.native_catalog(include_legacy=True)
    assert native['native:phases_analysis_3.ActivityPhaseV3']['category'] == 'analysis'
    assert native['native:phases3_selection.SelectionPhaseV3']['category'] == 'analysis'
    assert native['native:phases3.RecordPhaseV3']['category'] == 'experiment'
    assert native['native:phases3_loop.CartPolePhasWithViz']['category'] == 'experiment'
    assert all(entry['category'] in ('analysis', 'experiment') for entry in native.values())
    assert {entry['category'] for kind, entry in phase_catalog().items() if kind != 'custom_analysis'} == {'experiment'}
    assert phase_catalog()['custom_analysis']['category'] == 'analysis'
