import importlib
import sys
import types

import numpy as np
import pytest


MODULE = "braindance.experiments.real_time_sorting"


def test_import_has_no_optional_runtime_side_effects(monkeypatch):
    optional_modules = (
        "spikeinterface.extractors",
        "braindance.core.maxwell_env",
        "braindance.core.spikedetector.model2",
        "braindance.core.spikesorter.rt_sort",
        "braindance.utils.rt_linear_art_removal",
    )
    for name in (MODULE, *optional_modules):
        monkeypatch.delitem(sys.modules, name, raising=False)

    importlib.import_module(MODULE)

    assert all(name not in sys.modules for name in optional_modules)


def _install_runtime_fakes(monkeypatch, *, fail_during_cleaning=False):
    calls = {}

    class Recording:
        def __init__(self, path):
            calls["recording_path"] = path

        def get_num_channels(self):
            return 3

    class Sorter:
        def reset(self):
            calls["sorter_reset"] = True

        def running_sort(self, observations):
            calls.setdefault("sorter_observations", []).append(observations.copy())
            return [(4, 1.25)]

    class ArtifactRemover:
        def __init__(self, **kwargs):
            calls["artifact_init"] = kwargs

        def warmup(self):
            calls["artifact_warmup"] = True

        def fit_step(self, observations):
            calls.setdefault("artifact_observations", []).append(observations.copy())
            if fail_during_cleaning:
                raise RuntimeError("cleaning failed")
            return observations + 10, None, None

    class Environment:
        def __init__(self, **kwargs):
            calls["env_params"] = kwargs
            calls["env"] = self
            self.stim_dt = 0.4
            self.closed = False
            self.steps = 0

        def step(self, *, action, buffer_size):
            calls.setdefault("steps", []).append((action, buffer_size))
            self.steps += 1
            if self.steps == 1:
                self.stim_dt = 0.1
                return np.array([[1, 2, 3], [4, 5, 6]]), False
            return np.empty((0, 3)), True

        def close(self):
            self.closed = True

    def detect_sequences(recording, work_dir, model, **kwargs):
        calls["detect"] = (recording, work_dir, model, kwargs)
        calls["work_dir_existed"] = work_dir.is_dir()
        return Sorter()

    modules = {
        "spikeinterface.extractors": types.SimpleNamespace(
            MaxwellRecordingExtractor=Recording
        ),
        "braindance.core.maxwell_env": types.SimpleNamespace(MaxwellEnv=Environment),
        "braindance.core.params": types.SimpleNamespace(maxwell_params={"existing": 1}),
        "braindance.core.spikedetector.model2": types.SimpleNamespace(
            ModelSpikeSorter=types.SimpleNamespace(load_mea=lambda: "model")
        ),
        "braindance.core.spikesorter.rt_sort": types.SimpleNamespace(
            detect_sequences=detect_sequences
        ),
        "braindance.utils.rt_linear_art_removal": types.SimpleNamespace(
            LinearArtifactRemoval=ArtifactRemover
        ),
    }
    for name, fake in modules.items():
        monkeypatch.setitem(sys.modules, name, fake)
    return calls


def test_main_resolves_config_paths_and_runs_isolated_pipeline(tmp_path, monkeypatch):
    module = importlib.import_module(MODULE)
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    recording_path = data_dir / "recording.raw.h5"
    recording_path.parent.mkdir()
    recording_path.touch()
    monkeypatch.setattr(module, "get_data_dir", lambda: data_dir)
    monkeypatch.setattr(module, "get_output_dir", lambda: output_dir)
    calls = _install_runtime_fakes(monkeypatch)

    module.main(
        "recording.raw.h5",
        max_time_sec=12,
        stim_electrodes=[7, 8],
        recording_window_ms=(100, 200),
        buffer_size=32,
    )

    assert calls["recording_path"] == str(recording_path)
    _, work_dir, model, detect_kwargs = calls["detect"]
    assert calls["work_dir_existed"]
    assert work_dir.parent == output_dir / "real_time_sorting"
    assert not work_dir.exists()
    assert model == "model"
    assert detect_kwargs == {
        "return_spikes": False,
        "delete_inter": True,
        "recording_window_ms": (100, 200),
    }
    assert calls["env_params"] == {
        "existing": 1,
        "name": "test",
        "stim_electrodes": [7, 8],
        "max_time_sec": 12,
        "config": None,
        "observation_type": "raw",
        "dummy": str(recording_path),
    }
    assert calls["artifact_init"] == {"n_channels": 3, "batch_size": 3}
    assert calls["artifact_warmup"] and calls["sorter_reset"]
    assert calls["steps"] == [(([0], 150, 100), 32), (None, 32)]
    expected_artifact_input = np.array([[1, 4], [2, 5], [3, 6]])
    np.testing.assert_array_equal(calls["artifact_observations"][0], expected_artifact_input)
    np.testing.assert_array_equal(
        calls["sorter_observations"][0], expected_artifact_input.T + 10
    )
    assert calls["env"].closed


def test_main_closes_environment_when_processing_fails(tmp_path, monkeypatch):
    module = importlib.import_module(MODULE)
    recording_path = tmp_path / "recording.raw.h5"
    recording_path.touch()
    monkeypatch.setattr(module, "get_output_dir", lambda: tmp_path / "outputs")
    calls = _install_runtime_fakes(monkeypatch, fail_during_cleaning=True)

    with pytest.raises(RuntimeError, match="cleaning failed"):
        module.main(recording_path)

    assert calls["env"].closed
    assert not calls["detect"][1].exists()


def test_main_rejects_missing_recording_before_loading_optional_dependencies(
    tmp_path, monkeypatch
):
    module = importlib.import_module(MODULE)
    monkeypatch.setattr(module, "get_data_dir", lambda: tmp_path)

    with pytest.raises(FileNotFoundError, match="missing.raw.h5"):
        module.main("missing.raw.h5")
