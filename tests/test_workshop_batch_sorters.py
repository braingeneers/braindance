"""Exercise upload sorter dispatch and portable exports without expensive sorting."""
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from braindance.examples.streaming_workshop import batch_sorters as batch


def test_rt_sort_dispatch(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(batch, "run_sorting", lambda *args, **kwargs: calls.append((args, kwargs)) or {"ok": True})
    assert batch.run_batch_sorting("baseline.h5", [tmp_path / "a.h5"], tmp_path,
                                  params={"device": "cpu"}) == {"ok": True}
    assert calls[0][0][0] == "baseline.h5"
    assert calls[0][1]["params"] == {"device": "cpu"}


@pytest.fixture
def engine(monkeypatch, tmp_path):
    monkeypatch.setattr(batch.importlib.util, "find_spec", lambda name: object())
    paths = [tmp_path / "a.h5", tmp_path / "b.nwb"]
    for path in paths:
        path.touch()
    reads, calls = [], []
    recording = types.SimpleNamespace(get_num_segments=lambda: 1,
                                      get_sampling_frequency=lambda: 20000.,
                                      get_num_samples=lambda: 20000)
    sorting = types.SimpleNamespace(get_unit_ids=lambda: ["unit-a", "unit-b", "unit-c"],
                                    get_unit_spike_train=lambda unit, **kw: {
                                        "unit-a": np.array([600, 200]), "unit-b": np.array([]),
                                        "unit-c": np.array([400])}[unit])
    def read(path):
        reads.append(path)
        return recording
    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return sorting
    extractors = types.SimpleNamespace(read_maxwell=read, read_nwb=read)
    sorters = types.SimpleNamespace(installed_sorters=lambda: ["simple"],
                                   get_default_sorter_params=lambda name: {"threshold": 5}, run_sorter=run)
    monkeypatch.setitem(sys.modules, "spikeinterface", types.SimpleNamespace(extractors=extractors))
    monkeypatch.setitem(sys.modules, "spikeinterface.sorters", sorters)
    monkeypatch.setattr(batch, "sorting_capabilities", lambda: {"available": False, "reason": "No model"})
    return paths, reads, calls, sorting


def test_choices_show_installed_sorters_and_rt_reason(engine):
    assert batch.sorter_choices() == [
        dict(id="rt-sort", label="RT-Sort", available=False, reason="No model"),
        dict(id="spikeinterface:simple", label="simple (SpikeInterface)", available=True, reason="")]


def test_independent_batch_export_preserves_empty_units(engine, tmp_path):
    paths, reads, calls, _ = engine
    updates = []
    result = batch.run_batch_sorting("unused-baseline.h5", [*paths, paths[0]], tmp_path / "results",
                                    sorter="spikeinterface:simple", progress=lambda *a: updates.append(a))
    assert reads == list(map(str, paths))
    assert len(calls) == 2
    assert result["baseline_used"] is False
    assert result["unit_identity"] == "independent_per_recording"
    assert len(result["recordings"]) == 2
    for item in result["recordings"]:
        with np.load(item["spikes_path"], allow_pickle=False) as data:
            np.testing.assert_array_equal(data["times_ms"], [10, 20, 30])
            np.testing.assert_array_equal(data["unit_ids"], [0, 2, 0])
            np.testing.assert_array_equal(data["all_unit_ids"], [0, 1, 2])
            np.testing.assert_array_equal(data["original_unit_ids"], ["unit-a", "unit-b", "unit-c"])
            assert data["duration_ms"] == 1000
    assert updates[-1][0] == 1
    assert json.loads((Path(result["output_dir"]) / "manifest.json").read_text()) == result
    again = batch.run_batch_sorting(None, paths, tmp_path / "results", sorter="spikeinterface:simple")
    assert again["output_dir"] != result["output_dir"]


def test_empty_batch_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        batch.run_batch_sorting(None, [], tmp_path)


def test_unknown_sorter_and_params_rejected(engine, tmp_path):
    paths, _, calls, _ = engine
    with pytest.raises(ValueError, match="Unknown or unavailable"):
        batch.run_batch_sorting(None, paths, tmp_path, sorter="not-a-sorter")
    with pytest.raises(ValueError, match="Unknown sorting parameters"):
        batch.run_batch_sorting(None, paths, tmp_path, sorter="spikeinterface:simple", params={"typo": 4})
    assert calls == []


def test_zero_detected_units_still_exports(engine, tmp_path):
    paths, _, _, sorting = engine
    sorting.get_unit_ids = lambda: []
    result = batch.run_batch_sorting(None, paths[:1], tmp_path, sorter="spikeinterface:simple")
    with np.load(result["recordings"][0]["spikes_path"], allow_pickle=False) as data:
        assert data["times_ms"].size == data["all_unit_ids"].size == 0


def test_out_of_bounds_spikes_rejected(engine, tmp_path):
    paths, _, _, sorting = engine
    sorting.get_unit_spike_train = lambda *a, **kw: np.array([20000])
    with pytest.raises(RuntimeError, match="invalid spike times"):
        batch.run_batch_sorting(None, paths, tmp_path, sorter="spikeinterface:simple")


def test_builtin_missing_dependencies_are_not_reported_ready(engine, monkeypatch):
    monkeypatch.setattr(batch.importlib.util, "find_spec", lambda name: None if name == 'hdbscan' else object())
    choice = batch.sorter_choices()[1]
    assert not choice['available']
    assert 'hdbscan' in choice['reason']
