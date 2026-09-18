"""RT-Sort adapter contracts without expensive neural inference."""
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from braindance.examples.streaming_workshop import analysis_sorting as sorting


@pytest.mark.parametrize("params", [
    {"unknown": 3}, {"device": "magic"}, {"num_processes": 1.5},
    {"baseline_duration_ms": 0}, {"min_activity_hz": float("nan")},
    {"loose_thresh": .8, "stringent_thresh": .2}, {"num_processes": True},
])
def test_invalid_parameters(params):
    with pytest.raises(ValueError):
        sorting.validate_params(params)


class Recording:
    def __init__(self, ids=(0, 1), frequency=20000):
        self.ids = ids
        self.frequency = frequency

    def get_channel_ids(self):
        return self.ids

    def get_channel_locations(self):
        return np.array([[0, 0], [20, 20]])

    def get_sampling_frequency(self):
        return self.frequency

    def get_total_duration(self):
        return 1.0


@pytest.fixture
def engine(monkeypatch, tmp_path):
    baseline = tmp_path / "baseline.h5"
    target = tmp_path / "target.h5"
    baseline.touch()
    target.touch()
    recordings = {baseline: Recording(), target: Recording()}
    calls = []

    class Sorter:
        num_seqs = 3

        def save(self, path):
            path.write_bytes(b"sorter")

        def sort_offline(self, recording, **kwargs):
            calls.append(("sort", recording, kwargs))
            return [np.array([30., 10.]), np.array([]), np.array([20.])]

    def detect(*args, **kwargs):
        calls.append(("detect", args, kwargs))
        return Sorter()

    monkeypatch.setitem(sys.modules, "braindance.core.spikesorter.rt_sort",
                        types.SimpleNamespace(detect_sequences=detect, load_recording=recordings.__getitem__,
                                              save_traces=lambda rec, path, **kw: path / "scaled_traces.npy"))
    monkeypatch.setattr(sorting, "sorting_capabilities", lambda: {"available": True, "model_path": "model"})
    return baseline, target, recordings, calls


def test_sorts_targets_with_baseline_and_portable_output(engine, tmp_path):
    baseline, target, _, calls = engine
    updates = []
    result = sorting.run_sorting(baseline, [target, target], tmp_path / "results",
                                {"device": "cpu"}, lambda *args: updates.append(args))
    assert len(result["recordings"]) == 1
    assert calls[0][0] == "detect"
    assert calls[0][2]["recording_window_ms"] == (0, 1000)
    assert calls[1][2]["reset"] is True
    with np.load(result["recordings"][0]["spikes_path"], allow_pickle=False) as data:
        np.testing.assert_array_equal(data["times_ms"], [10, 20, 30])
        np.testing.assert_array_equal(data["unit_ids"], [0, 2, 0])
        np.testing.assert_array_equal(data["all_unit_ids"], [0, 1, 2])
        assert data["duration_ms"] == 1000
    assert updates[-1][0] == 1
    assert [item[0] for item in updates] == sorted(item[0] for item in updates)
    manifest = json.loads((Path(result["output_dir"]) / "manifest.json").read_text())
    assert manifest == result


@pytest.mark.parametrize("change", ["routing", "frequency"])
def test_rejects_incompatible_baseline_before_detection(engine, tmp_path, change):
    baseline, target, recordings, calls = engine
    recordings[target] = Recording(ids=(1, 0)) if change == "routing" else Recording(frequency=30000)
    with pytest.raises(ValueError, match="differs from baseline"):
        sorting.run_sorting(baseline, [target], tmp_path, {"device": "cpu"})
    assert calls == []


def test_rejects_empty_target_selection(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        sorting.run_sorting(tmp_path / "baseline.h5", [], tmp_path)


def test_rejects_baseline_window_outside_file(engine, tmp_path):
    baseline, target, _, calls = engine
    with pytest.raises(ValueError, match="starts after"):
        sorting.run_sorting(baseline, [target], tmp_path,
                            {"device": "cpu", "baseline_start_ms": 2000})
    assert calls == []


def test_no_units_gives_actionable_error(engine, tmp_path, monkeypatch):
    baseline, target, _, _ = engine
    module = sys.modules['braindance.core.spikesorter.rt_sort']
    monkeypatch.setattr(module, 'detect_sequences', lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="found no units"):
        sorting.run_sorting(baseline, [target], tmp_path, {"device": "cpu"})
    assert not list(tmp_path.rglob('*_spikes.npz'))


def test_repeated_runs_preserve_previous_results(engine, tmp_path):
    baseline, target, _, _ = engine
    first = sorting.run_sorting(baseline, [target], tmp_path, {"device": "cpu"})
    second = sorting.run_sorting(baseline, [target], tmp_path, {"device": "cpu"})
    assert first['output_dir'] != second['output_dir']
    assert Path(first['recordings'][0]['spikes_path']).is_file()
