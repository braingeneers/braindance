"""Value-level coverage for the catalog-based spatial latency example."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from spikelab import SpikeData

from braindance.utils.data_manager.tutorials import TODO_04_spatial_analysis as tutorial


@pytest.fixture
def recording(monkeypatch, tmp_path):
    times = np.arange(1, 31) * 1000.0
    plotted = []
    saved = []
    cleared = []
    rec = SimpleNamespace(
        identifier="fixture/chip/recording",
        mapping=object(),
        spike_locations=[(0, 0), (20, 0)],
        spikes=SpikeData([times + 11, []], length=32000),
        stim_log=pd.DataFrame({"time": times / 1000, "stim_electrodes": ["[42]"] * len(times)}),
        results={},
        pl=SimpleNamespace(spatial_latency=lambda **kwargs: plotted.append(kwargs)),
        save_results=lambda **kwargs: saved.append(kwargs),
        clear_cache=lambda: cleared.append(True),
    )
    selected = []
    catalog = SimpleNamespace(filter=lambda **kwargs: selected.append(kwargs) or [rec])
    monkeypatch.setattr(tutorial, "load_catalog", lambda: catalog)
    monkeypatch.setattr(tutorial, "get_output_dir", lambda: tmp_path)
    return rec, plotted, saved, cleared, selected


def test_measured_peak_values_routing_cache_and_output(recording, tmp_path):
    rec, plotted, saved, cleared, selected = recording
    results = tutorial.main("fixture", "chip", "recording", show=False)
    assert selected == [{"proj": "fixture", "chip": "chip", "experiment": "recording"}]
    assert set(results) == {42}
    assert set(results[42]) == {0}  # Silent neuron must not appear as a response.
    response = results[42][0]
    assert response["peak_latency"] == 11.0
    assert response["response_rate"] == 10.0  # One spike per 100 ms window.
    assert response["baseline_rate"] == 0.0
    assert response["n_trials"] == 30
    assert plotted[0]["electrode_id"] == 42
    assert plotted[0]["latency_results"] is results[42]
    assert plotted[0]["show"] is False
    assert plotted[0]["save_path"] == str(tmp_path / "spatial_latency_tutorial/fixture/chip/recording")
    assert len(saved) == len(cleared) == 1

    # Saved measured results are reused; recompute explicitly refreshes them.
    rec.spikes = SpikeData([np.arange(1, 31) * 1000.0 + 31, []], length=32000)
    assert tutorial.main("fixture", "chip", "recording", show=False)[42][0]["peak_latency"] == 11.0
    assert len(saved) == 1
    assert tutorial.main("fixture", "chip", "recording", recompute=True, show=False)[42][0]["peak_latency"] == 31.0
    assert len(saved) == 2
    assert len(cleared) == 3


def test_missing_spatial_data_exits_before_analysis_and_clears(recording):
    rec, plotted, saved, cleared, _ = recording
    rec.mapping = None
    assert tutorial.main(show=False) == {}
    assert plotted == saved == []
    assert cleared == [True]


def test_plot_failure_still_clears_recording(recording):
    rec, _, _, cleared, _ = recording

    def fail(**kwargs):
        raise RuntimeError("plot failed")

    rec.pl.spatial_latency = fail
    with pytest.raises(RuntimeError, match="plot failed"):
        tutorial.main(show=False)
    assert cleared == [True]
