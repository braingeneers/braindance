"""Regression coverage for SpikeLab and historical trusted recording pickles."""

import io
import pickle
import sys
import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("spikelab")

from spikelab import SpikeData
from braindance.spike_data import as_spike_data, load_spike_pickle
from braindance.utils.data_manager.utils.data_loading.recording import Recording
from braindance.utils.data_manager.utils.data_loading.data_context import DataContext


@pytest.fixture
def legacy_payload(monkeypatch, request):
    """Serialize an old class, then remove it before any compatibility load."""
    package = types.ModuleType("spikedata")
    module = types.ModuleType("spikedata.spikedata")
    legacy_class = type("SpikeData", (), {"__module__": module.__name__})
    module.SpikeData = legacy_class
    package.spikedata = module
    with monkeypatch.context() as temporary:
        temporary.setitem(sys.modules, "spikedata", package)
        temporary.setitem(sys.modules, module.__name__, module)
        legacy = legacy_class()
        legacy.__dict__.update(
            train=[np.array([1., 8.]), np.array([3., 9.])],
            N=2, length=12., metadata={"sample": "old recording"},
            neuron_attributes=[{"channel": 4}, {"channel": 8}],
            raw_data=np.arange(8.).reshape(2, 4), raw_time=np.arange(4.),
            custom_note={"retained": True},
        )
        contents = legacy if getattr(request, "param", None) == "direct" else {
            "spikes": legacy, "nested": [legacy]
        }
        payload = pickle.dumps(contents)
    # Block installed copies too: the test must not rely on the host environment.
    monkeypatch.setitem(sys.modules, "spikedata", None)
    monkeypatch.setitem(sys.modules, "spikedata.spikedata", None)
    return payload


def assert_legacy_fields(spikes):
    assert type(spikes) is SpikeData
    assert spikes.N == 2 and spikes.length == 12. and spikes.start_time == 0.
    np.testing.assert_array_equal(spikes.train[0], [1., 8.])
    np.testing.assert_array_equal(spikes.train[1], [3., 9.])
    np.testing.assert_array_equal(spikes.raw_data, np.arange(8.).reshape(2, 4))
    np.testing.assert_array_equal(spikes.raw_time, np.arange(4.))
    assert spikes.metadata == {"sample": "old recording"}
    assert spikes.neuron_attributes == [{"channel": 4}, {"channel": 8}]
    assert spikes.custom_note == {"retained": True}


def test_legacy_nested_pickle_without_old_package(legacy_payload):
    with pytest.raises(ModuleNotFoundError):
        pickle.loads(legacy_payload)
    restored = load_spike_pickle(io.BytesIO(legacy_payload))
    assert_legacy_fields(restored["spikes"])
    assert restored["nested"][0] is restored["spikes"]
    assert sys.modules.get("spikedata.spikedata") is None


def test_native_roundtrip_preserves_nonzero_origin():
    original = SpikeData([[-5., 2.], [0.]], start_time=-10., length=20.,
                         metadata={"event": 3}, neuron_attributes=[{}, {}])
    restored = load_spike_pickle(io.BytesIO(pickle.dumps(original)))
    assert type(restored) is SpikeData
    assert restored.start_time == -10. and restored.length == 20.
    assert restored.metadata == original.metadata
    np.testing.assert_array_equal(restored.train[0], original.train[0])
    assert as_spike_data(restored) is restored


@pytest.mark.parametrize("train_key", ["train", "spike_trains", None])
def test_recording_local_supported_formats(tmp_path, train_key):
    trains = [[1., 7.], [2., 8.]]
    data = trains if train_key is None else {
        train_key: trains, "N": 2, "length": 20., "start_time": -2.,
        "metadata": {"source": "dictionary"},
        "neuron_attributes": [{"channel": 0}, {"channel": 1}],
        "raw_data": np.arange(6.).reshape(2, 3), "raw_time": np.arange(3.),
    }
    path = tmp_path / "spikes.pkl"
    path.write_bytes(pickle.dumps(data))
    rec = Recording(pd.Series({"proj": "p", "chip": "c", "experiment": "e"}),
                    base_path=tmp_path, auto_upload=False)
    rec._paths_resolved = True
    rec._resolved_paths["spikes"] = path
    spikes = rec.spikes
    assert type(spikes) is SpikeData
    np.testing.assert_array_equal(spikes.train[0], trains[0])
    if train_key:
        assert spikes.start_time == -2. and spikes.length == 20.
        assert spikes.metadata == data["metadata"]
        assert spikes.neuron_attributes == data["neuron_attributes"]
        np.testing.assert_array_equal(spikes.raw_data, data["raw_data"])
        np.testing.assert_array_equal(spikes.raw_time, data["raw_time"])


def test_data_context_loads_nested_legacy_cache(tmp_path, legacy_payload):
    (tmp_path / "historical.pkl").write_bytes(legacy_payload)
    cached = DataContext(path=tmp_path)["historical"]
    assert_legacy_fields(cached["spikes"])


@pytest.mark.parametrize("legacy_payload", ["direct"], indirect=True)
@pytest.mark.parametrize("remote", [False, True])
def test_recording_loads_legacy_spikes(tmp_path, monkeypatch, legacy_payload, remote):
    from braindance.utils.data_manager.utils.data_loading import s3_loader

    rec = Recording(pd.Series({"proj": "p", "chip": "c", "experiment": "e"}),
                    base_path=tmp_path, auto_upload=False)
    rec._paths_resolved = True
    if remote:
        monkeypatch.setattr(s3_loader.boto3, "client", lambda *args, **kwargs: object())
        monkeypatch.setattr(s3_loader.smart_open, "open",
                            lambda *args, **kwargs: io.BytesIO(legacy_payload))
        rec._s3_loader = s3_loader.S3Loader(cache_dir=tmp_path)
        rec._resolved_paths["spikes"] = "s3://test-bucket/spikes.pkl"
    else:
        path = tmp_path / "spikes.pkl"
        path.write_bytes(legacy_payload)
        rec._resolved_paths["spikes"] = path
    assert_legacy_fields(rec.spikes)


def test_s3_download_and_disk_cache_convert_legacy(tmp_path, monkeypatch, legacy_payload):
    from braindance.utils.data_manager.utils.data_loading import s3_loader

    monkeypatch.setattr(s3_loader.boto3, "client", lambda *args, **kwargs: object())
    calls = []

    def remote_open(path, *args, **kwargs):
        calls.append(path)
        return io.BytesIO(legacy_payload)

    monkeypatch.setattr(s3_loader.smart_open, "open", remote_open)
    loader = s3_loader.S3Loader(cache_dir=tmp_path)
    path = "s3://test-bucket/historical.pkl"
    assert_legacy_fields(loader.load_pickle(path)["spikes"])
    assert_legacy_fields(loader.load_pickle(path)["spikes"])
    assert calls == [path]


def test_sttc_accessor_accepts_spikelab_result():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from braindance.utils.data_manager.utils.plotting.accessor import PlotAccessor

    spikes = SpikeData([[1., 5., 9.], [2., 6., 10.]], length=12.)
    record = types.SimpleNamespace(spikes=spikes, identifier="synthetic")
    figure, axes = PlotAccessor(record).sttc_matrix(delt=1., show=False)
    try:
        plotted = np.asarray(axes.images[0].get_array())
        expected = spikes.spike_time_tilings(1.).matrix.copy()
        np.fill_diagonal(expected, np.nan)  # Plotting intentionally hides self-correlation.
        np.testing.assert_allclose(plotted, expected)
    finally:
        plt.close(figure)
