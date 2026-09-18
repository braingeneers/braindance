"""CPU compatibility checks for the optional SpikeInterface sorting integration."""

from types import SimpleNamespace

import numpy as np
import pytest


def test_sorter_recording_apis(tmp_path):
    si = pytest.importorskip("spikeinterface.core")
    for dependency in ("torch", "diptest", "pynvml", "natsort"):
        pytest.importorskip(dependency)

    from braindance.core.spikesorter import kilosort2, rt_sort, sort
    from braindance.core.spikedetector import train

    # These were previously imported from extractors, which stopped exporting
    # BaseRecording in SpikeInterface 0.104.
    assert kilosort2.BaseRecording is si.BaseRecording
    assert sort.BaseRecording is si.BaseRecording
    assert train.BaseRecording is si.BaseRecording

    raw = np.arange(2000, dtype=np.int16).reshape(1000, 2)
    recording = si.NumpyRecording(raw, sampling_frequency=20000)
    recording.set_channel_gains([2.0, 3.0])
    recording.set_channel_offsets([1.0, -1.0])
    output = tmp_path / "traces.npy"
    np.save(output, np.zeros((2, 10), dtype=np.float32))
    for channel in range(2):
        rt_sort._save_traces_si(
            (SimpleNamespace(recording=recording), 10, 20, channel, output, np.float32)
        )
    np.testing.assert_allclose(
        np.load(output), (raw[10:20] * [2.0, 3.0] + [1.0, -1.0]).T
    )

    chunks = kilosort2.Curation.get_random_data_chunks(
        recording, return_scaled=True, num_chunks=2, chunk_size=10, seed=0
    )
    starts = np.random.RandomState(0).randint(0, 990, size=2)
    expected = np.concatenate([raw[start:start + 10] for start in starts])
    np.testing.assert_allclose(chunks, expected * [2.0, 3.0] + [1.0, -1.0])

    sorting = rt_sort.NumpySorting.from_samples_and_labels(
        [np.array([10, 20, 30])], [np.array([0, 1, 0])],
        20000, unit_ids=[0, 1],
    )
    sorting.register_recording(recording)
    np.testing.assert_array_equal(sorting.get_unit_spike_train(0), [10, 30])
