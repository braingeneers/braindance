"""CPU offline support and the live API boundary, without requiring CUDA."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from braindance import get_rt_sort_path
from braindance.core.spikedetector.model import ModelSpikeSorter
from braindance.core.spikesorter.rt_sort import RTSort, run_detection_model


@pytest.fixture
def sorter(tmp_path):
    model = ModelSpikeSorter.load_mea(device="cpu")
    sequence = SimpleNamespace(
        root_elec=0, spike_train=np.array([1.0]), comp_elecs=[0, 1],
        inner_loose_elecs=[0, 1], loose_elecs=[0, 1], min_loose_detections=2,
        all_latencies=np.array([0, 1]), all_amp_medians=np.array([4., 3.]),
        all_elec_probs=np.array([.9, .8]), root_to_amp_median_std={0: 1.})
    params = dict(
        samp_freq=20, elec_locs=np.array([[0., 0.], [10., 0.]]),
        model_inter_path=tmp_path, stringent_thresh=.275, loose_thresh=.1,
        inference_scaling_numerator=None, n_before=10, n_after=10,
        pre_median_frames=1000, inner_radius=50, min_elecs_for_array_noise=100,
        min_inner_loose_detections=2, max_latency_diff_spikes=3.5,
        clip_latency_diff_factor=2, max_amp_median_diff_spikes=.65,
        clip_amp_median_diff_factor=2, max_root_amp_median_std_spikes=2.5,
        repeated_detection_overlap_time=.2)
    return RTSort([sequence], model, params, device="cpu")


def test_cpu_offline_and_live_rejection(sorter):
    assert sorter.dtype == torch.float32
    assert next(sorter.model.parameters()).dtype == torch.float32
    traces = np.random.default_rng(0).normal(size=(2, 1203)).astype(np.float32)
    spikes = sorter.sort_offline(traces)
    assert spikes.shape == (1,)
    frame = sorter.latest_frame
    with pytest.raises(RuntimeError, match="offline sorting only"):
        sorter.running_sort(traces[:, :100].T)
    assert sorter.latest_frame == frame
    # Failure must not leave a persistent flag that interferes with offline use.
    repeated = sorter.sort_offline(traces)
    np.testing.assert_array_equal(spikes[0], repeated[0])


def test_gpu_live_dispatch_preserves_arguments():
    sorter = object.__new__(RTSort)
    sorter.device = "cuda"
    sorter._running_sort = lambda *args: args
    obs = np.zeros((100, 2))
    result = sorter.running_sort(obs, latest_frame=500, use_numba=True, remove_median=False)
    assert result[0] is obs
    assert result[1:] == (None, 500, True, False)


def test_saved_half_sorter_loads_as_cpu_float32(sorter, tmp_path):
    sorter.dtype = torch.float16
    for name, value in vars(sorter).items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            setattr(sorter, name, value.half())
    path = tmp_path / "sorter.pkl"
    sorter.save(path)
    restored = RTSort.load_from_file(path, get_rt_sort_path(), device="cpu")
    assert restored.dtype == torch.float32
    assert restored.comp_elecs.dtype == torch.int64
    assert restored.seq_no_overlap_mask.dtype == torch.bool
    assert all(v.device.type == "cpu" for v in vars(restored).values() if isinstance(v, torch.Tensor))
    assert next(restored.model.parameters()).dtype == torch.float32
    assert restored.sort_offline(np.zeros((2, 1100), dtype=np.float32)).shape == (1,)
    unchanged = RTSort.load_from_file(path)
    assert unchanged.dtype == torch.float16


@pytest.mark.parametrize("frames", [200, 320, 327])
def test_cpu_detector_including_tail(tmp_path, frames):
    model = ModelSpikeSorter.load_mea(device="cpu")
    traces = np.random.default_rng(1).normal(size=(2, frames)).astype(np.float16)
    path = tmp_path / "scaled_traces.npy"
    np.save(path, traces)
    run_detection_model(SimpleNamespace(get_num_channels=lambda: 2), model, path,
                        device="cpu", inference_scaling_numerator=None, verbose=False)
    outputs = np.load(tmp_path / "model_outputs.npy")
    assert outputs.shape == (2, frames - 80)
    assert np.isfinite(outputs).all()
    last = torch.tensor(traces[:, -200:], dtype=torch.float32)
    last -= last.median(dim=1, keepdim=True).values
    with torch.no_grad():
        expected = model.model.conv(last[:, None] * model.input_scale)[:, 0].numpy()
    count = 7 if frames == 327 else 120
    np.testing.assert_allclose(outputs[:, -count:], expected[:, -count:].astype(np.float16), atol=.004)


def test_cpu_detector_rejects_too_short_recording(tmp_path):
    path = tmp_path / "scaled_traces.npy"
    np.save(path, np.zeros((2, 199), dtype=np.float16))
    with pytest.raises(ValueError, match="at least 200 samples"):
        run_detection_model(SimpleNamespace(get_num_channels=lambda: 2),
                            ModelSpikeSorter.load_mea(device="cpu"), path,
                            device="cpu", verbose=False)


@pytest.mark.parametrize("chunk_start", [7, 17])
def test_maxwell_export_clips_chunk_to_requested_window(tmp_path, chunk_start):
    import h5py
    from braindance.core.spikesorter.rt_sort import _get_traces_mea_old, _save_traces_mea

    raw = np.arange(120, dtype=np.int16).reshape(3, 40)
    source = tmp_path / "recording.h5"
    with h5py.File(source, "w") as file:
        file["sig"] = raw
    target = tmp_path / "scaled_traces.npy"
    np.save(target, np.zeros((2, 13), dtype=np.float32))
    _save_traces_mea((source, target, 7, [0, 2], chunk_start, 100, 2.,
                      np.float32, _get_traces_mea_old))
    expected = np.zeros((2, 13), dtype=np.float32)
    expected[:, chunk_start-7:] = raw[[0, 2], chunk_start:20] * 2
    np.testing.assert_array_equal(np.load(target), expected)
