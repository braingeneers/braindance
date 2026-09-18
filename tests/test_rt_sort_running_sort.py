import numpy as np
import torch

from braindance.core.spikesorter.rt_sort import RTSort


def _capture_preprocessed_observations(remove_median, model_chunk=None):
    sorter = object.__new__(RTSort)
    sorter.chan_ids = None
    sorter.device = "cpu"
    sorter.dtype = torch.float32
    sorter.pre_median_frames = torch.zeros((2, 2), dtype=sorter.dtype)
    sorter.latest_frame = 0
    sorter.cur_num_pre_median_frames = 0
    sorter.total_num_pre_median_frames = 100
    sorter.pre_medians = torch.zeros(2)
    sorter.input_size = 2
    sorter.ignore_spikes_before_minuend = 0
    sorter.sort_chunk = lambda traces, **kwargs: traces.clone()

    kwargs = {} if remove_median is None else {"remove_median": remove_median}
    return sorter._running_sort(np.array([[1, 10], [3, 12]]), model_chunk=model_chunk, **kwargs)


def test_running_sort_median_centers_by_default_for_backward_compatibility():
    expected = torch.tensor([[0, 2], [0, 2]], dtype=torch.float32)
    assert torch.equal(_capture_preprocessed_observations(None), expected)
    assert torch.equal(_capture_preprocessed_observations(True), expected)


def test_running_sort_can_skip_median_centering():
    expected = torch.tensor([[1, 3], [10, 12]], dtype=torch.float32)
    assert torch.equal(_capture_preprocessed_observations(False), expected)


def test_supplied_model_outputs_skip_centering_like_the_old_implementation():
    expected = torch.tensor([[1, 3], [10, 12]], dtype=torch.float32)
    model_chunk = np.zeros((2, 2), dtype=np.float32)
    assert torch.equal(_capture_preprocessed_observations(None, model_chunk), expected)
    assert torch.equal(_capture_preprocessed_observations(True, model_chunk), expected)
