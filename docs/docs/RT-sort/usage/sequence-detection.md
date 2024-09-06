---
sidebar_position: 2
---

# Sequence Detection (Offline Use)

RT-Sort can be used for offline sequence detection with a pre-trained detection model. Here's how to do it for both Maxwell MEAs and Neuropixels.

## Maxwell MEAs

```python
from braindance.core.spikesorter.rt_sort import detect_sequences

rt_sort = detect_sequences(recording, inter_path, detection_model, **kwargs)
```

### Parameters

- `recording`: Recording loaded with SpikeInterface or a path to a saved recording (str or pathlib.Path).
- `inter_path`: Path to a folder where RT-Sort's intermediate cached data is stored.
- `detection_model`: ModelSpikeSorter object or a path to a folder containing a ModelSpikeSorter object's files.

### Optional Parameters

- `recording_window_ms`: Tuple (start_ms, end_ms) indicating which section of the recording to run RT-Sort. Default is None (entire duration).
- `delete_inter`: Whether to delete the directory inter_path and its contents. Default is False.
- `verbose`: Whether to print progress of RT-Sort. Default is True.
- `num_processes`: Number of CPU processes to use. Default is None (uses all available logical CPUs).

### Returns

- If `return_spikes=False` (default): Returns an RTSort object.
- If `return_spikes=True`: Returns a NumPy array of shape (num_sequences,) where each element is a NumPy array containing a sequence's spike train.

### Example

```python
# Detect sequences in the first 5 minutes of a recording
rt_sort = detect_sequences(recording, inter_path, detection_model, recording_window_ms=(0, 5*60*1000))

# Assign spikes in the next 5 minutes
sequence_spike_trains = rt_sort.sort_offline(recording, inter_path, recording_window_ms=(5*60*1000, 10*60*1000), verbose=True)
```

## Neuropixels

The process for Neuropixels is similar to Maxwell MEAs, with one key difference:

```python
from braindance.core.spikesorter.rt_sort import detect_sequences, neuropixels_params

rt_sort = detect_sequences(recording, inter_path, detection_model, **neuropixels_params, **kwargs)
```

Use the `neuropixels_params` dictionary to pass Neuropixels-specific RT-Sort parameters. It's also recommended to use a Neuropixels-specific detection model.

## Next Steps

After detecting sequences, you can proceed to [real-time application](real-time-application) or [training your own models](training-models).