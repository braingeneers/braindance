---
sidebar_position: 5
---

# API Reference

This page provides a detailed reference for the main classes and functions in RT-Sort.

## ModelSpikeSorter

### `ModelSpikeSorter.load(detection_model_path)`

Loads a pre-trained detection model.

- **Parameters:**
  - `detection_model_path`: Path to a folder containing a ModelSpikeSorter object's files.
- **Returns:** ModelSpikeSorter object

## Sequence Detection

### `detect_sequences(recording, inter_path, detection_model, **kwargs)`

Detects sequences in a recording.

- **Parameters:**
  - `recording`: Recording loaded with SpikeInterface or path to a saved recording.
  - `inter_path`: Path to a folder for intermediate cached data.
  - `detection_model`: ModelSpikeSorter object or path to its folder.
  - `**kwargs`: Additional optional parameters.
    - `recording_window_ms`: Tuple (start_ms, end_ms) for section of recording to process. Default is None (entire duration).
    - `delete_inter`: Whether to delete the intermediate directory. Default is False.
    - `verbose`: Whether to print progress. Default is True.
    - `num_processes`: Number of CPU processes to use. Default is None (uses all available).
- **Returns:** RTSort object or NumPy array of spike trains.

## RTSort

### `reset()`

Resets the RTSort object for online use.

- **Parameters:** None
- **Returns:** None

### `running_sort(obs)`

Sorts a stream of data in real-time.

- **Parameters:**
  - `obs`: NumPy array with shape (num_frames, num_electrodes).
- **Returns:** List of tuples containing sorted spike data.

## Training

### `train_detection_model(recordings, **kwargs)`

Trains a new detection model.

- **Parameters:**
  - `recordings`: List of recordings for training.
  - `**kwargs`: Additional optional parameters.
    - `kilosort_path`: Folder where Kilosort2 is installed. Default is None.
    - `input_scale`: Multiplier for recording traces. Default is 0.01.
    - `learning_rate`: Learning rate for training. Default is 7.76e-4.
    - `validation_recording`: Recording to use as validation. Default is None.
- **Returns:** ModelSpikeSorter object

## Kilosort2 Integration

### `run_kilosort2(recording_files, **kwargs)`

Runs Kilosort2 on a set of recordings.

- **Parameters:**
  - `recording_files`: List of recordings to spike sort.
  - `**kwargs`: Additional optional parameters.
    - `kilosort_path`: Folder where Kilosort2 is installed.
    - `results_folders`: List of folders to store sorted results. Default is None.
    - `compile_to_npz`: Whether to save results as "sorted.npz". Default is True.
    - `compile_to_mat`: Whether to save results as "sorted.mat". Default is False.
    - `intermediate_folders`: List of folders for intermediate results. Default is None.
    - `delete_inter`: Whether to delete intermediate folders after sorting. Default is True.
- **Returns:** None (results are saved to files)

## Usage Examples

### Offline Sequence Detection

```python
rt_sort = detect_sequences(recording, inter_path, detection_model, recording_window_ms=(0, 5*60*1000))
sequence_spike_trains = rt_sort.sort_offline(recording, inter_path, recording_window_ms=(5*60*1000, 10*60*1000), verbose=True)
```

### Online Sorting

```python
rt_sort.reset()
while not done:
    obs, done = env.step()
    sequence_detections = rt_sort.running_sort(obs)
    # Process sequence_detections
```

For more detailed information on each function and its parameters, please refer to the respective sections in the user manual.