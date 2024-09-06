---
sidebar_position: 4
---

# Training Your Own Detection Models

RT-Sort allows you to train your own detection models using leave-one-out cross-validation. Here's how to do it:

## Training Detection Models

```python
from braindance.core.spikedetector.train import train_detection_model

model = train_detection_model(recordings)
```

### Parameters

- `recordings`: A list containing the recordings to use for training the detection model (at least two).

### Recommended Parameters

- `kilosort_path`: Folder where Kilosort2 is installed (if not set in environment variable "KILOSORT_PATH").
- `input_scale`: Multiplier for recording traces before input into the detection model (default: 0.01).
- `learning_rate`: Learning rate for training (default: 7.76e-4).
- `validation_recording`: Recording to use as validation (default: None).

### Returns

- `ModelSpikeSorter` object

## Sorting Recordings Offline with Kilosort2

If you prefer to sort recordings with Kilosort2 separately before training detection models, you can use:

```python
from braindance.core.spikesorter.kilosort2 import run_kilosort2

run_kilosort2(recording_files)
```

### Parameters

- `recordings`: A list containing the recordings to spike sort.

### Recommended Parameters

- `kilosort_path`: Folder where Kilosort2 is installed.
- `results_folders`: List of folders to store sorted results (default: None).
- `compile_to_npz`: Whether to save results as "sorted.npz" (default: True).
- `compile_to_mat`: Whether to save results as "sorted.mat" (default: False).
- `intermediate_folders`: List of folders to store intermediate results (default: None).
- `delete_inter`: Whether to delete intermediate folders after sorting (default: True).

### Returns

None (results are saved to files)

## Notes on Training

- The training process is divided into two main steps: spike sorting with Kilosort2 and training the detection model.
- Adjust the learning rate based on the loss curve created at the end of training.
- A straight loss curve might indicate a learning rate that is too low, while a curve that decreases initially but then strongly increases might indicate a rate that is too high.

For more details on parameters and advanced usage, refer to the [API Reference](../api-reference).