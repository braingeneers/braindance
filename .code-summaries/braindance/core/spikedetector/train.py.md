# train.py

**Path:** `braindance/core/spikedetector/train.py`
**Module:** `braindance.core.spikedetector.train`
**Feature Area:** `Spike Detection`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Builds waveform/noise datasets from one or more recordings or prepared deep-learning folders and trains a one-channel `ModelSpikeSorter`. It prepares Kilosort-derived folders when needed, optionally reserves one recording for validation, retries NaN training with reduced input scale, and saves the trained model.

## Connections
- **Uses:** `data` from `braindance.core.spikedetector` — imports (static evidence).
- **Uses:** `ModelSpikeSorter` from `braindance.core.spikedetector.model` — imports (static evidence).
- **Uses:** `utils` from `braindance.core.spikedetector` — imports (static evidence).
- **Shared data:** Uses `spikedetector.data.setup_dl_folders`/`MultiRecordingDataset`, `utils.random_seed`, PyTorch DataLoader, SpikeInterface `BaseRecording`, and `ModelSpikeSorter.fit/save`.

## Dependencies
- `braindance.core.spikedetector.data` — intra-repo import; source import evidence.
- `braindance.core.spikedetector.model.ModelSpikeSorter` — intra-repo import; source import evidence.
- `braindance.core.spikedetector.utils` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `spikeinterface.core.BaseRecording` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.
- `torch.utils.data.DataLoader` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `train_detection_model(recordings: list, dl_folder_name='dl_folder', validation_recording=None, thresh_amp=18.88275, thresh_std=0.6, sample_size_ms=10, recording_spike_before_ms=2, recording_spike_after_ms=2, samples_per_waveform=2, num_wfs_probs=[0.5, 0.3, 0.12, 0.06, 0.02], isi_wf_min_ms=0.2, isi_wf_max_ms=None, learning_rate=0.000776, momentum=0.85, training_thresh=0.01, learning_rate_patience=5, learning_rate_decay=0.4, epoch_patience=10, max_num_epochs=200, batch_size=1, num_workers=0, shuffle=True, training_random_seed=231, input_scale=0.01, input_scale_decay=0.1, device='cuda', dtype=torch.float16, **run_kilosort2_kwargs)`
> Prepare training/validation datasets, construct and fit a detection model with early stopping/LR decay, recover from NaN loss by scaling inputs, save it, and return it.
> **Called by:** braindance/core/spikedetector/train.py:221 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/core/spikedetector/train.py:11`
### `main()`
> Run a site-specific training job using hardcoded prepared recording folders and Kilosort2 installation.
> **Called by:** braindance/core/spikedetector/train.py:229 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/train.py:220`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Dataset defaults include 10 ms samples, 2 ms front/end buffers, 18.88275 microvolt amplitude threshold, 0.6 waveform-ratio standard-deviation threshold, and 0.2 ms minimum waveform ISI. |
| Training defaults: learning rate 7.76e-4, momentum 0.85, batch size 1, 200 epochs, epoch patience 10, LR patience 5/decay 0.4, seed 231, CUDA float16. |
| Arbitrary `run_kilosort2_kwargs` pass to dataset-folder preparation; CLI `main` hardcodes six `/data/MEAprojects/...` folders and a `/home/mea/.../Kilosort2` path. |

## Data Shapes
- Each prepared folder must contain `sorted.npz` with sampling frequency `fs` and `scaled_traces.npy`; sample counts derive from rounded kHz sampling frequency.
- The model consumes one input channel, with sample length `sample_size_ms * kHz` and 2 ms buffers; `num_wfs_probs` defines probabilities for 1..N waveforms per synthesized sample.

## Notes
- Providing validation mutates the caller's `recordings` list by appending the validation recording.
- After processing the loop, `model.save(rec, verbose=True)` uses the last original loop item rather than a dedicated output path, which may be unintended.
- The NaN retry loop has no retry limit and repeatedly reinitializes weights while shrinking `input_scale`.
