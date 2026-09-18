# spikedetector.py

**Path:** `braindance/core/spikedetector/spikedetector.py`
**Module:** `braindance.core.spikedetector.spikedetector`
**Feature Area:** `Spike Detection`
**Entry point:** no — library or imported component

## Overview
Wraps a TorchScript detector with frame and chunk buffering interfaces. The single-frame path currently returns zero predictions while the chunk path invokes the loaded model.

## Connections
- **Uses:** `ModelSpikeSorter` from `braindance.core.spikedetector.model` — imports (static evidence).

## Dependencies
- `braindance.core.spikedetector.model.ModelSpikeSorter` — intra-repo import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.
- `torch_tensorrt9` — external or unresolved local import; source import evidence.

## Classes
### SpikeDetector()
> unclear — see source
**Source:** `braindance/core/spikedetector/spikedetector.py:17`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, model_path, n_channels=256, n_frames=200, device='cuda')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `data_slice` | inferred at runtime | `torch.zeros(n_channels, 1, n_frames).to('cpu').float()` |
| `ind` | inferred at runtime | `0` |
| `model` | inferred at runtime | `torch.jit.load(model_path, map_location=device)` |
| `n_channels` | inferred at runtime | `n_channels` |
| `n_frames` | inferred at runtime | `n_frames` |
**Methods:**
#### `detect(self, data_frame)`
> Takes in the current frame of data, adds it to the data slice, and returns the spike predictions Parameters ---------- data_frame : np.array of shape (n_channels, 1) The current frame of data Returns ------- spike_preds : np.array of shape (n_channels, 120) The spike predictions for each channel, for the previous 6ms, or 120 frames
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikedetector/spikedetector.py:28`
#### `detect_chunk(self, data_chunk)`
> Takes in a chunk of data, and returns the spike predictions for each channel for the last 6ms of data
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikedetector/spikedetector.py:63`
#### `reset(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikedetector/spikedetector.py:104`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| model_path, n_channels=256, n_frames=200, device=cuda |

## Data Shapes
- Buffer shape (channels,1,frames); documented predictions (channels,120).

## Notes
- Constructor moves model and buffer to CPU, but detect_chunk/reset explicitly use CUDA.
- Import probes torch_tensorrt9 and prints selected backend.
