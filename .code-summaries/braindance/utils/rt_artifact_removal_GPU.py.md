# rt_artifact_removal_GPU.py

**Path:** `braindance/utils/rt_artifact_removal_GPU.py`
**Module:** `braindance.utils.rt_artifact_removal_GPU`
**Feature Area:** `Artifact Removal`
**Entry point:** no — library or imported component

## Overview
Implements a PyTorch cubic-baseline artifact remover with channel state tensors and batched input storage. Uses initialization, active fitting, and depegging passes and can return NumPy or tensor outputs matching input kind.

## Connections
None

## Dependencies
- `numpy` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.

## Classes
### TorchArtifactRemovalOptimized()
> Highly parallelized PyTorch implementation for GPU-accelerated artifact removal.
**Source:** `braindance/utils/rt_artifact_removal_GPU.py:5`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, n_channels, N=60, nc_start=60, min_val=-100, max_val=100, batch_size=64, device='cuda')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `N` | inferred at runtime | `N` |
| `S_matrices` | inferred at runtime | `self._compute_S_matrices(N, nc_start)` |
| `artifact_mask` | inferred at runtime | `torch.zeros(n_channels, dtype=torch.bool, device=self.device)` |
| `batch_count` | inferred at runtime | `0` |
| `batch_size` | inferred at runtime | `batch_size` |
| `buffer_size` | inferred at runtime | `buffer_size` |
| `buffers` | inferred at runtime | `torch.zeros((n_channels, buffer_size), dtype=torch.float32, device=self.device)` |
| `depeg_counts` | inferred at runtime | `torch.zeros(n_channels, dtype=torch.int32, device=self.device)` |
| `device` | inferred at runtime | `torch.device('cpu')` |
| `init_counts` | inferred at runtime | `torch.zeros(n_channels, dtype=torch.int32, device=self.device)` |
| `input_batch` | inferred at runtime | `torch.zeros((n_channels, batch_size), dtype=torch.float32, device=self.device)` |
| `is_initialized` | inferred at runtime | `torch.zeros(n_channels, dtype=torch.bool, device=self.device)` |
| `max_val` | inferred at runtime | `max_val` |
| `min_val` | inferred at runtime | `min_val` |
| `moving_means` | inferred at runtime | `torch.zeros(n_channels, dtype=torch.float32, device=self.device)` |
| `n_channels` | inferred at runtime | `n_channels` |
| `n_powers` | inferred at runtime | `torch.zeros((buffer_size, 4), dtype=torch.float32, device=self.device)` |
| `output_artifacts` | inferred at runtime | `torch.zeros((n_channels, batch_size), dtype=torch.float32, device=self.device)` |
| `output_clean` | inferred at runtime | `torch.zeros((n_channels, batch_size), dtype=torch.float32, device=self.device)` |
| `output_spikes` | inferred at runtime | `torch.zeros((n_channels, batch_size), dtype=torch.bool, device=self.device)` |
| `spike_flags` | inferred at runtime | `torch.zeros(n_channels, dtype=torch.bool, device=self.device)` |
| `temp_W` | inferred at runtime | `torch.zeros((n_channels, 4), dtype=torch.float32, device=self.device)` |
| `temp_a` | inferred at runtime | `torch.zeros((n_channels, 4), dtype=torch.float32, device=self.device)` |
| `temp_buffers` | inferred at runtime | `torch.zeros_like(self.buffers)` |
| `timing` | inferred at runtime | `{'total': 0, 'compute': 0}` |
**Methods:**
#### `_compute_S_matrices(self, N, nc_start)`
> Compute S matrices for polynomial fitting.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal_GPU.py:81`
#### `_update_initialized_masks(self, batch_idx=None)`
> Update initialization masks based on init counts.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_artifact_removal_GPU.py:98`
#### `_vectorized_shift_buffers(self, frame, mask=None)`
> Shift buffers for selected channels using vectorized operations.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_artifact_removal_GPU.py:109`
#### `_compute_W_vectorized(self, mask=None)`
> Compute W vectors using vectorized operations.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_artifact_removal_GPU.py:136`
#### `_compute_coefficients(self, mask=None)`
> Compute polynomial coefficients.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_artifact_removal_GPU.py:169`
#### `_detect_artifacts(self, frame)`
> Detect artifacts in the current frame.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal_GPU.py:199`
#### `_detect_spikes(self, clean_values, output_spikes)`
> Detect spikes in clean values.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_artifact_removal_GPU.py:216`
#### `_update_moving_means(self, frame, mask=None)`
> Update moving means for specified channels.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_artifact_removal_GPU.py:243`
#### `_initialize_new_channels(self, frame, batch_idx)`
> Process initialization for channels that need it.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_artifact_removal_GPU.py:264`
#### `_process_active_channels(self, frame, batch_idx)`
> Process channels that are active (initialized and not in depeg).
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_artifact_removal_GPU.py:314`
#### `_process_depeg_channels(self, frame, batch_idx)`
> Process channels in depeg mode.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_artifact_removal_GPU.py:383`
#### `process_batch(self, frames)`
> Run three channel-state passes per frame using shared tensor buffers.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal_GPU.py:445`
#### `fit_step(self, frame, artifact_width=50, remove_frames_before=8)`
> Process one frame across all channels with batch support.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_artifact_removal_GPU.py:481`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| device='cuda' falls back to CPU if unavailable; N=60,min_val=-100,max_val=100,batch_size=64. |

## Data Shapes
- fit_step input channel vector; returns clean/artifact/spikes vectors; internal tensors float32 and bool.

## Notes
- fit_step processes individual frames and then reprocesses accumulated full batches, advancing state twice for those frames.
- artifact_width/remove_frames_before are accepted but internal recovery is fixed at 50.
- Masked _detect_artifacts subtracts full moving_means, so partial active-channel masks can mismatch shapes.
