# rt_artifact_removal_JIT.py

**Path:** `braindance/utils/rt_artifact_removal_JIT.py`
**Module:** `braindance.utils.rt_artifact_removal_JIT`
**Feature Area:** `Artifact Removal`
**Entry point:** no — library or imported component

## Overview
Implements a stateful cubic-baseline Numba remover that processes complete channel-by-frame blocks. Supports threshold recovery blanking and a rail-based saturation mode that returns NaNs until the fitting window is valid.

## Connections
None

## Dependencies
- `numba.boolean` — external or unresolved local import; source import evidence.
- `numba.experimental.jitclass` — external or unresolved local import; source import evidence.
- `numba.float32` — external or unresolved local import; source import evidence.
- `numba.float64` — external or unresolved local import; source import evidence.
- `numba.int64` — external or unresolved local import; source import evidence.
- `numba.njit` — external or unresolved local import; source import evidence.
- `numba.prange` — external or unresolved local import; source import evidence.
- `numba.types` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### NumbaOptimizedMultiChannelArtifactRemovalMinibatch()
> Multi-channel cubic artifact removal that processes multiple frames at once.
**Source:** `braindance/utils/rt_artifact_removal_JIT.py:351`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, n_channels, N=60, nc_start=60, min_val=-100, max_val=100, batch_size=None, *, blanking_mode='threshold', rail_min=None, rail_max=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `N` | inferred at runtime | `N` |
| `S_matrices` | inferred at runtime | `np.zeros((n_channels, 4, 4), dtype=np.float64)` |
| `W_vectors` | inferred at runtime | `np.zeros((n_channels, 4), dtype=np.float64)` |
| `Wp_vectors` | inferred at runtime | `np.zeros((n_channels, 4), dtype=np.float64)` |
| `batch_size` | inferred at runtime | `8 * ((n_channels + 7) // 8)` |
| `blanking_mode` | inferred at runtime | `blanking_mode` |
| `buffers` | inferred at runtime | `np.zeros((n_channels, buffer_size), dtype=np.float64)` |
| `channel_batches` | inferred at runtime | `[]` |
| `init_inds` | inferred at runtime | `np.zeros(n_channels, dtype=np.int64)` |
| `max_val` | inferred at runtime | `max_val` |
| `min_val` | inferred at runtime | `min_val` |
| `moving_means` | inferred at runtime | `np.zeros(n_channels, dtype=np.float64)` |
| `n_channels` | inferred at runtime | `n_channels` |
| `nc_start` | inferred at runtime | `nc_start` |
| `need_resets` | inferred at runtime | `np.zeros(n_channels, dtype=np.int64)` |
| `rail_counts` | inferred at runtime | `np.zeros(n_channels, dtype=np.int64)` |
| `spike_flags` | inferred at runtime | `np.zeros(n_channels, dtype=np.bool_)` |
| `states` | inferred at runtime | `np.zeros(n_channels, dtype=np.int64)` |
| `timing` | inferred at runtime | `{'total': 0, 'processing': 0}` |
**Methods:**
#### `fit_step(self, frames_batch, artifact_width=60, remove_frames_before=8)`
> Select blanking kernel and advance all channel filter states.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_artifact_removal_JIT.py:430`
### Timer()
> unclear — see source
**Source:** `braindance/utils/rt_artifact_removal_JIT.py:489`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, name='Timer')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `end_time` | inferred at runtime | `0` |
| `name` | inferred at runtime | `name` |
| `start_time` | inferred at runtime | `None` |
**Methods:**
#### `__enter__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_artifact_removal_JIT.py:495`
#### `__exit__(self, exc_type, exc_val, exc_tb)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_artifact_removal_JIT.py:499`

## Functions
### `fast_factorial(n)`
> Fast factorial calculation using lookup table.
> **Called by:** braindance/utils/rt_artifact_removal_JIT.py:97 (named-call hint); braindance/utils/rt_artifact_removal_JIT.py:98 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal_JIT.py:19`
### `fast_mmean(q, frame)`
> Exponential moving mean update.
> **Called by:** braindance/utils/rt_artifact_removal_JIT.py:233 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal_JIT.py:26`
### `compute_T(nc, N)`
> Compute the T vector for polynomial fit.
> **Called by:** braindance/utils/rt_artifact_removal_JIT.py:416 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal_JIT.py:33`
### `compute_S(T)`
> Compute and invert the S matrix.
> **Called by:** braindance/utils/rt_artifact_removal_JIT.py:417 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal_JIT.py:42`
### `compute_W(v, nc, N)`
> Compute initial W vector.
> **Called by:** braindance/utils/rt_artifact_removal_JIT.py:174 (named-call hint); braindance/utils/rt_artifact_removal_JIT.py:287 (named-call hint); braindance/utils/rt_artifact_removal_JIT.py:333 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal_JIT.py:51`
### `compute_W_rec(Wp, N, nc, v, outgoing)`
> Compute W recursively using previous W.
> **Called by:** braindance/utils/rt_artifact_removal_JIT.py:241 (named-call hint); braindance/utils/rt_artifact_removal_JIT.py:337 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal_JIT.py:90`
### `compute_a(S, W, a)`
> Compute polynomial coefficients.
> **Called by:** braindance/utils/rt_artifact_removal_JIT.py:246 (named-call hint); braindance/utils/rt_artifact_removal_JIT.py:338 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal_JIT.py:111`
### `shift_buffer(buffer, new_val)`
> Shift values in buffer left and add new value at the end.
> **Called by:** braindance/utils/rt_artifact_removal_JIT.py:204 (named-call hint); braindance/utils/rt_artifact_removal_JIT.py:269 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal_JIT.py:121`
### `process_channel_minibatch_numba(channels, frames_batch, clean_values, artifacts, spikes, artifact_width, remove_frames_before, states, init_inds, moving_means, need_resets, spike_flags, buffers, W_vectors, Wp_vectors, S_matrices, min_val, max_val)`
> Process a minibatch of frames for multiple channels in parallel using Numba.
> **Called by:** braindance/utils/rt_artifact_removal_JIT.py:475 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal_JIT.py:130`
### `process_saturation_cubic(frames, clean, artifact, spikes, states, init_inds, rail_counts, spike_flags, buffers, W_current, W_prev, S, rail_min, rail_max, spike_min, spike_max, N)`
> Use the original polynomial algebra, excluding rail-contaminated windows.
> **Called by:** braindance/utils/rt_artifact_removal_JIT.py:463 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal_JIT.py:298`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| N=60,min_val=-100,max_val=100,batch_size; blanking_mode='threshold'\|'saturation', finite rail_min/rail_max required for saturation. |
| fit_step artifact_width=60; spike detection range fixed -20<clean<-3.5. |

## Data Shapes
- Input (n_channels,n_frames); output tuple clean/artifact float64 and spikes bool of same shape.

## Notes
- Threshold mode startup/recovery emits zeros; saturation mode emits NaNs for excluded windows.
- Maintains state across blocks; remove_frames_before accepted but unused by threshold kernel.
