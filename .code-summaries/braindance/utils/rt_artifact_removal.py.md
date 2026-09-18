# rt_artifact_removal.py

**Path:** `braindance/utils/rt_artifact_removal.py`
**Module:** `braindance.utils.rt_artifact_removal`
**Feature Area:** `Artifact Removal`
**Entry point:** no — library or imported component

## Overview
Implements a legacy cubic-baseline artifact remover with per-channel initialization, fitting, and recovery states. Buffers single frames into minibatches and dispatches channel groups through a thread pool and parallel Numba kernel.

## Connections
None

## Dependencies
- `numba.boolean` — external or unresolved local import; source import evidence.
- `numba.float64` — external or unresolved local import; source import evidence.
- `numba.int64` — external or unresolved local import; source import evidence.
- `numba.njit` — external or unresolved local import; source import evidence.
- `numba.prange` — external or unresolved local import; source import evidence.
- `numba.types` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### MinibatchNumbaArtifactRemoval()
> High-performance minibatch artifact removal for 500+ channels.
**Source:** `braindance/utils/rt_artifact_removal.py:309`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, n_channels, N=60, nc_start=60, min_val=-50, max_val=50, minibatch_size=40, channel_batch_size=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `N` | inferred at runtime | `N` |
| `S_matrices` | inferred at runtime | `np.zeros((n_channels, 4, 4), dtype=np.float64)` |
| `W_vectors` | inferred at runtime | `np.zeros((n_channels, 4), dtype=np.float64)` |
| `Wp_vectors` | inferred at runtime | `np.zeros((n_channels, 4), dtype=np.float64)` |
| `artifacts` | inferred at runtime | `np.zeros(n_channels, dtype=np.float64)` |
| `artifacts_batch` | inferred at runtime | `np.zeros((n_channels, minibatch_size), dtype=np.float64)` |
| `batch_count` | inferred at runtime | `0` |
| `buffers` | inferred at runtime | `np.zeros((n_channels, buffer_size), dtype=np.float64)` |
| `channel_batch_size` | inferred at runtime | `8 * ((n_channels + 7) // 8 // 4)` |
| `channel_batches` | inferred at runtime | `[]` |
| `clean_batch` | inferred at runtime | `np.zeros((n_channels, minibatch_size), dtype=np.float64)` |
| `clean_values` | inferred at runtime | `np.zeros(n_channels, dtype=np.float64)` |
| `frames_batch` | inferred at runtime | `np.zeros((n_channels, minibatch_size), dtype=np.float64)` |
| `futures` | inferred at runtime | `[]` |
| `init_inds` | inferred at runtime | `np.zeros(n_channels, dtype=np.int64)` |
| `last_output_buffers` | inferred at runtime | `np.zeros((n_channels, 3), dtype=np.float64)` |
| `max_val` | inferred at runtime | `max_val` |
| `min_val` | inferred at runtime | `min_val` |
| `minibatch_size` | inferred at runtime | `minibatch_size` |
| `moving_means` | inferred at runtime | `np.zeros(n_channels, dtype=np.float64)` |
| `n_channels` | inferred at runtime | `n_channels` |
| `nc_start` | inferred at runtime | `nc_start` |
| `need_resets` | inferred at runtime | `np.zeros(n_channels, dtype=np.int64)` |
| `needs_W_init` | inferred at runtime | `np.ones(n_channels, dtype=np.int64)` |
| `spike_flags` | inferred at runtime | `np.zeros(n_channels, dtype=np.bool_)` |
| `spikes` | inferred at runtime | `np.zeros(n_channels, dtype=np.bool_)` |
| `spikes_batch` | inferred at runtime | `np.zeros((n_channels, minibatch_size), dtype=np.bool_)` |
| `states` | inferred at runtime | `np.zeros(n_channels, dtype=np.int64)` |
| `thread_pool` | inferred at runtime | `ThreadPoolExecutor(max_workers=min(32, len(self.channel_batches)))` |
| `timing` | inferred at runtime | `{'total': 0, 'processing': 0}` |
**Methods:**
#### `_process_channel_batch(self, channel_batch)`
> Process a batch of channels for the current minibatch.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal.py:400`
#### `fit_step(self, frames, artifact_width=50, remove_frames_before=8)`
> Buffer a frame and process full minibatches, returning last channel outputs.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_artifact_removal.py:413`
#### `flush(self)`
> Process any remaining frames in the minibatch.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_artifact_removal.py:473`
#### `shutdown(self)`
> Join and shut down worker threads.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal.py:505`
### Timer()
> unclear — see source
**Source:** `braindance/utils/rt_artifact_removal.py:510`
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
**Source:** `braindance/utils/rt_artifact_removal.py:516`
#### `__exit__(self, exc_type, exc_val, exc_tb)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_artifact_removal.py:520`

## Functions
### `fast_factorial(n)`
> Fast factorial calculation using lookup table.
> **Called by:** braindance/utils/rt_artifact_removal.py:70 (named-call hint); braindance/utils/rt_artifact_removal.py:71 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal.py:16`
### `fast_mmean(q, frame)`
> Exponential moving mean update.
> **Called by:** braindance/utils/rt_artifact_removal.py:196 (named-call hint); braindance/utils/rt_artifact_removal.py:279 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal.py:23`
### `compute_T(nc, N)`
> Compute the T vector for polynomial fit.
> **Called by:** braindance/utils/rt_artifact_removal.py:368 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal.py:29`
### `compute_S(T)`
> Compute and invert the S matrix.
> **Called by:** braindance/utils/rt_artifact_removal.py:369 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal.py:38`
### `compute_W(v, nc, N)`
> Compute initial W vector.
> **Called by:** braindance/utils/rt_artifact_removal.py:164 (named-call hint); braindance/utils/rt_artifact_removal.py:253 (named-call hint); braindance/utils/rt_artifact_removal.py:276 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal.py:47`
### `compute_W_rec(W, Wp, N, nc, v)`
> Compute W recursively using previous W.
> **Called by:** braindance/utils/rt_artifact_removal.py:203 (named-call hint); braindance/utils/rt_artifact_removal.py:282 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal.py:63`
### `compute_a(S, W, a)`
> Compute polynomial coefficients.
> **Called by:** braindance/utils/rt_artifact_removal.py:207 (named-call hint); braindance/utils/rt_artifact_removal.py:284 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal.py:83`
### `shift_buffer(buffer, new_val)`
> Shift values in buffer left and add new value at the end.
> **Called by:** braindance/utils/rt_artifact_removal.py:184 (named-call hint); braindance/utils/rt_artifact_removal.py:199 (named-call hint); braindance/utils/rt_artifact_removal.py:238 (named-call hint); braindance/utils/rt_artifact_removal.py:261 (named-call hint); braindance/utils/rt_artifact_removal.py:280 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal.py:93`
### `process_minibatch_numba(channels, frames_batch, n_batch_frames, clean_batch, artifacts_batch, spikes_batch, artifact_width, remove_frames_before, states, init_inds, moving_means, need_resets, spike_flags, buffers, W_vectors, Wp_vectors, S_matrices, needs_W_init, last_output_buffers)`
> Process a minibatch of frames for a batch of channels in parallel.
> **Called by:** braindance/utils/rt_artifact_removal.py:402 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_artifact_removal.py:101`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| n_channels,N=60,minibatch_size=40,channel_batch_size; kernel uses fixed threshold ±50 and recovery width 50. |

## Data Shapes
- fit_step input vector(n_channels); returns clean/artifact float64 vectors and spike bool vector; internal batches (channels,minibatch_size).

## Notes
- Between full batches fit_step repeats previous outputs; flush processes remaining input and returns final sample only.
- Call shutdown to release thread pool; fit_step artifact_width/remove_frames_before and constructor min_val/max_val do not reach kernel settings.
- Default channel_batch_size can become zero for small channel counts.
