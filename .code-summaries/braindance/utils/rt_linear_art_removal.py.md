# rt_linear_art_removal.py

**Path:** `braindance/utils/rt_linear_art_removal.py`
**Module:** `braindance.utils.rt_linear_art_removal`
**Feature Area:** `Artifact Removal`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Provides original and optimized stateful linear-baseline removers with Numba channel parallelism. Optimized version reuses output buffers and supports rail-based saturation blanking alongside legacy threshold recovery.

## Connections
- **Used by:** `braindance.core.phases2` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases2_loop` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.real_time_sorting` — import consumer hint; not a proven runtime call.

## Dependencies
- `numba` — external or unresolved local import; source import evidence.
- `numba.njit` — external or unresolved local import; source import evidence.
- `numba.prange` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### LinearArtifactRemoval()
> Optimized multi-channel artifact removal with linear fitting.
**Source:** `braindance/utils/rt_linear_art_removal.py:195`
**Kind:** class. **Instantiated by:** braindance/core/phases2.py:1256 (named-call hint); braindance/core/phases2_loop.py:289 (named-call hint); braindance/core/phases2_loop.py:88 (named-call hint); braindance/experiments/real_time_sorting.py:47 (named-call hint); braindance/utils/rt_linear_art_removal.py:753 (named-call hint)
**Constructor:** `__init__(self, n_channels, N=60, nc_start=60, min_val=-100, max_val=100, spike_thresh=[-3.5, 20], batch_size=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `N` | inferred at runtime | `N` |
| `S_matrices` | inferred at runtime | `np.zeros((n_channels, 2, 2), dtype=np.float64)` |
| `W_vectors` | inferred at runtime | `np.zeros((n_channels, 2), dtype=np.float64)` |
| `Wp_vectors` | inferred at runtime | `np.zeros((n_channels, 2), dtype=np.float64)` |
| `batch_size` | inferred at runtime | `8 * ((n_channels + 7) // 8)` |
| `buffers` | inferred at runtime | `np.zeros((n_channels, buffer_size), dtype=np.float64)` |
| `channel_batches` | inferred at runtime | `[]` |
| `init_inds` | inferred at runtime | `np.zeros(n_channels, dtype=np.int64)` |
| `max_val` | inferred at runtime | `max_val` |
| `min_val` | inferred at runtime | `min_val` |
| `moving_means` | inferred at runtime | `np.zeros(n_channels, dtype=np.float64)` |
| `n_channels` | inferred at runtime | `n_channels` |
| `nc_start` | inferred at runtime | `nc_start` |
| `need_resets` | inferred at runtime | `np.zeros(n_channels, dtype=np.int64)` |
| `spike_flags` | inferred at runtime | `np.zeros(n_channels, dtype=np.bool_)` |
| `spike_thresh_max` | inferred at runtime | `spike_thresh[1]` |
| `spike_thresh_min` | inferred at runtime | `spike_thresh[0]` |
| `states` | inferred at runtime | `np.zeros(n_channels, dtype=np.int64)` |
| `timing` | inferred at runtime | `{'total': 0, 'processing': 0}` |
**Methods:**
#### `fit_step(self, frames_batch, artifact_width=60, remove_frames_before=8)`
> Process multiple frames across all channels using batch parallelism.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_linear_art_removal.py:263`
#### `warmup(self, n_samples=1000, minibatch_size=40)`
> Warm up the artifact removal by processing dummy data.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_linear_art_removal.py:311`
### Timer()
> unclear — see source
**Source:** `braindance/utils/rt_linear_art_removal.py:327`
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
**Source:** `braindance/utils/rt_linear_art_removal.py:333`
#### `__exit__(self, exc_type, exc_val, exc_tb)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_linear_art_removal.py:337`
### LinearArtifactRemoval2()
> Highly optimized multi-channel artifact removal with linear fitting.
**Source:** `braindance/utils/rt_linear_art_removal.py:547`
**Kind:** class. **Instantiated by:** braindance/utils/rt_linear_art_removal.py:781 (named-call hint)
**Constructor:** `__init__(self, n_channels, N=60, nc_start=60, min_val=-100, max_val=100, spike_thresh=[-3.5, 20], n_threads=None, *, blanking_mode='threshold', rail_min=None, rail_max=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `N` | inferred at runtime | `N` |
| `S_inv_params` | inferred at runtime | `np.zeros((n_channels, 3), dtype=np.float64)` |
| `W_current` | inferred at runtime | `np.zeros((n_channels, 2), dtype=np.float64, order='C')` |
| `W_prev` | inferred at runtime | `np.zeros((n_channels, 2), dtype=np.float64, order='C')` |
| `_artifacts_buffer` | inferred at runtime | `None` |
| `_clean_buffer` | inferred at runtime | `None` |
| `_spikes_buffer` | inferred at runtime | `None` |
| `artifact_width` | inferred at runtime | `60` |
| `blanking_mode` | inferred at runtime | `blanking_mode` |
| `buffers` | inferred at runtime | `np.zeros((n_channels, buffer_size), dtype=np.float64, order='C')` |
| `init_inds` | inferred at runtime | `np.zeros(n_channels, dtype=np.int64)` |
| `max_val` | inferred at runtime | `max_val` |
| `min_val` | inferred at runtime | `min_val` |
| `moving_means` | inferred at runtime | `np.zeros(n_channels, dtype=np.float64)` |
| `n_channels` | inferred at runtime | `n_channels` |
| `nc_start` | inferred at runtime | `nc_start` |
| `need_resets` | inferred at runtime | `np.zeros(n_channels, dtype=np.int64)` |
| `rail_counts` | inferred at runtime | `np.zeros(n_channels, dtype=np.int64)` |
| `spike_flags` | inferred at runtime | `np.zeros(n_channels, dtype=np.bool_)` |
| `spike_thresh_max` | inferred at runtime | `spike_thresh[1]` |
| `spike_thresh_min` | inferred at runtime | `spike_thresh[0]` |
| `states` | inferred at runtime | `np.zeros(n_channels, dtype=np.int64)` |
| `timing` | inferred at runtime | `{'total': 0, 'processing': 0}` |
**Methods:**
#### `fit_step(self, frames_batch, artifact_width=60, remove_frames_before=8)`
> Process channel blocks with reusable buffers and selected blanking mode.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_linear_art_removal.py:641`
#### `warmup(self, n_samples=1000, minibatch_size=200)`
> Warm up the processor with dummy data.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/rt_linear_art_removal.py:704`

## Functions
### `fast_factorial(n)`
> Fast factorial calculation using lookup table.
> **Called by:** braindance/utils/rt_linear_art_removal.py:63 (named-call hint); braindance/utils/rt_linear_art_removal.py:64 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_linear_art_removal.py:9`
### `fast_mmean(q, frame)`
> Exponential moving mean update.
> **Called by:** braindance/utils/rt_linear_art_removal.py:150 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_linear_art_removal.py:16`
### `compute_T(nc, N)`
> Compute the T vector for linear fit.
> **Called by:** braindance/utils/rt_linear_art_removal.py:249 (named-call hint); braindance/utils/rt_linear_art_removal.py:623 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_linear_art_removal.py:23`
### `compute_S(T)`
> Compute and invert the S matrix for linear fit.
> **Called by:** braindance/utils/rt_linear_art_removal.py:250 (named-call hint); braindance/utils/rt_linear_art_removal.py:624 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_linear_art_removal.py:32`
### `compute_W(v, nc, N)`
> Compute initial W vector for linear fit.
> **Called by:** braindance/utils/rt_linear_art_removal.py:118 (named-call hint); braindance/utils/rt_linear_art_removal.py:192 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_linear_art_removal.py:41`
### `compute_W_rec(Wp, N, nc, v, outgoing)`
> Compute W recursively using previous W for linear fit.
> **Called by:** braindance/utils/rt_linear_art_removal.py:154 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_linear_art_removal.py:56`
### `compute_a(S, W, a)`
> Compute linear fit coefficients.
> **Called by:** braindance/utils/rt_linear_art_removal.py:158 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_linear_art_removal.py:72`
### `shift_buffer(buffer, new_val)`
> Shift values in buffer left and add new value at the end.
> **Called by:** braindance/utils/rt_linear_art_removal.py:136 (named-call hint); braindance/utils/rt_linear_art_removal.py:179 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_linear_art_removal.py:82`
### `process_channel_minibatch_numba(channels, frames_batch, clean_values, artifacts, spikes, artifact_width, remove_frames_before, states, init_inds, moving_means, need_resets, spike_flags, buffers, W_vectors, Wp_vectors, S_matrices, min_val, max_val, spike_thresh_min, spike_thresh_max)`
> Process a minibatch of frames for multiple channels in parallel using Numba. Simplified for linear fit.
> **Called by:** braindance/utils/rt_linear_art_removal.py:297 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_linear_art_removal.py:89`
### `compute_W_vectorized(v, nc, N, W_out)`
> Vectorized W computation with pre-allocated output.
> **Called by:** braindance/utils/rt_linear_art_removal.py:414 (named-call hint); braindance/utils/rt_linear_art_removal.py:484 (named-call hint); braindance/utils/rt_linear_art_removal.py:530 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_linear_art_removal.py:347`
### `compute_W_rec_optimized(Wp, N, nc, v, W_out, outgoing)`
> Optimized recursive W computation.
> **Called by:** braindance/utils/rt_linear_art_removal.py:449 (named-call hint); braindance/utils/rt_linear_art_removal.py:534 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_linear_art_removal.py:360`
### `compute_linear_fit_inline(S_inv_00, S_inv_01, S_inv_11, W0, W1)`
> Inline computation of linear fit coefficients using pre-computed S inverse.
> **Called by:** braindance/utils/rt_linear_art_removal.py:452 (named-call hint); braindance/utils/rt_linear_art_removal.py:535 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_linear_art_removal.py:371`
### `process_channels_optimized(channels_data, clean_out, artifacts_out, spikes_out, states, init_inds, moving_means, need_resets, spike_flags, buffers, W_current, W_prev, S_inv_params, min_val, max_val, spike_thresh_min, spike_thresh_max, artifact_width, N)`
> Optimized processing with better memory access patterns.
> **Called by:** braindance/utils/rt_linear_art_removal.py:690 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_linear_art_removal.py:381`
### `process_saturation_linear(frames, clean, artifact, spikes, states, init_inds, rail_counts, spike_flags, buffers, W_current, W_prev, S, rail_min, rail_max, spike_min, spike_max, N)`
> Track rail-invalid samples and resume fitting only on valid windows.
> **Called by:** braindance/utils/rt_linear_art_removal.py:682 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/rt_linear_art_removal.py:495`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| N=60,min_val=-100,max_val=100,spike_thresh=[-3.5,20]; optimized n_threads and blanking_mode/rail_min/rail_max; artifact_width=60. |

## Data Shapes
- Input (n_channels,n_frames); output clean/artifact float64 and spike bool arrays; optimized returns copies.

## Notes
- Default spike threshold comparison requires clean<-3.5 and clean>20, so default cannot flag spikes.
- warmup advances persistent filter state; optimized n_threads changes Numba thread count.
- Saturation mode uses NaNs for startup/rail windows; legacy mode zeros startup and recovery; remove_frames_before unused.
- __main__ runs performance/output-comparison benchmark.
