# artifact_removal.py

**Path:** `braindance/core/artifact_removal.py`
**Module:** `braindance.core.artifact_removal`
**Feature Area:** `Artifact Removal`
**Entry point:** no — library or imported component

## Overview
Implements recursive cubic baseline subtraction for traces and streaming samples. Stateful saturation handling can blank samples and detect negative threshold crossings while parallel fitting processes channel rows.

## Connections
- **Used by:** `braindance.analysis.causal_connectivity` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases2` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_analysis` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_env` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.analysis_evoked` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.stim_responses` — import consumer hint; not a proven runtime call.

## Dependencies
- `numba.jit` — external or unresolved local import; source import evidence.
- `numba.njit` — external or unresolved local import; source import evidence.
- `numba.prange` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `tqdm.tqdm` — external or unresolved local import; source import evidence.

## Classes
### ArtifactRemoval()
> unclear — see source
**Source:** `braindance/core/artifact_removal.py:164`
**Kind:** class. **Instantiated by:** braindance/core/phases.py:617 (named-call hint); braindance/core/phases2.py:747 (named-call hint); braindance/core/phases_analysis.py:624 (named-call hint); braindance/core/phases_env.py:98 (named-call hint); braindance/experiments/stim_responses.py:52 (named-call hint); braindance/experiments/stim_responses.py:55 (named-call hint)
**Constructor:** `__init__(self, N, nc_start=0, min_val=-100, max_val=100, spike_thresh=[-3.5, -20])`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `N` | inferred at runtime | `N` |
| `S` | inferred at runtime | `self._compute_S(self.T)` |
| `T` | inferred at runtime | `self._compute_T(nc_start)` |
| `W` | inferred at runtime | `np.zeros(4)` |
| `Wp` | inferred at runtime | `np.zeros(4)` |
| `depeg_count` | inferred at runtime | `0` |
| `init_ind` | inferred at runtime | `0` |
| `max_val` | inferred at runtime | `max_val` |
| `min_val` | inferred at runtime | `min_val` |
| `moving_mean` | inferred at runtime | `0` |
| `nc_start` | inferred at runtime | `nc_start` |
| `prev_a` | inferred at runtime | `None` |
| `prev_nc` | inferred at runtime | `nc_start` |
| `spike` | inferred at runtime | `False` |
| `spike_thresh_max` | inferred at runtime | `spike_thresh[1]` |
| `spike_thresh_min` | inferred at runtime | `spike_thresh[0]` |
| `state` | inferred at runtime | `'init'` |
| `v` | inferred at runtime | `np.zeros(2 * N + 2)` |
**Methods:**
#### `_compute_T(self, nc)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:197`
#### `_compute_S(self, T)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:204`
#### `_compute_W(self, v, nc)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:211`
#### `_compute_W_step(self, v)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:223`
#### `_compute_W_rec(W, Wp, v, nc, N)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:236`
#### `_compute_a(S, W)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:253`
#### `fit(self, v, T, S)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/artifact_removal.py:263`
#### `fit_step(self, frame)`
> Advance initialization, fit, or depeg state and detect a new negative threshold crossing.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/artifact_removal.py:299`
#### `shift_ind(arr, val)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:383`
#### `run(self, data, return_artifacts=False, return_spikes=False, progress_bar=False, n_workers=1)`
> Runs the process for the whole data input. If data is a 1D array, it will return the cleaned data If data is a 2D array, it will run for each row
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:388`
### ArtifactRemoval2()
> unclear — see source
**Source:** `braindance/core/artifact_removal.py:447`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, N, nc_start=0, min_val=-100, max_val=100)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `N` | inferred at runtime | `N` |
| `S` | inferred at runtime | `self._compute_S(self.T)` |
| `T` | inferred at runtime | `self._comput1e_T(nc_start)` |
| `W` | inferred at runtime | `np.zeros(4)` |
| `Wp` | inferred at runtime | `np.zeros(4)` |
| `init_ind` | inferred at runtime | `0` |
| `max_val` | inferred at runtime | `100` |
| `min_val` | inferred at runtime | `-100` |
| `nc_start` | inferred at runtime | `nc_start` |
| `prev_a` | inferred at runtime | `None` |
| `prev_nc` | inferred at runtime | `nc_start` |
| `state` | inferred at runtime | `'init'` |
| `v` | inferred at runtime | `np.zeros(2 * N + 2)` |
| `v_ind` | inferred at runtime | `0` |
| `v_len` | inferred at runtime | `self.v.shape[0]` |
**Methods:**
#### `_compute_T(self, nc)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:472`
#### `_compute_S(self, T)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:479`
#### `_compute_W(self, v, nc)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:486`
#### `_compute_W_rec(Wp, v, v_ind, v_size, N)`
> v_next is v[nc+N + 1] v_prev is v[nc - N]
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:501`
#### `_compute_a(S, W)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:520`
#### `fit(self, v, T, S)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/artifact_removal.py:530`
#### `fit_step(self, frame)`
> Recursively computes the new W then solves for the artifact of the center frame when the new frame is added
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/artifact_removal.py:574`
### Timer()
> unclear — see source
**Source:** `braindance/core/artifact_removal.py:661`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, name='bob')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `end_time` | inferred at runtime | `-1` |
| `name` | inferred at runtime | `name` |
| `start_time` | inferred at runtime | `None` |
**Methods:**
#### `__enter__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/artifact_removal.py:667`
#### `__exit__(self, exc_type, exc_val, exc_tb)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/artifact_removal.py:671`

## Functions
### `cubic_fit5(v, N=60, nc_start=0)`
> unclear — see source
> **Called by:** braindance/analysis/causal_connectivity.py:87 (named-call hint); braindance/core/artifact_removal.py:120 (named-call hint); braindance/core/phases_analysis.py:663 (named-call hint); braindance/examples/streaming_workshop/analysis_evoked.py:78 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:5`
### `cubic_fit5_parallel(data, N=60, nc_start=60)`
> Apply cubic baseline fitting independently across channel rows.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:115`
### `fast_factorial(n)`
> unclear — see source
> **Called by:** braindance/core/artifact_removal.py:241 (named-call hint); braindance/core/artifact_removal.py:242 (named-call hint); braindance/core/artifact_removal.py:511 (named-call hint); braindance/core/artifact_removal.py:512 (named-call hint); braindance/core/artifact_removal.py:55 (named-call hint); braindance/core/artifact_removal.py:56 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:143`
### `fast_median(a)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:149`
### `fast_mmean(q, frame)`
> unclear — see source
> **Called by:** braindance/core/artifact_removal.py:331 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal.py:153`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Window radius N, nc_start, saturation bounds, spike thresholds |

## Data Shapes
- Batch input is (samples,) or (channels,samples); outputs cleaned traces plus optional artifacts and spike-index lists.
- fit_step returns (cleaned value, artifact estimate, new_spike).

## Notes
- ArtifactRemoval retains state across calls; ArtifactRemoval2 constructor references misspelled _comput1e_T.
