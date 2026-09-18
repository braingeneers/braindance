# artifact_removal_archive.py

**Path:** `braindance/core/artifact_removal_archive.py`
**Module:** `braindance.core.artifact_removal_archive`
**Feature Area:** `Artifact Removal`
**Entry point:** no — library or imported component

## Overview
Preserves earlier cubic-fit implementations and stateful streaming removal prototypes. Includes recurrence lookup tables, saturation recovery, and timing instrumentation for algorithm comparison.

## Connections
None

## Dependencies
- `numba.jit` — external or unresolved local import; source import evidence.
- `numba.njit` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### ArtifactRemoval()
> unclear — see source
**Source:** `braindance/core/artifact_removal_archive.py:359`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, N, nc_start=0, min_val=-100, max_val=100)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `N` | inferred at runtime | `N` |
| `S` | inferred at runtime | `self._compute_S(self.T)` |
| `T` | inferred at runtime | `self._compute_T(nc_start)` |
| `W` | inferred at runtime | `np.zeros(4)` |
| `Wp` | inferred at runtime | `np.zeros(4)` |
| `init_ind` | inferred at runtime | `0` |
| `max_val` | inferred at runtime | `100` |
| `min_val` | inferred at runtime | `-100` |
| `moving_mean` | inferred at runtime | `0` |
| `nc_start` | inferred at runtime | `nc_start` |
| `prev_a` | inferred at runtime | `None` |
| `prev_nc` | inferred at runtime | `nc_start` |
| `state` | inferred at runtime | `'init'` |
| `state_timers` | inferred at runtime | `{'init': 0, 'depeg': 0, 'fit': 0, 'init-fit': 0, 'fit-depeg': 0, 'depeg-init': 0, 'fit-shifting': 0, 'fit-median': 0}` |
| `v` | inferred at runtime | `np.zeros(2 * N + 2)` |
**Methods:**
#### `_compute_T(self, nc)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:386`
#### `_compute_S(self, T)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:393`
#### `_compute_W(self, v, nc)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:400`
#### `_compute_W_step(self, v)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:412`
#### `_compute_W_rec(Wp, v, nc, N)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:425`
#### `_compute_W_rec_precomp(Wp, v, nc, N)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:442`
#### `_compute_a(S, W, a)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:453`
#### `fit(self, v, T, S)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/artifact_removal_archive.py:463`
#### `fit_step(self, frame)`
> Recursively computes the new W then solves for the artifact of the center frame when the new frame is added
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/artifact_removal_archive.py:497`
#### `shift_ind(arr, val)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:562`
### ArtifactRemoval2()
> unclear — see source
**Source:** `braindance/core/artifact_removal_archive.py:578`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, N, nc_start=0, min_val=-100, max_val=100)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `N` | inferred at runtime | `N` |
| `S` | inferred at runtime | `self._compute_S(self.T)` |
| `T` | inferred at runtime | `self._compute_T(nc_start)` |
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
**Source:** `braindance/core/artifact_removal_archive.py:603`
#### `_compute_S(self, T)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:610`
#### `_compute_W(self, v, nc)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:617`
#### `_compute_W_rec(Wp, v, v_ind, v_size, N)`
> v_next is v[nc+N + 1] v_prev is v[nc - N]
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:632`
#### `_compute_a(S, W)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:651`
#### `fit(self, v, T, S)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/artifact_removal_archive.py:661`
#### `fit_step(self, frame)`
> Recursively computes the new W then solves for the artifact of the center frame when the new frame is added
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/artifact_removal_archive.py:705`
### Timer()
> unclear — see source
**Source:** `braindance/core/artifact_removal_archive.py:769`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, name='bob')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `name` | inferred at runtime | `name` |
| `start_time` | inferred at runtime | `None` |
**Methods:**
#### `__enter__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/artifact_removal_archive.py:774`
#### `__exit__(self, exc_type, exc_val, exc_tb)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:778`

## Functions
### `cubic_fit(v, N, nc_start=0)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:5`
### `cubic_fit4(v, N, nc_start=0)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:106`
### `cubic_fit5(v, N, nc_start=0)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:207`
### `precompute_n_to_k(N=20)`
> unclear — see source
> **Called by:** braindance/core/artifact_removal_archive.py:350 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:300`
### `precompute_np_to_k(N=20)`
> unclear — see source
> **Called by:** braindance/core/artifact_removal_archive.py:351 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:306`
### `fast_factorial(n)`
> unclear — see source
> **Called by:** braindance/core/artifact_removal_archive.py:343 (named-call hint); braindance/core/artifact_removal_archive.py:344 (named-call hint); braindance/core/artifact_removal_archive.py:430 (named-call hint); braindance/core/artifact_removal_archive.py:431 (named-call hint); braindance/core/artifact_removal_archive.py:642 (named-call hint); braindance/core/artifact_removal_archive.py:643 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:323`
### `fast_median(a)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:329`
### `fast_mmean(q, frame)`
> unclear — see source
> **Called by:** braindance/core/artifact_removal_archive.py:526 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:333`
### `precompute_num_den()`
> unclear — see source
> **Called by:** braindance/core/artifact_removal_archive.py:349 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_archive.py:338`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| N and nc_start; lookup powers precomputed for N=20 |

## Data Shapes
- Trace fit outputs cleaned and artifact arrays; streaming fit_step returns value/artifact pair.

## Notes
- Archived cubic_fit has no active return; precomputed recurrence powers remain fixed at N=20.
- Some constructor saturation bounds are replaced by hardcoded -100 and 100.
