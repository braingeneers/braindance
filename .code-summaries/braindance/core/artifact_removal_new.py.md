# artifact_removal_new.py

**Path:** `braindance/core/artifact_removal_new.py`
**Module:** `braindance.core.artifact_removal_new`
**Feature Area:** `Artifact Removal`
**Entry point:** no — library or imported component

## Overview
Adds saturation blanking and reinitialization to recursive cubic trace fitting. Multi-channel fitting uses changes in the across-channel mean to detect shared artifacts.

## Connections
None

## Dependencies
- `numba.jit` — external or unresolved local import; source import evidence.
- `numba.njit` — external or unresolved local import; source import evidence.
- `numba.prange` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `scipy.signal.savgol_filter` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `cubic_fit5(v, N=60, nc_start=60, mean_arr=None, mean_std=None)`
> unclear — see source
> **Called by:** braindance/core/artifact_removal_new.py:205 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new.py:5`
### `mean_numba(a)`
> unclear — see source
> **Called by:** braindance/core/artifact_removal_new.py:200 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new.py:187`
### `cubic_fit2d(data, N=60, nc_start=60)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new.py:197`
### `fast_factorial(n)`
> unclear — see source
> **Called by:** braindance/core/artifact_removal_new.py:298 (named-call hint); braindance/core/artifact_removal_new.py:299 (named-call hint); braindance/core/artifact_removal_new.py:55 (named-call hint); braindance/core/artifact_removal_new.py:56 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new.py:233`
### `fast_median(a)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new.py:239`
### `fast_mmean(q, frame)`
> unclear — see source
> **Called by:** braindance/core/artifact_removal_new.py:167 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new.py:243`
### `_compute_T(N, nc)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new.py:253`
### `_compute_S(T)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new.py:261`
### `_compute_W(v, nc, N)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new.py:269`
### `_compute_a(S, W)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new.py:282`
### `_compute_W_rec(W, Wp, v, nc, N)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new.py:293`
### `savgol_filter(np_data, window_length=60, polyorder=3)`
> unclear — see source
> **Called by:** braindance/core/artifact_removal_new.py:402 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new.py:398`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| N=60; nc_start=60; optional mean_arr/mean_std |

## Data Shapes
- cubic_fit2d expects (channels,samples) and returns cleaned and artifact arrays.

## Notes
- savgol_filter wrapper shadows its imported SciPy function and recursively calls itself.
- Saturation triggers a fixed 20-frame blanking period and additional nc_start-frame reset gap.
