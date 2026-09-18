# artifact_removal_new2.py

**Path:** `braindance/core/artifact_removal_new2.py`
**Module:** `braindance.core.artifact_removal_new2`
**Feature Area:** `Artifact Removal`
**Entry point:** no — library or imported component

## Overview
Provides configurable artifact blanking around recursive cubic baseline removal. Multi-channel processing detects mean-signal jumps or uses an override artifact signal and optionally returns artifact estimates or recovery markers.

## Connections
- **Used by:** `braindance.core.phases_analysis_2` — import consumer hint; not a proven runtime call.

## Dependencies
- `numba.jit` — external or unresolved local import; source import evidence.
- `numba.njit` — external or unresolved local import; source import evidence.
- `numba.prange` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `cubic_fit5(v, N=60, nc_start=60, mean_arr=None, mean_std=None, artifact_width=50, remove_frames_before=8, return_counts=False)`
> unclear — see source
> **Called by:** braindance/core/artifact_removal_new2.py:238 (named-call hint); braindance/core/artifact_removal_new2.py:241 (named-call hint); braindance/core/artifact_removal_new2.py:245 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new2.py:5`
### `mean_numba(a)`
> unclear — see source
> **Called by:** braindance/core/artifact_removal_new2.py:219 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new2.py:201`
### `cubic_fit2d(data, N=60, nc_start=60, return_artifacts=False, artifact_width=50, remove_frames_before=8, n_stds=2, return_artifact_count=False, art_array_override=None, threshold=None)`
> Determine shared artifact threshold and fit channel rows in parallel.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new2.py:211`
### `fast_factorial(n)`
> unclear — see source
> **Called by:** braindance/core/artifact_removal_new2.py:81 (named-call hint); braindance/core/artifact_removal_new2.py:82 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new2.py:266`
### `fast_median(a)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new2.py:272`
### `fast_mmean(q, frame)`
> unclear — see source
> **Called by:** braindance/core/artifact_removal_new2.py:182 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/artifact_removal_new2.py:276`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| N, nc_start, artifact_width, remove_frames_before, n_stds, threshold, art_array_override |

## Data Shapes
- Input (channels,samples); output (cleaned float32 array, artifact array or None).
- return_artifact_count marks recovery endpoints per channel rather than returning a scalar count.

## Notes
- return_artifacts and return_artifact_count are mutually exclusive.
