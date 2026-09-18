# artifact_filters.py

**Path:** `braindance/utils/artifact_filters.py`
**Module:** `braindance.utils.artifact_filters`
**Feature Area:** `Artifact Removal`
**Entry point:** no — library or imported component

## Overview
Provides stateful linear and cubic centered polynomial baseline removal with excursion and rail exclusions. Numba kernels update rolling moments and return cleaned samples, baseline estimates, and explicit validity masks across acquisition blocks.

## Connections
None

## Dependencies
- `numba.njit` — external or unresolved local import; source import evidence.
- `numba.prange` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### FilterResult()
> unclear — see source
**Source:** `braindance/utils/artifact_filters.py:14`
**Kind:** class. **Instantiated by:** braindance/utils/artifact_filters.py:185 (named-call hint)
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
None
### ArtifactRemover(ABC)
> Common API; output column t describes input column t-delay_samples.
**Source:** `braindance/utils/artifact_filters.py:110`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, n_channels, half_window=60, excursion_threshold=100.0, recovery_samples=60, rail_min=-np.inf, rail_max=np.inf, *, parallel=True, rebase_interval=256)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `anchors` | inferred at runtime | `np.zeros(self.n_channels)` |
| `bad` | inferred at runtime | `np.zeros(self.buffer.shape, dtype=bool)` |
| `bad_counts` | inferred at runtime | `np.zeros(self.n_channels, dtype=np.int64)` |
| `buffer` | inferred at runtime | `np.zeros((self.n_channels, len(self.weights)))` |
| `delay_samples` | inferred at runtime | `half_window` |
| `excursion_threshold` | inferred at runtime | `float(excursion_threshold)` |
| `holds` | inferred at runtime | `np.zeros(self.n_channels, dtype=np.int64)` |
| `means` | inferred at runtime | `np.zeros(self.n_channels)` |
| `moments` | inferred at runtime | `np.zeros((self.n_channels, 3))` |
| `n_channels` | inferred at runtime | `n_channels` |
| `parallel` | inferred at runtime | `bool(parallel)` |
| `position` | inferred at runtime | `0` |
| `rebase_interval` | inferred at runtime | `rebase_interval` |
| `recovery_samples` | inferred at runtime | `recovery_samples` |
| `seen` | inferred at runtime | `0` |
| `weights` | inferred at runtime | `np.ascontiguousarray(np.linalg.pinv(design)[0])` |
**Methods:**
#### `degree(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/artifact_filters.py:118`
#### `reset(self)`
> Discard recording state without changing parameters.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/artifact_filters.py:155`
#### `process(self, frames)`
> Advance filter state and return delayed clean/artifact/valid arrays.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/artifact_filters.py:167`
#### `fit_step(self, frames)`
> Alias for process; returns FilterResult, not legacy spike flags.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/artifact_filters.py:187`
#### `warmup(self)`
> Compile kernels using a disposable filter without changing current state.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/artifact_filters.py:191`
### SalpaArtifactRemover(ArtifactRemover)
> SALPA-style cubic fit with the shared conservative exclusion policy.
**Source:** `braindance/utils/artifact_filters.py:201`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
None
### LinearFitArtifactRemover(ArtifactRemover)
> Local degree-one least-squares fit with the shared exclusion policy.
**Source:** `braindance/utils/artifact_filters.py:206`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
None

## Functions
### `_process_channel(ch, x, buffer, bad, means, holds, anchors, moments, bad_counts, position, seen, threshold, recovery_samples, rail_min, rail_max, c0, c2, rebase_interval, clean, artifact, valid)`
> Update one channel; the cubic intercept needs only moments 0, 1, 2.
> **Called by:** braindance/utils/artifact_filters.py:105 (named-call hint); braindance/utils/artifact_filters.py:95 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/artifact_filters.py:21`
### `_process_parallel(x, buffer, bad, means, holds, anchors, moments, bad_counts, position, seen, threshold, recovery_samples, rail_min, rail_max, c0, c2, rebase_interval, clean, artifact, valid)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/artifact_filters.py:91`
### `_process_serial(x, buffer, bad, means, holds, anchors, moments, bad_counts, position, seen, threshold, recovery_samples, rail_min, rail_max, c0, c2, rebase_interval, clean, artifact, valid)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/artifact_filters.py:101`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| n_channels, half_window=60, excursion_threshold=100, recovery_samples=60, rail_min/max, parallel=True, rebase_interval=256. |

## Data Shapes
- Input float-compatible (n_channels,n_samples); FilterResult.clean/artifact float64 and valid bool have matching shape.

## Notes
- Output column t represents input t-half_window; startup/excluded outputs are NaN and invalid.
- Keep instance across blocks; reset clears state; cubic filter is SALPA-style, not the original SALPA state machine.
