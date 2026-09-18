# thresh.py

**Path:** `braindance/core/spikesorter/thresh.py`
**Module:** `braindance.core.spikesorter.thresh`
**Feature Area:** `Spike Sorting`
**Entry point:** no — library or imported component

## Overview
Implements a per-channel streaming voltage threshold sorter with periodically refreshed noise estimates. It median-centers incoming frames and emits channel identities and millisecond times for local maxima below a negative threshold.

## Connections
- **Uses:** `Sort` from `braindance.core.spikesorter.sort` — imports (static evidence).
- **Shared data:** Extends Sort for offline execution.

## Dependencies
- `braindance.core.spikesorter.sort.Sort` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### ThreshSort(Sort)
> Use threshold crossings (such as 5RMS) for spike sorting Instead of sequences' spikes, get electrodes' spikes
**Source:** `braindance/core/spikesorter/thresh.py:5`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, num_elecs, thresh, samp_freq, noise_ms=50, dtype='float32')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `cache_frame` | inferred at runtime | `2` |
| `dtype` | inferred at runtime | `dtype` |
| `latest_frame` | inferred at runtime | `env.latest_frame + 1` |
| `thresh` | inferred at runtime | `None` |
| `total_noise_frames` | inferred at runtime | `round(noise_ms * samp_freq)` |
| `traces_cache` | inferred at runtime | `np.zeros((self.total_noise_frames + 2, num_elecs), dtype=dtype)` |
| `x_thresh` | inferred at runtime | `thresh` |
**Methods:**
#### `reset(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/thresh.py:34`
#### `running_sort(self, obs, env=None)`
> Update rolling voltage/noise caches and detect threshold events.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/thresh.py:39`
#### `sort_chunk(self, chunk, spike_times_frame_offset=0)`
> Find local maxima below each channel threshold and convert frame positions to milliseconds.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/thresh.py:70`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| noise_ms sets noise-estimation window; x_thresh scales channel standard deviations. |

## Data Shapes
- Input: frame-by-channel voltage matrix.
- Output: two-column detections containing channel index and time in milliseconds.

## Notes
- Threshold refresh uses exact cache-boundary equality, so chunk sizes must align with the noise window.
- Peak comparisons select local maxima despite testing a negative threshold.
