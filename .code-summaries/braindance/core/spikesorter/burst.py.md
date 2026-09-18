# burst.py

**Path:** `braindance/core/spikesorter/burst.py`
**Module:** `braindance.core.spikesorter.burst`
**Feature Area:** `Spike Sorting`
**Entry point:** no — library or imported component

## Overview
Provides two population-burst detectors that wrap an existing spike sorter. BurstDetector thresholds spike counts in a rolling window, while SmoothBurstDetector calibrates and applies a causal half-Gaussian population-rate predictor.

## Connections
- **Uses:** `save_traces` from `braindance.core.spikesorter.rt_sort` — imports (static evidence).
- **Uses:** `Sort` from `braindance.core.spikesorter.sort` — imports (static evidence).
- **Shared data:** Wraps `Sort` or another sorter and calls its `running_sort`/`sort_offline` methods.
- **Shared data:** SmoothBurstDetector uses `spikelab.SpikeData.raster` and `scipy.signal.find_peaks` during calibration.

## Dependencies
- `braindance.core.spikesorter.rt_sort.save_traces` — intra-repo import; source import evidence.
- `braindance.core.spikesorter.sort.Sort` — intra-repo import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `scipy.signal.find_peaks` — external or unresolved local import; source import evidence.
- `spikelab.SpikeData` — external or unresolved local import; source import evidence.
- `tqdm.tqdm` — external or unresolved local import; source import evidence.

## Classes
### BurstDetector(Sort)
> Detect population bursts by thresholding the number of sorted spikes in a rolling time window.
**Source:** `braindance/core/spikesorter/burst.py:14`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, sorter, burst_window_ms=200, burst_rms_thresh=3, min_ms_between_bursts=800)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `burst_rms_thresh` | inferred at runtime | `burst_rms_thresh` |
| `burst_window_frames` | inferred at runtime | `burst_window_ms * self.samp_freq` |
| `calibrate` | inferred at runtime | `False` |
| `last_burst` | inferred at runtime | `-np.inf` |
| `min_burst_spikes` | inferred at runtime | `None` |
| `min_ms_between_bursts` | inferred at runtime | `min_ms_between_bursts` |
| `min_num_obs` | inferred at runtime | `None` |
| `num_window_spikes` | inferred at runtime | `0` |
| `sorter` | inferred at runtime | `sorter` |
| `spike_counts` | inferred at runtime | `[]` |
| `window_spike_counts` | inferred at runtime | `[]` |
**Methods:**
#### `reset(self)`
> Clear the rolling spike-count window, elapsed-time clock, threshold-crossing state, and saved spike trains before a new stream.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/burst.py:43`
#### `sort_offline(self, recording, buffer_size, inter_path=None, recording_window_ms=None, reset=True, verbose=False, calibrate=False)`
> Run the wrapped sorter over a recording and optionally calibrate an RMS-derived burst-count threshold.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/burst.py:51`
#### `running_sort(self, obs)`
> Update the rolling spike count for one observation and emit a burst event when threshold and refractory rules pass.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/burst.py:117`
### SmoothBurstDetector()
> Predict bursts from a causal half-Gaussian-smoothed population firing rate.
**Source:** `braindance/core/spikesorter/burst.py:153`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, sorter, rms_peak_threshold=3, rms_trough_threshold=1, future_bins=25, ibi_threshold_ms=800, burst_duration_thresh_f=0.1)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `bin_size_ms` | inferred at runtime | `round(self.buffer_size / self.samp_freq)` |
| `buffer_size` | inferred at runtime | `sorter.buffer_size` |
| `burst_duration_thresh_f` | inferred at runtime | `burst_duration_thresh_f` |
| `calibrate` | inferred at runtime | `True` |
| `future_bins` | inferred at runtime | `future_bins` |
| `half_gaussian` | inferred at runtime | `np.exp(-(x / future_bins) ** 2 / 2) * norm` |
| `ibi_threshold_ms` | inferred at runtime | `ibi_threshold_ms` |
| `last_burst` | inferred at runtime | `-np.inf` |
| `pop_rate_cache` | inferred at runtime | `[]` |
| `pop_rate_threshold` | inferred at runtime | `None` |
| `remaining_burst_time_ms` | inferred at runtime | `None` |
| `rms_peak_threshold` | inferred at runtime | `rms_peak_threshold` |
| `rms_trough_threshold` | inferred at runtime | `rms_trough_threshold` |
| `samp_freq` | inferred at runtime | `sorter.samp_freq` |
| `sorter` | inferred at runtime | `sorter` |
**Methods:**
#### `reset(self)`
> Clear the sorter, causal smoothing cache, elapsed-time clock, current burst flag, and accumulated spike trains for a new prediction run.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/burst.py:207`
#### `offline_predict(self, recording, inter_path=None, recording_window_ms=None, reset=True, verbose=False, calibrate=False, plot=False)`
> Calibrate rate and duration thresholds and return detected burst times from a recording or spike trains.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/burst.py:212`
#### `running_predict(self, obs)`
> Sort one observation, update the causal rate cache, and report whether a burst is predicted.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/burst.py:308`
#### `is_bursting(self)`
> Report whether the current sorter time remains inside the calibrated burst duration.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/burst.py:343`
#### `smooth_half_gaussian(signal, half_gaussian_size)`
> Apply causal half-Gaussian convolution to a one-dimensional signal.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/burst.py:350`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `BurstDetector(burst_window_ms=200, burst_rms_thresh=3, min_ms_between_bursts=800)` controls the count window, RMS multiple, and refractory interval. |
| `SmoothBurstDetector(rms_peak_threshold=3, rms_trough_threshold=1, future_bins=25, ibi_threshold_ms=800, burst_duration_thresh_f=0.1)` controls calibration and online prediction. |

## Data Shapes
- Wrapped sorters return detections as `(sequence_index, spike_time_ms)` pairs; BurstDetector exposes one synthetic unit.
- SmoothBurstDetector accepts spike-train collections or recordings and reduces a SpikeData raster to a one-dimensional population-rate trace.

## Notes
- Both detectors require offline calibration before online use; otherwise their thresholds remain `None`.
- BurstDetector converts milliseconds to frames by multiplication with `samp_freq`, whose convention here is frames per millisecond.
