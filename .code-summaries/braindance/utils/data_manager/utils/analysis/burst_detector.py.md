# burst_detector.py

**Path:** `braindance/utils/data_manager/utils/analysis/burst_detector.py`
**Module:** `braindance.utils.data_manager.utils.analysis.burst_detector`
**Feature Area:** `Neural Analysis`
**Entry point:** no — library or imported component

## Overview
Detects population bursts by smoothing binned spike counts, finding peaks, and extending threshold-based edges. Computes neuron burst involvement, rigid-backbone membership, and optional burst summary metrics with result-cache support.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.analysis.burst_latency_analyzer` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.data_loading.recording` — import consumer hint; not a proven runtime call.
- **Shared data:** Recording.detect_bursts creates BurstDetector; BurstLatencyAnalyzer uses peaks; ResultsCache persists results.

## Dependencies
- `numpy` — external or unresolved local import; source import evidence.
- `scipy.signal.find_peaks` — external or unresolved local import; source import evidence.
- `scipy.stats.variation` — external or unresolved local import; source import evidence.

## Classes
### BurstDetector()
> A modular class for detecting and analyzing bursts in neural spike data.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:14`
**Kind:** class. **Instantiated by:** braindance/utils/data_manager/utils/analysis/burst_latency_analyzer.py:76 (named-call hint); braindance/utils/data_manager/utils/data_loading/recording.py:2447 (named-call hint)
**Constructor:** `__init__(self, spike_data, bin_size: float=1.0, smoothing_window: int=50, burst_detection_params: Optional[Dict]=None, burst_edge_params: Optional[Dict]=None, backbone_threshold: float=0.9, compute_all_metrics: bool=True)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_population_activity` | inferred at runtime | `None` |
| `_results` | inferred at runtime | `None` |
| `_smoothed_activity` | inferred at runtime | `None` |
| `backbone_threshold` | inferred at runtime | `backbone_threshold` |
| `bin_size` | inferred at runtime | `bin_size` |
| `burst_detection_params` | inferred at runtime | `burst_detection_params or {'baseline_percentile': 25, 'peak_threshold_factor': 2.5, 'peak_distance': 200, 'peak_prominence': 0.5}` |
| `burst_edge_params` | inferred at runtime | `burst_edge_params or {'edge_threshold_factor': 0.2, 'min_burst_width': 20, 'max_burst_width': 500}` |
| `compute_all_metrics` | inferred at runtime | `compute_all_metrics` |
| `smoothing_window` | inferred at runtime | `smoothing_window` |
| `spike_data` | inferred at runtime | `spike_data` |
**Methods:**
#### `_get_cache_params(self) -> Dict`
> Generate parameters dict for cache key.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:79`
#### `compute_population_activity(self)`
> Compute the population activity from spike trains.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:101`
#### `smooth_population_activity(self)`
> Apply smoothing to the population activity.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:120`
#### `find_burst_peaks(self)`
> Find peaks in the smoothed population activity that correspond to bursts.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:141`
#### `find_burst_edges(self, peak_indices, peak_info)`
> Find the edges of bursts based on peak indices.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:173`
#### `calculate_burst_involvement(self, burst_edges)`
> Calculate burst involvement coefficient (BIC) for each neuron.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:226`
#### `detect_bursts(self, cache=None, use_cache: bool=True, check_s3: bool=False, force_recompute: bool=False) -> 'BurstResults'`
> Detect bursts and calculate burst metrics with automatic caching.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:276`
#### `_compute_bursts(self) -> Dict[str, Any]`
> Internal method to compute burst detection results.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:330`
#### `get_burst_times(self) -> np.ndarray`
> Get start and end times of all detected bursts.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:397`
#### `get_burst_spikes(self) -> List[List[float]]`
> Get spikes occurring during bursts for each neuron.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:411`
#### `get_summary_metrics(self) -> Dict[str, Any]`
> Get a summary of burst metrics as a dictionary.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:455`
#### `reset(self)`
> Reset all cached calculations.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:487`
### BurstResults()
> Wrapper class for burst detection results with clean dot-notation access.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:498`
**Kind:** class. **Instantiated by:** braindance/utils/data_manager/utils/analysis/burst_detector.py:320 (named-call hint); braindance/utils/data_manager/utils/analysis/burst_detector.py:324 (named-call hint); braindance/utils/data_manager/utils/analysis/burst_detector.py:328 (named-call hint)
**Constructor:** `__init__(self, results_dict: Dict[str, Any], bin_size: float)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_bin_size` | inferred at runtime | `bin_size` |
| `_results` | inferred at runtime | `results_dict` |
**Methods:**
#### `time_bins(self) -> np.ndarray`
> Array of time values in ms for each bin.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:537`
#### `smoothed_activity(self) -> np.ndarray`
> Smoothed population firing rate.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:543`
#### `burst_times(self) -> np.ndarray`
> Array of (start, end) times in ms for each burst.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:548`
#### `n_bursts(self) -> int`
> Number of detected bursts.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:553`
#### `burst_widths(self) -> Optional[np.ndarray]`
> Width of each burst in ms.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:558`
#### `burst_amplitudes(self) -> Optional[np.ndarray]`
> Amplitude (peak height) of each burst.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:563`
#### `bic_matrix(self) -> Optional[np.ndarray]`
> Burst involvement coefficient for each neuron.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:568`
#### `backbone_classification(self) -> Optional[Dict[str, List[int]]]`
> Dictionary with 'rigid' and 'nonrigid' neuron lists.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:573`
#### `peak_indices(self) -> np.ndarray`
> Indices of detected burst peaks.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:578`
#### `mean_burst_width(self) -> Optional[float]`
> Mean burst width in ms.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:583`
#### `std_burst_width(self) -> Optional[float]`
> Standard deviation of burst width in ms.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:588`
#### `mean_burst_amplitude(self) -> Optional[float]`
> Mean burst amplitude.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:593`
#### `std_burst_amplitude(self) -> Optional[float]`
> Standard deviation of burst amplitude.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:598`
#### `burst_frequency(self) -> Optional[float]`
> Burst frequency in Hz.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:603`
#### `mean_population_rate(self) -> Optional[float]`
> Mean population firing rate.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:608`
#### `cv_population_rate(self) -> Optional[float]`
> Coefficient of variation of population rate.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:613`
#### `n_rigid_units(self) -> Optional[int]`
> Number of rigid (backbone) neurons.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:618`
#### `__repr__(self) -> str`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_detector.py:622`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| bin_size=1ms, smoothing_window=50; peak and edge dictionaries; backbone_threshold=0.9; compute_all_metrics=True. |

## Data Shapes
- SpikeData requires length,N,idces_times(); BurstResults wraps population_rate, peak_indices/info, burst_edges, n_bursts and optional metrics.
- burst_edges shape(bursts,2); bic_matrix actually per-neuron involvement fractions; backbone_classification maps rigid/nonrigid to neuron indices.

## Notes
- Edges are bin indices but involvement/spike extraction compare directly against millisecond spike times; non-1ms bins need care.
- check_s3 is accepted but not used; cache object's own auto_download governs S3.
- Peak amplitudes include all peaks even if edge validation rejects some.
