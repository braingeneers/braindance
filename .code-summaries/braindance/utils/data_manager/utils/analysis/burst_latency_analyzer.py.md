# burst_latency_analyzer.py

**Path:** `braindance/utils/data_manager/utils/analysis/burst_latency_analyzer.py`
**Module:** `braindance.utils.data_manager.utils.analysis.burst_latency_analyzer`
**Feature Area:** `Neural Analysis`
**Entry point:** no — library or imported component

## Overview
Associates population burst peaks with each electrode's stimulation times and compares evoked burst counts against prestimulus windows. Reports latency distributions, immediate/late categories, and per-electrode validation criteria with optional local caching.

## Connections
- **Uses:** `BurstDetector` from `braindance.utils.data_manager.utils.analysis.burst_detector` — imports (static evidence).
- **Uses:** `UltraOptimizedLatencyHelper` from `braindance.utils.data_manager.utils.analysis.ultra_optimized_latency_helper` — imports (static evidence).
- **Shared data:** Uses BurstDetector for peaks and UltraOptimizedLatencyHelper for stimulus grouping; ResultsCache.load_local/save_local store analysis.

## Dependencies
- `braindance.utils.data_manager.utils.analysis.burst_detector.BurstDetector` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.analysis.ultra_optimized_latency_helper.UltraOptimizedLatencyHelper` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.
- `scipy.stats.binom_test` — external or unresolved local import; source import evidence.
- `scipy.stats.binomtest` — external or unresolved local import; source import evidence.
- `scipy.stats.chi2_contingency` — external or unresolved local import; source import evidence.

## Classes
### BurstLatencyAnalyzer()
> Analyze burst latencies in response to electrical stimulation.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_latency_analyzer.py:26`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, spike_data, stim_log: pd.DataFrame, baseline_window_ms: float=500.0, analysis_window_ms: float=200.0, burst_detection_params: Optional[Dict]=None, validation_params: Optional[Dict]=None, use_time_mod: bool=True, verbose: bool=True, cache=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_burst_results` | inferred at runtime | `None` |
| `_electrode_groups` | inferred at runtime | `None` |
| `analysis_window_ms` | inferred at runtime | `analysis_window_ms` |
| `baseline_window_ms` | inferred at runtime | `baseline_window_ms` |
| `burst_detector` | inferred at runtime | `BurstDetector(spike_data, burst_detection_params=burst_detection_params)` |
| `cache` | inferred at runtime | `cache` |
| `latency_helper` | inferred at runtime | `UltraOptimizedLatencyHelper(verbose=verbose)` |
| `spike_data` | inferred at runtime | `spike_data` |
| `stim_log` | inferred at runtime | `stim_log` |
| `use_time_mod` | inferred at runtime | `use_time_mod` |
| `validation_params` | inferred at runtime | `{'probability_increase_factor': 1.5, 'max_p_value': 0.01, 'min_evoked_bursts': 3, 'max_mean_latency_ms': 100.0, **(validation_par…` |
| `verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `_get_cache_params(self) -> dict`
> Generate parameters dict for cache key.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_latency_analyzer.py:103`
#### `analyze_burst_latencies(self, use_cache: bool=True, force_recompute: bool=False) -> Dict[int, Dict[str, Any]]`
> Analyze burst latencies for all electrodes with automatic caching.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_latency_analyzer.py:114`
#### `_get_burst_peak_times(self) -> np.ndarray`
> Get burst peak times from the burst detector results.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_latency_analyzer.py:209`
#### `_analyze_electrode_burst_latencies(self, electrode_id: int, stim_times: np.ndarray, burst_times: np.ndarray) -> Optional[Dict[str, Any]]`
> Analyze burst latencies for a single electrode.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_latency_analyzer.py:234`
#### `_calculate_baseline_burst_stats(self, stim_times: np.ndarray, burst_peak_times: np.ndarray) -> Dict[str, Any]`
> Calculate baseline burst statistics using pre-stimulus periods.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_latency_analyzer.py:315`
#### `_validate_burst_responses(self, n_evoked_bursts: int, n_stimuli: int, baseline_stats: Dict[str, Any], mean_latency: float) -> Dict[str, Any]`
> Validate burst responses using combination approach.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_latency_analyzer.py:370`
#### `_classify_burst_latencies(self, burst_latencies: List[float]) -> Dict[str, Any]`
> Classify burst latencies into categories.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_latency_analyzer.py:464`
#### `get_summary_statistics(self) -> Dict[str, Any]`
> Get summary statistics across all electrodes.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/burst_latency_analyzer.py:506`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| baseline_window_ms=500, analysis_window_ms=200; validation defaults probability factor=1.5, p<0.01, >=3 bursts, mean latency<=100ms. |

## Data Shapes
- Input SpikeData and stim DataFrame; output dict[electrode_id] with burst_latencies, stimulus_burst_pairs, baseline_stats, validation, validated.

## Notes
- Counts every peak inside each window, so 'probability' may exceed one for multiple bursts.
- Cache parameter set omits burst detection parameters and use_time_mod; cache stores pickled results in an object array.
- Modern binomtest handling can fall through to failed validation because pvalue is checked on function rather than result.
