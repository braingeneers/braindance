# ultra_optimized_latency_helper.py

**Path:** `braindance/utils/data_manager/utils/analysis/ultra_optimized_latency_helper.py`
**Module:** `braindance.utils.data_manager.utils.analysis.ultra_optimized_latency_helper`
**Feature Area:** `Neural Analysis`
**Entry point:** no — library or imported component

## Overview
Computes stimulation-aligned per-neuron PSTHs in electrode batches and statistically validates evoked electrode-neuron pairs. It uses 2 ms bins, memory-aware batching, response/baseline rate ratios, and a one-sided Mann-Whitney U test; reported latency is the response-window peak bin.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.analysis.burst_latency_analyzer` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.analysis.latency_helper` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.plotting.spatial_latency` — import consumer hint; not a proven runtime call.
- **Shared data:** Consumes SpikeLab SpikeData and pandas stimulation logs; called by higher-level latency helpers and Recording analysis APIs.

## Dependencies
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.
- `psutil` — external or unresolved local import; source import evidence.
- `scipy.ndimage` — external or unresolved local import; source import evidence.
- `scipy.signal.find_peaks` — external or unresolved local import; source import evidence.
- `scipy.stats` — external or unresolved local import; source import evidence.
- `scipy.stats.mannwhitneyu` — external or unresolved local import; source import evidence.
- `spikelab.SpikeData` — external or unresolved local import; source import evidence.

## Classes
### UltraOptimizedLatencyHelper()
> Batch stimulation-aligned PSTH calculation and statistical validation for evoked neural responses.
**Source:** `braindance/utils/data_manager/utils/analysis/ultra_optimized_latency_helper.py:40`
**Kind:** class. **Instantiated by:** braindance/utils/data_manager/utils/analysis/burst_latency_analyzer.py:82 (named-call hint); braindance/utils/data_manager/utils/analysis/latency_helper.py:47 (named-call hint); braindance/utils/data_manager/utils/analysis/latency_helper.py:80 (named-call hint); braindance/utils/data_manager/utils/plotting/spatial_latency.py:64 (named-call hint)
**Constructor:** `__init__(self, default_window_ms: float=100.0, default_method: str='psth_only', immediate_window: Tuple[float, float]=(0.0, 25.0), late_window: Tuple[float, float]=(25.0, 100.0), memory_usage_target: float=0.85, verbose: bool=True)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `default_method` | inferred at runtime | `default_method` |
| `default_window_ms` | inferred at runtime | `default_window_ms` |
| `immediate_window` | inferred at runtime | `immediate_window` |
| `late_window` | inferred at runtime | `late_window` |
| `memory_usage_target` | inferred at runtime | `memory_usage_target` |
| `verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `calculate_latencies_by_electrode_batched(self, spike_data: SpikeData, stim_log: pd.DataFrame, window_ms: Optional[float]=None, method: Optional[str]=None, batch_size: Optional[int]=None, use_time_mod: bool=True, **kwargs) -> Dict[int, Dict[int, Dict[str, Any]]]`
> Group stimulations, choose a batch size, and compute every electrode-neuron PSTH profile.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/ultra_optimized_latency_helper.py:81`
#### `_calculate_ultra_batch_size(self, electrode_groups: Dict[int, np.ndarray], spike_data: SpikeData, window_ms: float) -> int`
> Calculate aggressive batch size for maximum speed.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/ultra_optimized_latency_helper.py:154`
#### `_ultra_process_electrode_batch(self, spike_data: SpikeData, electrode_groups: Dict[int, np.ndarray], window_ms: float, method: str, **kwargs) -> Dict[int, Dict[int, Dict[str, Any]]]`
> Ultra-optimized batch processing.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/ultra_optimized_latency_helper.py:196`
#### `_ultra_calculate_batch_psth(self, spike_data: SpikeData, all_stim_times: np.ndarray, electrode_mapping: np.ndarray, window_ms: float, **kwargs) -> Dict[int, Dict[str, Any]]`
> Ultra-optimized PSTH calculation with speed improvements from optimized2.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/ultra_optimized_latency_helper.py:233`
#### `_slice_batch_results_ultra(self, batch_psth_data: Dict[int, Dict[str, Any]], electrode_groups: Dict[int, np.ndarray], spike_data=None, **kwargs) -> Dict[int, Dict[int, Dict[str, Any]]]`
> Slice batch results with preserved accuracy for onset detection.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/ultra_optimized_latency_helper.py:329`
#### `_analyze_electrode_psth_ultra(self, mean_psth: np.ndarray, time_bins: np.ndarray, psth_trials: np.ndarray, spike_data=None, stim_times=None, neuron_idx=None) -> Dict[str, Any]`
> Package PSTH data for downstream statistical validation.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/ultra_optimized_latency_helper.py:378`
#### `_prepare_batch_stimulations(self, electrode_groups: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]`
> Prepare batch stimulations.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/ultra_optimized_latency_helper.py:416`
#### `group_stimulations_by_electrode(self, stim_log: pd.DataFrame, use_time_mod: bool=True) -> Dict[int, np.ndarray]`
> Parse stimulation electrode fields and return sorted millisecond times per electrode.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/ultra_optimized_latency_helper.py:429`
#### `calculate_latencies_with_statistical_validation(self, spike_data: SpikeData, stim_log: pd.DataFrame, min_response_ratio: float=1.5, max_p_value: float=0.0001, baseline_window: Tuple[float, float]=(-100, 0), response_window: Tuple[float, float]=(0, 100), use_time_mod: bool=True, **kwargs) -> Dict[str, Dict[str, Any]]`
> Compute PSTH candidates and retain pairs that pass rate-ratio and Mann-Whitney criteria.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/ultra_optimized_latency_helper.py:512`
#### `_validate_candidates_from_psth_data(self, candidate_results: Dict[int, Dict[int, Dict[str, Any]]], min_response_ratio: float, max_p_value: float, baseline_window: Tuple[float, float], response_window: Tuple[float, float]) -> Dict[str, Dict[str, Any]]`
> Derive per-trial baseline/response counts, test them, and package significant pairs with peak latency and PSTH data.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/ultra_optimized_latency_helper.py:587`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Constructor defaults: 100 ms half-window, `psth_only`, immediate 0-25 ms, late 25-100 ms, memory target 0.85, verbose logging. |
| Validation defaults: response ratio >=1.5, p <=0.0001, baseline -100..0 ms, response 0..100 ms, and `time_mod` preference. |

## Data Shapes
- Per-neuron trial PSTHs are int32 arrays shaped stimulations x 2-ms bins; mean PSTH/time-bin centers are one-dimensional.
- Candidate results are nested electrode -> neuron -> profile; validated results are keyed `electrode_<id>_neuron_<idx>`.
- Stim-log times are read in seconds and converted to milliseconds; multi-electrode entries may be strings, lists, arrays, or scalars.

## Notes
- Auto batch sizing targets 90% of currently available memory despite the constructor's stored 85% target.
- All electrode-neuron PSTH profiles are candidates; statistical validation is the actual responsiveness gate.
- Onset detection was intentionally removed in favor of peak latency because a consecutive-bin rule rejected sharp responses.
