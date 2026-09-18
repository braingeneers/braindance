# latency_helper.py

**Path:** `braindance/utils/data_manager/utils/analysis/latency_helper.py`
**Module:** `braindance.utils.data_manager.utils.analysis.latency_helper`
**Feature Area:** `Neural Analysis`
**Entry point:** no — library or imported component

## Overview
Exposes concise public wrappers for statistically validated stimulus responses and electrode grouping. Delegates all computation to UltraOptimizedLatencyHelper.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.data_loading.recording` — import consumer hint; not a proven runtime call.
- **Uses:** `UltraOptimizedLatencyHelper` from `braindance.utils.data_manager.utils.analysis.ultra_optimized_latency_helper` — imports (static evidence).
- **Shared data:** Recording.calculate_latencies and public data_manager imports call these wrappers; implementation lives in ultra_optimized_latency_helper.

## Dependencies
- `braindance.utils.data_manager.utils.analysis.ultra_optimized_latency_helper.UltraOptimizedLatencyHelper` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `calculate_latencies(spike_data, stim_log: pd.DataFrame, min_response_ratio: float=1.5, max_p_value: float=0.0001, baseline_window: Tuple[float, float]=(-100, 0), response_window: Tuple[float, float]=(0, 100), use_time_mod: bool=True, verbose: bool=False, **kwargs) -> Dict[str, Dict[str, Any]]`
> Detect stimulus-evoked neural responses with statistical validation.
> **Called by:** braindance/utils/data_manager/utils/data_loading/recording.py:2612 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/latency_helper.py:15`
### `group_stimulations_by_electrode(stim_log: pd.DataFrame, use_time_mod: bool=True) -> Dict[int, np.ndarray]`
> Group stimulation times by electrode ID.
> **Called by:** braindance/utils/data_manager/utils/data_loading/recording.py:2658 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/latency_helper.py:63`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| min_response_ratio=1.5, max_p_value=0.0001, baseline_window=(-100,0), response_window=(0,100), use_time_mod=True. |

## Data Shapes
- SpikeData plus stimulation DataFrame → evoked-pair dict; grouping → dict[electrode_id, stimulus_times_ms].

## Notes
None
