# spatial_latency.py

**Path:** `braindance/utils/data_manager/utils/plotting/spatial_latency.py`
**Module:** `braindance.utils.data_manager.utils.plotting.spatial_latency`
**Feature Area:** `Visualization`
**Entry point:** no — library or imported component

## Overview
Plots neuron coordinates colored by supplied peak latency and sized by response ratio with a stimulation-electrode marker. Response-map and animation entry points are placeholders that print a message and return None.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.plotting.accessor` — import consumer hint; not a proven runtime call.
- **Uses:** `UltraOptimizedLatencyHelper` from `braindance.utils.data_manager.utils.analysis.ultra_optimized_latency_helper` — imports (static evidence).
- **Uses:** `Recording` from `braindance.utils.data_manager.utils.data_loading.recording` — imports (static evidence).
- **Uses:** `Styler` from `braindance.utils.data_manager.utils.plotting.plot_styler` — imports (static evidence).
- **Shared data:** PlotAccessor exposes these methods; mapping comes from analysis.Mapping and locations from Recording.

## Dependencies
- `braindance.utils.data_manager.utils.analysis.ultra_optimized_latency_helper.UltraOptimizedLatencyHelper` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.data_loading.recording.Recording` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.plot_styler.Styler` — intra-repo import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `scipy.ndimage.uniform_filter1d` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `spatial_latency(recording: 'Recording', electrode_id: Optional[int]=None, latency_results: Optional[Dict]=None, time_window: Tuple[float, float]=(0, 100), styler: Optional['Styler']=None, save_path: Optional[str]=None, filename: Optional[str]=None, show: bool=False) -> Any`
> Plot spatial distribution of onset latencies. Creates a scatter plot of neurons on the electrode array, color-coded by onset latency and sized by response strength. Args: recording: Recording object electrode_id: Stimulation electrode ID to analyze latency_results: Optional pre-computed latency results time_window: (min, max) latency range for color mapping (ms) styler: Optional Styler for plot styling save_path: Optional directory to save figur…
> **Called by:** braindance/utils/data_manager/utils/plotting/accessor.py:264 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/spatial_latency.py:20`
### `spatial_response_map(recording: 'Recording', electrode_id: int, metric: str='latency', styler: Optional['Styler']=None, save_path: Optional[str]=None, filename: Optional[str]=None, show: bool=False) -> Any`
> Plot spatial map of neural responses to stimulation. Args: recording: Recording object electrode_id: Stimulation electrode to analyze metric: What to visualize - 'latency', 'strength', or 'change' styler: Optional Styler for plot styling save_path: Optional directory to save figure filename: Optional filename show: Whether to display the plot Returns: Matplotlib figure.
> **Called by:** braindance/utils/data_manager/utils/plotting/accessor.py:286 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/spatial_latency.py:184`
### `spatial_animation(recording: 'Recording', electrode_id: int, time_window: Tuple[float, float]=(-50, 100), smoothing_ms: float=20, styler: Optional['Styler']=None, save_path: Optional[str]=None, filename: Optional[str]=None) -> Any`
> Create animation of neural response propagation. Shows how activity spreads across the electrode array over time after stimulation. Args: recording: Recording object electrode_id: Stimulation electrode ID time_window: (start, end) time window for animation (ms) smoothing_ms: Smoothing window size (ms) styler: Optional Styler for plot styling save_path: Optional directory to save animation filename: Optional filename (will add .gif extension) Ret…
> **Called by:** braindance/utils/data_manager/utils/plotting/accessor.py:307 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/spatial_latency.py:214`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| electrode_id, time_window=(0,100) ms, latency_results, save_path/filename/show; spatial_animation uses time_window=(-50,100) ms and smoothing_ms=20. |

## Data Shapes
- Recording.mapping.get_positions and spike_locations provide (x,y) coordinates; latency_results is keyed by integer neuron index; plotting returns Figure or None.

## Notes
- Automatic latency computation is unimplemented; caller must transform flat evoked-pair results to neuron-index mapping.
- Responsive-data branch references undefined onset_latencies and raises NameError.
- Default Styler is explicitly journal="draft".
