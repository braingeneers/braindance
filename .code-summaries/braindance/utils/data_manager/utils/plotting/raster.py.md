# raster.py

**Path:** `braindance/utils/data_manager/utils/plotting/raster.py`
**Module:** `braindance.utils.data_manager.utils.plotting.raster`
**Feature Area:** `Visualization`
**Entry point:** no — library or imported component

## Overview
Plots neuron spike rasters with optional firing-rate ordering and population-rate overlays, stimulation markers, selectable corrected or raw stimulation timestamps, unit conversion, and Styler export.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.plotting.accessor` — import consumer hint; not a proven runtime call.
- **Uses:** `Recording` from `braindance.utils.data_manager.utils.data_loading.recording` — imports (static evidence).
- **Uses:** `Styler` from `braindance.utils.data_manager.utils.plotting.plot_styler` — imports (static evidence).
- **Shared data:** PlotAccessor.raster_with_pop forwards rec.spikes/stim_log; Styler handles size and export.

## Dependencies
- `braindance.utils.data_manager.utils.data_loading.recording.Recording` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.plot_styler.Styler` — intra-repo import; source import evidence.
- `matplotlib.lines.Line2D` — external or unresolved local import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `plot_raster_with_pop(spike_data, title: Optional[str]=None, time_window: Optional[Tuple[float, float]]=None, y_lim: Optional[Tuple[float, float]]=None, styler: Optional['Styler']=None, width_pt: Optional[float]=None, height_pt: Optional[float]=None, size_preset: str='single', save_path: Optional[str]=None, filename: Optional[str]=None, smoothing_window: float=100, sort_by_fr: bool=False, time_unit: str='seconds', pop_rate_unit: str='Hz', raster_color: str='black', stim_log=None, show_pop_rate: bool=True, show_stim_markers: bool=False, stim_marker_style: str='line', stim_color: Optional[str]=None, stim_alpha: float=0.3, stim_time_col: str='auto', show: bool=True) -> Tuple[plt.Figure, Tuple[plt.Axes, Optional[plt.Axes]]]`
> Plot a styled raster plot with population firing rate overlay.
> **Called by:** braindance/utils/data_manager/utils/plotting/accessor.py:86 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/raster.py:16`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| smoothing_window=100 ms sets population histogram bin width; time_unit='seconds'\|'minutes'; pop_rate_unit='Hz'\|'kHz'; stim_time_col='auto' prefers time_mod then time; show_pop_rate/show_stim_markers control overlays. |

## Data Shapes
- SpikeData exposes idces_times/train/length/N; stim_log times are seconds; returns (Figure, (raster Axes, population Axes or None)).

## Notes
- Stimulation markers default to corrected artifact-aligned time_mod when available; explicit stim_time_col="time" selects the raw software clock.
- Population rate sums all neurons rather than averaging; time_window is interpreted in the selected displayed unit.
- Default Styler is explicitly journal="draft".
