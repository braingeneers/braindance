# evoked_response.py

**Path:** `braindance/utils/data_manager/utils/plotting/evoked_response.py`
**Module:** `braindance.utils.data_manager.utils.plotting.evoked_response`
**Feature Area:** `Visualization`
**Entry point:** no — library or imported component

## Overview
Plots single-neuron trial rasters and trial-normalized PSTHs aligned to one stimulation electrode. Can annotate validated peak latency, baseline/response rates, and response statistics.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.plotting.accessor` — import consumer hint; not a proven runtime call.
- **Uses:** `Recording` from `braindance.utils.data_manager.utils.data_loading.recording` — imports (static evidence).
- **Uses:** `Styler` from `braindance.utils.data_manager.utils.plotting.plot_styler` — imports (static evidence).
- **Shared data:** PlotAccessor.evoked_raster/evoked_psth supply recording data and optional cached pair information.

## Dependencies
- `braindance.utils.data_manager.utils.data_loading.recording.Recording` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.plot_styler.Styler` — intra-repo import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `plot_evoked_raster(spike_data, stim_log, electrode_id: int, neuron_idx: int, plot_window: Tuple[float, float]=(-100, 100), max_trials: int=400, latency_data: Optional[dict]=None, styler: Optional['Styler']=None, raster_color: str='black', response_color: Optional[str]=None, save_path: Optional[str]=None, use_time_mod: bool=True, show: bool=True, **kwargs) -> plt.Figure`
> Plot evoked response raster for a neuron-electrode pair. Shows trial-by-trial spike responses aligned to stimulation onset. Each row represents one trial, with spikes plotted as markers.
> **Called by:** braindance/utils/data_manager/utils/plotting/accessor.py:204 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/utils/data_manager/utils/plotting/evoked_response.py:16`
### `plot_evoked_psth(spike_data, stim_log, electrode_id: int, neuron_idx: int, plot_window: Tuple[float, float]=(-100, 100), max_trials: int=400, latency_data: Optional[dict]=None, styler: Optional['Styler']=None, psth_color: Optional[str]=None, response_color: Optional[str]=None, psth_bins: int=75, save_path: Optional[str]=None, use_time_mod: bool=True, show: bool=True, **kwargs) -> plt.Figure`
> Plot evoked response PSTH (peri-stimulus time histogram) for a neuron-electrode pair. Shows average firing rate across trials in time bins aligned to stimulation onset.
> **Called by:** braindance/utils/data_manager/utils/plotting/accessor.py:238 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/utils/data_manager/utils/plotting/evoked_response.py:181`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| plot_window=(-100,100) ms, max_trials=400, use_time_mod=True, psth_bins=75; save_path is treated as a filename stem. |

## Data Shapes
- SpikeData.train times are milliseconds; stim_log times are seconds and electrodes may be list literals; latency_data is an evoked-pair mapping; returns a Figure.

## Notes
- Uses first max_trials stimuli and warns when corrected time_mod is unavailable.
- Writes PNG and SVG when save_path is provided; default Styler is explicitly journal="draft".
