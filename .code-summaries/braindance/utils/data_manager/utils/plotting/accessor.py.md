# accessor.py

**Path:** `braindance/utils/data_manager/utils/plotting/accessor.py`
**Module:** `braindance.utils.data_manager.utils.plotting.accessor`
**Feature Area:** `Visualization`
**Entry point:** no — library or imported component

## Overview
Provides the `rec.pl` plotting facade bound to a Recording. It lazily creates a default Styler and forwards recording data and user options to raster, population, evoked-response, and spatial plotting functions.

## Connections
- **Uses:** `Recording` from `braindance.utils.data_manager.utils.data_loading.recording` — imports (static evidence).
- **Uses:** `plot_evoked_psth` from `braindance.utils.data_manager.utils.plotting.evoked_response` — imports (static evidence).
- **Uses:** `plot_evoked_raster` from `braindance.utils.data_manager.utils.plotting.evoked_response` — imports (static evidence).
- **Uses:** `Styler` from `braindance.utils.data_manager.utils.plotting.plot_styler` — imports (static evidence).
- **Uses:** `plot_firing_rate_histogram` from `braindance.utils.data_manager.utils.plotting.population` — imports (static evidence).
- **Uses:** `plot_sttc_matrix` from `braindance.utils.data_manager.utils.plotting.population` — imports (static evidence).
- **Uses:** `plot_raster_with_pop` from `braindance.utils.data_manager.utils.plotting.raster` — imports (static evidence).
- **Uses:** `spatial_animation` from `braindance.utils.data_manager.utils.plotting.spatial_latency` — imports (static evidence).
- **Uses:** `spatial_latency` from `braindance.utils.data_manager.utils.plotting.spatial_latency` — imports (static evidence).
- **Uses:** `spatial_response_map` from `braindance.utils.data_manager.utils.plotting.spatial_latency` — imports (static evidence).
- **Shared data:** Recording.pl constructs this accessor; methods delegate to plotting.raster, evoked_response, population, spatial_latency, and plot_styler modules.

## Dependencies
- `braindance.utils.data_manager.utils.data_loading.recording.Recording` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.evoked_response.plot_evoked_psth` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.evoked_response.plot_evoked_raster` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.plot_styler.Styler` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.population.plot_firing_rate_histogram` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.population.plot_sttc_matrix` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.raster.plot_raster_with_pop` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.spatial_latency.spatial_animation` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.spatial_latency.spatial_latency` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.spatial_latency.spatial_response_map` — intra-repo import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### PlotAccessor()
> Expose recording-aware plotting methods through `rec.pl`.
**Source:** `braindance/utils/data_manager/utils/plotting/accessor.py:23`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, recording: 'Recording')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_recording` | inferred at runtime | `recording` |
| `_styler` | inferred at runtime | `None` |
**Methods:**
#### `styler(self) -> 'Styler'`
> Get or create the default styler.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/plotting/accessor.py:42`
#### `styler(self, value: 'Styler')`
> Set a custom styler.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/plotting/accessor.py:50`
#### `raster_with_pop(self, title: Optional[str]=None, time_window: Optional[Tuple[float, float]]=None, y_lim: Optional[Tuple[float, float]]=None, size_preset: str='single', smoothing_window: float=100, sort_by_fr: bool=False, time_unit: str='seconds', pop_rate_unit: str='Hz', stim_log=None, show_pop_rate: bool=True, show_stim_markers: bool=False, stim_marker_style: str='line', stim_color: Optional[str]=None, stim_alpha: float=0.3, stim_time_col: str='auto', save_path: Optional[str]=None, filename: Optional[str]=None, show: bool=True, **kwargs) -> Tuple[plt.Figure, Tuple[plt.Axes, Optional[plt.Axes]]]`
> Plot spike raster, optional population rate, and optional stimulation markers.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/accessor.py:54`
#### `sttc_matrix(self, sttc_matrix: Optional[np.ndarray]=None, title: Optional[str]=None, size_preset: str='single', cmap: str='RdBu_r', vmin: Optional[float]=None, vmax: Optional[float]=None, delt: float=20.0, save_path: Optional[str]=None, filename: Optional[str]=None, show: bool=True, **kwargs) -> Tuple[plt.Figure, plt.Axes]`
> Plot a supplied, cached, or automatically computed STTC connectivity matrix.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/accessor.py:110`
#### `firing_rate_hist(self, title: Optional[str]=None, size_preset: str='single', bins: int=30, log_scale: bool=True, save_path: Optional[str]=None, filename: Optional[str]=None, show: bool=True, **kwargs) -> Tuple[plt.Figure, plt.Axes]`
> Plot the recording's neuron firing-rate distribution.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/accessor.py:157`
#### `evoked_raster(self, electrode_id: int, neuron_idx: int, plot_window: Tuple[float, float]=(-100, 100), max_trials: int=400, save_path: Optional[str]=None, show: bool=True, **kwargs) -> plt.Figure`
> Plot trial-aligned spikes for one stimulation-electrode/neuron pair with optional cached latency overlay.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/accessor.py:184`
#### `evoked_psth(self, electrode_id: int, neuron_idx: int, plot_window: Tuple[float, float]=(-100, 100), max_trials: int=400, psth_bins: int=75, save_path: Optional[str]=None, show: bool=True, **kwargs) -> plt.Figure`
> Plot the peri-stimulus histogram for one electrode-neuron pair.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/accessor.py:217`
#### `spatial_latency(self, electrode_id: Optional[int]=None, latency_results: Optional[Dict]=None, time_window: Tuple[float, float]=(0, 100), save_path: Optional[str]=None, filename: Optional[str]=None, show: bool=False) -> Any`
> Delegate spatial onset-latency visualization for selected results.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/accessor.py:252`
#### `spatial_response_map(self, electrode_id: int, metric: str='latency', save_path: Optional[str]=None, filename: Optional[str]=None, show: bool=False) -> Any`
> Plot a spatial response metric map for one stimulation electrode.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/accessor.py:275`
#### `spatial_animation(self, electrode_id: int, time_window: Tuple[float, float]=(-50, 100), smoothing_ms: float=20, save_path: Optional[str]=None, filename: Optional[str]=None) -> Any`
> Animate spatial response propagation over a selected time window.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/accessor.py:296`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Default Styler uses journal preset `draft` and may be replaced through the writable `styler` property. |
| Plot methods expose time windows/units, style presets, stimulation markers, saving, display, and method-specific controls. |

## Data Shapes
- STTC input is a square neuron-by-neuron matrix or an object exposing `.matrix`.
- Evoked plots select one electrode-neuron pair and may consume cached latency data keyed `electrode_<id>_neuron_<idx>`.

## Notes
- STTC is auto-computed from `recording.spikes.spike_time_tilings` when neither an argument nor cached result is available.
- Raster plotting auto-fetches the Recording stimulation log when no explicit log is supplied.
