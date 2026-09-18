# plot_helper.py

**Path:** `braindance/analysis/plot_helper.py`
**Module:** `braindance.analysis.plot_helper`
**Feature Area:** `Visualization`
**Entry point:** no — library or imported component

## Overview
Plots spatial spike heatmaps, electrode waveform footprints, evoked traces and first/multi-order causal connectivity. Supports legacy AnalysisDAO and experiment-result interfaces and can save figures plus derived matrices.

## Connections
- **Used by:** `braindance.core.phases_analysis` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_analysis_2` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.paper_cartpole.causal` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.paper_cartpole.recording` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.neural_config_analysis` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.gui.experiment_launcher` — import consumer hint; not a proven runtime call.
- **Uses:** `FootprintPhase` from `braindance.core.phases_analysis` — imports (static evidence).
- **Shared data:** get_footprints invokes core.phases_analysis.FootprintPhase.
- **Shared data:** GUI launcher and neural_config_analysis use footprint plots; causal_connectivity analysis provides connectivity metrics.

## Dependencies
- `braindance.core.phases_analysis.FootprintPhase` — intra-repo import; source import evidence.
- `matplotlib` — external or unresolved local import; source import evidence.
- `matplotlib.cm` — external or unresolved local import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `mpl_toolkits.axes_grid1.make_axes_locatable` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `scipy.ndimage.filters.gaussian_filter` — external or unresolved local import; source import evidence.
- `scipy.signal.find_peaks` — external or unresolved local import; source import evidence.
- `seaborn` — external or unresolved local import; source import evidence.
- `tqdm.tqdm` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `plot_heatmap_spikes(spikes, mapping, title=None, cmap='magma', save=False, filename=None)`
> Plot heatmap of spikes, first converting to a list of positions, then calling plot_heatmap
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/plot_helper.py:9`
### `plot_heatmap(x, y, sigma=6, bins=1000, v_min=None, v_max=None, fig=None, ax=None, plt_show=True, cmap='viridis', conversion_factor=1, units='units', cbar=True, log=False, alpha=1)`
> unclear — see source
> **Called by:** braindance/analysis/plot_helper.py:28 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/plot_helper.py:32`
### `plot_footprint(x, y, footprint, fig=None, ax=None, x_scale=0.01, y_scale=0.1, window_len=30, **kwargs)`
> Plot a waveform footprint at the corresponding channel
> **Called by:** braindance/analysis/plot_helper.py:148 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/plot_helper.py:91`
### `plot_footprints(footprint_chs, footprint_waves, mapping, selected_channels, invert_y=True, save_dir=None, annotate=True, **kwargs)`
> unclear — see source
> **Called by:** braindance/core/phases_analysis_2.py:683 (named-call hint); braindance/examples/paper_cartpole/recording.py:76 (named-call hint); braindance/experiments/neural_config_analysis.py:58 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/analysis/plot_helper.py:126`
### `plot_reactivity(analysis, file_path=None, fig=None, ax=None, **kwargs)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/plot_helper.py:168`
### `plot_reactivity_traces(analysis, save_dir=None, fig=None, ax=None, time_scatter=False, **kwargs)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** write call (static hint).
**Source:** `braindance/analysis/plot_helper.py:202`
### `plot_causal_connectivity(analysis, save_dir=None, fig=None, ax=None, first_order_ms=30, multi_order_ms=100, save_mats=True, **kwargs)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** write call (static hint).
**Source:** `braindance/analysis/plot_helper.py:301`
### `plot_reactivity_traces_experiment(experiment, clean_data=None, stim_electrodes=None, channels_of_interest=None, save_dir=None, fig=None, ax=None, time_scatter=False, **kwargs)`
> Plot reactivity traces using data from an Experiment object. Similar to plot_reactivity_traces but uses experiment instead of analysis object.
> **Called by:** braindance/core/phases_analysis_2.py:896 (named-call hint); braindance/core/phases_analysis_2.py:905 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/analysis/plot_helper.py:491`
### `plot_causal_connectivity_experiment(experiment, info=None, save_dir=None, fig=None, ax=None, first_order_ms=30, multi_order_ms=100, save_mats=True, **kwargs)`
> Plot causal connectivity using data from an Experiment object. Similar to plot_causal_connectivity but uses experiment instead of analysis object.
> **Called by:** braindance/core/phases_analysis_2.py:887 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/analysis/plot_helper.py:626`
### `get_footprints(analysis, selected_electrodes=None, file_path=None)`
> Return the footprint channels and waveforms for the given electrodes.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/plot_helper.py:906`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Heatmap sigma,bins,cmap,linear/log normalization; footprint x/y scale and invert_y |
| Connectivity first_order_ms=30,multi_order_ms=100,save_mats=True |

## Data Shapes
- Mapping requires channel,electrode,x,y; footprints channel lists with aligned waveforms
- clean_data indexed repeat,stimulus,response_channel,time; reactivity_times stimulus Ã— response object arrays
- Saved derived matrices include raw/z-scored connectivity and per-response mean/std

## Notes
- Reactivity time conversions hardcode 20 frames/ms.
- plot_heatmap cbar branch is empty; legacy connectivity save_mats=True assumes save_dir.
- Experiment connectivity plot requires precomputed info metrics despite optional info signature.
