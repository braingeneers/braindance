# population.py

**Path:** `braindance/utils/data_manager/utils/plotting/population.py`
**Module:** `braindance.utils.data_manager.utils.plotting.population`
**Feature Area:** `Visualization`
**Entry point:** no — library or imported component

## Overview
Renders STTC connectivity matrices and neuron firing-rate distributions. Adds configurable heatmap labels/color scales and mean/median rate markers with Styler export.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.plotting.accessor` — import consumer hint; not a proven runtime call.
- **Uses:** `Recording` from `braindance.utils.data_manager.utils.data_loading.recording` — imports (static evidence).
- **Uses:** `Styler` from `braindance.utils.data_manager.utils.plotting.plot_styler` — imports (static evidence).
- **Shared data:** PlotAccessor.sttc_matrix and firing_rate_hist call these helpers.

## Dependencies
- `braindance.utils.data_manager.utils.data_loading.recording.Recording` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.plot_styler.Styler` — intra-repo import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `plot_sttc_matrix(sttc_matrix: np.ndarray, title: Optional[str]=None, styler: Optional['Styler']=None, width_pt: Optional[float]=None, height_pt: Optional[float]=None, size_preset: str='single', save_path: Optional[str]=None, filename: Optional[str]=None, cmap: str='RdBu_r', vmin: Optional[float]=None, vmax: Optional[float]=None, show_colorbar: bool=True, show_diagonal: bool=False, neuron_labels: Optional[list]=None, show: bool=True) -> Tuple[plt.Figure, plt.Axes]`
> Plot a Spike Time Tiling Coefficient (STTC) correlation matrix.
> **Called by:** braindance/utils/data_manager/utils/plotting/accessor.py:143 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/population.py:15`
### `plot_firing_rate_histogram(spike_data, title: Optional[str]=None, styler: Optional['Styler']=None, size_preset: str='single', bins: int=30, log_scale: bool=True, save_path: Optional[str]=None, filename: Optional[str]=None, show: bool=True) -> Tuple[plt.Figure, plt.Axes]`
> Plot histogram of firing rates across neurons.
> **Called by:** braindance/utils/data_manager/utils/plotting/accessor.py:171 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/plotting/population.py:140`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| plot_sttc_matrix defaults cmap='RdBu_r', show_diagonal=False; plot_firing_rate_histogram defaults bins=30 and log_scale=True; save_path/filename/show control output. |

## Data Shapes
- STTC input is a square numeric matrix; firing-rate input is SpikeData.rates(unit="Hz"); functions return Figure/Axes.

## Notes
- Log-scale histogram excludes zero rates but mean/median use all rates.
- STTC diagonal is replaced with NaN on a copied matrix; input must support NaN assignment.
- Default Styler is explicitly journal="draft".
