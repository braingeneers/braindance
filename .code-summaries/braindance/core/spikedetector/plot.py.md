# plot.py

**Path:** `braindance/core/spikedetector/plot.py`
**Module:** `braindance.core.spikedetector.plot`
**Feature Area:** `Visualization`
**Entry point:** no — library or imported component

## Overview
Formats detector training examples, waveform overlays, prediction probabilities, and error distributions. Helpers customize Matplotlib ticks, legends, DPI, and percentage histogram labels.

## Connections
- **Used by:** `braindance.core.spikedetector.data` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.spikedetector.model2` — import consumer hint; not a proven runtime call.
- **Uses:** `utils` from `braindance.core.spikedetector` — imports (static evidence).

## Dependencies
- `braindance.core.spikedetector.utils` — intra-repo import; source import evidence.
- `matplotlib` — external or unresolved local import; source import evidence.
- `matplotlib.axes._axes.Axes` — external or unresolved local import; source import evidence.
- `matplotlib.lines.Line2D` — external or unresolved local import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `matplotlib.ticker.PercentFormatter` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `plot_waveform(waveform, peak_idx, wf_len, wf_alpha, wf_trace_loc, axis, xlim=None, ylim_diff=None, title='Waveform', **wf_line_kwargs)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/plot.py:14`
### `plot_hist_loc_mad(loc_deviations, n_bins=15)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/plot.py:38`
### `plot_hist_percent_abs_error(percent_abs_errors, n_bins=10)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/plot.py:55`
### `get_yticks_lim(trace, anchor=0, increment=5, buffer_min=5, buffer_max=3)`
> Get lim and ticks for y-axis when trace is plotted
> **Called by:** braindance/core/spikedetector/plot.py:115 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/plot.py:71`
### `set_ticks(subplots: Tuple[Axes], trace: np.array, increment=10, buffer_min=10, buffer_max=10, center_xticks=False)`
> Set x and y ticks for subplots
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/plot.py:99`
### `set_dpi(dpi)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/plot.py:136`
### `get_empty_line(label)`
> unclear — see source
> **Called by:** braindance/core/spikedetector/plot.py:149 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/plot.py:141`
### `display_prob_spike(spike_output, axis)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/plot.py:146`
### `unscaled_ticks_to_uv(subplot)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/plot.py:157`
### `plot_hist_percents(data, ax=None, **hist_kwargs)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/plot.py:163`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Histogram bins, tick increment/buffers, global figure DPI |

## Data Shapes
- Waveform arrays are aligned using peak_idx and wf_trace_loc.

## Notes
- set_ticks references undefined samp_freq_khz; unscaled_ticks_to_uv references absent utils.FACTOR_UV.
