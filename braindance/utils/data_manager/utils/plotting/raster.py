"""
Raster plotting utilities for BrainDance.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple, Union, Dict, Any, TYPE_CHECKING
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from braindance.utils.data_manager.utils.data_loading.recording import Recording
    from braindance.utils.data_manager.utils.plotting.plot_styler import Styler


def plot_raster_with_pop(
    spike_data,
    title: Optional[str] = None,
    time_window: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    styler: Optional['Styler'] = None,
    width_pt: Optional[float] = None,
    height_pt: Optional[float] = None,
    size_preset: str = 'single',
    save_path: Optional[str] = None,
    filename: Optional[str] = None,
    smoothing_window: float = 100,
    sort_by_fr: bool = False,
    time_unit: str = 'seconds',
    pop_rate_unit: str = 'Hz',
    raster_color: str = 'black',
    stim_log = None,
    show_pop_rate: bool = True,
    show_stim_markers: bool = False,
    stim_marker_style: str = 'line',
    stim_color: Optional[str] = None,
    stim_alpha: float = 0.3,
    stim_time_col: str = 'auto',
    show: bool = True
) -> Tuple[plt.Figure, Tuple[plt.Axes, Optional[plt.Axes]]]:
    """
    Plot a styled raster plot with population firing rate overlay.
    
    Parameters
    ----------
    spike_data : SpikeData
        SpikeData object containing spike times and neuron data
    title : str, optional
        Title for the plot
    time_window : tuple of float, optional
        Time window to display (start, end) in units specified by time_unit
    y_lim : tuple of float, optional
        Y-axis limits for population rate (min, max)
    styler : Styler, optional
        Styler object for plot formatting. If None, uses default styling.
    width_pt : float, optional
        Width of the figure in points (overrides size_preset)
    height_pt : float, optional
        Height of the figure in points (overrides size_preset)
    size_preset : str, default='single'
        Figure size preset: 'single', '1.5', or 'double'
    save_path : str, optional
        Path to save the figure (directory or file path)
    filename : str, optional
        Filename for saved figure (without extension)
    smoothing_window : float, default=100
        Bin size in ms for population rate calculation
    sort_by_fr : bool, default=False
        Whether to sort neurons by firing rate (highest at top)
    time_unit : str, default='seconds'
        Unit for x-axis: 'seconds' or 'minutes'
    pop_rate_unit : str, default='Hz'
        Unit for population rate: 'Hz' or 'kHz'
    raster_color : str, default='black'
        Color for raster spikes
    stim_log : DataFrame, optional
        Stimulation log with a 'time' (and usually 'time_mod') column, in
        seconds. If provided with show_stim_markers=True, will plot
        stimulation timing markers
    show_pop_rate : bool, default=True
        Whether to show population rate overlay on twin axis
    show_stim_markers : bool, default=False
        Whether to show stimulation timing markers (requires stim_log)
    stim_marker_style : str, default='line'
        Style for stimulation markers: 'line' (vertical lines) or 'scatter' (points)
    stim_color : str, optional
        Color for stimulation markers (defaults to styler.colors[7] - red)
    stim_alpha : float, default=0.3
        Alpha transparency for stimulation markers
    stim_time_col : str, default='auto'
        Which stim_log column carries the stimulation times. 'auto' uses
        'time_mod' when present and falls back to 'time'. Pass 'time'
        explicitly only if you specifically want the uncorrected software
        clock -- it is a median 13.7 ms off the real artifact.
    show : bool, default=True
        Whether to display the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : tuple of (ax_raster, ax_rate)
        Tuple of axes objects. ax_rate is None if show_pop_rate=False or no spikes present.
    """
    # Validate inputs
    if time_unit not in ['seconds', 'minutes']:
        raise ValueError("time_unit must be either 'seconds' or 'minutes'")
    if pop_rate_unit not in ['Hz', 'kHz']:
        raise ValueError("pop_rate_unit must be either 'Hz' or 'kHz'")
    
    # Get spike times and cluster IDs from spike_data
    cluster_ids, spike_times = spike_data.idces_times()
    
    # Sort by firing rate if requested
    if sort_by_fr and len(spike_times) > 0:
        rates = []
        for i in range(spike_data.N):
            spikes = spike_data.train[i]
            rate = len(spikes) / (spike_data.length / 1000) if spike_data.length > 0 else 0
            rates.append(rate)
        
        # Get sorted indices (descending order - highest FR at top)
        sort_indices = np.argsort(rates)[::-1]
        
        # Create mapping from original to sorted indices
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(sort_indices)}
        
        # Map original cluster IDs to sorted IDs
        cluster_ids = np.array([id_mapping[cid] for cid in cluster_ids])
    
    # Create figure
    if styler is None:
        from braindance.utils.data_manager.utils.plotting.plot_styler import Styler
        styler = Styler(journal="draft")
    
    if width_pt and height_pt:
        fig = plt.figure(figsize=styler.get_figure_size(width_pt=width_pt, height_pt=height_pt))
    else:
        fig = plt.figure(figsize=styler.get_figure_size(size_preset=size_preset))
    
    # Calculate scale factor for marker sizes
    fig_width, fig_height = fig.get_size_inches()
    reference_area = 3.346 * 2.51  # Reference figure area
    scale_factor = np.sqrt(fig_width * fig_height) / np.sqrt(reference_area)
    
    ax = fig.add_subplot(111)
    
    # Convert spike times to selected time unit
    time_divisor = 1000 if time_unit == 'seconds' else 60000
    normalized_times = spike_times / time_divisor
    
    # Plot raster
    ax.scatter(
        normalized_times, 
        cluster_ids,
        s=0.25 * scale_factor,
        marker='|',
        c=raster_color,
        alpha=0.8,
        linewidths=0.5 * scale_factor
    )
    
    # Set labels
    ax.set_xlabel(f'Time ({time_unit})')
    ylabel = 'Neuron ID (sorted by FR)' if sort_by_fr else 'Neuron ID'
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='y')
    
    if title:
        ax.set_title(title)
    
    # Add stimulation markers if requested
    if show_stim_markers and stim_log is not None:
        # Set stim color
        if stim_color is None:
            stim_marker_color = styler.colors[7]  # Red
        else:
            stim_marker_color = stim_color

        # Extract stimulation times (in seconds) and convert to selected time_unit.
        #
        # Prefer `time_mod` over `time`. `time` is the software `perf_counter()`
        # stamp written live by maxwell_env, and it sits a median 13.7 ms -- worst
        # 79 ms -- from the real stimulation artifact in the trace. `time_mod` is
        # that stamp snapped onto `spike_data.metadata['artifact_times']` by
        # `Recording._adjust_stim_times()`, so it is the one that lines up with the
        # spikes being plotted next to it. Every other stim-time consumer in this
        # repo already prefers it (evoked.py, latency_helper.py,
        # burst_latency_analyzer.py); this function used to be the odd one out.
        # `time_mod` is absent whenever `artifact_times` is, hence the fallback.
        if stim_time_col == 'auto':
            resolved_col = 'time_mod' if 'time_mod' in stim_log.columns else 'time'
        else:
            resolved_col = stim_time_col
        if resolved_col not in stim_log.columns:
            raise ValueError(
                f"stim_time_col={resolved_col!r} not in stim_log columns "
                f"{list(stim_log.columns)}"
            )
        stim_times_s = stim_log[resolved_col].values
        stim_times_ms = stim_times_s * 1000  # Convert to milliseconds
        stim_times_normalized = stim_times_ms / time_divisor

        # Filter by time window if specified
        if time_window is not None:
            mask = (stim_times_normalized >= time_window[0]) & (stim_times_normalized <= time_window[1])
            stim_times_normalized = stim_times_normalized[mask]

        # Plot markers
        if stim_marker_style == 'line':
            for stim_time in stim_times_normalized:
                ax.axvline(x=stim_time, color=stim_marker_color, linestyle='-',
                          alpha=stim_alpha, linewidth=0.5)
        elif stim_marker_style == 'scatter':
            y_min = ax.get_ylim()[0]
            ax.scatter(stim_times_normalized, [y_min] * len(stim_times_normalized),
                      c=stim_marker_color, s=2, alpha=stim_alpha, marker='|')

        # Add to legend if markers were added
        if len(stim_times_normalized) > 0:
            legend_elements = [Line2D([0], [0], color=stim_marker_color, linestyle='-',
                                    alpha=stim_alpha, label='Stimulation')]
            if ax.get_legend() is not None:
                # Append to existing legend
                handles, labels = ax.get_legend_handles_labels()
                handles.extend(legend_elements)
                ax.legend(handles=handles, fontsize=6)
            else:
                ax.legend(handles=legend_elements, fontsize=6)

    # Add population firing rate overlay
    ax2 = None
    if show_pop_rate and len(spike_times) > 0:
        ax2 = ax.twinx()
        
        # Calculate population rate
        bin_size = smoothing_window  # ms
        bins = np.arange(spike_times.min(), spike_times.max() + bin_size, bin_size)
        spike_counts, _ = np.histogram(spike_times, bins=bins)
        pop_rate = spike_counts / (bin_size / 1000)  # Convert to Hz
        
        # Convert to kHz if requested
        if pop_rate_unit == 'kHz':
            pop_rate = pop_rate / 1000.0
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_centers_normalized = bin_centers / time_divisor
        
        # Apply time window mask
        if time_window is not None:
            multiplier = 1000 if time_unit == 'seconds' else 60000
            time_window_ms = (time_window[0] * multiplier, time_window[1] * multiplier)
            mask = (bin_centers >= time_window_ms[0]) & (bin_centers <= time_window_ms[1])
            bin_centers_normalized = bin_centers_normalized[mask]
            pop_rate = pop_rate[mask]
        
        # Plot population rate
        pop_rate_color = styler.colors[5]  # Dark blue
        ax2.plot(
            bin_centers_normalized, 
            pop_rate,
            color=pop_rate_color,
            linewidth=1 * scale_factor
        )
        
        ax2.set_ylabel(f'Population Rate ({pop_rate_unit})', color=pop_rate_color)
        ax2.tick_params(axis='y', labelcolor=pop_rate_color)
        ax2.spines['right'].set_visible(True)
        ax2.spines['right'].set_color(pop_rate_color)
        
        if y_lim is not None:
            ax2.set_ylim(y_lim)
    
    # Set time window limits
    if time_window is not None:
        ax.set_xlim(time_window)
    
    # Handle saving
    if save_path is not None:
        if os.path.isdir(save_path):
            save_dir = save_path
            name = filename or 'raster_with_pop'
        else:
            save_dir = os.path.dirname(save_path) or '.'
            name = filename or os.path.splitext(os.path.basename(save_path))[0]
        styler.finish_plot(True, save_dir, name, show=False)
    else:
        plt.tight_layout()
        if show:
            plt.show()
    
    return fig, (ax, ax2)
