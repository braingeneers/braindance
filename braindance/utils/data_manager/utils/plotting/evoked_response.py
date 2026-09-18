"""
Evoked response plotting utilities for BrainDance.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import ast
from typing import Optional, Tuple, Union, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from braindance.utils.data_manager.utils.data_loading.recording import Recording
    from braindance.utils.data_manager.utils.plotting.plot_styler import Styler


def plot_evoked_raster(
    spike_data,
    stim_log,
    electrode_id: int,
    neuron_idx: int,
    plot_window: Tuple[float, float] = (-100, 100),
    max_trials: int = 400,
    latency_data: Optional[dict] = None,
    styler: Optional['Styler'] = None,
    raster_color: str = 'black',
    response_color: Optional[str] = None,
    save_path: Optional[str] = None,
    use_time_mod: bool = True,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot evoked response raster for a neuron-electrode pair.

    Shows trial-by-trial spike responses aligned to stimulation onset.
    Each row represents one trial, with spikes plotted as markers.

    Parameters
    ----------
    spike_data : SpikeData
        SpikeData object containing spike times
    stim_log : DataFrame
        Stimulation log with 'time' and 'stim_electrodes' columns
    electrode_id : int
        ID of the stimulation electrode
    neuron_idx : int
        Index of the neuron to plot
    plot_window : tuple of float, default=(-100, 100)
        Time window around stimulus in milliseconds (start, end)
    max_trials : int, default=400
        Maximum number of trials to plot (for performance)
    latency_data : dict, optional
        Dictionary with latency analysis results. Should contain keys:
        'peak_latency', 'response_ratio', 'p_value', 'n_trials'
    styler : Styler, optional
        Styler object for consistent formatting
    raster_color : str, default='black'
        Color for spike markers in raster
    response_color : str, optional
        Color for first post-stimulus spike highlight (defaults to styler.colors[7] - red)
    save_path : str, optional
        Path to save figure
    use_time_mod : bool, default=True
        Use artifact-corrected 'time_mod' column if available, otherwise use 'time'.
        This is important for accurate stimulus alignment.
    show : bool, default=True
        Whether to display the plot
    **kwargs
        Additional arguments (reserved for future use)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Raster plot figure showing trial-by-trial responses
    """
    # Initialize styler
    if styler is None:
        from braindance.utils.data_manager.utils.plotting.plot_styler import Styler
        styler = Styler(journal="draft")

    # Set response color
    if response_color is None:
        response_color = styler.colors[7]  # Red

    # Find stimulation times for this electrode
    electrode_mask = stim_log['stim_electrodes'].apply(
        lambda x: electrode_id in x if isinstance(x, list) else (
            electrode_id in ast.literal_eval(x) if isinstance(x, str) else False
        )
    )

    if not electrode_mask.any():
        raise ValueError(f"No stimulations found for electrode {electrode_id}")

    # Get stimulation times - USE time_mod if available for accurate alignment
    electrode_stims = stim_log[electrode_mask]
    if use_time_mod and 'time_mod' in electrode_stims.columns:
        stim_times_sec = electrode_stims['time_mod'].values
    else:
        stim_times_sec = electrode_stims['time'].values
        if use_time_mod:
            import warnings
            warnings.warn(
                "'time_mod' column not found in stim_log. Using 'time' column instead. "
                "This may cause stimulus alignment issues if artifact removal was applied. "
                "Set use_time_mod=False to suppress this warning.",
                UserWarning
            )
    
    # Convert to milliseconds
    electrode_stims_ms = stim_times_sec * 1000

    # Get spike times for this neuron (already in milliseconds)
    neuron_spikes_ms = np.array(spike_data.train[neuron_idx])

    # Limit stimulations for plotting performance
    max_stims_plot = min(max_trials, len(electrode_stims_ms))
    stims_to_plot = electrode_stims_ms[:max_stims_plot]

    # Create Raster Plot
    fig_raster, ax_raster = styler.create_figure(size_preset='single')

    spike_times_rel = []

    for i, stim_time in enumerate(stims_to_plot):
        # Find spikes in window around this stimulation
        stim_spikes = neuron_spikes_ms[(neuron_spikes_ms >= stim_time + plot_window[0]) &
                                     (neuron_spikes_ms < stim_time + plot_window[1])]

        if len(stim_spikes) > 0:
            relative_times = stim_spikes - stim_time

            # Find first spike after stimulation for highlighting
            post_stim_spikes = relative_times[relative_times >= 0]

            # Plot all spikes
            ax_raster.scatter(relative_times, [i] * len(relative_times),
                            c=raster_color, s=1, alpha=0.8)

            # Highlight first post-stimulus spike
            if len(post_stim_spikes) > 0:
                first_spike_time = post_stim_spikes[0]
                ax_raster.scatter([first_spike_time], [i], c=response_color, s=3, alpha=1.0)

            spike_times_rel.extend(relative_times)

    # Format raster plot
    ax_raster.axvline(x=0, color=styler.colors[5], linestyle='--', alpha=0.7, label='Stimulation')

    # Mark peak latency if available
    if latency_data and 'peak_latency' in latency_data:
        peak_latency = latency_data['peak_latency']
        ax_raster.axvline(x=peak_latency, color=response_color, linestyle='-',
                        alpha=0.8, label=f'Peak: {peak_latency:.1f}ms')

    ax_raster.set_xlim(plot_window)
    ax_raster.set_xlabel('Time from stimulation (ms)')
    ax_raster.set_ylabel('Trial #')
    ax_raster.set_title(f'Evoked Response - Electrode {electrode_id}, Neuron {neuron_idx}')
    ax_raster.legend(fontsize=6)
    ax_raster.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot if requested
    if save_path:
        import pathlib
        save_dir = pathlib.Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        fig_raster.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        fig_raster.savefig(f"{save_path}.svg", format='svg', bbox_inches='tight', pad_inches=0.1)
        print(f"Raster plot saved to {save_path}")

    if show:
        plt.show()

    return fig_raster


def plot_evoked_psth(
    spike_data,
    stim_log,
    electrode_id: int,
    neuron_idx: int,
    plot_window: Tuple[float, float] = (-100, 100),
    max_trials: int = 400,
    latency_data: Optional[dict] = None,
    styler: Optional['Styler'] = None,
    psth_color: Optional[str] = None,
    response_color: Optional[str] = None,
    psth_bins: int = 75,
    save_path: Optional[str] = None,
    use_time_mod: bool = True,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot evoked response PSTH (peri-stimulus time histogram) for a neuron-electrode pair.

    Shows average firing rate across trials in time bins aligned to stimulation onset.

    Parameters
    ----------
    spike_data : SpikeData
        SpikeData object containing spike times
    stim_log : DataFrame
        Stimulation log with 'time' and 'stim_electrodes' columns
    electrode_id : int
        ID of the stimulation electrode
    neuron_idx : int
        Index of the neuron to plot
    plot_window : tuple of float, default=(-100, 100)
        Time window around stimulus in milliseconds (start, end)
    max_trials : int, default=400
        Maximum number of trials to use (for performance)
    latency_data : dict, optional
        Dictionary with latency analysis results. Should contain keys:
        'peak_latency', 'baseline_rate', 'response_rate', 'response_ratio', 'p_value'
    styler : Styler, optional
        Styler object for consistent formatting
    psth_color : str, optional
        Color for PSTH bars (defaults to styler.colors[8] - black)
    response_color : str, optional
        Color for onset latency line (defaults to styler.colors[7] - red)
    psth_bins : int, default=75
        Number of bins for histogram
    save_path : str, optional
        Path to save figure
    use_time_mod : bool, default=True
        Use artifact-corrected 'time_mod' column if available, otherwise use 'time'.
        This is important for accurate stimulus alignment.
    show : bool, default=True
        Whether to display the plot
    **kwargs
        Additional arguments (reserved for future use)

    Returns
    -------
    fig : matplotlib.figure.Figure
        PSTH plot figure showing average firing rate
    """
    # Initialize styler
    if styler is None:
        from braindance.utils.data_manager.utils.plotting.plot_styler import Styler
        styler = Styler(journal="draft")

    # Set colors
    if psth_color is None:
        psth_color = styler.colors[8]  # Black
    if response_color is None:
        response_color = styler.colors[7]  # Red

    # Find stimulation times for this electrode
    electrode_mask = stim_log['stim_electrodes'].apply(
        lambda x: electrode_id in x if isinstance(x, list) else (
            electrode_id in ast.literal_eval(x) if isinstance(x, str) else False
        )
    )

    if not electrode_mask.any():
        raise ValueError(f"No stimulations found for electrode {electrode_id}")

    # Get stimulation times - USE time_mod if available for accurate alignment
    electrode_stims = stim_log[electrode_mask]
    if use_time_mod and 'time_mod' in electrode_stims.columns:
        stim_times_sec = electrode_stims['time_mod'].values
    else:
        stim_times_sec = electrode_stims['time'].values
        if use_time_mod:
            import warnings
            warnings.warn(
                "'time_mod' column not found in stim_log. Using 'time' column instead. "
                "This may cause stimulus alignment issues if artifact removal was applied. "
                "Set use_time_mod=False to suppress this warning.",
                UserWarning
            )
    
    # Convert to milliseconds
    electrode_stims_ms = stim_times_sec * 1000

    # Get spike times for this neuron (already in milliseconds)
    neuron_spikes_ms = np.array(spike_data.train[neuron_idx])

    # Limit stimulations for processing
    max_stims_plot = min(max_trials, len(electrode_stims_ms))
    stims_to_plot = electrode_stims_ms[:max_stims_plot]

    # Collect all spike times relative to stimulation
    spike_times_rel = []
    for stim_time in stims_to_plot:
        stim_spikes = neuron_spikes_ms[(neuron_spikes_ms >= stim_time + plot_window[0]) &
                                     (neuron_spikes_ms < stim_time + plot_window[1])]
        if len(stim_spikes) > 0:
            relative_times = stim_spikes - stim_time
            spike_times_rel.extend(relative_times)

    # Create PSTH figure
    fig_psth, ax_psth = styler.create_figure(size_preset='single')

    # Create PSTH
    if spike_times_rel and len(spike_times_rel) > 0:
        # Create PSTH histogram
        hist, bin_edges = np.histogram(spike_times_rel, bins=psth_bins, range=plot_window)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Convert to Hz
        bin_width_s = (bin_edges[1] - bin_edges[0]) / 1000  # Convert ms to seconds
        hist_hz = hist / (max_stims_plot * bin_width_s)

        # Plot as histogram bars
        ax_psth.bar(bin_centers, hist_hz, width=(bin_centers[1]-bin_centers[0])*0.8,
                   alpha=0.7, color=psth_color, edgecolor='black', linewidth=0.5)

        # Add baseline and response rate lines if available
        if latency_data:
            if 'baseline_rate' in latency_data and latency_data['baseline_rate'] is not None:
                ax_psth.axhline(latency_data['baseline_rate'], color=styler.colors[4], linestyle=':',
                               alpha=0.8, linewidth=1, label=f'Baseline: {latency_data["baseline_rate"]:.1f} Hz')
            if 'response_rate' in latency_data and latency_data['response_rate'] is not None:
                ax_psth.axhline(latency_data['response_rate'], color=styler.colors[1], linestyle=':',
                               alpha=0.8, linewidth=1, label=f'Response: {latency_data["response_rate"]:.1f} Hz')

        # Set appropriate Y-axis limits
        max_rate = np.max(hist_hz)
        ax_psth.set_ylim(0, max_rate * 1.1)

    else:
        ax_psth.text(0.5, 0.5, 'No spikes found in time window',
                   transform=ax_psth.transAxes, ha='center', va='center', fontsize=7)
        ax_psth.set_ylim(0, 1)

    # Format PSTH plot
    ax_psth.axvline(x=0, color=styler.colors[5], linestyle='--', alpha=0.7, label='Stimulation')

    # Mark peak latency if available
    if latency_data and 'peak_latency' in latency_data:
        peak_latency = latency_data['peak_latency']
        ax_psth.axvline(x=peak_latency, color=response_color, linestyle='-', alpha=0.8,
                      label=f'Peak: {peak_latency:.1f}ms')

    ax_psth.set_xlim(plot_window)
    ax_psth.set_xlabel('Time from stimulation (ms)')
    ax_psth.set_ylabel('Spike rate (Hz)')
    ax_psth.set_title(f'PSTH - Electrode {electrode_id}, Neuron {neuron_idx}')
    ax_psth.legend(loc='upper right', fontsize=6)
    ax_psth.grid(True, alpha=0.3)

    # Add statistics text box if available
    if latency_data:
        stats_text = f"Electrode: {electrode_id}, Neuron: {neuron_idx}\n"
        if 'peak_latency' in latency_data:
            stats_text += f"Peak latency: {latency_data['peak_latency']:.1f}ms\n"
        if 'response_ratio' in latency_data:
            stats_text += f"Response ratio: {latency_data['response_ratio']:.2f}\n"
        if 'p_value' in latency_data:
            stats_text += f"P-value: {latency_data['p_value']:.4f}\n"
        if 'n_trials' in latency_data:
            stats_text += f"Trials: {latency_data['n_trials']}"

        ax_psth.text(0.02, 0.98, stats_text, transform=ax_psth.transAxes,
                    verticalalignment='top', fontsize=6,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save plot if requested
    if save_path:
        import pathlib
        save_dir = pathlib.Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        fig_psth.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        fig_psth.savefig(f"{save_path}.svg", format='svg', bbox_inches='tight', pad_inches=0.1)
        print(f"PSTH plot saved to {save_path}")

    if show:
        plt.show()

    return fig_psth
