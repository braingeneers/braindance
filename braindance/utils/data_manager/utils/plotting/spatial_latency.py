"""
Spatial visualization methods for neural recordings.

Provides methods for visualizing neural activity in spatial coordinates,
including latency maps, response propagation, and electrode-neuron relationships.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List, Any, TYPE_CHECKING
from pathlib import Path
from scipy.ndimage import uniform_filter1d

from braindance.utils.data_manager.utils.plotting.plot_styler import Styler

if TYPE_CHECKING:
    from braindance.utils.data_manager.utils.data_loading.recording import Recording


def spatial_latency(
    recording: 'Recording',
    electrode_id: Optional[int] = None,
    latency_results: Optional[Dict] = None,
    time_window: Tuple[float, float] = (0, 100),
    styler: Optional['Styler'] = None,
    save_path: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False
) -> Any:
    """
    Plot spatial distribution of onset latencies.
    
    Creates a scatter plot of neurons on the electrode array, color-coded by
    onset latency and sized by response strength.
    
    Args:
        recording: Recording object
        electrode_id: Stimulation electrode ID to analyze
        latency_results: Optional pre-computed latency results
        time_window: (min, max) latency range for color mapping (ms)
        styler: Optional Styler for plot styling
        save_path: Optional directory to save figure
        filename: Optional filename (without extension)
        show: Whether to display the plot
    
    Returns:
        Matplotlib figure
    """
    # Get required data
    mapping = recording.mapping
    spike_locations = recording.spike_locations
    
    if mapping is None or spike_locations is None:
        print("✗ Spatial plotting requires both mapping and spike_locations")
        print("  mapping available:", mapping is not None)
        print("  spike_locations available:", spike_locations is not None)
        return None
    
    # Get latency results if not provided
    if latency_results is None:
        # Try to calculate latencies
        try:
            from braindance.utils.data_manager.utils.analysis.ultra_optimized_latency_helper import UltraOptimizedLatencyHelper
            helper = UltraOptimizedLatencyHelper(verbose=False)
            # This would need the actual implementation
            print("⚠ Automatic latency calculation not yet fully implemented")
            print("  Please provide latency_results dictionary")
            return None
        except:
            print("✗ Could not calculate latencies - please provide latency_results")
            return None
    
    # Get stim electrode position
    stim_pos = mapping.get_positions(electrodes=[electrode_id])[0] if electrode_id is not None else None
    
    # Create figure
    if styler:
        fig, ax = styler.create_figure(size_preset='single')
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract neuron positions
    n_neurons = len(spike_locations)
    all_x = np.array([loc[0] for loc in spike_locations])
    all_y = np.array([loc[1] for loc in spike_locations])
    
    # Extract latency and response data
    peak_latencies = np.full(n_neurons, np.nan)
    response_ratios = np.zeros(n_neurons)
    has_data = np.zeros(n_neurons, dtype=bool)

    for neuron_idx in range(n_neurons):
        if neuron_idx in latency_results:
            peak_latencies[neuron_idx] = latency_results[neuron_idx].get('peak_latency', np.nan)
            # response_ratio replaces the removed response_strength for size weighting
            response_ratios[neuron_idx] = latency_results[neuron_idx].get('response_ratio', 0)
            if response_ratios[neuron_idx] > 0:
                has_data[neuron_idx] = True

    # Size by response_ratio (clip runaway inf ratios)
    ratios_clipped = np.clip(response_ratios, 0, 20)
    max_ratio = np.max(ratios_clipped) if np.max(ratios_clipped) > 0 else 1
    sizes = 30 + 270 * (ratios_clipped / max_ratio)

    # Identify responsive neurons (have latency in time window)
    responsive_mask = ~np.isnan(peak_latencies) & (peak_latencies >= time_window[0]) & (peak_latencies <= time_window[1])
    no_data_mask = ~has_data
    non_responsive_mask = has_data & ~responsive_mask
    
    # Plot neurons without data
    if np.any(no_data_mask):
        ax.scatter(all_x[no_data_mask], all_y[no_data_mask],
                  s=15, c='lightgray', alpha=0.3, zorder=1,
                  edgecolors='gray', linewidth=0.5)
    
    # Plot non-responsive neurons
    if np.any(non_responsive_mask):
        ax.scatter(all_x[non_responsive_mask], all_y[non_responsive_mask],
                  s=25, c='gray', alpha=0.5, zorder=1,
                  edgecolors='darkgray', linewidth=0.5)
    
    # Plot responsive neurons with latency coloring
    if np.any(responsive_mask):
        scatter = ax.scatter(all_x[responsive_mask], all_y[responsive_mask],
                            c=peak_latencies[responsive_mask],
                            s=sizes[responsive_mask],
                            cmap='bwr',
                            vmin=time_window[0], vmax=time_window[1],
                            alpha=0.8,
                            edgecolors='black',
                            linewidth=0.5,
                            zorder=2)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Peak Latency (ms)', rotation=270, labelpad=20)
    
    # Plot stimulation electrode if provided
    if stim_pos is not None:
        ax.scatter(stim_pos[0], stim_pos[1], s=400, c='gold', marker='*', 
                   label=f'Stim Electrode {electrode_id}', zorder=5, 
                   edgecolors='black', linewidth=2)
        ax.legend(loc='upper right')
    
    # Set labels and title
    ax.set_xlabel('X Position (μm)')
    ax.set_ylabel('Y Position (μm)')
    title = f'Spatial Latency Response'
    if electrode_id:
        title += f' (Electrode {electrode_id})'
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Statistics text
    n_responsive = np.sum(responsive_mask)
    if n_responsive > 0:
        mean_latency = np.nanmean(onset_latencies[responsive_mask])
        stats_text = (f'Total neurons: {n_neurons}\n'
                     f'Responsive neurons: {n_responsive}\n'
                     f'Mean latency: {mean_latency:.1f}ms')
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, verticalalignment='bottom')
    
    # Save and show
    if styler is None:
        styler = Styler(journal="draft")
    
    if save_path:
        styler.finish_plot(
            save_plots=True,
            save_dir=save_path,
            name=filename or 'spatial_latency',
            show=show
        )
    elif show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def spatial_response_map(
    recording: 'Recording',
    electrode_id: int,
    metric: str = 'latency',
    styler: Optional['Styler'] = None,
    save_path: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False
) -> Any:
    """
    Plot spatial map of neural responses to stimulation.
    
    Args:
        recording: Recording object
        electrode_id: Stimulation electrode to analyze
        metric: What to visualize - 'latency', 'strength', or 'change'
        styler: Optional Styler for plot styling
        save_path: Optional directory to save figure
        filename: Optional filename
        show: Whether to display the plot
    
    Returns:
        Matplotlib figure
    """
    print(f"Spatial response map - metric: {metric}")
    print("⚠ Full implementation coming soon - this is a placeholder")
    # TODO: Implement based on spatial_response.py create_activity_difference_plot
    return None


def spatial_animation(
    recording: 'Recording',
    electrode_id: int,
    time_window: Tuple[float, float] = (-50, 100),
    smoothing_ms: float = 20,
    styler: Optional['Styler'] = None,
    save_path: Optional[str] = None,
    filename: Optional[str] = None
) -> Any:
    """
    Create animation of neural response propagation.
    
    Shows how activity spreads across the electrode array over time after stimulation.
    
    Args:
        recording: Recording object
        electrode_id: Stimulation electrode ID
        time_window: (start, end) time window for animation (ms)
        smoothing_ms: Smoothing window size (ms)
        styler: Optional Styler for plot styling
        save_path: Optional directory to save animation
        filename: Optional filename (will add .gif extension)
    
    Returns:
        Animation object
    """
    print(f"Spatial animation - electrode {electrode_id}, window {time_window}")
    print("⚠ Full implementation coming soon - this is a placeholder")
    # TODO: Implement based on spatial_response.py create_spatial_animation
    return None
