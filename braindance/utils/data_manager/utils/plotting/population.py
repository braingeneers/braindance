"""
Population-level plotting utilities for BrainDance.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple, Union, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from braindance.utils.data_manager.utils.data_loading.recording import Recording
    from braindance.utils.data_manager.utils.plotting.plot_styler import Styler


def plot_sttc_matrix(
    sttc_matrix: np.ndarray,
    title: Optional[str] = None,
    styler: Optional['Styler'] = None,
    width_pt: Optional[float] = None,
    height_pt: Optional[float] = None,
    size_preset: str = 'single',
    save_path: Optional[str] = None,
    filename: Optional[str] = None,
    cmap: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
    show_diagonal: bool = False,
    neuron_labels: Optional[list] = None,
    show: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a Spike Time Tiling Coefficient (STTC) correlation matrix.
    
    Parameters
    ----------
    sttc_matrix : np.ndarray
        Square matrix of STTC values (N x N neurons)
    title : str, optional
        Title for the plot
    styler : Styler, optional
        Styler object for plot formatting
    width_pt : float, optional
        Width of the figure in points
    height_pt : float, optional
        Height of the figure in points
    size_preset : str, default='single'
        Figure size preset: 'single', '1.5', or 'double'
    save_path : str, optional
        Path to save the figure
    filename : str, optional
        Filename for saved figure (without extension)
    cmap : str, default='RdBu_r'
        Colormap for the matrix
    vmin : float, optional
        Minimum value for color scale. If None, uses data minimum.
    vmax : float, optional
        Maximum value for color scale. If None, uses data maximum.
    show_colorbar : bool, default=True
        Whether to show the colorbar
    show_diagonal : bool, default=False
        Whether to show the diagonal (self-correlations)
    neuron_labels : list, optional
        Labels for neurons (if None, uses indices)
    show : bool, default=True
        Whether to display the plot
    """
    # Create copy to avoid modifying original
    matrix = sttc_matrix.copy()
    
    # Auto-compute vmin/vmax from data if not provided
    if vmin is None:
        vmin = np.nanmin(matrix) if not np.all(np.isnan(matrix)) else -1
    if vmax is None:
        vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1
    
    # Mask diagonal if requested
    if not show_diagonal:
        np.fill_diagonal(matrix, np.nan)
    
    # Create figure
    if styler is None:
        from braindance.utils.data_manager.utils.plotting.plot_styler import Styler
        styler = Styler(journal="draft")
    
    if width_pt and height_pt:
        fig, ax = styler.create_figure(width_pt=width_pt, height_pt=height_pt)
    else:
        fig, ax = styler.create_figure(size_preset=size_preset)
    
    # Plot matrix
    im = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='equal',
        interpolation='nearest'
    )
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('STTC')
    
    # Labels
    ax.set_xlabel('Neuron')
    ax.set_ylabel('Neuron')
    
    if title:
        ax.set_title(title)
    
    # Custom neuron labels if provided
    if neuron_labels is not None:
        ax.set_xticks(range(len(neuron_labels)))
        ax.set_yticks(range(len(neuron_labels)))
        ax.set_xticklabels(neuron_labels, rotation=45, ha='right')
        ax.set_yticklabels(neuron_labels)
    
    # Style the axes
    styler.style_heatmap(ax)
    
    # Handle saving
    if save_path is not None:
        if os.path.isdir(save_path):
            save_dir = save_path
            name = filename or 'sttc_matrix'
        else:
            save_dir = os.path.dirname(save_path) or '.'
            name = filename or os.path.splitext(os.path.basename(save_path))[0]
        styler.finish_plot(True, save_dir, name, show=False)
    else:
        plt.tight_layout()
        if show:
            plt.show()
    
    return fig, ax


def plot_firing_rate_histogram(
    spike_data,
    title: Optional[str] = None,
    styler: Optional['Styler'] = None,
    size_preset: str = 'single',
    bins: int = 30,
    log_scale: bool = True,
    save_path: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot histogram of firing rates across neurons.
    
    Parameters
    ----------
    spike_data : SpikeData
        SpikeData object
    title : str, optional
        Title for the plot
    styler : Styler, optional
        Styler object for formatting
    size_preset : str, default='single'
        Figure size preset
    bins : int, default=30
        Number of histogram bins
    log_scale : bool, default=True
        Whether to use log scale for x-axis
    save_path : str, optional
        Path to save figure
    filename : str, optional
        Filename for saved figure
    show : bool, default=True
        Whether to display the plot
    """
    if styler is None:
        from braindance.utils.data_manager.utils.plotting.plot_styler import Styler
        styler = Styler(journal="draft")
    
    fig, ax = styler.create_figure(size_preset=size_preset)
    
    # Calculate firing rates
    rates = spike_data.rates(unit='Hz')
    
    # Filter out zeros for log scale
    if log_scale:
        rates_plot = rates[rates > 0]
        if len(rates_plot) > 0:
            ax.hist(rates_plot, bins=bins, color=styler.colors[0], edgecolor='white', linewidth=0.5)
            ax.set_xscale('log')
        else:
            ax.text(0.5, 0.5, 'No non-zero rates for log scale', ha='center', va='center')
    else:
        ax.hist(rates, bins=bins, color=styler.colors[0], edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Firing Rate (Hz)')
    ax.set_ylabel('Count')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Firing Rate Distribution (n={len(rates)} neurons)')
    
    # Add summary stats
    if len(rates) > 0:
        mean_rate = np.mean(rates)
        median_rate = np.median(rates)
        ax.axvline(mean_rate, color=styler.colors[4], linestyle='--', linewidth=1, label=f'Mean: {mean_rate:.2f} Hz')
        ax.axvline(median_rate, color=styler.colors[1], linestyle='--', linewidth=1, label=f'Median: {median_rate:.2f} Hz')
        ax.legend(fontsize=6)
    
    # Handle saving
    if save_path is not None:
        if os.path.isdir(save_path):
            save_dir = save_path
            name = filename or 'firing_rate_hist'
        else:
            save_dir = os.path.dirname(save_path) or '.'
            name = filename or os.path.splitext(os.path.basename(save_path))[0]
        styler.finish_plot(True, save_dir, name, show=False)
    else:
        plt.tight_layout()
        if show:
            plt.show()
    
    return fig, ax
