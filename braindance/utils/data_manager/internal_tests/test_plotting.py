"""
Tests for plotting utilities and the Recording.pl accessor.

These are "smoke tests" that verify the plotting functions run without 
crashing and return the expected matplotlib objects.
"""

import pytest
import matplotlib.pyplot as plt
import numpy as np
from braindance.utils.data_manager import (
    plot_raster_with_pop, 
    plot_evoked_raster, 
    plot_evoked_psth,
    plot_sttc_matrix,
    plot_firing_rate_histogram,
    Styler
)

def test_imports_from_top_level():
    """Verify plotting functions can be imported from the top level."""
    from braindance.utils.data_manager import (
        plot_raster_with_pop,
        plot_evoked_raster,
        plot_evoked_psth,
        plot_sttc_matrix,
        plot_firing_rate_histogram,
        spatial_latency,
        Styler,
        PlotAccessor
    )
    assert plot_raster_with_pop is not None
    assert spatial_latency is not None

def test_raster_with_pop_execution(mock_spike_data):
    """Test standalone plot_raster_with_pop."""
    fig, axes = plot_raster_with_pop(mock_spike_data, show=False)
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 2
    plt.close(fig)

def test_firing_rate_hist_execution(mock_spike_data):
    """Test standalone plot_firing_rate_histogram."""
    fig, ax = plot_firing_rate_histogram(mock_spike_data, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_sttc_matrix_execution():
    """Test standalone plot_sttc_matrix."""
    matrix = np.random.rand(10, 10)
    fig, ax = plot_sttc_matrix(matrix, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_evoked_raster_execution(mock_spike_data, mock_stim_log):
    """Test standalone plot_evoked_raster."""
    # Electrode 1 is in our mock_stim_log
    fig = plot_evoked_raster(
        mock_spike_data, mock_stim_log, 
        electrode_id=1, neuron_idx=0, show=False
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_evoked_psth_execution(mock_spike_data, mock_stim_log):
    """Test standalone plot_evoked_psth."""
    fig = plot_evoked_psth(
        mock_spike_data, mock_stim_log, 
        electrode_id=1, neuron_idx=0, show=False
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_recording_pl_accessor(mock_recording, mock_stim_log):
    """Test the rec.pl accessor methods."""
    # Inject stim_log manually since conftest.py doesn't create the file on disk
    mock_recording._data._data['stim_log'] = mock_stim_log
    
    # Test raster_with_pop
    fig, axes = mock_recording.pl.raster_with_pop(show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test firing_rate_hist
    fig, ax = mock_recording.pl.firing_rate_hist(show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test sttc_matrix (with specific matrix to avoid heavy computation/errors)
    matrix = np.random.rand(5, 5)
    fig, ax = mock_recording.pl.sttc_matrix(sttc_matrix=matrix, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test evoked plots
    fig = mock_recording.pl.evoked_raster(electrode_id=1, neuron_idx=0, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    fig = mock_recording.pl.evoked_psth(electrode_id=1, neuron_idx=0, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_styler_customization(mock_spike_data):
    """Verify Styler can be customized and passed to plots."""
    custom_styler = Styler()
    assert hasattr(custom_styler, 'create_figure')
    
    fig, _ = plot_raster_with_pop(
        mock_spike_data, 
        styler=custom_styler, 
        size_preset='double',
        show=False
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_spatial_latency_execution(mock_recording):
    """Test spatial_latency through the accessor."""
    # Create mock latency results
    latency_results = {
        0: {'onset_latency': 10.5, 'response_strength': 2.0},
        1: {'onset_latency': 15.2, 'response_strength': 1.5}
    }
    
    # We need to mock spike_locations and mapping for this to work
    mock_recording.spike_locations = [(0, 0), (100, 100), (200, 200), (300, 300), (400, 400)]
    
    class MockMapping:
        def get_positions(self, electrodes=None):
            return [(0, 0)]
    mock_recording.mapping = MockMapping()
    
    fig = mock_recording.pl.spatial_latency(
        electrode_id=1, 
        latency_results=latency_results,
        show=False
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
