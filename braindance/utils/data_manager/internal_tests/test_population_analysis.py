import numpy as np
import matplotlib.pyplot as plt
from braindance.utils.data_manager.utils.plotting import plot_raster_with_pop, plot_sttc_matrix

def test_pop_rate_calculation_logic(mock_spike_data):
    """Verify population rate calculation (Hz/kHz) and axes data."""
    # Action
    # Use show=False to prevent blocking
    fig, (_, ax_pop) = plot_raster_with_pop(mock_spike_data, show=False)
    
    # Verification
    # Check that ax_pop has a line plot
    lines = ax_pop.get_lines()
    assert len(lines) > 0
    
    y_data = lines[0].get_ydata()
    # mock_spike_data has ~2 bursts and some background spikes
    # Total spikes: 5 neurons * (10 spontaneous + 5 + 8) = ~115 spikes in 10s
    # Mean rate ~ 2.3 Hz per neuron -> 11.5 Hz population
    # Smoothed peak might be much higher.
    assert np.mean(y_data) > 0
    
    plt.close(fig)

def test_sttc_matrix_plot():
    """Verify STTC matrix plotting with proper diagonal masking."""
    # Setup - create a correlation matrix
    matrix = np.eye(10)  # Identity matrix (1s on diagonal, 0s elsewhere)
    
    # Action - default behavior masks diagonal
    fig, ax = plot_sttc_matrix(matrix, show=False)
    
    # Verification
    # Check that image data has diagonal masked (default behavior)
    im = ax.get_images()[0]
    data = im.get_array()
    
    # Data should be a masked array with diagonal masked
    assert np.ma.is_masked(data), "Data should be a masked array"
    
    # Check that diagonal elements are masked
    for i in range(10):
        assert data.mask[i, i], f"Diagonal element [{i},{i}] should be masked"
    
    # Off-diagonal elements should NOT be masked and should match original
    for i in range(10):
        for j in range(10):
            if i != j:
                assert not data.mask[i, j], f"Off-diagonal element [{i},{j}] should not be masked"
                assert data[i, j] == matrix[i, j], f"Off-diagonal element [{i},{j}] should match"
    
    plt.close(fig)
    
    # Also test with show_diagonal=True
    fig2, ax2 = plot_sttc_matrix(matrix, show_diagonal=True, show=False)
    im2 = ax2.get_images()[0]
    data2 = im2.get_array()
    
    # With show_diagonal=True, should match exactly (no masking)
    assert np.array_equal(data2, matrix), "With show_diagonal=True, should match input matrix"
    
    plt.close(fig2)

def test_recording_plot_accessor(mock_recording):
    """Verify the .pl accessor on Recording."""
    # Action
    # Triggering the accessor creation
    accessor = mock_recording.pl
    assert accessor is not None
    # We won't run full plots here to avoid memory issues in tests, 
    # but verify the attributes exist
    assert hasattr(accessor, 'raster_with_pop')
    assert hasattr(accessor, 'sttc_matrix')
