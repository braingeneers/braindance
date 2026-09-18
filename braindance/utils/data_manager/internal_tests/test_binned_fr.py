import pytest
import numpy as np
from braindance.utils.data_manager.utils.analysis.vectorized_binning import bin_spike_data_vectorized

def test_binning_exact_counts():
    """Verify bin counts match manual placement."""
    # Neuron 0: Spikes at 10ms, 50ms (Bin 0: 0-100ms) -> count 2
    # Neuron 1: Spike at 150ms (Bin 1: 100-200ms) -> count 1
    # Note: bin_spike_data_vectorized expects list of sorted np.arrays
    spikes = [np.array([10.0, 50.0]), np.array([150.0])] 
    
    # Action
    # bin_size_ms=100
    binned, times = bin_spike_data_vectorized(spikes, bin_size_ms=100.0, time_range=(0.0, 200.0))
    
    # Verification
    # Shape: (2 bins, 2 neurons)
    assert binned.shape == (2, 2)
    assert binned[0, 0] == 2 # Neuron 0, Bin 0
    assert binned[0, 1] == 0 # Neuron 1, Bin 0
    assert binned[1, 1] == 1 # Neuron 1, Bin 1
    assert binned[1, 0] == 0 # Neuron 0, Bin 1
    
    # Times should be bin centers: 50.0, 150.0
    assert np.allclose(times, [50.0, 150.0])

def test_binning_invalid_range():
    """Verify behavior with invalid time range."""
    spikes = [np.array([10.0])]
    # time_range=(100, 0) is invalid
    with pytest.raises(ValueError):
        bin_spike_data_vectorized(spikes, bin_size_ms=10, time_range=(100, 0))

def test_binning_caching(mock_recording):
    """Verify that Recording.get_binned_fr uses the ResultsCache."""
    # Setup
    mock_recording.clear_cache()
    
    # Action 1: First call computes and caches
    res1 = mock_recording.get_binned_fr(bin_ms=100.0)
    assert 'rates' in res1
    
    # Action 2: Second call should be from cache
    # We can verify by checking the cache object
    assert mock_recording.cache.exists_local('binned_fr', {'bin_ms': 100.0})
    
    res2 = mock_recording.get_binned_fr(bin_ms=100.0)
    assert np.array_equal(res1['rates'], res2['rates'])
