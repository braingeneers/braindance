import numpy as np
from braindance.utils.data_manager.utils.analysis.burst_detector import BurstDetector

def test_burst_detection_basic(mock_spike_data):
    """Verify that known bursts are detected."""
    # Setup - use small bin size for high resolution in test
    detector = BurstDetector(mock_spike_data, bin_size=1.0, smoothing_window=50)
    
    # Action
    results = detector.detect_bursts(use_cache=False)
    
    # Verification
    # We expect at least 2 bursts (at 2000 and 5000ms)
    assert results.n_bursts >= 2
    
    # Check if a burst exists near 2000ms
    burst_times = results.burst_times
    found_2000 = any(np.isclose(t[0], 2000, atol=100) for t in burst_times)
    assert found_2000, f"Burst at 2000ms not found in {burst_times}"

def test_burst_metrics_consistency(mock_spike_data):
    """Verify that derived metrics match manual calculations."""
    detector = BurstDetector(mock_spike_data, compute_all_metrics=True)
    results = detector.detect_bursts(use_cache=False)
    
    # Verify burst_frequency = n_bursts / duration_sec
    duration_sec = mock_spike_data.length / 1000.0
    expected_freq = results.n_bursts / duration_sec
    assert np.isclose(results.burst_frequency, expected_freq)
    
    # Mean width matches average of individual widths
    if len(results.burst_widths) > 0:
        assert np.isclose(results.mean_burst_width, np.mean(results.burst_widths))

def test_bic_matrix_calculation(mock_spike_data):
    """Verify Burst Involvement Coefficient (BIC) matrix shape and values."""
    detector = BurstDetector(mock_spike_data)
    results = detector.detect_bursts(use_cache=False)
    
    # BIC matrix is 1D: (n_neurons,) - represents involvement coefficient per neuron
    bic = results.bic_matrix
    assert bic.shape == (mock_spike_data.N,)
    
    # Values should be between 0 and 1
    assert np.all((bic >= 0) & (bic <= 1))

from spikelab import SpikeData

def test_burst_detection_no_spikes():
    """Verify robust handling of empty recordings."""
    # Use a real SpikeData object but with no spikes
    spike_data = SpikeData([np.array([]) for _ in range(5)], N=5, length=1000)
    
    detector = BurstDetector(spike_data)
    results = detector.detect_bursts(use_cache=False)
    
    assert results.n_bursts == 0
    assert len(results.burst_times) == 0
    # Metrics should be NaN, 0, or None, not crash
    assert results.burst_frequency == 0 or results.burst_frequency is None
    assert np.isnan(results.mean_burst_width) or results.mean_burst_width is None

def test_burst_detection_short_recording(mock_spike_data):
    """Verify handling of recording shorter than smoothing window."""
    # Mock a very short recording
    mock_spike_data.length = 10 
    detector = BurstDetector(mock_spike_data, smoothing_window=50)
    
    results = detector.detect_bursts(use_cache=False)
    assert results.n_bursts == 0
