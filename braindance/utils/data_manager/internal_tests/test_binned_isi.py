import numpy as np
from braindance.utils.data_manager.utils.analysis.vectorized_binning import (
    compute_binned_isi_vectorized,
    compute_binned_fr_isi_vectorized
)

def test_isi_exact_values():
    """Verify ISI calculation with manual placement."""
    # Neuron 0: 3 spikes in Bin 0 (0-100ms) at [10, 40, 80]
    # ISIs: (40-10)=30, (80-40)=40. Mean ISI = (30+40)/2 = 35.0ms
    # After normalization (bin_size=100): 35 / 100 = 0.35
    
    # Neuron 1: 2 spikes in Bin 1 (100-200ms) at [120, 150]
    # ISIs: (150-120)=30. Mean ISI = 30.0ms
    # After normalization: 30 / 100 = 0.3
    
    # Neuron 2: 1 spike in Bin 0 [50]
    # No ISI defined, should use sentinel (bin_size=100)
    # After normalization: 100 / 100 = 1.0
    
    spikes = [
        np.array([10.0, 40.0, 80.0]), # Neuron 0
        np.array([120.0, 150.0]),      # Neuron 1
        np.array([50.0])               # Neuron 2
    ]
    
    # Action
    bin_size = 100.0
    isi, _ = compute_binned_isi_vectorized(spikes, bin_size_ms=bin_size, time_range=(0.0, 200.0), normalize=True)
    
    # Verification
    # Shape: (2 bins, 3 neurons)
    assert isi.shape == (2, 3)
    
    # Neuron 0, Bin 0: Expect 0.35
    assert np.isclose(isi[0, 0], 0.35)
    
    # Neuron 1, Bin 1: Expect 0.3
    assert np.isclose(isi[1, 1], 0.3)
    
    # Neuron 2, Bin 0: Expect 1.0 (sentinel)
    assert np.isclose(isi[0, 2], 1.0)
    
    # Empty bins: Expect 1.0 (sentinel)
    assert np.isclose(isi[0, 1], 1.0)
    assert np.isclose(isi[1, 0], 1.0)
    assert np.isclose(isi[1, 2], 1.0)

def test_isi_no_normalization():
    """Verify ISI calculation without normalization."""
    spikes = [np.array([10.0, 40.0, 80.0])]
    isi, _ = compute_binned_isi_vectorized(spikes, bin_size_ms=100.0, time_range=(0.0, 100.0), normalize=False)
    
    assert np.isclose(isi[0, 0], 35.0)

def test_isi_custom_sentinel():
    """Verify custom sentinel value handling."""
    spikes = [np.array([10.0])] # 1 spike -> no ISI
    isi, _ = compute_binned_isi_vectorized(spikes, bin_size_ms=100.0, time_range=(0.0, 100.0), sentinel_value=-1.0, normalize=False)
    
    assert isi[0, 0] == -1.0

def test_recording_get_binned_isi(mock_recording):
    """Verify that Recording.get_binned_isi works and uses cache."""
    mock_recording.clear_cache()
    
    # Action 1: Compute
    res1 = mock_recording.get_binned_isi(bin_ms=100.0)
    assert 'isi' in res1
    assert 'time_axis_ms' in res1
    
    # Action 2: Check cache
    assert mock_recording.cache.exists_local('binned_isi', {'bin_ms': 100.0})
    
    # Action 3: Load from cache
    res2 = mock_recording.get_binned_isi(bin_ms=100.0)
    assert np.array_equal(res1['isi'], res2['isi'])

def test_recording_get_binned_fr_and_isi(mock_recording):
    """Verify joint FR and ISI computation."""
    mock_recording.clear_cache()
    
    res = mock_recording.get_binned_fr_and_isi(bin_ms=100.0)
    assert 'rates' in res
    assert 'isi' in res
    assert 'time_axis_ms' in res
    
    # Check shape compatibility
    assert res['rates'].shape == res['isi'].shape
    
    # Verify cache
    assert mock_recording.cache.exists_local('binned_fr_isi', {'bin_ms': 100.0})

def test_isi_joint_exact_values():
    """Verify compute_binned_fr_isi_vectorized matches individual calls."""
    spikes = [
        np.array([10.0, 40.0, 80.0]),
        np.array([120.0, 150.0]),
        np.array([50.0])
    ]
    bin_size = 100.0
    
    # Action
    fr, isi, times = compute_binned_fr_isi_vectorized(
        spikes, bin_size_ms=bin_size, time_range=(0.0, 200.0), normalize_isi=True
    )
    
    # Verification
    assert fr.shape == (2, 3)
    assert isi.shape == (2, 3)
    
    # FR checks
    assert fr[0, 0] == 3 # Neuron 0, Bin 0: [10, 40, 80]
    assert fr[1, 1] == 2 # Neuron 1, Bin 1: [120, 150]
    assert fr[0, 2] == 1 # Neuron 2, Bin 0: [50]
    
    # ISI checks (same as test_isi_exact_values)
    assert np.isclose(isi[0, 0], 0.35)
    assert np.isclose(isi[1, 1], 0.3)
    assert np.isclose(isi[0, 2], 1.0) # Sentinel

def test_isi_performance():
    """Verify that ISI computation is fast (vetorized) and not brute-force."""
    import time
    
    # Generate 500 neurons, ~1 million spikes total over 10 minutes
    n_neurons = 500
    duration_ms = 600_000 # 10 minutes
    n_spikes_per_neuron = 2000
    
    print(f"\n  Generating {n_neurons * n_spikes_per_neuron} synthetic spikes...")
    spikes = [
        np.sort(np.random.uniform(0, duration_ms, n_spikes_per_neuron))
        for _ in range(n_neurons)
    ]
    
    # Time the joint computation
    start_time = time.time()
    fr, isi, times = compute_binned_fr_isi_vectorized(
        spikes, bin_size_ms=20.0, time_range=(0, duration_ms), verbose=False
    )
    duration = time.time() - start_time
    
    print(f"  Processed {n_neurons} neurons, {len(times)} bins in {duration:.4f}s")
    
    # Requirement: Should definitely be under 500ms for 1M spikes on modern HW
    # Brute force (nested loops) would take MINUTES for this scale.
    assert duration < 0.5, f"ISI computation too slow ({duration:.4f}s). Is it still vectorized?"
    assert fr.shape == (len(times), n_neurons)
    assert isi.shape == (len(times), n_neurons)
