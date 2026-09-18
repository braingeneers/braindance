"""
Vectorized spike binning utilities for LSTM preprocessing.

This module provides highly optimized functions for converting spike trains
into binned count data suitable for LSTM models. Uses numpy vectorization
and sparse matrices for maximum performance.

Author: BrainGenEERS Lab
Date: 2024
"""

import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Optional


def bin_spike_data_vectorized(spike_trains: List[np.ndarray], 
                             bin_size_ms: float = 100.0,
                             time_range: Optional[Tuple[float, float]] = None,
                             return_sparse: bool = False,
                             verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized binning of multiple spike trains into count matrix.
    
    This function is optimized for speed and memory efficiency using numpy
    vectorization and optional sparse matrix representation.
    
    Args:
        spike_trains: List of spike time arrays (in milliseconds)
        bin_size_ms: Size of each time bin in milliseconds
        time_range: Optional (start_time, end_time) in ms. If None, uses data range
        return_sparse: Whether to return sparse matrix (saves memory for sparse data)
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (binned_data, time_axis) where:
        - binned_data: (n_time_bins, n_neurons) count matrix
        - time_axis: Array of bin center times in milliseconds
    """
    if not spike_trains:
        raise ValueError("Empty spike_trains list provided")
    
    # Determine time range
    if time_range is None:
        all_spikes = np.concatenate([train for train in spike_trains if len(train) > 0])
        if len(all_spikes) == 0:
            raise ValueError("No spikes found in any train")
        start_time = float(np.min(all_spikes))
        end_time = float(np.max(all_spikes))
    else:
        start_time, end_time = time_range
    
    # Create time bins
    bin_edges = np.arange(start_time, end_time + bin_size_ms, bin_size_ms)
    n_bins = len(bin_edges) - 1
    n_neurons = len(spike_trains)
    
    if verbose:
        print(f"Binning {n_neurons} neurons over {n_bins} time bins "
              f"({bin_size_ms}ms bins, {(end_time-start_time)/1000:.1f}s total)")
    
    # Initialize output matrix
    if return_sparse:
        # Use sparse matrix for memory efficiency
        binned_data = sp.lil_matrix((n_bins, n_neurons), dtype=np.int32)
    else:
        binned_data = np.zeros((n_bins, n_neurons), dtype=np.int32)
    
    # Process each neuron
    for neuron_idx, spike_times in enumerate(spike_trains):
        if len(spike_times) == 0:
            continue
            
        # Filter spikes within time range
        valid_spikes = spike_times[(spike_times >= start_time) & (spike_times < end_time)]
        
        if len(valid_spikes) == 0:
            continue
        
        # Vectorized binning using histogram
        counts, _ = np.histogram(valid_spikes, bins=bin_edges)
        
        # Store in output matrix
        if return_sparse:
            binned_data[:, neuron_idx] = counts.reshape(-1, 1)
        else:
            binned_data[:, neuron_idx] = counts
    
    # Convert sparse matrix to dense if needed
    if return_sparse:
        binned_data = binned_data.tocsr()  # Convert to CSR for efficient operations
    
    # Create time axis (bin centers)
    time_axis = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    if verbose:
        total_spikes = np.sum(binned_data)
        density = (total_spikes / (n_bins * n_neurons)) * 100
        print(f"Binning complete: {total_spikes} total spikes, "
              f"{density:.2f}% density")
    
    return binned_data, time_axis


def bin_spike_data_chunked(spike_trains: List[np.ndarray],
                          bin_size_ms: float = 100.0,
                          time_range: Optional[Tuple[float, float]] = None,
                          chunk_size_neurons: int = 1000,
                          return_sparse: bool = False,
                          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-efficient chunked binning for very large datasets.
    
    Processes neurons in chunks to avoid memory overflow with large datasets.
    
    Args:
        spike_trains: List of spike time arrays (in milliseconds)
        bin_size_ms: Size of each time bin in milliseconds
        time_range: Optional (start_time, end_time) in ms
        chunk_size_neurons: Number of neurons to process at once
        return_sparse: Whether to return sparse matrix
        verbose: Whether to print progress
        
    Returns:
        Tuple of (binned_data, time_axis)
    """
    n_neurons = len(spike_trains)
    
    if n_neurons <= chunk_size_neurons:
        # Use regular binning for small datasets
        return bin_spike_data_vectorized(spike_trains, bin_size_ms, time_range, 
                                       return_sparse, verbose)
    
    # Determine time range first
    if time_range is None:
        all_spikes = np.concatenate([train for train in spike_trains if len(train) > 0])
        if len(all_spikes) == 0:
            raise ValueError("No spikes found in any train")
        start_time = float(np.min(all_spikes))
        end_time = float(np.max(all_spikes))
    else:
        start_time, end_time = time_range
    
    # Create time bins
    bin_edges = np.arange(start_time, end_time + bin_size_ms, bin_size_ms)
    n_bins = len(bin_edges) - 1
    time_axis = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    if verbose:
        print(f"Chunked binning: {n_neurons} neurons, {n_bins} time bins, "
              f"chunk size: {chunk_size_neurons}")
    
    # Process in chunks
    chunks = []
    for chunk_start in range(0, n_neurons, chunk_size_neurons):
        chunk_end = min(chunk_start + chunk_size_neurons, n_neurons)
        chunk_trains = spike_trains[chunk_start:chunk_end]
        
        if verbose:
            print(f"Processing chunk {chunk_start//chunk_size_neurons + 1}/"
                  f"{(n_neurons-1)//chunk_size_neurons + 1} "
                  f"(neurons {chunk_start}-{chunk_end-1})")
        
        # Bin this chunk
        chunk_binned, _ = bin_spike_data_vectorized(
            chunk_trains, bin_size_ms, (start_time, end_time), 
            return_sparse=False, verbose=False
        )
        
        chunks.append(chunk_binned)
    
    # Concatenate all chunks
    binned_data = np.hstack(chunks)
    
    # Convert to sparse if requested
    if return_sparse:
        binned_data = sp.csr_matrix(binned_data)
    
    if verbose:
        total_spikes = np.sum(binned_data)
        density = (total_spikes / (n_bins * n_neurons)) * 100
        print(f"Chunked binning complete: {total_spikes} total spikes, "
              f"{density:.2f}% density")
    
    return binned_data, time_axis


def create_sliding_windows(data: np.ndarray, 
                          window_size: int, 
                          step_size: int = 1,
                          return_sparse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series data.
    
    Optimized for creating input sequences for LSTM models.
    
    Args:
        data: Input data array (n_timepoints, n_features)
        window_size: Size of each window
        step_size: Step size between windows
        return_sparse: Whether to return sparse arrays
        
    Returns:
        Tuple of (windows, targets) where:
        - windows: (n_windows, window_size, n_features) array
        - targets: (n_windows, n_features) array (next timepoint after each window)
    """
    n_timepoints, n_features = data.shape
    
    # Calculate number of windows
    n_windows = (n_timepoints - window_size) // step_size
    
    if n_windows <= 0:
        raise ValueError(f"Not enough timepoints ({n_timepoints}) for window_size ({window_size})")
    
    # Pre-allocate arrays
    if return_sparse and hasattr(data, 'toarray'):
        # Convert sparse data to dense for windowing, then back to sparse
        data_dense = data.toarray()
        windows = np.zeros((n_windows, window_size, n_features), dtype=data_dense.dtype)
        targets = np.zeros((n_windows, n_features), dtype=data_dense.dtype)
    else:
        windows = np.zeros((n_windows, window_size, n_features), dtype=data.dtype)
        targets = np.zeros((n_windows, n_features), dtype=data.dtype)
    
    # Vectorized window creation
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        target_idx = end_idx
        
        if return_sparse and hasattr(data, 'toarray'):
            windows[i] = data_dense[start_idx:end_idx]
            targets[i] = data_dense[target_idx]
        else:
            windows[i] = data[start_idx:end_idx]
            targets[i] = data[target_idx]
    
    return windows, targets



def validate_spike_data(spike_trains: List[np.ndarray], 
                       expected_length_ms: Optional[float] = None,
                       verbose: bool = True) -> dict:
    """
    Validate spike train data quality and characteristics.
    
    Args:
        spike_trains: List of spike time arrays
        expected_length_ms: Expected recording length in milliseconds
        verbose: Whether to print validation results
        
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'n_neurons': len(spike_trains),
        'total_spikes': 0,
        'empty_neurons': 0,
        'min_spike_time': float('inf'),
        'max_spike_time': float('-inf'),
        'firing_rates_hz': [],
        'issues': []
    }
    
    for i, train in enumerate(spike_trains):
        if len(train) == 0:
            stats['empty_neurons'] += 1
            continue
            
        # Check for valid spike times
        if np.any(train < 0):
            stats['issues'].append(f"Neuron {i}: negative spike times")
        
        if np.any(np.diff(train) < 0):
            stats['issues'].append(f"Neuron {i}: unsorted spike times")
        
        # Update statistics
        stats['total_spikes'] += len(train)
        stats['min_spike_time'] = min(stats['min_spike_time'], np.min(train))
        stats['max_spike_time'] = max(stats['max_spike_time'], np.max(train))
        
        # Calculate firing rate
        if expected_length_ms:
            firing_rate = len(train) / (expected_length_ms / 1000.0)
            stats['firing_rates_hz'].append(firing_rate)
    
    # Convert to numpy arrays
    stats['firing_rates_hz'] = np.array(stats['firing_rates_hz'])
    
    # Calculate derived statistics
    recording_length_ms = stats['max_spike_time'] - stats['min_spike_time']
    stats['recording_length_ms'] = recording_length_ms
    stats['overall_firing_rate_hz'] = stats['total_spikes'] / (recording_length_ms / 1000.0)
    
    if verbose:
        print(f"Spike data validation:")
        print(f"  Neurons: {stats['n_neurons']} ({stats['empty_neurons']} empty)")
        print(f"  Total spikes: {stats['total_spikes']}")
        print(f"  Recording length: {recording_length_ms/1000:.1f}s")
        print(f"  Overall firing rate: {stats['overall_firing_rate_hz']:.2f} Hz")
        
        if len(stats['firing_rates_hz']) > 0:
            print(f"  Firing rate stats: {np.mean(stats['firing_rates_hz']):.2f} ± "
                  f"{np.std(stats['firing_rates_hz']):.2f} Hz")
        
        if stats['issues']:
            print(f"  Issues found: {len(stats['issues'])}")
            for issue in stats['issues'][:5]:  # Show first 5 issues
                print(f"    {issue}")
    
    return stats


def compute_binned_isi_vectorized(
    spike_trains: List[np.ndarray],
    bin_size_ms: float = 20.0,
    time_range: Optional[Tuple[float, float]] = None,
    sentinel_value: Optional[float] = None,
    normalize: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean ISI per neuron per time bin using fully vectorized operations.
    
    This algorithm avoids per-neuron/per-bin loops by processing all spikes 
    simultaneously using lexsort and bincount.
    
    Complexity: O(N log N) where N is total spikes (limited by sort).
    
    Args:
        spike_trains: List of spike time arrays (in milliseconds)
        bin_size_ms: Size of each time bin in milliseconds
        time_range: Optional (start_time, end_time) in ms. If None, uses data range.
        sentinel_value: Value for bins with < 2 spikes. Default: bin_size_ms.
        normalize: If True, divides result by bin_size_ms (result in [0, 1] range).
        verbose: Print progress information.
        
    Returns:
        Tuple of (isi_per_bin, time_axis) where:
        - isi_per_bin: (n_time_bins, n_neurons) mean ISI matrix
        - time_axis: Array of bin center times in milliseconds
    """
    if not spike_trains:
        raise ValueError("Empty spike_trains list provided")

    # Determine time range
    if time_range is None:
        all_spikes_flat = np.concatenate([train for train in spike_trains if len(train) > 0])
        if len(all_spikes_flat) == 0:
            raise ValueError("No spikes found in any train")
        start_time = float(np.min(all_spikes_flat))
        end_time = float(np.max(all_spikes_flat))
    else:
        start_time, end_time = time_range

    # Create time bins
    bin_edges = np.arange(start_time, end_time + bin_size_ms, bin_size_ms)
    n_bins = len(bin_edges) - 1
    n_neurons = len(spike_trains)
    
    if sentinel_value is None:
        sentinel_value = bin_size_ms

    if verbose:
        print(f"Computing ISI for {n_neurons} neurons over {n_bins} bins "
              f"({bin_size_ms}ms bins, {(end_time-start_time)/1000:.1f}s total)")

    # Prepare data for vectorization: Flatten all spikes with neuron labels
    neuron_ids = []
    spike_times = []
    for i, train in enumerate(spike_trains):
        if len(train) > 0:
            neuron_ids.append(np.full(len(train), i, dtype=np.int32))
            spike_times.append(train)
    
    if not neuron_ids:
        # All neurons empty
        return np.full((n_bins, n_neurons), sentinel_value / bin_size_ms if normalize else sentinel_value, dtype=np.float32), (bin_edges[:-1] + bin_edges[1:]) / 2

    all_neuron_ids = np.concatenate(neuron_ids)
    all_spike_times = np.concatenate(spike_times)

    # Step 1: Sort spikes by neuron_id, then by time
    sort_indices = np.lexsort((all_spike_times, all_neuron_ids))
    sorted_neuron_ids = all_neuron_ids[sort_indices]
    sorted_spike_times = all_spike_times[sort_indices]

    # Step 2: Calculate ISIs between consecutive spikes (within same neuron only)
    same_neuron_mask = sorted_neuron_ids[:-1] == sorted_neuron_ids[1:]
    all_isis = np.diff(sorted_spike_times)

    # Only keep ISIs where both spikes are from same neuron
    valid_isis = all_isis[same_neuron_mask]
    valid_neuron_ids = sorted_neuron_ids[1:][same_neuron_mask]
    valid_second_spike_times = sorted_spike_times[1:][same_neuron_mask]

    # Initialize with sentinel value
    # If normalizing, sentinel_value / bin_size_ms (e.g. 20/20 = 1.0)
    fill_val = sentinel_value / bin_size_ms if normalize else sentinel_value
    isi_per_bin = np.full((n_bins, n_neurons), fill_val, dtype=np.float32)

    if len(valid_isis) > 0:
        # Step 3: Assign each ISI to its (bin, neuron) position
        # Filter spikes within time range
        in_range = (valid_second_spike_times >= start_time) & (valid_second_spike_times < end_time)
        valid_isis = valid_isis[in_range]
        valid_neuron_ids = valid_neuron_ids[in_range]
        valid_times = valid_second_spike_times[in_range]
        
        if len(valid_isis) > 0:
            spike_bins = np.digitize(valid_times, bin_edges) - 1
            spike_bins = np.clip(spike_bins, 0, n_bins - 1)

            # Step 4: Use bincount to accumulate ISI sums and counts
            flat_indices = spike_bins * n_neurons + valid_neuron_ids
            
            isi_counts = np.bincount(flat_indices, minlength=n_bins * n_neurons)
            isi_sums = np.bincount(flat_indices, weights=valid_isis, minlength=n_bins * n_neurons)
            
            isi_counts = isi_counts.reshape(n_bins, n_neurons)
            isi_sums = isi_sums.reshape(n_bins, n_neurons)

            # Compute mean ISI per bin where counts > 0
            has_isi = isi_counts > 0
            isi_per_bin[has_isi] = isi_sums[has_isi] / isi_counts[has_isi]
            
            # Normalization
            if normalize:
                isi_per_bin[has_isi] = isi_per_bin[has_isi] / bin_size_ms

    # Create time axis (bin centers)
    time_axis = (bin_edges[:-1] + bin_edges[1:]) / 2

    return isi_per_bin, time_axis


def compute_binned_fr_isi_vectorized(
    spike_trains: List[np.ndarray],
    bin_size_ms: float = 20.0,
    time_range: Optional[Tuple[float, float]] = None,
    isi_sentinel_value: Optional[float] = None,
    normalize_isi: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute BOTH firing rates and mean ISI in a single efficient pass.
    
    Shares the expensive sorting and digitizing operations.
    Recommended for model training where both features are needed.
    
    Args:
        spike_trains: List of spike time arrays (in milliseconds)
        bin_size_ms: Size of each time bin in milliseconds
        time_range: Optional (start_time, end_time) in ms
        isi_sentinel_value: Value for bins with < 2 spikes. Default: bin_size_ms.
        normalize_isi: If True, divides ISI by bin_size_ms.
        verbose: Print progress information.
        
    Returns:
        Tuple of (binned_fr, binned_isi, time_axis) where:
        - binned_fr: (n_time_bins, n_neurons) spike count matrix
        - binned_isi: (n_time_bins, n_neurons) mean ISI matrix
        - time_axis: Array of bin center times in milliseconds
    """
    if not spike_trains:
        raise ValueError("Empty spike_trains list provided")

    # Determine time range
    if time_range is None:
        all_spikes_flat = np.concatenate([train for train in spike_trains if len(train) > 0])
        if len(all_spikes_flat) == 0:
            raise ValueError("No spikes found in any train")
        start_time = float(np.min(all_spikes_flat))
        end_time = float(np.max(all_spikes_flat))
    else:
        start_time, end_time = time_range

    # Create time bins
    bin_edges = np.arange(start_time, end_time + bin_size_ms, bin_size_ms)
    n_bins = len(bin_edges) - 1
    n_neurons = len(spike_trains)
    
    if isi_sentinel_value is None:
        isi_sentinel_value = bin_size_ms

    if verbose:
        print(f"Jointly computing FR/ISI for {n_neurons} neurons over {n_bins} bins "
              f"({bin_size_ms}ms bins, {(end_time-start_time)/1000:.1f}s total)")

    # Prepare data for vectorization
    neuron_ids_list = []
    spike_times_list = []
    for i, train in enumerate(spike_trains):
        if len(train) > 0:
            neuron_ids_list.append(np.full(len(train), i, dtype=np.int32))
            spike_times_list.append(train)
    
    if not neuron_ids_list:
        # All neurons empty
        fr_empty = np.zeros((n_bins, n_neurons), dtype=np.int32)
        isi_empty = np.full((n_bins, n_neurons), isi_sentinel_value / bin_size_ms if normalize_isi else isi_sentinel_value, dtype=np.float32)
        return fr_empty, isi_empty, (bin_edges[:-1] + bin_edges[1:]) / 2

    all_neuron_ids = np.concatenate(neuron_ids_list)
    all_spike_times = np.concatenate(spike_times_list)

    # Step 1: Sort ALL spikes by neuron_id, then by time
    sort_indices = np.lexsort((all_spike_times, all_neuron_ids))
    sorted_neuron_ids = all_neuron_ids[sort_indices]
    sorted_spike_times = all_spike_times[sort_indices]

    # Step 2: Firing Rate calculation (counts per bin)
    # Digitize all spikes
    all_spike_bins = np.digitize(sorted_spike_times, bin_edges) - 1
    
    # Filter for spikes in range
    valid_mask = (all_spike_bins >= 0) & (all_spike_bins < n_bins)
    valid_bins = all_spike_bins[valid_mask]
    valid_neuron_ids_fr = sorted_neuron_ids[valid_mask]
    
    # Use bincount for FR
    flat_indices_fr = valid_bins * n_neurons + valid_neuron_ids_fr
    binned_fr = np.bincount(flat_indices_fr, minlength=n_bins * n_neurons).reshape(n_bins, n_neurons).astype(np.int32)

    # Step 3: ISI calculation (mean time diff per bin)
    # Consecutive spikes within SAME neuron
    same_neuron_mask = sorted_neuron_ids[:-1] == sorted_neuron_ids[1:]
    all_isis = np.diff(sorted_spike_times)

    # Only keep ISIs where both spikes are from same neuron
    valid_isis = all_isis[same_neuron_mask]
    valid_neuron_ids_isi = sorted_neuron_ids[1:][same_neuron_mask]
    valid_second_spike_times = sorted_spike_times[1:][same_neuron_mask]

    # Initialize ISI with sentinel value
    fill_val = isi_sentinel_value / bin_size_ms if normalize_isi else isi_sentinel_value
    binned_isi = np.full((n_bins, n_neurons), fill_val, dtype=np.float32)

    if len(valid_isis) > 0:
        # Digitizing second spikes of pairs
        isi_spike_bins = np.digitize(valid_second_spike_times, bin_edges) - 1
        isi_in_range = (isi_spike_bins >= 0) & (isi_spike_bins < n_bins)
        
        if np.any(isi_in_range):
            v_isis = valid_isis[isi_in_range]
            v_bins = isi_spike_bins[isi_in_range]
            v_neurons = valid_neuron_ids_isi[isi_in_range]
            
            flat_indices_isi = v_bins * n_neurons + v_neurons
            
            isi_counts = np.bincount(flat_indices_isi, minlength=n_bins * n_neurons)
            isi_sums = np.bincount(flat_indices_isi, weights=v_isis, minlength=n_bins * n_neurons)
            
            isi_counts = isi_counts.reshape(n_bins, n_neurons)
            isi_sums = isi_sums.reshape(n_bins, n_neurons)
            
            has_isi = isi_counts > 0
            binned_isi[has_isi] = isi_sums[has_isi] / isi_counts[has_isi]
            
            if normalize_isi:
                binned_isi[has_isi] = binned_isi[has_isi] / bin_size_ms

    # Time axis
    time_axis = (bin_edges[:-1] + bin_edges[1:]) / 2

    return binned_fr, binned_isi, time_axis