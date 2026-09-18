"""
Burst detection and analysis for neural spike data.

This module provides methods to detect bursts in spike data, calculate burst metrics,
and classify neurons based on their involvement in bursts.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import variation
from typing import Dict, List, Optional, Any


class BurstDetector:
    """
    A modular class for detecting and analyzing bursts in neural spike data.
    
    This class provides methods to detect bursts in spike data, calculate burst metrics,
    and classify neurons based on their involvement in bursts.
    
    Parameters
    ----------
    spike_data : SpikeData
        SpikeData object containing spike trains for multiple neurons
    bin_size : float, optional
        Size of time bins in ms for population activity calculation, default=1.0
    smoothing_window : int, optional
        Size of the smoothing window for population activity, default=50
    burst_detection_params : dict, optional
        Parameters for burst detection, including:
        - baseline_percentile: percentile for baseline calculation (default=25)
        - peak_threshold_factor: factor for peak threshold (default=2.5)
        - peak_distance: minimum distance between peaks (default=200)
        - peak_prominence: minimum prominence for peaks (default=0.5)
    burst_edge_params : dict, optional
        Parameters for burst edge detection, including:
        - edge_threshold_factor: factor for edge threshold (default=0.2)
        - min_burst_width: minimum width of a burst in ms (default=20)
        - max_burst_width: maximum width of a burst in ms (default=500)
    backbone_threshold : float, optional
        Threshold for classifying a neuron as rigid, default=0.9
    compute_all_metrics : bool, optional
        Whether to compute all metrics or just the basic ones, default=True
    """
    
    def __init__(self, spike_data, 
                 bin_size: float = 1.0, 
                 smoothing_window: int = 50, 
                 burst_detection_params: Optional[Dict] = None,
                 burst_edge_params: Optional[Dict] = None,
                 backbone_threshold: float = 0.9,
                 compute_all_metrics: bool = True):
        
        self.spike_data = spike_data
        self.bin_size = bin_size
        self.smoothing_window = smoothing_window
        self.backbone_threshold = backbone_threshold
        self.compute_all_metrics = compute_all_metrics
        
        # Set default parameters if not provided
        self.burst_detection_params = burst_detection_params or {
            'baseline_percentile': 25,
            'peak_threshold_factor': 2.5,
            'peak_distance': 200,
            'peak_prominence': 0.5
        }
        
        self.burst_edge_params = burst_edge_params or {
            'edge_threshold_factor': 0.2,
            'min_burst_width': 20,
            'max_burst_width': 500
        }
        
        # Cache for storing results
        self._population_activity = None
        self._smoothed_activity = None
        self._results = None

    def _get_cache_params(self) -> Dict:
        """
        Generate parameters dict for cache key.

        Returns
        -------
        dict
            Dictionary of parameters that uniquely identify this burst detection configuration
        """
        return {
            'bin_size': float(self.bin_size),
            'smoothing_window': int(self.smoothing_window),
            'baseline_percentile': float(self.burst_detection_params['baseline_percentile']),
            'peak_threshold_factor': float(self.burst_detection_params['peak_threshold_factor']),
            'peak_distance': int(self.burst_detection_params['peak_distance']),
            'peak_prominence': float(self.burst_detection_params['peak_prominence']),
            'edge_threshold_factor': float(self.burst_edge_params['edge_threshold_factor']),
            'min_burst_width': int(self.burst_edge_params['min_burst_width']),
            'max_burst_width': int(self.burst_edge_params['max_burst_width']),
            'backbone_threshold': float(self.backbone_threshold)
        }

    def compute_population_activity(self):
        """
        Compute the population activity from spike trains.

        Returns
        -------
        numpy.ndarray
            Population activity histogram
        """
        duration = self.spike_data.length
        n_bins = int(duration / self.bin_size)
        bins = np.linspace(0, duration, n_bins + 1)

        _, all_spikes = self.spike_data.idces_times()
        population_activity, _ = np.histogram(all_spikes, bins=bins)

        self._population_activity = population_activity
        return population_activity
    
    def smooth_population_activity(self):
        """
        Apply smoothing to the population activity.
        
        Returns
        -------
        numpy.ndarray
            Smoothed population activity
        """
        if self._population_activity is None:
            self.compute_population_activity()
        
        # Create smoothing kernel
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        
        # Apply convolution for smoothing
        smoothed_activity = np.convolve(self._population_activity, kernel, mode='same')
        
        self._smoothed_activity = smoothed_activity
        return smoothed_activity
    
    def find_burst_peaks(self):
        """
        Find peaks in the smoothed population activity that correspond to bursts.
        
        Returns
        -------
        tuple
            (peak_indices, peak_info)
        """
        if self._smoothed_activity is None:
            self.smooth_population_activity()
        
        # Get burst detection parameters
        baseline_percentile = self.burst_detection_params['baseline_percentile']
        peak_threshold_factor = self.burst_detection_params['peak_threshold_factor']
        peak_distance = self.burst_detection_params['peak_distance']
        peak_prominence = self.burst_detection_params['peak_prominence']
        
        # Calculate baseline and threshold
        baseline = np.percentile(self._smoothed_activity, baseline_percentile)
        peak_threshold = baseline + peak_threshold_factor * np.std(self._smoothed_activity)
        
        # Find peaks
        peak_indices, peak_info = find_peaks(
            self._smoothed_activity,
            height=peak_threshold,
            distance=peak_distance,
            prominence=peak_prominence
        )
        
        return peak_indices, peak_info
    
    def find_burst_edges(self, peak_indices, peak_info):
        """
        Find the edges of bursts based on peak indices.
        
        Parameters
        ----------
        peak_indices : numpy.ndarray
            Indices of burst peaks
        peak_info : dict
            Information about peaks returned by find_peaks
        
        Returns
        -------
        numpy.ndarray
            Array of burst edges with shape (n_bursts, 2)
        """
        if self._smoothed_activity is None:
            self.smooth_population_activity()
        
        # Get burst edge parameters
        edge_threshold_factor = self.burst_edge_params['edge_threshold_factor']
        min_burst_width = self.burst_edge_params['min_burst_width']
        max_burst_width = self.burst_edge_params['max_burst_width']
        
        # Calculate edge thresholds based on peak heights and baseline
        baseline = np.percentile(self._smoothed_activity, self.burst_detection_params['baseline_percentile'])
        edge_thresholds = baseline + edge_threshold_factor * (peak_info['peak_heights'] - baseline)
        
        burst_edges = []
        
        for peak_i, edge_thr in zip(peak_indices, edge_thresholds):
            # Look for burst start
            burst_start = None
            for i in range(peak_i, max(peak_i - max_burst_width, 0), -1):
                if all(self._smoothed_activity[max(0, i-5):i+1] < edge_thr):
                    burst_start = i
                    break
            
            # Look for burst end
            burst_end = None
            for i in range(peak_i, min(peak_i + max_burst_width, len(self._smoothed_activity))):
                if all(self._smoothed_activity[i:min(i+5, len(self._smoothed_activity))] < edge_thr):
                    burst_end = i
                    break
            
            # Add burst if it meets criteria
            if (burst_start is not None and 
                burst_end is not None and 
                min_burst_width <= (burst_end - burst_start) <= max_burst_width):
                burst_edges.append([burst_start, burst_end])
        
        return np.array(burst_edges)
    
    def calculate_burst_involvement(self, burst_edges):
        """
        Calculate burst involvement coefficient (BIC) for each neuron.

        Parameters
        ----------
        burst_edges : numpy.ndarray
            Array of burst edges with shape (n_bursts, 2)

        Returns
        -------
        tuple
            (bic_matrix, backbone_classification)
        """
        if len(burst_edges) == 0:
            return np.array([]), {'rigid': [], 'nonrigid': []}

        total_bursts = len(burst_edges)
        total_neurons = self.spike_data.N

        # Initialize matrix to count neuron involvement in each burst
        burst_involvement_matrix = np.zeros((total_neurons, total_bursts), dtype=int)

        # Optimized: Get all spikes with their neuron indices at once
        idces, times = self.spike_data.idces_times()

        # For each burst, vectorize the check for spike involvement
        for i, (burst_start, burst_end) in enumerate(burst_edges):
            # Find all spikes within this burst window
            in_burst_mask = (times >= burst_start) & (times <= burst_end)
            burst_spike_idces = idces[in_burst_mask]

            # Count spikes per neuron in this burst
            spike_counts = np.bincount(burst_spike_idces, minlength=total_neurons)

            # Mark as involved if at least 2 spikes occur during burst
            burst_involvement_matrix[:, i] = (spike_counts >= 2).astype(int)

        # Calculate BIC for each neuron
        bic_matrix = np.sum(burst_involvement_matrix, axis=1) / total_bursts

        # Classify neurons as rigid or nonrigid using vectorized operations
        rigid_mask = bic_matrix >= self.backbone_threshold
        backbone_class = {
            'rigid': np.where(rigid_mask)[0].tolist(),
            'nonrigid': np.where(~rigid_mask)[0].tolist()
        }

        return bic_matrix, backbone_class
    
    def detect_bursts(
        self,
        cache=None,
        use_cache: bool = True,
        check_s3: bool = False,
        force_recompute: bool = False
    ) -> 'BurstResults':
        """
        Detect bursts and calculate burst metrics with automatic caching.

        Parameters
        ----------
        cache : ResultsCache, optional
            ResultsCache instance for caching results. If None, no caching is used.
        use_cache : bool, default=True
            Whether to use cached results if available
        check_s3 : bool, default=False
            Whether to check S3 for cached results (may be slower than recompute)
        force_recompute : bool, default=False
            Force recomputation bypassing cache

        Returns
        -------
        BurstResults
            Object containing burst detection results with clean property access
        """
        # Handle caching if cache is provided and use_cache is True
        if cache is not None and use_cache and not force_recompute:
            cache_params = self._get_cache_params()

            # Try to load from cache
            def compute_fn():
                return self._compute_bursts()

            try:
                cached_data = cache.get_or_compute(
                    'bursts',
                    params=cache_params,
                    compute_fn=compute_fn,
                    force_recompute=force_recompute
                )
                # Store results in detector instance for method access
                self._results = cached_data
                # Reconstruct BurstResults from cached data
                return BurstResults(cached_data, self.bin_size)
            except Exception as e:
                # If caching fails, fall back to direct computation
                print(f"Warning: Cache operation failed ({e}), computing directly")
                return BurstResults(self._compute_bursts(), self.bin_size)

        # No caching - compute directly
        results_data = self._compute_bursts()
        return BurstResults(results_data, self.bin_size)

    def _compute_bursts(self) -> Dict[str, Any]:
        """
        Internal method to compute burst detection results.

        Returns
        -------
        dict
            Dictionary containing all burst detection results
        """
        # Ensure population activity is calculated
        if self._smoothed_activity is None:
            self.smooth_population_activity()

        # Find burst peaks
        peak_indices, peak_info = self.find_burst_peaks()
        
        # Initialize results dictionary
        results = {
            'population_rate': self._smoothed_activity,
            'peak_indices': peak_indices,
            'peak_info': peak_info,
            'burst_edges': np.array([]).reshape(0, 2),
            'n_bursts': 0
        }
        
        # If we found peaks, find burst edges
        if len(peak_indices) > 0:
            burst_edges = self.find_burst_edges(peak_indices, peak_info)
            results['burst_edges'] = burst_edges
            results['n_bursts'] = len(burst_edges)
            
            # Calculate burst widths and amplitudes
            if len(burst_edges) > 0:
                burst_widths = burst_edges[:, 1] - burst_edges[:, 0]
                results['burst_widths'] = burst_widths
                results['burst_amplitudes'] = peak_info['peak_heights']
                
                # Calculate burst involvement and backbone classification
                bic_matrix, backbone_class = self.calculate_burst_involvement(burst_edges)
                results['bic_matrix'] = bic_matrix
                results['backbone_classification'] = backbone_class
                
                # Calculate additional metrics if requested
                if self.compute_all_metrics:
                    # Calculate recording duration in seconds
                    duration_ms = self.spike_data.length
                    duration_sec = duration_ms / 1000.0
                    
                    # Calculate burst frequency
                    results['burst_frequency'] = len(burst_edges) / duration_sec
                    
                    # Calculate population rate statistics
                    results['mean_population_rate'] = np.mean(self._smoothed_activity)
                    results['cv_population_rate'] = variation(self._smoothed_activity)
                    
                    # Calculate mean and std of burst properties
                    results['mean_burst_width'] = np.mean(burst_widths)
                    results['std_burst_width'] = np.std(burst_widths)
                    results['mean_burst_amplitude'] = np.mean(results['burst_amplitudes'])
                    results['std_burst_amplitude'] = np.std(results['burst_amplitudes'])
                    results['n_rigid_units'] = len(backbone_class['rigid'])
        
        self._results = results

        # Return results dictionary (BurstResults wrapper is created in detect_bursts)
        return results
    
    def get_burst_times(self) -> np.ndarray:
        """
        Get start and end times of all detected bursts.
        
        Returns
        -------
        numpy.ndarray
            Array of burst edges with shape (n_bursts, 2)
        """
        if self._results is None:
            self.detect_bursts()
        
        return self._results['burst_edges']
    
    def get_burst_spikes(self) -> List[List[float]]:
        """
        Get spikes occurring during bursts for each neuron.

        Returns
        -------
        list of lists
            List of spike times during bursts for each neuron
        """
        if self._results is None:
            self.detect_bursts()

        burst_edges = self._results['burst_edges']

        # Handle case with no bursts
        if len(burst_edges) == 0:
            return [[] for _ in range(self.spike_data.N)]

        idces, times = self.spike_data.idces_times()

        # Vectorized burst checking using broadcasting
        burst_starts = burst_edges[:, 0]
        burst_ends = burst_edges[:, 1]

        # Create matrix where [i, j] = True if spike i is in burst j
        in_burst_matrix = (times[:, None] >= burst_starts) & (times[:, None] <= burst_ends)
        in_any_burst = in_burst_matrix.any(axis=1)

        # Filter to only burst spikes
        burst_idces = idces[in_any_burst]
        burst_times_filtered = times[in_any_burst]

        # Sort by neuron index for efficient splitting
        sort_idx = np.argsort(burst_idces)
        sorted_idces = burst_idces[sort_idx]
        sorted_times = burst_times_filtered[sort_idx]

        # Find split points for each neuron
        split_points = np.searchsorted(sorted_idces, np.arange(self.spike_data.N + 1))
        burst_spikes = [sorted_times[split_points[i]:split_points[i+1]].tolist()
                        for i in range(self.spike_data.N)]

        return burst_spikes
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Get a summary of burst metrics as a dictionary.
        
        Returns
        -------
        dict
            Dictionary of summary burst metrics
        """
        if self._results is None:
            self.detect_bursts()
        
        results = self._results
        
        summary = {
            'n_bursts': results['n_bursts']
        }
        
        if results['n_bursts'] > 0 and self.compute_all_metrics:
            summary.update({
                'mean_burst_width': results.get('mean_burst_width', np.nan),
                'std_burst_width': results.get('std_burst_width', np.nan),
                'mean_burst_amplitude': results.get('mean_burst_amplitude', np.nan),
                'std_burst_amplitude': results.get('std_burst_amplitude', np.nan),
                'burst_frequency': results.get('burst_frequency', np.nan),
                'mean_population_rate': results.get('mean_population_rate', np.nan),
                'cv_population_rate': results.get('cv_population_rate', np.nan),
                'n_rigid_units': results.get('n_rigid_units', 0),
            })
        
        return summary
    
    def reset(self):
        """
        Reset all cached calculations.
        """
        self._population_activity = None
        self._smoothed_activity = None
        self._results = None




class BurstResults:
    """
    Wrapper class for burst detection results with clean dot-notation access.
    
    This class wraps the results dictionary from BurstDetector and provides
    easy access to all burst detection data through properties.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of burst detection results from BurstDetector
    bin_size : float
        Bin size in ms used for population activity calculation
    
    Properties
    ----------
    time_bins : np.ndarray
        Array of time values in ms for each bin
    smoothed_activity : np.ndarray
        Smoothed population firing rate
    burst_times : np.ndarray
        Array of (start, end) times in ms for each burst, shape (n_bursts, 2)
    n_bursts : int
        Number of detected bursts
    burst_widths : np.ndarray
        Width of each burst in ms
    burst_amplitudes : np.ndarray
        Amplitude (peak height) of each burst
    bic_matrix : np.ndarray
        Burst involvement coefficient for each neuron
    backbone_classification : dict
        Dictionary with 'rigid' and 'nonrigid' neuron lists
    """
    
    def __init__(self, results_dict: Dict[str, Any], bin_size: float):
        self._results = results_dict
        self._bin_size = bin_size
    
    @property
    def time_bins(self) -> np.ndarray:
        """Array of time values in ms for each bin."""
        n_bins = len(self._results['population_rate'])
        return np.arange(n_bins) * self._bin_size
    
    @property
    def smoothed_activity(self) -> np.ndarray:
        """Smoothed population firing rate."""
        return self._results['population_rate']
    
    @property
    def burst_times(self) -> np.ndarray:
        """Array of (start, end) times in ms for each burst."""
        return self._results['burst_edges']
    
    @property
    def n_bursts(self) -> int:
        """Number of detected bursts."""
        return self._results['n_bursts']
    
    @property
    def burst_widths(self) -> Optional[np.ndarray]:
        """Width of each burst in ms."""
        return self._results.get('burst_widths')
    
    @property
    def burst_amplitudes(self) -> Optional[np.ndarray]:
        """Amplitude (peak height) of each burst."""
        return self._results.get('burst_amplitudes')
    
    @property
    def bic_matrix(self) -> Optional[np.ndarray]:
        """Burst involvement coefficient for each neuron."""
        return self._results.get('bic_matrix')
    
    @property
    def backbone_classification(self) -> Optional[Dict[str, List[int]]]:
        """Dictionary with 'rigid' and 'nonrigid' neuron lists."""
        return self._results.get('backbone_classification')
    
    @property
    def peak_indices(self) -> np.ndarray:
        """Indices of detected burst peaks."""
        return self._results['peak_indices']
    
    @property
    def mean_burst_width(self) -> Optional[float]:
        """Mean burst width in ms."""
        return self._results.get('mean_burst_width', np.nan)
    
    @property
    def std_burst_width(self) -> Optional[float]:
        """Standard deviation of burst width in ms."""
        return self._results.get('std_burst_width', np.nan)
    
    @property
    def mean_burst_amplitude(self) -> Optional[float]:
        """Mean burst amplitude."""
        return self._results.get('mean_burst_amplitude', np.nan)
    
    @property
    def std_burst_amplitude(self) -> Optional[float]:
        """Standard deviation of burst amplitude."""
        return self._results.get('std_burst_amplitude', np.nan)
    
    @property
    def burst_frequency(self) -> Optional[float]:
        """Burst frequency in Hz."""
        return self._results.get('burst_frequency', 0.0)
    
    @property
    def mean_population_rate(self) -> Optional[float]:
        """Mean population firing rate."""
        return self._results.get('mean_population_rate')
    
    @property
    def cv_population_rate(self) -> Optional[float]:
        """Coefficient of variation of population rate."""
        return self._results.get('cv_population_rate')
    
    @property
    def n_rigid_units(self) -> Optional[int]:
        """Number of rigid (backbone) neurons."""
        return self._results.get('n_rigid_units')
    
    def __repr__(self) -> str:
        return f"BurstResults(n_bursts={self.n_bursts})"

