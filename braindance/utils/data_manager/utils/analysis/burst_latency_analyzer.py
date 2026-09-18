"""
BurstLatencyAnalyzer: Analyze burst latencies in response to electrical stimulation.

This class combines burst detection from BurstDetector with timing analysis to measure
the latency between electrical stimulation and evoked network bursts.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import chi2_contingency

try:
    from scipy.stats import binomtest
except ImportError:
    # Fallback for older scipy versions
    from scipy.stats import binom_test as binomtest

from .burst_detector import BurstDetector
from .ultra_optimized_latency_helper import UltraOptimizedLatencyHelper

logger = logging.getLogger(__name__)


class BurstLatencyAnalyzer:
    """
    Analyze burst latencies in response to electrical stimulation.
    
    This class detects network bursts and measures their timing relative to
    electrical stimulation events, providing statistical validation of
    stimulus-evoked bursting.
    
    Parameters
    ----------
    spike_data : SpikeData
        SpikeData object containing spike trains
    stim_log : pd.DataFrame
        DataFrame containing stimulation timing and electrode information
    baseline_window_ms : float, optional
        Duration of baseline window before stimulus (default=500ms)
    analysis_window_ms : float, optional
        Duration of analysis window after stimulus (default=200ms)
    burst_detection_params : dict, optional
        Parameters for burst detection (passed to BurstDetector)
    validation_params : dict, optional
        Parameters for statistical validation
    use_time_mod : bool, optional
        Use time_mod (corrected stim times) if available, otherwise use time (default=True)
    verbose : bool, optional
        Enable verbose logging (default=True)
    cache : ResultsCache, optional
        Cache instance for storing/loading burst detection results (default=None)
    """
    
    def __init__(self,
                 spike_data,
                 stim_log: pd.DataFrame,
                 baseline_window_ms: float = 500.0,
                 analysis_window_ms: float = 200.0,
                 burst_detection_params: Optional[Dict] = None,
                 validation_params: Optional[Dict] = None,
                 use_time_mod: bool = True,
                 verbose: bool = True,
                 cache=None):

        self.spike_data = spike_data
        self.stim_log = stim_log
        self.baseline_window_ms = baseline_window_ms
        self.analysis_window_ms = analysis_window_ms
        self.use_time_mod = use_time_mod
        self.verbose = verbose
        self.cache = cache

        # Initialize burst detector
        self.burst_detector = BurstDetector(
            spike_data,
            burst_detection_params=burst_detection_params
        )

        # Initialize latency helper for utility functions
        self.latency_helper = UltraOptimizedLatencyHelper(verbose=verbose)
        
        # Set default validation parameters
        self.validation_params = {
            'probability_increase_factor': 1.5,  # 50% increase required
            'max_p_value': 0.01,                 # Looser than neuron-level
            'min_evoked_bursts': 3,              # At least 3 instances
            'max_mean_latency_ms': 100.0,        # Reasonable timing window
            **(validation_params or {})
        }
        
        # Cache for results
        self._burst_results = None
        self._electrode_groups = None
        
        if verbose:
            logger.info(f"Initialized BurstLatencyAnalyzer:")
            logger.info(f"  - Baseline window: {baseline_window_ms}ms")
            logger.info(f"  - Analysis window: {analysis_window_ms}ms")
            logger.info(f"  - Validation criteria: {self.validation_params}")
    
    def _get_cache_params(self) -> dict:
        """Generate parameters dict for cache key."""
        return {
            'baseline_window_ms': self.baseline_window_ms,
            'analysis_window_ms': self.analysis_window_ms,
            'probability_increase_factor': self.validation_params.get('probability_increase_factor', 1.5),
            'max_p_value': self.validation_params.get('max_p_value', 0.01),
            'min_evoked_bursts': self.validation_params.get('min_evoked_bursts', 3),
            'max_mean_latency_ms': self.validation_params.get('max_mean_latency_ms', 100.0),
        }

    def analyze_burst_latencies(self, use_cache: bool = True, force_recompute: bool = False) -> Dict[int, Dict[str, Any]]:
        """
        Analyze burst latencies for all electrodes with automatic caching.

        Parameters
        ----------
        use_cache : bool, default=True
            Whether to use cached results if available
        force_recompute : bool, default=False
            Force recomputation bypassing cache

        Returns
        -------
        Dict[int, Dict[str, Any]]
            Results dictionary with electrode IDs as keys
        """
        # Check cache first
        if self.cache is not None and use_cache and not force_recompute:
            cache_params = self._get_cache_params()
            cached_data = self.cache.load_local('burst_latency', cache_params)
            if cached_data is not None:
                import pickle
                # Extract pickled bytes from numpy array (unwrapped by load_local via .item())
                pickled_bytes = cached_data['results']
                if isinstance(pickled_bytes, bytes):
                    results = pickle.loads(pickled_bytes)
                else:
                    # If it's still wrapped in numpy array (shouldn't happen with our load_local fix)
                    results = pickle.loads(pickled_bytes.item() if hasattr(pickled_bytes, 'item') else pickled_bytes)

                # Still need to detect bursts to populate self._burst_results for summary stats
                # This will be fast since burst detection is also cached
                self._burst_results = self.burst_detector.detect_bursts(cache=self.cache)

                return results

        if self.verbose:
            logger.info("Starting burst latency analysis...")

        # Step 1: Detect all bursts in the recording
        if self.verbose:
            logger.info("Detecting network bursts...")

        burst_results = self.burst_detector.detect_bursts(cache=self.cache)
        burst_times = self.burst_detector.get_burst_times()  # Shape: (n_bursts, 2)
        
        if len(burst_times) == 0:
            logger.warning("No bursts detected in recording")
            return {}
        
        if self.verbose:
            logger.info(f"Detected {len(burst_times)} network bursts")
        
        # Step 2: Group stimulations by electrode
        electrode_groups = self.latency_helper.group_stimulations_by_electrode(
            self.stim_log, use_time_mod=self.use_time_mod
        )
        
        if len(electrode_groups) == 0:
            logger.warning("No electrode groups found")
            return {}
        
        # Step 3: Analyze each electrode
        results = {}
        for electrode_id, stim_times in electrode_groups.items():
            if self.verbose:
                logger.info(f"Analyzing electrode {electrode_id} ({len(stim_times)} stimuli)")
            
            electrode_result = self._analyze_electrode_burst_latencies(
                electrode_id, stim_times, burst_times
            )
            
            if electrode_result is not None:
                results[electrode_id] = electrode_result
        
        self._burst_results = burst_results
        self._electrode_groups = electrode_groups

        if self.verbose:
            validated_electrodes = sum(1 for r in results.values() if r.get('validated', False))
            logger.info(f"Burst latency analysis complete: {validated_electrodes}/{len(results)} electrodes validated")

        # Save to cache if enabled
        if self.cache is not None and use_cache:
            cache_params = self._get_cache_params()
            # Convert results dict to cache-friendly format
            import pickle
            import numpy as np
            cache_data = {
                'results': np.array(pickle.dumps(results), dtype=object),  # Wrap pickled data in numpy array
            }
            self.cache.save_local('burst_latency', cache_params, cache_data)

        return results
    
    def _get_burst_peak_times(self) -> np.ndarray:
        """
        Get burst peak times from the burst detector results.
        
        Returns
        -------
        np.ndarray
            Array of burst peak times in ms
        """
        if self._burst_results is None:
            burst_results = self.burst_detector.detect_bursts()
            self._burst_results = burst_results
        else:
            burst_results = self._burst_results
        
        peak_indices = burst_results.peak_indices
        
        if len(peak_indices) == 0:
            return np.array([])
        
        # Convert peak indices to times using bin_size
        peak_times = peak_indices * self.burst_detector.bin_size
        
        return peak_times
    
    def _analyze_electrode_burst_latencies(self, 
                                         electrode_id: int,
                                         stim_times: np.ndarray,
                                         burst_times: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Analyze burst latencies for a single electrode.
        
        Parameters
        ----------
        electrode_id : int
            Electrode identifier
        stim_times : np.ndarray
            Array of stimulation times in ms
        burst_times : np.ndarray
            Array of burst start/end times, shape (n_bursts, 2)
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Analysis results or None if insufficient data
        """
        if len(stim_times) < 3:  # Need minimum stimuli for statistics
            return None
        
        # Get burst peak times instead of start times
        burst_peak_times = self._get_burst_peak_times()
        
        # Find burst-stimulus associations
        burst_latencies = []
        stimulus_burst_pairs = []
        
        for i, stim_time in enumerate(stim_times):
            # Find bursts within analysis window after this stimulus
            post_stim_bursts = burst_peak_times[
                (burst_peak_times >= stim_time) & 
                (burst_peak_times <= stim_time + self.analysis_window_ms)
            ]
            
            # Record all bursts for this stimulus
            for burst_time in post_stim_bursts:
                latency = burst_time - stim_time
                burst_latencies.append(latency)
                stimulus_burst_pairs.append((i, stim_time, burst_time, latency))
        
        # Calculate baseline burst statistics
        baseline_stats = self._calculate_baseline_burst_stats(
            stim_times, burst_peak_times
        )
        
        # Prepare results
        n_evoked_bursts = len(burst_latencies)
        n_stimuli = len(stim_times)
        evoked_burst_probability = n_evoked_bursts / n_stimuli
        
        # Calculate statistics
        mean_latency = np.mean(burst_latencies) if burst_latencies else np.nan
        std_latency = np.std(burst_latencies) if len(burst_latencies) > 1 else np.nan
        
        # Perform statistical validation
        validation_result = self._validate_burst_responses(
            n_evoked_bursts, n_stimuli, baseline_stats, mean_latency
        )
        
        # Classify burst types based on latency
        burst_types = self._classify_burst_latencies(burst_latencies)
        
        return {
            'electrode_id': electrode_id,
            'n_stimuli': n_stimuli,
            'n_evoked_bursts': n_evoked_bursts,
            'evoked_burst_probability': evoked_burst_probability,
            'burst_latencies': burst_latencies,
            'mean_burst_latency': mean_latency,
            'std_burst_latency': std_latency,
            'burst_types': burst_types,
            'stimulus_burst_pairs': stimulus_burst_pairs,
            'baseline_stats': baseline_stats,
            'validation': validation_result,
            'validated': validation_result['is_validated']
        }
    
    def _calculate_baseline_burst_stats(self, 
                                      stim_times: np.ndarray,
                                      burst_peak_times: np.ndarray) -> Dict[str, Any]:
        """
        Calculate baseline burst statistics using pre-stimulus periods.
        
        Parameters
        ----------
        stim_times : np.ndarray
            Array of stimulation times
        burst_peak_times : np.ndarray
            Array of all burst peak times
        
        Returns
        -------
        Dict[str, Any]
            Baseline statistics
        """
        # Count bursts in baseline periods before each stimulus
        baseline_bursts = 0
        total_baseline_duration = 0
        
        for stim_time in stim_times:
            baseline_start = stim_time - self.baseline_window_ms
            baseline_end = stim_time
            
            # Skip if baseline extends before recording start
            if baseline_start < 0:
                continue
            
            # Count bursts in this baseline period
            baseline_period_bursts = np.sum(
                (burst_peak_times >= baseline_start) & 
                (burst_peak_times < baseline_end)
            )
            
            baseline_bursts += baseline_period_bursts
            total_baseline_duration += self.baseline_window_ms
        
        # Calculate baseline statistics
        baseline_duration_sec = total_baseline_duration / 1000.0
        baseline_burst_rate = baseline_bursts / baseline_duration_sec if baseline_duration_sec > 0 else 0
        
        # Calculate baseline probability per window
        n_baseline_windows = len(stim_times)
        baseline_burst_probability = baseline_bursts / n_baseline_windows if n_baseline_windows > 0 else 0
        
        return {
            'n_baseline_bursts': baseline_bursts,
            'baseline_duration_sec': baseline_duration_sec,
            'baseline_burst_rate': baseline_burst_rate,
            'baseline_burst_probability': baseline_burst_probability,
            'n_baseline_windows': n_baseline_windows
        }
    
    def _validate_burst_responses(self, 
                                n_evoked_bursts: int,
                                n_stimuli: int,
                                baseline_stats: Dict[str, Any],
                                mean_latency: float) -> Dict[str, Any]:
        """
        Validate burst responses using combination approach.
        
        Parameters
        ----------
        n_evoked_bursts : int
            Number of evoked bursts
        n_stimuli : int
            Number of stimuli
        baseline_stats : Dict[str, Any]
            Baseline burst statistics
        mean_latency : float
            Mean burst latency
        
        Returns
        -------
        Dict[str, Any]
            Validation results
        """
        evoked_prob = n_evoked_bursts / n_stimuli
        baseline_prob = baseline_stats['baseline_burst_probability']
        
        # Validation criteria
        criteria = {}
        
        # 1. Burst probability increase
        if baseline_prob > 0:
            prob_increase = evoked_prob / baseline_prob
            criteria['burst_probability_increase'] = prob_increase >= self.validation_params['probability_increase_factor']
        else:
            # If no baseline bursts, any evoked burst is significant
            criteria['burst_probability_increase'] = evoked_prob > 0
        
        # 2. Statistical significance (Chi-square test)
        if baseline_stats['n_baseline_bursts'] > 0:
            # 2x2 contingency table: [stimulus_period, baseline_period] x [burst, no_burst]
            evoked_no_bursts = n_stimuli - n_evoked_bursts
            baseline_bursts = baseline_stats['n_baseline_bursts']
            baseline_no_bursts = baseline_stats['n_baseline_windows'] - baseline_bursts
            
            contingency_table = [
                [n_evoked_bursts, evoked_no_bursts],
                [baseline_bursts, baseline_no_bursts]
            ]
            
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                criteria['statistical_significance'] = p_value < self.validation_params['max_p_value']
            except ValueError:
                # Handle edge cases
                p_value = 1.0
                criteria['statistical_significance'] = False
        else:
            # Use binomial test if no baseline bursts
            try:
                # Use binomtest (newer scipy) or binom_test (older scipy)
                if hasattr(binomtest, 'pvalue'):
                    # New scipy.stats.binomtest object
                    result = binomtest(n_evoked_bursts, n_stimuli, 0.01)
                    p_value = result.pvalue
                else:
                    # Old scipy.stats.binom_test function
                    p_value = binomtest(n_evoked_bursts, n_stimuli, 0.01)
                criteria['statistical_significance'] = p_value < self.validation_params['max_p_value']
            except:
                p_value = 1.0
                criteria['statistical_significance'] = False
        
        # 3. Minimum evoked bursts
        criteria['minimum_evoked_bursts'] = n_evoked_bursts >= self.validation_params['min_evoked_bursts']
        
        # 4. Temporal precision
        if not np.isnan(mean_latency):
            criteria['temporal_precision'] = mean_latency <= self.validation_params['max_mean_latency_ms']
        else:
            criteria['temporal_precision'] = False
        
        # Overall validation
        is_validated = all(criteria.values())
        
        return {
            'is_validated': is_validated,
            'criteria': criteria,
            'p_value': p_value,
            'evoked_probability': evoked_prob,
            'baseline_probability': baseline_prob,
            'probability_ratio': evoked_prob / baseline_prob if baseline_prob > 0 else np.inf
        }
    
    def _classify_burst_latencies(self, burst_latencies: List[float]) -> Dict[str, Any]:
        """
        Classify burst latencies into categories.
        
        Parameters
        ----------
        burst_latencies : List[float]
            List of burst latencies in ms
        
        Returns
        -------
        Dict[str, Any]
            Classification results
        """
        if not burst_latencies:
            return {'immediate': 0, 'late': 0, 'categories': []}
        
        latencies = np.array(burst_latencies)
        
        # Classify based on timing
        immediate_mask = latencies <= 25.0  # 0-25ms
        late_mask = latencies > 25.0        # >25ms
        
        immediate_count = np.sum(immediate_mask)
        late_count = np.sum(late_mask)
        
        # Create category labels
        categories = []
        for latency in latencies:
            if latency <= 25.0:
                categories.append('immediate')
            else:
                categories.append('late')
        
        return {
            'immediate': immediate_count,
            'late': late_count,
            'categories': categories,
            'immediate_fraction': immediate_count / len(latencies),
            'late_fraction': late_count / len(latencies)
        }
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all electrodes.
        
        Returns
        -------
        Dict[str, Any]
            Summary statistics
        """
        if self._burst_results is None:
            return {}
        
        return {
            'total_bursts_detected': self._burst_results.n_bursts,
            'burst_detection_params': self.burst_detector.burst_detection_params,
            'validation_params': self.validation_params
        }
    
