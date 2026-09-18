"""
Latency analysis functions for evoked response detection.

Provides convenience wrappers around UltraOptimizedLatencyHelper for
integration with the BrainDance data manager.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from .ultra_optimized_latency_helper import UltraOptimizedLatencyHelper


def calculate_latencies(
    spike_data,
    stim_log: pd.DataFrame,
    min_response_ratio: float = 1.5,
    max_p_value: float = 0.0001,
    baseline_window: Tuple[float, float] = (-100, 0),
    response_window: Tuple[float, float] = (0, 100),
    use_time_mod: bool = True,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Detect stimulus-evoked neural responses with statistical validation.

    Args:
        spike_data: SpikeData object with neuron spike trains
        stim_log: DataFrame with stimulation times and electrode info
        min_response_ratio: Minimum response/baseline firing rate ratio
        max_p_value: Maximum p-value for Mann-Whitney U test
        baseline_window: (start, end) times in ms for baseline period
        response_window: (start, end) times in ms for response period
        use_time_mod: Use artifact-corrected times if available
        verbose: Print progress information
        **kwargs: Additional arguments for UltraOptimizedLatencyHelper

    Returns:
        Dict mapping "electrode_{id}_neuron_{idx}" to latency data with:
        - peak_latency (time of max response in response_window)
        - baseline_rate, response_rate, response_ratio, p_value
        - mean_psth, time_bins (for visualization)
    """
    # Initialize helper
    helper = UltraOptimizedLatencyHelper(verbose=verbose)

    # Run analysis with full validation
    evoked_pairs = helper.calculate_latencies_with_statistical_validation(
        spike_data, stim_log,
        min_response_ratio=min_response_ratio,
        max_p_value=max_p_value,
        baseline_window=baseline_window,
        response_window=response_window,
        use_time_mod=use_time_mod,
        **kwargs
    )

    return evoked_pairs


def group_stimulations_by_electrode(
    stim_log: pd.DataFrame,
    use_time_mod: bool = True
) -> Dict[int, np.ndarray]:
    """
    Group stimulation times by electrode ID.

    Useful for custom analysis or electrode-specific plotting.

    Args:
        stim_log: DataFrame with stimulation information
        use_time_mod: Use artifact-corrected times if available

    Returns:
        Dict mapping electrode_id -> array of stim times (in ms)
    """

    helper = UltraOptimizedLatencyHelper(verbose=False)
    return helper.group_stimulations_by_electrode(stim_log, use_time_mod=use_time_mod)
