"""
Validators for Catalog Generation

Contains functions for validating and categorizing experiments:
- Baseline detection
- Frequency calculation and categorization
- Experiment type extraction
- Project extraction from paths
"""

import re
import numpy as np
import pandas as pd


def is_baseline_with_reason(experiment, freq=None):
    """
    Determine if an experiment is a baseline and return the reason.

    Args:
        experiment: Experiment path string (e.g., 'exp1/exp1_cont_12')
        freq: Optional frequency value (not used in current logic)

    Returns:
        tuple: (is_baseline: bool, reason: str)

    Special cases:
        - Only BL1 variants are baselines (BL1, bl_1, bl-1)
        - BL2+ variants are NOT baselines
        - Causal variants are NOT baselines
        - frequencystimphase experiments are NOT baselines
        - recordphase_recording without _X suffix is a baseline
        - Basic experiments without '_cont' are baselines
    """
    if pd.isna(experiment) or not isinstance(experiment, str):
        return False, "Invalid experiment name"

    # Extract just the filename part (after the last slash)
    experiment_name = experiment.split("/")[-1]

    # Special case: Only BL1 variants are baselines (regardless of freq)
    bl1_patterns = ["BL1", "bl_1", "bl-1"]
    if experiment_name in bl1_patterns:
        return True, "BL1 variant"

    # Numbered BL variants (BL2, bl-2, bl_2, bl-3, bl-4, etc.) are NOT baselines
    # Plain "bl" or "BL" falls through to the folder-name-match check below
    if re.match(r'^[Bb][Ll][-_]?\d', experiment_name):
        return False, "BL variant (not BL1)"
    if "causal" in experiment_name.lower():
        return False, "Contains 'causal'"

    # Special case: frequencystimphase experiments are NEVER baselines
    if "frequencystimphase" in experiment_name:
        return False, "Frequency stimulation phase"

    # Special case: recordphase_recording patterns
    if "recordphase_recording" in experiment_name:
        if re.search(r"recordphase_recording_\d+", experiment_name):
            return False, "Numbered recording phase"
        elif experiment_name.endswith("recordphase_recording"):
            return True, "Base recording phase"

    # Check if experiment filename matches folder name exactly (e.g., "david/david", "exp1/exp1")
    # This is the primary baseline pattern
    parts = experiment.split("/")
    if len(parts) >= 2:
        folder_name = parts[-2]
        if experiment_name == folder_name:
            return True, "Experiment name matches folder name exactly"

    # For experiments with _cont suffix, not a baseline
    if "_cont" in experiment_name:
        return False, "Contains '_cont'"

    # Any other suffix means it's not a baseline
    return False, "Has suffix or doesn't match folder name"


def is_baseline(experiment, freq=None):
    """
    Determine if an experiment is a baseline.

    This is a simplified wrapper around is_baseline_with_reason that
    returns just the boolean result.

    Args:
        experiment: Experiment path string
        freq: Optional frequency value (not used)

    Returns:
        bool: True if experiment is a baseline, False otherwise

    Examples:
        >>> is_baseline('exp1/exp1')
        True
        >>> is_baseline('david/david')
        True
        >>> is_baseline('david/david_causal')
        False
        >>> is_baseline('exp1/exp1_cont_5')
        False
        >>> is_baseline('exp1/exp1_cartpole_F_1')
        False
        >>> is_baseline('BL1/BL1')
        True
        >>> is_baseline('BL2/BL2')
        False
    """
    result, _ = is_baseline_with_reason(experiment, freq)
    return result


def calculate_stim_frequency(log):
    """
    Calculate stimulation frequency from a stimulus log DataFrame.

    Analyzes the timing of stimulations to determine the frequency.

    Args:
        log: DataFrame with a 'time' column containing stimulus times

    Returns:
        float or None: Calculated frequency in Hz, or None if cannot be calculated
    """
    if log is None or len(log) < 2:
        return None

    if "time" not in log.columns:
        return None

    try:
        time_values = pd.to_numeric(log["time"], errors="coerce")
        time_values = time_values.dropna()

        if len(time_values) < 2:
            return None
    except Exception:
        return None

    # Calculate time differences between consecutive stimuli
    time_diffs = np.diff(time_values.values)

    # Remove outliers (gaps > 10 seconds might indicate pauses)
    clean_diffs = time_diffs[time_diffs < 10]

    if len(clean_diffs) == 0:
        return None

    # Calculate frequency as 1/mean_interval
    mean_interval = np.mean(clean_diffs)
    frequency = 1.0 / mean_interval

    return frequency


def categorize_frequency(freq):
    """
    Categorize a frequency into predefined stimulation paradigms.

    Maps a raw frequency value to the closest standard frequency
    used in the experimental paradigm.

    Args:
        freq: Raw frequency value in Hz

    Returns:
        float or None: Closest standard frequency (0.5, 1, 2, 4, or 8 Hz)

    Example:
        >>> categorize_frequency(1.97)
        2.0
        >>> categorize_frequency(0.48)
        0.5
    """
    if freq is None:
        return None

    target_freqs = [0.5, 1, 2, 4, 8]

    distances = [abs(freq - target) for target in target_freqs]
    closest_idx = np.argmin(distances)

    return target_freqs[closest_idx]


def calculate_freq_from_stim_count(n_stims):
    """
    Estimate frequency based on stimulation count.

    Uses an approximate mapping based on expected stim counts
    for 35-minute recordings at different frequencies.

    Args:
        n_stims: Number of stimulations

    Returns:
        float: Estimated frequency in Hz

    Mapping:
        1050 stims → 0.5 Hz
        2100 stims → 1 Hz
        4200 stims → 2 Hz
        8400 stims → 4 Hz
        16800 stims → 8 Hz
    """
    stim_to_freq = {
        1050: 0.5,
        2100: 1.0,
        4200: 2.0,
        8400: 4.0,
        16800: 8.0,
    }

    if n_stims == 0:
        return 0.0

    closest_stim = min(stim_to_freq.keys(), key=lambda x: abs(x - n_stims))
    return stim_to_freq[closest_stim]


def extract_proj_from_s3_path(s3_path):
    """
    Extract project identifier from an S3 path.

    Args:
        s3_path: S3 path string

    Returns:
        str or None: Project identifier

    Examples:
        >>> extract_proj_from_s3_path('s3://braingeneers/braindance/25-02-25_busybees/')
        '25-02-25_busybees'
        >>> extract_proj_from_s3_path('s3://braingeneersdev/asrobbin/braindance_data/24-08-16_busybee/')
        '24-08-16_busybee'
    """
    if pd.isna(s3_path) or not isinstance(s3_path, str):
        return None

    # Remove trailing slash
    s3_path = s3_path.rstrip("/")

    # Get the last part of the path
    parts = s3_path.split("/")
    if not parts:
        return None

    last_part = parts[-1]

    # Check if it matches the pattern (date_project)
    if "_" in last_part and any(c.isdigit() for c in last_part):
        return last_part

    return None


def extract_exp_from_experiment_path(experiment):
    """
    Extract experiment type from experiment path.

    Args:
        experiment: Experiment path string

    Returns:
        str or None: Experiment type (first part before /)

    Examples:
        >>> extract_exp_from_experiment_path('freqs/freqs_cont_12')
        'freqs'
        >>> extract_exp_from_experiment_path('exp1/exp1_cont_26')
        'exp1'
    """
    if pd.isna(experiment) or not isinstance(experiment, str):
        return None

    return experiment.split("/")[0]


def count_digits(string):
    """Count the number of digits in a string."""
    return sum(c.isdigit() for c in string)


def extract_number(file_string, regex=r"[_.](\d+)[_.]"):
    """
    Extract a number from a file string.

    Used for sorting files by numeric component.

    Args:
        file_string: String to extract number from
        regex: Regular expression pattern

    Returns:
        int: Extracted number or -1 if not found
    """
    match = re.search(regex, file_string)
    return int(match.group(1)) if match else -1


def parse_exp_number(exp_name):
    """
    Extract numeric value from exp name for chronological ordering.

    Handles edge cases:
    - 'exp1' -> 1
    - 'exp10' -> 10
    - 'exp1_repeat' -> 1 (same as exp1, use alphabetical tiebreaker)
    - 'david' -> None (alphabetical fallback)
    - 'BL2' -> 2
    - 'RL_k252a' -> None (no clear numeric component)

    Args:
        exp_name: Experiment name string

    Returns:
        int or None: Numeric value if found, None otherwise

    Examples:
        >>> parse_exp_number('exp1')
        1
        >>> parse_exp_number('exp10')
        10
        >>> parse_exp_number('david')
        None
    """
    if pd.isna(exp_name) or not isinstance(exp_name, str):
        return None

    match = re.search(r'\d+', exp_name)
    if match:
        return int(match.group())
    return None


def compare_exp_names(exp_a, exp_b):
    """
    Return True if exp_a is "earlier" than exp_b.

    Comparison order:
    1. Numeric comparison if both have numbers
    2. If tied numerically, alphabetical comparison
    3. If one is numeric and other isn't, numeric comes first
    4. If neither numeric, alphabetical comparison

    Args:
        exp_a: First experiment name
        exp_b: Second experiment name

    Returns:
        bool: True if exp_a is earlier than exp_b

    Examples:
        >>> compare_exp_names('exp1', 'exp2')
        True
        >>> compare_exp_names('exp10', 'exp2')
        False
        >>> compare_exp_names('exp1', 'exp1_repeat')
        True
    """
    num_a = parse_exp_number(exp_a)
    num_b = parse_exp_number(exp_b)

    if num_a is not None and num_b is not None:
        if num_a != num_b:
            return num_a < num_b
        # Tie: use alphabetical
        return exp_a < exp_b
    elif num_a is not None:
        return True  # numeric before non-numeric
    elif num_b is not None:
        return False
    else:
        return exp_a < exp_b  # both non-numeric
