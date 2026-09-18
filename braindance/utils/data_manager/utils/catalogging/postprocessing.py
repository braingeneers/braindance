"""
Postprocessing Functions for Catalog Generation

Contains all the "gross" manual fixes, corrupted file removal,
and one-off corrections that need to be applied to the catalog.
These are preserved from the original amend_catalog.py.
"""

import pandas as pd
from .validators import (
    is_baseline,
    extract_proj_from_s3_path,
    extract_exp_from_experiment_path,
    compare_exp_names
)


# ============================================================================
# CORRUPTED FILE LISTS
# ============================================================================

CORRUPTED_FILES_25178IC = [
    "exp1_cont_59", "exp1_cont_40", "exp1_cont_45", "exp1_cont_41", "exp1_cont_25",
    "exp1_cont_43", "exp1_cont_46", "exp1_cont_38", "exp1_cont_19", "exp1_cont_39",
    "exp1_cont_57", "exp1_cont_52", "exp1_cont_62", "exp1_cont_34", "exp1_cont_36",
    "exp1_cont_37", "exp1_cont_28", "exp1_cont_42", "exp1_cont_54", "exp1_cont_61",
    "exp1_cont_31", "exp1_cont_64", "exp1_cont_13", "exp1_cont_51", "exp1_cont_66",
    "exp1_cont_63", "exp1_cont_53", "exp1_cont_16", "exp1_cont_30", "exp1_cont_65",
    "exp1_cont_29", "exp1_cont_18", "exp1_cont_22", "exp1_cont_32", "exp1_cont_20",
    "exp1_cont_48", "exp1_cont_33", "exp1_cont_17", "exp1_cont_60", "exp1_cont_47",
    "exp1_cont_23", "exp1_cont_21", "exp1_cont_26", "exp1_cont_49", "exp1_cont_50",
    "exp1_cont_55", "exp1_cont_15", "exp1_cont_56", "exp1_cont_35", "exp1_cont_58",
    "exp1_cont_24exp1_cont_27", "exp1_cont_44", "exp1_cont_14",
]

CORRUPTED_FILES_21985IC = [
    "exp1_cont_40", "exp1_cont_45", "exp1_cont_41", "exp1_cont_25", "exp1_cont_43",
    "exp1_cont_38", "exp1_cont_39", "exp1_cont_34", "exp1_cont_36", "exp1_cont_37",
    "exp1_cont_28", "exp1_cont_42", "exp1_cont_31", "exp1_cont_30", "exp1_cont_29",
    "exp1_cont_22", "exp1_cont_32", "exp1_cont_33", "exp1_cont_23", "exp1_cont_21",
    "exp1_cont_26", "exp1_cont_35", "exp1_cont_24", "exp1_cont_27", "exp1_cont_44",
]


# ============================================================================
# SPECIFIC EXPERIMENTS TO REMOVE
# These are one-off removals for various reasons (duplicates, bad data, etc.)
# ============================================================================

EXPERIMENTS_TO_REMOVE = [
    # (proj, chip, experiment) tuples
    ("2024-05-06_butterfly", "22097", "rl_only/rl_only_1"),
    ("24-04-18_butterfly", "23138", "RL2_k252a/RL2_k252a_1"),
    ("24-04-18_butterfly", "p001237", "exp6/exp6_1"),
    ("23-11-22_cartpole", "c22064", "r1-e1/e1_0"),
    ("24-03-25_plasticity", "p001237", "RL_k252a/RL_k252a_1"),
    ("24-03-25_plasticity", "p001237", "exp6/exp6_1"),
    ("24-03-25_plasticity", "p001280", "RL_pharma/RL_pharma_1"),
    ("25-02-2025_cp_drugs_gpu", "25123ic", "full1/full1_1"),
    ("25-02-2025_cp_drugs_gpu", "25178ic", "drug3/drug3_1"),
    ("25-02-2025_cp_drugs_gpu", "25178ic", "drug4/drug4_1"),
    ("25-02-2025_cp_drugs_gpu", "25178ic", "full3/full3_1"),
    ("2025-03-24_cp_full_drugs", "25245lc", "full5/full5_1"),
    ("24-03-25_plasticity", "p001384", "exp1/exp1"),
    ("24-03-25_plasticity", "p001384", "exp1/exp1_1"),
    ("24-03-25_plasticity", "p001384", "exp1/exp1_2"),
    ("23-11-22_cartpole", "c22064", "r1-e1/e1"),
    ("24-03-25_plasticity", "20247", "Bl1/Bl1"),
    
    # Experiments without baseline recordings (no baseline exists in data)
    ("23-05-10_drug_causal", "21985", "func-2/func-2_causal"),
    ("23-05-10_drug_causal", "22097", "causal-3/causal-3_causal"),
]


def apply_corrupted_file_removal(df, verbose=True):
    """
    Remove known corrupted files from the catalog.
    
    Args:
        df: Catalog DataFrame
        verbose: Whether to print progress
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    initial_count = len(df)
    
    # Remove corrupted files from chip 25178ic
    mask_25178ic = (
        (df['chip'] == '25178ic') & 
        (df['experiment'].str.contains('|'.join(CORRUPTED_FILES_25178IC), na=False))
    )
    df = df[~mask_25178ic]
    
    # Remove corrupted files from chip 21985ic
    mask_21985ic = (
        (df['chip'] == '21985ic') &
        (df['experiment'].str.contains('|'.join(CORRUPTED_FILES_21985IC), na=False))
    )
    df = df[~mask_21985ic]
    
    if verbose:
        removed = initial_count - len(df)
        print(f"Removed {removed} corrupted file entries")
    
    return df


def apply_specific_removals(df, verbose=True):
    """
    Remove specific experiments from the catalog.
    
    Args:
        df: Catalog DataFrame
        verbose: Whether to print progress
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    initial_count = len(df)
    
    for proj, chip, experiment in EXPERIMENTS_TO_REMOVE:
        mask = (
            (df['proj'] == proj) &
            (df['chip'] == chip) &
            (df['experiment'] == experiment)
        )
        df = df[~mask]
    
    if verbose:
        removed = initial_count - len(df)
        print(f"Removed {removed} specific experiment entries")
    
    return df


def apply_pattern_removals(df, verbose=True):
    """
    Remove experiments matching certain patterns.
    
    This includes problematic cartpole files, plasticity experiments, etc.
    
    Args:
        df: Catalog DataFrame
        verbose: Whether to print progress
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    initial_count = len(df)
    
    # 23-11-30_cartpole chip 22064: remove exp1, exp2, exp4, exp5, exp6 (keep exp3 and 'long')
    mask1 = (
        (df['proj'] == '23-11-30_cartpole') &
        (df['chip'] == '22064') &
        (df['experiment'].str.contains('exp1[/_]|exp2[/_]|exp4[/_]|exp5[/_]|exp6[/_]', na=False, regex=True)) &
        (~df['experiment'].str.contains('long', na=False))
    )
    df = df[~mask1]
    
    # 23-11-30_cartpole chip 22064: also remove test/ and data/ folders (no baselines exist)
    mask1b = (
        (df['proj'] == '23-11-30_cartpole') &
        (df['chip'] == '22064') &
        (df['experiment'].str.contains('^test/|^data/', na=False, regex=True))
    )
    df = df[~mask1b]
    
    # 23-11-30_cartpole chip P01354: remove exp1, exp2
    mask2 = (
        (df['proj'] == '23-11-30_cartpole') &
        (df['chip'] == 'P01354') &
        (df['experiment'].str.contains('exp1[/_]|exp2[/_]', na=False, regex=True))
    )
    df = df[~mask2]
    
    # 24-03-25_plasticity chip 20247: remove Bl2 variants
    mask3 = (
        (df['proj'] == '24-03-25_plasticity') &
        (df['chip'] == '20247') &
        (df['experiment'].str.contains('Bl2|bl-2|bl_2|bl2', na=False, regex=True))
    )
    df = df[~mask3]
    
    # 23-11-22_cartpole chip c22064: remove r1-e1/e1_0
    mask4 = (
        (df['proj'] == '23-11-22_cartpole') &
        (df['chip'] == 'c22064') &
        (df['experiment'].str.contains('r1-e1/e1_0', na=False, regex=True))
    )
    df = df[~mask4]
    
    # 24-04-18_butterfly chip 23138: remove RL2_k252a variants
    mask5 = (
        (df['proj'] == '24-04-18_butterfly') &
        (df['chip'] == '23138') &
        (df['experiment'].str.contains('RL2_k252a/RL2_k252a_causal|RL2_k252a/RL2_k252a', na=False, regex=True))
    )
    df = df[~mask5]
    
    # 24-04-18_butterfly chip p001237: remove specific cartpole longs
    mask6 = (
        (df['proj'] == '24-04-18_butterfly') &
        (df['chip'] == 'p001237') &
        (df['experiment'].str.contains('exp3/exp3_cartpole_long|exp3/exp3_cartpole_long_1', na=False, regex=True))
    )
    df = df[~mask6]
    
    # 25-02-2025_cp_drugs_gpu chip 25178ic: remove exp1/exp1
    mask7 = (
        (df['proj'] == '25-02-2025_cp_drugs_gpu') &
        (df['chip'] == '25178ic') &
        (df['experiment'].str.contains('exp1/exp1', na=False, regex=True))
    )
    df = df[~mask7]
    
    # Remove cartpole experiments with drugs
    mask8 = (
        (df['experiment'].str.contains('cartpole', na=False)) &
        (df.get('drug', pd.Series([None]*len(df))).notna())
    )
    if 'drug' in df.columns:
        df = df[~mask8]
    
    # Remove experiments with 'Trace' in name
    mask9 = df['experiment'].str.contains('Trace', na=False)
    df = df[~mask9]
    
    if verbose:
        removed = initial_count - len(df)
        print(f"Removed {removed} entries matching exclusion patterns")
    
    return df


def apply_path_fixes(df, verbose=True):
    """
    Apply path corrections to the catalog.
    
    Args:
        df: Catalog DataFrame
        verbose: Whether to print progress
        
    Returns:
        pd.DataFrame: Fixed DataFrame
    """
    # Fix 25_04_15_busiest_bee chip 25244ic: remove stale exp/ entries (exp1/ is correct)
    mask = (
        (df['proj'] == '25_04_15_busiest_bee') &
        (df['chip'] == '25244ic') &
        (df['experiment'].str.startswith('exp/'))
    )
    if mask.any():
        if verbose:
            print(f"Removed {mask.sum()} stale exp/ entries for 25_04_15_busiest_bee/25244ic")
        df = df[~mask]
    
    # Append .raw.h5 extension to full_path if missing
    def append_raw_h5(row):
        if pd.notna(row['full_path']):
            if not str(row['full_path']).endswith('.raw.h5'):
                return str(row['full_path']) + '.raw.h5'
        return row['full_path']
    
    df['full_path'] = df.apply(append_raw_h5, axis=1)
    
    return df


def apply_frequency_fixes(df, verbose=True):
    """
    Apply specific frequency corrections.
    
    Args:
        df: Catalog DataFrame
        verbose: Whether to print progress
        
    Returns:
        pd.DataFrame: Fixed DataFrame
    """
    # Fix exp1_causal on chip 25123ic: set freq=0.5, type=stim
    mask = (df['chip'] == '25123ic') & (df['experiment'] == 'exp1/exp1_causal')
    
    if mask.any():
        df.loc[mask, 'freq'] = 0.5
        df.loc[mask, 'type'] = 'stim'
        if verbose:
            print("Fixed exp1_causal on chip 25123ic: freq=0.5Hz, type=stim")
    
    return df


def clean_baseline_experiments(df, verbose=True):
    """
    Clean baseline experiments by removing spurious stimulation data.
    
    Baseline experiments should always be spontaneous activity with no stimulation.
    
    Args:
        df: Catalog DataFrame
        verbose: Whether to print progress
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Identify baseline experiments
    baseline_mask = df['experiment'].apply(is_baseline)
    
    # For baseline experiments, clear stimulation fields
    df.loc[baseline_mask, 'n_stims'] = 0
    df.loc[baseline_mask, 'freq'] = 0.0
    if 'calculated_freq' in df.columns:
        df.loc[baseline_mask, 'calculated_freq'] = None
    if 'log_available' in df.columns:
        df.loc[baseline_mask, 'log_available'] = False
    df.loc[baseline_mask, 'type'] = 'spontaneous'
    
    # Note: Cartpole baselines (first in series without numeric suffix, e.g., exp2_cartpole_long)
    # ARE valid baselines and are correctly detected by is_baseline()
    
    if verbose:
        print(f"Cleaned {baseline_mask.sum()} baseline experiments")
    
    return df


def add_derived_columns(df):
    """
    Add derived columns (proj, exp, baseline) if not present.

    Args:
        df: Catalog DataFrame

    Returns:
        pd.DataFrame: DataFrame with derived columns
    """
    if 'proj' not in df.columns:
        df['proj'] = df['base_path'].apply(extract_proj_from_s3_path)

    if 'exp' not in df.columns:
        df['exp'] = df['experiment'].apply(extract_exp_from_experiment_path)

    if 'baseline' not in df.columns:
        df['baseline'] = df['experiment'].apply(is_baseline)

    return df


def resolve_baseline_inheritance(df):
    """
    Add 'inherited_baseline_from' column showing which exp value provides baseline.

    For each (proj, chip, exp) group without a baseline:
    - Find all earlier exp values for same (proj, chip) that have baselines
    - Use the most recent (numerically largest) earlier exp
    - Mark with inherited_baseline_from = 'exp1' (or None if direct baseline)

    Algorithm:
    1. Group by (proj, chip)
    2. For each chip group:
       a. Sort exp values numerically (using compare_exp_names)
       b. Build baseline_map: {exp -> has_direct_baseline}
       c. For each exp without baseline, find max(earlier_exps_with_baseline)
       d. Assign inherited_baseline_from

    Args:
        df: Catalog DataFrame with 'proj', 'chip', 'exp', and 'baseline' columns

    Returns:
        pd.DataFrame: DataFrame with new 'inherited_baseline_from' column

    Examples:
        If exp1 has baseline and exp2 doesn't:
        - exp1 recordings: inherited_baseline_from = None
        - exp2 recordings: inherited_baseline_from = 'exp1'
    """
    # Initialize the column
    df['inherited_baseline_from'] = None

    # Group by (proj, chip) to handle inheritance within each chip
    for (proj, chip), chip_group in df.groupby(['proj', 'chip']):
        # Get unique exp values for this chip
        exps_in_chip = chip_group['exp'].unique()

        # Build baseline_map: {exp -> has_direct_baseline}
        baseline_map = {}
        for exp in exps_in_chip:
            exp_group = chip_group[chip_group['exp'] == exp]
            baseline_map[exp] = exp_group['baseline'].any()

        # For each exp, determine inheritance
        for exp in exps_in_chip:
            if baseline_map[exp]:
                # Has direct baseline, no inheritance needed
                continue

            # Find all earlier exps with baselines
            earlier_exps_with_baseline = [
                earlier_exp for earlier_exp in exps_in_chip
                if baseline_map[earlier_exp] and compare_exp_names(earlier_exp, exp)
            ]

            if earlier_exps_with_baseline:
                # Sort to find the most recent (largest) earlier exp
                earlier_exps_with_baseline.sort(key=lambda x: (x,), reverse=True)
                # Use custom comparison to find the "latest" earlier exp
                most_recent = earlier_exps_with_baseline[0]
                for candidate in earlier_exps_with_baseline[1:]:
                    if compare_exp_names(most_recent, candidate):
                        most_recent = candidate

                # Assign inheritance to all recordings in this (proj, chip, exp)
                mask = (df['proj'] == proj) & (df['chip'] == chip) & (df['exp'] == exp)
                df.loc[mask, 'inherited_baseline_from'] = most_recent

    return df


def apply_catalog_fixes(df, verbose=True):
    """
    Apply all catalog fixes in the correct order.

    This is the main entry point for postprocessing a catalog.

    Args:
        df: Catalog DataFrame
        verbose: Whether to print progress

    Returns:
        pd.DataFrame: Fully processed DataFrame
    """
    if verbose:
        print(f"Starting postprocessing with {len(df)} entries")

    # Add derived columns first
    df = add_derived_columns(df)

    # Remove failed experiments
    if 'success' in df.columns:
        initial = len(df)
        df = df[df['success'] == True]
        if verbose:
            print(f"Removed {initial - len(df)} failed experiments")

    # Apply fixes in order
    df = apply_path_fixes(df, verbose)
    df = apply_corrupted_file_removal(df, verbose)
    df = apply_pattern_removals(df, verbose)
    df = apply_specific_removals(df, verbose)
    df = apply_frequency_fixes(df, verbose)
    df = clean_baseline_experiments(df, verbose)

    # Resolve baseline inheritance AFTER all other fixes
    df = resolve_baseline_inheritance(df)
    if verbose:
        inherited_count = df['inherited_baseline_from'].notna().sum()
        if inherited_count > 0:
            print(f"Resolved baseline inheritance for {inherited_count} recordings")

    if verbose:
        print(f"Postprocessing complete: {len(df)} entries remaining")

    return df
