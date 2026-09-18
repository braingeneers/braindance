"""
Catalog Generation Module for BrainDance Data Manager

This module provides tools for generating and maintaining catalogs of
BrainDance experiments stored in S3. It scans S3 paths, discovers experiments,
extracts metadata (stimulation counts, frequencies, etc.), and produces
CSV catalogs compatible with RecordingCatalog.

Simple Usage:
    from braindance.utils.data_manager.utils.catalogging import (
        generate_catalog, add_metadata, add_units
    )
    
    # Generate catalog from all configured S3 paths
    catalog = generate_catalog()
    
    # Add organoid metadata (age, etc.)
    catalog = add_metadata(catalog)
    
    # Add spike sorting unit counts
    catalog = add_units(catalog, verbose=True)
    
    # Save
    catalog.to_csv('my_catalog.csv', index=False)

S3 paths are configured in config.py - edit DEFAULT_S3_BASES to add new data sources.
"""

import pandas as pd

from .generator import CatalogGenerator, DataPathManager
from .experiment import BraindanceExperiment
from .s3_helpers import (
    construct_log_path,
    construct_spike_data_path,
    load_log_from_s3,
    get_s3_client,
    parse_s3_path,
    check_s3_file_exists,
    get_num_units_for_experiment_folder,
)
from .validators import (
    is_baseline,
    is_baseline_with_reason,
    calculate_stim_frequency,
    categorize_frequency,
    extract_proj_from_s3_path,
    extract_exp_from_experiment_path,
)
from .metadata import load_org_metadata, merge_metadata, calculate_organoid_age, add_drug_column
from .postprocessing import apply_catalog_fixes, apply_corrupted_file_removal
from .config import DEFAULT_S3_BASES, get_default_catalog_path


# =============================================================================
# SIMPLE API - Start here!
# =============================================================================

def generate_catalog(s3_paths=None, verbose=True):
    """
    Generate a catalog from S3 paths.
    
    Args:
        s3_paths: List of S3 base paths to scan. If None, uses DEFAULT_S3_BASES from config.py
        verbose: Print progress messages
        
    Returns:
        pandas.DataFrame: Catalog with columns like chip, experiment, freq, type, etc.
        
    Example:
        catalog = generate_catalog()
        catalog = generate_catalog(s3_paths=['s3://mybucket/mydata/'])
    """
    generator = CatalogGenerator(s3_bases=s3_paths)
    catalog = generator.generate(verbose=verbose, expand_sub_experiments=True, include_units=False)
    
    if verbose:
        print(f"\n✓ Generated catalog with {len(catalog)} experiments")
        print(f"  Chips: {catalog['chip'].nunique()}")
        if 'type' in catalog.columns:
            print(f"  Types: {catalog['type'].value_counts().to_dict()}")
    
    return catalog


def add_metadata(catalog, metadata_path=None, verbose=True):
    """
    Add organoid metadata (age, org_id, etc.) to a catalog.

    Args:
        catalog: DataFrame or path to CSV
        metadata_path: Path to metadata CSV. If None, uses bundled metadata.
        verbose: Print progress messages

    Returns:
        pandas.DataFrame: Catalog with metadata columns added

    Example:
        catalog = add_metadata(catalog)
    """
    if isinstance(catalog, str):
        from pathlib import Path
        catalog_path = Path(catalog)
        if not catalog_path.exists():
            raise ValueError(
                f"Catalog file not found: {catalog_path}\n"
                "To generate a catalog, run:\n"
                "  python -m braindance.utils.data_manager.catalogging generate"
            )
        catalog = pd.read_csv(catalog_path)
    
    catalog = merge_metadata(catalog, metadata_path=metadata_path)
    catalog = add_drug_column(catalog)
    catalog = apply_catalog_fixes(catalog, verbose=verbose)
    
    if verbose:
        if 'age_days' in catalog.columns:
            with_age = catalog['age_days'].notna().sum()
            print(f"✓ Added metadata: {with_age}/{len(catalog)} have age info")
    
    return catalog


def add_units(catalog, verbose=True):
    """
    Add num_units column by reading spike sorting files from S3.

    Searches for spike data at the experiment folder level (e.g., drug1/, drug3/)
    since spike sorting is done per experiment folder, not per chip.
    
    This function is incremental: if the catalog already has a 'num_units' column,
    it will only fetch units for entries that don't have them yet.

    Args:
        catalog: DataFrame or path to CSV
        verbose: Print progress for each experiment folder

    Returns:
        pandas.DataFrame: Catalog with num_units column added

    Example:
        catalog = add_units(catalog, verbose=True)
    """
    if isinstance(catalog, str):
        from pathlib import Path
        catalog_path = Path(catalog)
        if not catalog_path.exists():
            raise ValueError(
                f"Catalog file not found: {catalog_path}\n"
                "To generate a catalog, run:\n"
                "  python -m braindance.utils.data_manager.catalogging generate"
            )
        catalog = pd.read_csv(catalog_path)
    
    # Extract experiment folder from experiment path
    def get_exp_folder(exp):
        if pd.isna(exp):
            return None
        return exp.split('/')[0] if '/' in exp else exp
    
    catalog = catalog.copy()
    catalog['_exp_folder'] = catalog['experiment'].apply(get_exp_folder)
    
    # Initialize num_units column if it doesn't exist
    if 'num_units' not in catalog.columns:
        catalog['num_units'] = None
    
    # Get unique (base_path, chip, experiment_folder) combinations
    exp_combos = catalog[['base_path', 'chip', '_exp_folder']].drop_duplicates().reset_index(drop=True)
    
    # Filter to only process entries that don't have num_units yet
    # Check which experiment folders already have unit counts
    existing_units = {}
    for _, row in catalog.iterrows():
        key = (row['base_path'], row['chip'], row['_exp_folder'])
        if pd.notna(row['num_units']):
            existing_units[key] = row['num_units']
    
    # Filter exp_combos to only include those without existing units
    exp_combos_to_fetch = []
    for _, row in exp_combos.iterrows():
        key = (row['base_path'], row['chip'], row['_exp_folder'])
        if key not in existing_units:
            exp_combos_to_fetch.append(row)
    
    if verbose:
        print(f"Found {len(exp_combos)} unique (chip, experiment_folder) combinations")
        if existing_units:
            print(f"Skipping {len(existing_units)} that already have unit counts")
        print(f"Fetching units for {len(exp_combos_to_fetch)} new combinations\n")
    
    # Build unit count mapping (start with existing)
    unit_counts = existing_units.copy()
    
    for idx, row in enumerate(exp_combos_to_fetch):
        base_path, chip, exp_folder = row['base_path'], row['chip'], row['_exp_folder']
        
        if verbose:
            path_parts = base_path.rstrip('/').split('/')
            proj = path_parts[-1] if path_parts else base_path
            print(f"[{idx+1}/{len(exp_combos_to_fetch)}] {chip}/{exp_folder} ({proj})")

        num_units = get_num_units_for_experiment_folder(base_path, chip, exp_folder, verbose=verbose)
        unit_counts[(base_path, chip, exp_folder)] = num_units

        if verbose:
            if num_units:
                print(f"→ {num_units} units")
            else:
                print(f"→ no spike data")
    
    # Apply to catalog
    catalog['num_units'] = catalog.apply(
        lambda r: unit_counts.get((r['base_path'], r['chip'], r['_exp_folder'])), axis=1
    )
    catalog = catalog.drop(columns=['_exp_folder'])
    
    if verbose:
        with_units = catalog['num_units'].notna().sum()
        print(f"\n✓ Total entries with unit counts: {with_units}/{len(catalog)}")
    
    return catalog

def validate_recording_files(catalog, verbose=True):
    """
    Check that each recording's specific spike_data file exists on S3.

    add_units() assigns unit counts at the experiment-folder level, so recordings
    whose individual spike_data file doesn't exist still get num_units > 0.
    This function validates per-recording and sets num_units = None for missing files,
    causing them to land in problematic_recordings.csv.

    Args:
        catalog: DataFrame with base_path, chip, experiment, num_units columns
        verbose: Print progress

    Returns:
        pandas.DataFrame: Catalog with num_units cleared for missing files
    """
    catalog = catalog.copy()

    # Only check rows that currently have num_units (i.e., supposedly have spike data)
    to_check = catalog[catalog['num_units'].notna()].copy()

    if len(to_check) == 0:
        if verbose:
            print("No recordings with num_units to validate")
        return catalog

    # Build unique S3 paths to check (many recordings share the same spike file)
    path_map = {}  # s3_path -> list of catalog indices
    for idx, row in to_check.iterrows():
        s3_path = construct_spike_data_path(row['base_path'], row['chip'], row['experiment'])
        if s3_path not in path_map:
            path_map[s3_path] = []
        path_map[s3_path].append(idx)

    if verbose:
        print(f"Validating {len(path_map)} unique spike_data paths for {len(to_check)} recordings...")

    # Check each unique path
    # Group by bucket to reuse S3 clients
    from collections import defaultdict
    bucket_paths = defaultdict(list)
    for s3_path in path_map:
        bucket, _ = parse_s3_path(s3_path)
        bucket_paths[bucket].append(s3_path)

    missing_count = 0
    checked = 0
    for bucket, paths in bucket_paths.items():
        s3_client = get_s3_client(bucket)
        for s3_path in paths:
            checked += 1
            if verbose and checked % 100 == 0:
                print(f"  Checked {checked}/{len(path_map)} paths ({missing_count} missing)...")

            if not check_s3_file_exists(s3_path, s3_client=s3_client):
                # File missing — clear num_units for all recordings pointing to this path
                for cat_idx in path_map[s3_path]:
                    catalog.at[cat_idx, 'num_units'] = None
                missing_count += 1

    if verbose:
        invalidated = (to_check['num_units'].notna().sum() -
                       catalog.loc[to_check.index, 'num_units'].notna().sum())
        print(f"✓ Validation complete: {missing_count}/{len(path_map)} paths missing "
              f"({invalidated} recordings invalidated)")

    return catalog


__all__ = [
    # Simple API - start here!
    'generate_catalog',
    'add_metadata',
    'add_units',
    'validate_recording_files',
    # Config
    'DEFAULT_S3_BASES',
    'get_default_catalog_path',
    # Advanced: Main classes
    'CatalogGenerator',
    'DataPathManager', 
    'BraindanceExperiment',
    # Advanced: S3 helpers
    'construct_log_path',
    'load_log_from_s3',
    'get_s3_client',
    'parse_s3_path',
    'check_s3_file_exists',
    # Advanced: Validators
    'is_baseline',
    'is_baseline_with_reason',
    'calculate_stim_frequency',
    'categorize_frequency',
    'extract_proj_from_s3_path',
    'extract_exp_from_experiment_path',
    # Advanced: Metadata
    'load_org_metadata',
    'merge_metadata',
    'calculate_organoid_age',
    # Advanced: Postprocessing
    'apply_catalog_fixes',
    'apply_corrupted_file_removal',
]
