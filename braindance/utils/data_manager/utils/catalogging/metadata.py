"""
Metadata Functions for Catalog Generation

Handles loading organoid metadata and merging it with catalog data,
including age calculation based on experiment dates.
"""

import re
import pandas as pd
from datetime import datetime
from pathlib import Path


def load_org_metadata(metadata_path=None):
    """
    Load organoid metadata from CSV file.
    
    Args:
        metadata_path: Path to metadata CSV. If None, uses bundled file.
        
    Returns:
        pd.DataFrame: Metadata DataFrame with columns like Chip, Org_Day_0, etc.
    """
    if metadata_path is None:
        # Use bundled metadata file
        module_dir = Path(__file__).parent
        metadata_path = module_dir / 'org_metadata.csv'
    
    if not Path(metadata_path).exists():
        print(f"Warning: Metadata file not found at {metadata_path}")
        return pd.DataFrame()
    
    return pd.read_csv(metadata_path)


def extract_date_from_path(path):
    """
    Extract experiment date from S3 path.
    
    Looks for patterns like YY-MM-DD or YY_MM_DD in paths.
    
    Args:
        path: S3 path or file path string
        
    Returns:
        datetime or None: Extracted date or None if not found
    """
    if pd.isna(path):
        return None
    
    pattern = r"(\d{2})[-_](\d{2})[-_](\d{2})"
    match = re.search(pattern, path)
    if match:
        year, month, day = match.groups()
        try:
            return datetime(2000 + int(year), int(month), int(day))
        except ValueError:
            return None
    return None


def parse_date(date_str):
    """
    Parse a date string in MM/DD/YY format.
    
    Args:
        date_str: Date string to parse
        
    Returns:
        datetime or None: Parsed date or None if failed
    """
    if pd.isna(date_str):
        return None
    try:
        return datetime.strptime(date_str, "%m/%d/%y")
    except (ValueError, TypeError):
        return None


def calculate_organoid_age(row, org_day_0_col='Org_Day_0', path_col='base_path', 
                           existing_age_col='Org_Age'):
    """
    Calculate organoid age at experiment time.
    
    Args:
        row: DataFrame row
        org_day_0_col: Column name for organoid day 0 date
        path_col: Column name for path containing experiment date
        existing_age_col: Column name for existing age (fallback)
        
    Returns:
        int or None: Age in days, or existing value if calculation fails
    """
    org_day_0 = parse_date(row.get(org_day_0_col))
    exp_date = extract_date_from_path(row.get(path_col))
    
    if org_day_0 and exp_date:
        delta = exp_date - org_day_0
        return delta.days
    
    return row.get(existing_age_col)


def merge_metadata(catalog_df, metadata_df=None, metadata_path=None):
    """
    Merge catalog with organoid metadata.
    
    Performs a left join on chip ID and calculates organoid age.
    
    Args:
        catalog_df: Catalog DataFrame with 'chip' column
        metadata_df: Optional metadata DataFrame (if not provided, loads from file)
        metadata_path: Optional path to metadata CSV
        
    Returns:
        pd.DataFrame: Merged DataFrame with metadata columns
    """
    if metadata_df is None:
        metadata_df = load_org_metadata(metadata_path)
    
    if metadata_df.empty:
        print("Warning: No metadata to merge")
        return catalog_df
    
    # Handle chip ID typo (temporary fix for merge)
    original_chips = catalog_df['chip'].copy()
    catalog_df['chip'] = catalog_df['chip'].replace("25245lc", "25245ic")
    
    # Merge on chip columns
    merged = pd.merge(
        catalog_df, 
        metadata_df, 
        left_on='chip', 
        right_on='Chip', 
        how='left'
    )
    
    # Drop duplicate Chip column if present
    if 'Chip' in merged.columns:
        merged = merged.drop(columns=['Chip'])
    
    # Calculate organoid age
    if 'Org_Day_0' in merged.columns:
        merged['Org_Age'] = merged.apply(calculate_organoid_age, axis=1)
    
    # Revert chip ID fix
    merged['chip'] = merged['chip'].replace("25245ic", "25245lc")
    
    return merged


def extract_drug_from_experiment(experiment, drug_list=None):
    """
    Extract drug name from experiment string if any known drug is found.
    
    Args:
        experiment: Experiment name string
        drug_list: List of drug names to search for
        
    Returns:
        str or None: Drug name if found, None otherwise
    """
    if drug_list is None:
        drug_list = ["k252a", "thc", "nbqx_apv", "gaba-1", "gabazine", "bdnf", "nbqx", "apv"]
    
    if pd.isna(experiment) or not isinstance(experiment, str):
        return None
    
    experiment_lower = experiment.lower()
    
    for drug in drug_list:
        if drug.lower() in experiment_lower:
            return drug
    
    return None


def add_drug_column(catalog_df):
    """
    Add a 'drug' column to the catalog based on experiment names.
    
    Args:
        catalog_df: Catalog DataFrame with 'experiment' column
        
    Returns:
        pd.DataFrame: DataFrame with added 'drug' column
    """
    catalog_df['drug'] = catalog_df['experiment'].apply(extract_drug_from_experiment)
    return catalog_df
