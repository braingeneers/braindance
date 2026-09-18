"""
S3 Helper Functions for Catalog Generation

Provides utilities for interacting with S3 storage, including:
- Client creation with proper endpoint configuration
- Path parsing and construction
- File existence checking
- Log file loading
"""

import os
import re
import boto3
import pandas as pd
from io import StringIO
from urllib.parse import urlparse
from botocore.client import Config

from .config import S3_ENDPOINTS
from braindance.spike_data import load_spike_pickle


def get_s3_client(bucket_name=None, endpoint_url=None):
    """
    Get an S3 client configured for the appropriate endpoint.
    
    Args:
        bucket_name: Name of the bucket (used to determine endpoint)
        endpoint_url: Explicit endpoint URL (overrides bucket_name lookup)
        
    Returns:
        boto3.client: Configured S3 client
    """
    if endpoint_url is None and bucket_name is not None:
        endpoint_url = S3_ENDPOINTS.get(bucket_name)
    
    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        config=Config(signature_version='s3v4')
    )


def parse_s3_path(s3_path):
    """
    Parse an S3 path into bucket and key components.
    
    Args:
        s3_path: Full S3 path (e.g., 's3://bucket/path/to/file')
        
    Returns:
        tuple: (bucket_name, key)
        
    Raises:
        ValueError: If path is not a valid S3 path
    """
    if not s3_path.startswith('s3://'):
        raise ValueError(f"Invalid S3 path: {s3_path}")
    
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    
    return bucket, key


def check_s3_file_exists(s3_path, s3_client=None):
    """
    Check if a file exists in S3.
    
    Args:
        s3_path: Full S3 path to the file
        s3_client: Optional pre-configured S3 client
        
    Returns:
        bool: True if file exists, False otherwise
    """
    try:
        bucket, key = parse_s3_path(s3_path)
        
        if s3_client is None:
            s3_client = get_s3_client(bucket)
        
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def construct_log_path(base_path, chip, experiment):
    """
    Construct the S3 path for a stimulus log file.
    
    Args:
        base_path: S3 base path (e.g., 's3://braingeneersdev/asrobbin/braindance_data/24-08-16_busybee/')
        chip: Chip ID (e.g., '22194a')
        experiment: Experiment path (e.g., 'freqs/freqs_cont_24')
        
    Returns:
        str or None: S3 path to the log file, or None if inputs are invalid
        
    Example:
        >>> construct_log_path(
        ...     's3://braingeneersdev/asrobbin/braindance_data/24-08-16_busybee/',
        ...     '22194a',
        ...     'freqs/freqs_cont_24'
        ... )
        's3://braingeneersdev/asrobbin/braindance_data/24-08-16_busybee/22194a/freqs/freqs_cont_24_log.csv'
    """
    if pd.isna(base_path) or pd.isna(chip) or pd.isna(experiment):
        return None
    
    # Clean up paths
    base_path = base_path.rstrip('/')
    experiment = experiment.strip()
    
    # Extract the experiment name from the full experiment path
    exp_name = experiment.split('/')[-1]  # Gets 'freqs_cont_24' from 'freqs/freqs_cont_24'
    exp_dir = experiment.split('/')[0]     # Gets 'freqs' from 'freqs/freqs_cont_24'
    
    # Construct the log path
    log_path = f"{base_path}/{chip}/{exp_dir}/{exp_name}_log.csv"
    
    return log_path


def load_log_from_s3(s3_path, s3_client=None):
    """
    Load a stimulus log CSV from S3.
    
    Args:
        s3_path: Full S3 path to the log file
        s3_client: Optional pre-configured S3 client
        
    Returns:
        pd.DataFrame or None: Log data as DataFrame, or None if failed
    """
    if s3_path is None:
        return None
    
    try:
        bucket, key = parse_s3_path(s3_path)
        
        if s3_client is None:
            s3_client = get_s3_client(bucket)
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        
        return pd.read_csv(StringIO(content))
        
    except Exception:
        # Don't print errors for every failed log load to reduce noise
        return None


def construct_spike_data_path(base_path, chip, experiment):
    """
    Construct the S3 path for a spike_data pickle file.
    
    Args:
        base_path: S3 base path
        chip: Chip ID
        experiment: Experiment path (e.g., 'exp1/exp1_cont_5')
        
    Returns:
        str: S3 path to the spike_data file
    """
    base_path = base_path.rstrip('/')
    exp_dir = experiment.split('/')[0]
    exp_name = experiment.split('/')[-1]
    
    return f"{base_path}/{chip}/{exp_dir}/spike_data/{exp_name}_spike_data.pkl"


def construct_rt_sort_path(base_path, chip, experiment):
    """
    Construct the S3 path for an rt_sort pickle file.
    
    Args:
        base_path: S3 base path
        chip: Chip ID  
        experiment: Experiment path (e.g., 'exp1/exp1_cont_5')
        
    Returns:
        str: S3 path to the rt_sort file
    """
    base_path = base_path.rstrip('/')
    exp_dir = experiment.split('/')[0]
    exp_name = experiment.split('/')[-1]
    
    return f"{base_path}/{chip}/{exp_dir}/rt_sort/{exp_name}_rt_sort.pickle"


def load_spike_data_from_s3(s3_path, verbose=False):
    """
    Load a SpikeData object from S3 by downloading complete file first.

    Avoids streaming truncation issues by downloading the full file to disk,
    then loading from the complete local file.

    Args:
        s3_path: Full S3 path to the pickle file
        verbose: If True, print error messages

    Returns:
        SpikeData or None: Loaded SpikeData object, or None if failed
    """
    try:
        from pathlib import Path
        from braindance.utils.data_manager.utils.data_loading.s3_loader import S3Loader
        from braindance import get_data_dir

        # Determine endpoint from bucket name
        bucket, _ = parse_s3_path(s3_path)
        endpoint_url = S3_ENDPOINTS.get(bucket)

        # Create S3Loader with cache directory and correct endpoint
        cache_dir = get_data_dir() / ".s3_cache"
        loader = S3Loader(endpoint_url=endpoint_url, cache_dir=cache_dir)

        # Generate cache path for this file
        cache_path = cache_dir / Path(s3_path).name

        # Check if already cached
        if cache_path.exists():
            if verbose:
                print(f"    Using cached file: {cache_path.name}", flush=True)
        else:
            # Download complete file first (avoids streaming truncation)
            if verbose:
                print(f"    Downloading: {Path(s3_path).name}", flush=True)
            if not loader.download_file(s3_path, cache_path):
                return None

        # Load from complete local file (no streaming)
        with open(cache_path, 'rb') as f:
            spike_data = load_spike_pickle(f)

        return spike_data

    except Exception as e:
        if verbose:
            print(f"    ERROR loading pickle from {s3_path}: {e}", flush=True)
        return None


def get_num_units_for_chip(base_path, chip, sample_experiment=None, verbose=False):
    """
    Get the number of units for a chip by loading one spike_data file.
    
    Since all recordings from the same chip share the same rt_sort spike sorting,
    we only need to load one file per chip.
    
    Args:
        base_path: S3 base path
        chip: Chip ID
        sample_experiment: Optional specific experiment to check
        verbose: If True, print progress messages
        
    Returns:
        int or None: Number of units, or None if unavailable
    """
    try:
        import braingeneers.utils.s3wrangler as wr
        
        # Try to find a spike_data or rt_sort file for this chip
        chip_path = f"{base_path.rstrip('/')}/{chip}/"
        
        if verbose:
            print(f"  Searching: {chip_path}", flush=True)
        
        # First try spike_data files
        try:
            if verbose:
                print(f"  Looking for: *_spike_data.pkl", flush=True)
            spike_data_files = wr.list_objects(chip_path, suffix='_spike_data.pkl')
            if spike_data_files:
                if verbose:
                    print(f"  Found: {spike_data_files[0]}", flush=True)
                spike_data = load_spike_data_from_s3(spike_data_files[0], verbose=verbose)
                if spike_data is not None and hasattr(spike_data, 'N'):
                    return spike_data.N
        except Exception as e:
            if verbose:
                print(f"  No spike_data files: {e}", flush=True)
        
        # Try rt_sort files
        try:
            if verbose:
                print(f"  Looking for: *_rt_sort.pickle", flush=True)
            rt_sort_files = wr.list_objects(chip_path, suffix='_rt_sort.pickle')
            if rt_sort_files:
                if verbose:
                    print(f"  Found: {rt_sort_files[0]}", flush=True)
                spike_data = load_spike_data_from_s3(rt_sort_files[0], verbose=verbose)
                if spike_data is not None and hasattr(spike_data, 'N'):
                    return spike_data.N
        except Exception as e:
            if verbose:
                print(f"  No rt_sort files: {e}", flush=True)
        
        return None
        
    except Exception as e:
        if verbose:
            print(f"  Error: {e}", flush=True)
        return None


def get_num_units_for_experiment_folder(base_path, chip, experiment_folder, verbose=False):
    """
    Get the number of units for a specific experiment folder within a chip.
    
    Spike sorting is done per experiment folder (e.g., drug1/, drug3/, full3/),
    so each folder may have different unit counts.
    
    Args:
        base_path: S3 base path
        chip: Chip ID
        experiment_folder: Top-level experiment folder (e.g., 'drug1', 'drug3', 'full3')
        verbose: If True, print progress messages
        
    Returns:
        int or None: Number of units, or None if unavailable
    """
    try:
        import braingeneers.utils.s3wrangler as wr
        
        # Search within the experiment folder
        exp_folder_path = f"{base_path.rstrip('/')}/{chip}/{experiment_folder}/"
        
        if verbose:
            print(f"    Searching: {exp_folder_path}", flush=True)
        
        # First try spike_data files in this experiment folder (including spike_data/ subdirectory)
        try:
            # Try the spike_data/ subdirectory first (common location)
            spike_data_subdir = f"{exp_folder_path}spike_data/"
            if verbose:
                print(f"    DEBUG: Checking spike_data subdir: {spike_data_subdir}", flush=True)
            spike_data_files = wr.list_objects(spike_data_subdir, suffix='_spike_data.pkl')
            if verbose:
                print(f"    DEBUG: Found {len(spike_data_files) if spike_data_files else 0} files in subdir", flush=True)

            # If not found in subdirectory, try the experiment folder root
            if not spike_data_files:
                if verbose:
                    print(f"    DEBUG: Checking exp folder root: {exp_folder_path}", flush=True)
                spike_data_files = wr.list_objects(exp_folder_path, suffix='_spike_data.pkl')
                if verbose:
                    print(f"    DEBUG: Found {len(spike_data_files) if spike_data_files else 0} files in root", flush=True)

            if spike_data_files:
                if verbose:
                    print(f"    Found spike_data: {spike_data_files[0]}", flush=True)
                spike_data = load_spike_data_from_s3(spike_data_files[0], verbose=verbose)
                if spike_data is not None and hasattr(spike_data, 'N'):
                    if verbose:
                        print(f"    DEBUG: Successfully loaded, N={spike_data.N}", flush=True)
                    return spike_data.N
                elif verbose:
                    print(f"    DEBUG: Loaded but no N attribute or None", flush=True)
        except Exception as e:
            if verbose:
                print(f"    ERROR in spike_data search for {experiment_folder}: {e}", flush=True)

        return None
        
    except Exception as e:
        if verbose:
            print(f"    Error searching {experiment_folder}: {e}", flush=True)
        return None
