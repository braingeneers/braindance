"""
Configuration for Catalog Generation

Contains S3 bucket configurations and utility functions for accessing configuration values.
S3 paths are imported from the canonical source in internal_usage/catalog/config.py.
"""

import os
from pathlib import Path


# Import S3 paths from the canonical location
try:
    import sys
    _catalog_dir = Path(__file__).parent.parent.parent / 'internal_usage' / 'catalog'
    sys.path.insert(0, str(_catalog_dir))
    from config import S3_BASES as DEFAULT_S3_BASES
    sys.path.pop(0)
except ImportError:
    # Fallback if internal_usage not available
    DEFAULT_S3_BASES = []


# S3 endpoint configuration by bucket
S3_ENDPOINTS = {
    'braingeneersdev': 'https://s3-west.nrp-nautilus.io',
    'braingeneers': 'https://s3-west.nrp-nautilus.io',
}


# Log file suffixes to search for
LOG_SUFFIXES = [
    '_log.csv',
    '_game_log.csv', 
    '_pattern_log.csv',
    '_reward_log.csv',
    '_causal_log.csv',
]


# Blacklist patterns for directories that should not be considered experiments
EXPERIMENT_BLACKLIST_PATTERNS = [
    '_meta',
    'metadata',
    'configs',
    'config',
    '.git',
    '__pycache__',
    'temp',
    'tmp',
]


# Stimulation count to frequency mapping
STIM_TO_FREQ_MAP = {
    1050: 0.5,
    2100: 1.0,
    4200: 2.0,
    8400: 4.0,
    16800: 8.0,
}


# Target frequencies for categorization
TARGET_FREQUENCIES = [0.5, 1, 2, 4, 8]


def get_default_catalog_path():
    """
    Get the default path for the catalog file.
    
    Returns the path from braindance config if available,
    otherwise returns a default location.
    
    Returns:
        Path: Default catalog file path
    """
    # Try to get from braindance config
    config_dir = Path.home() / '.braindance'
    if config_dir.exists():
        config_file = config_dir / 'config.json'
        if config_file.exists():
            import json
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    if 'catalog' in config:
                        return Path(config['catalog'])
            except Exception:
                pass
    
    # Default fallback
    return Path.cwd() / 'braindance_catalog.csv'


def get_default_metadata_path():
    """
    Get the default path for the organoid metadata file.
    
    Returns:
        Path: Default metadata file path
    """
    # The metadata file is bundled with this module
    module_dir = Path(__file__).parent
    return module_dir / 'org_metadata.csv'


def get_s3_endpoint(bucket_name):
    """
    Get the S3 endpoint URL for a given bucket.
    
    Args:
        bucket_name: Name of the S3 bucket
        
    Returns:
        str or None: Endpoint URL or None for default AWS
    """
    return S3_ENDPOINTS.get(bucket_name)
