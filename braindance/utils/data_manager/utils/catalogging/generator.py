"""
Catalog Generator

Main module for scanning S3 paths and generating experiment catalogs.
Contains DataPathManager for S3 operations and CatalogGenerator for
orchestrating the catalog generation process.
"""

import os
import re
import pandas as pd
import boto3
from urllib.parse import urlparse
from botocore.client import Config

from .config import (
    DEFAULT_S3_BASES, 
    EXPERIMENT_BLACKLIST_PATTERNS,
    LOG_SUFFIXES,
)
from .experiment import BraindanceExperiment
from .validators import count_digits, extract_number
from .s3_helpers import get_s3_client, get_num_units_for_chip

# Import braingeneers S3 utilities
try:
    import braingeneers.utils.s3wrangler as wr
except ImportError:
    wr = None
    print("Warning: braingeneers.utils.s3wrangler not found. S3 operations will be limited.")


class DataPathManager:
    """
    Manages data paths across different S3 structures and local paths.
    
    Handles the complexity of different bucket endpoints, path structures,
    and provides caching for improved performance.
    
    Attributes:
        s3_bases: List of S3 base paths to search
        chip_cache: Cache of chip directories by base path
        experiment_cache: Cache of experiment directories by chip path
    """
    
    def __init__(self, s3_bases=None):
        """
        Initialize with a list of S3 base paths to search.
        
        Args:
            s3_bases: List of S3 base paths. Defaults to DEFAULT_S3_BASES.
        """
        self.s3_bases = s3_bases or DEFAULT_S3_BASES
        self.chip_cache = {}
        self.experiment_cache = {}
    
    def get_s3_client(self, bucket_name):
        """Get appropriate S3 client based on bucket name."""
        return get_s3_client(bucket_name)
    
    def parse_s3_path(self, s3_path):
        """Parse S3 path into bucket and key."""
        parsed = urlparse(s3_path)
        return parsed.netloc, parsed.path.lstrip('/')
    
    def list_chips(self, base_path):
        """
        List all chips (UUIDs) under a base path.
        
        Args:
            base_path: S3 base path to search
            
        Returns:
            list: List of chip directory names
        """
        if base_path in self.chip_cache:
            return self.chip_cache[base_path]
        
        if base_path.startswith('s3://'):
            if wr is None:
                print(f"Cannot list S3 path without braingeneers utilities: {base_path}")
                return []
            try:
                chips = wr.list_directories(base_path)
                chips = [chip.rstrip('/').split('/')[-1] for chip in chips]
                # Filter to chip-like directories (5+ digits)
                chips = [chip for chip in chips 
                        if re.match(r'^\d{5}[a-z]{1,2}$', chip) or count_digits(chip) >= 5]
                self.chip_cache[base_path] = chips
                return chips
            except Exception as e:
                print(f"Error listing chips for {base_path}: {e}")
                return []
        else:
            # Local path
            try:
                chips = [d for d in os.listdir(base_path) 
                        if os.path.isdir(os.path.join(base_path, d)) and 
                        (re.match(r'^\d{5}[a-z]{1,2}$', d) or count_digits(d) >= 5)]
                self.chip_cache[base_path] = chips
                return chips
            except Exception as e:
                print(f"Error listing chips for {base_path}: {e}")
                return []
    
    def list_experiments(self, base_path, chip):
        """
        List all experiments for a specific chip.
        
        Uses a blacklist approach to exclude known non-experiment directories.
        
        Args:
            base_path: S3 base path
            chip: Chip identifier
            
        Returns:
            list: List of experiment directory names
        """
        chip_path = f"{base_path}{chip}/"
        
        if chip_path in self.experiment_cache:
            return self.experiment_cache[chip_path]
        
        def is_experiment_dir(dirname):
            """Check if directory should be considered an experiment."""
            dirname_lower = dirname.lower()
            
            for pattern in EXPERIMENT_BLACKLIST_PATTERNS:
                if pattern in dirname_lower:
                    return False
            
            if dirname.startswith('.'):
                return False
            
            return True
        
        if chip_path.startswith('s3://'):
            if wr is None:
                return []
            try:
                exp_dirs = wr.list_directories(chip_path)
                exp_dirs = [d.rstrip('/').split('/')[-1] for d in exp_dirs]
                exp_dirs = [d for d in exp_dirs if is_experiment_dir(d)]
                self.experiment_cache[chip_path] = exp_dirs
                return exp_dirs
            except Exception as e:
                print(f"Error listing experiments for {chip_path}: {e}")
                return []
        else:
            try:
                exp_dirs = [d for d in os.listdir(chip_path)
                           if os.path.isdir(os.path.join(chip_path, d)) and
                           is_experiment_dir(d)]
                self.experiment_cache[chip_path] = exp_dirs
                return exp_dirs
            except Exception as e:
                print(f"Error listing experiments for {chip_path}: {e}")
                return []
    
    def list_sub_experiments(self, exp_path):
        """
        List all sub-experiments in an experiment directory.
        
        For example, finding all freqs_cont_N in a freqs directory.
        
        Args:
            exp_path: Full path to experiment directory
            
        Returns:
            list: List of sub-experiment identifiers (sorted by number)
        """
        raw_files = self.get_raw_data_files(exp_path)
        
        sub_exps = set()
        for raw_file in raw_files:
            filename = os.path.basename(raw_file)
            if filename.endswith('.raw.h5'):
                sub_exp = filename[:-7]  # Remove '.raw.h5'
                sub_exps.add(sub_exp)
        
        return sorted(list(sub_exps), key=extract_number)
    
    def get_experiment_path(self, base_path, chip, exp):
        """Get the full path to an experiment directory."""
        return f"{base_path}{chip}/{exp}/"
    
    def list_log_files(self, exp_path, log_suffixes=None):
        """
        List all log files in an experiment directory.
        
        Args:
            exp_path: Experiment path
            log_suffixes: List of log file suffixes to filter by
            
        Returns:
            list: List of log file paths
        """
        if log_suffixes is None:
            log_suffixes = LOG_SUFFIXES
        
        all_logs = []
        
        if exp_path.startswith('s3://'):
            if wr is None:
                return []
            for suffix in log_suffixes:
                try:
                    logs = wr.list_objects(exp_path, suffix=suffix)
                    all_logs.extend(logs)
                except Exception as e:
                    pass  # Silently skip missing log types
        else:
            try:
                files = os.listdir(exp_path)
                for suffix in log_suffixes:
                    logs = [os.path.join(exp_path, f) for f in files if f.endswith(suffix)]
                    all_logs.extend(logs)
            except Exception as e:
                print(f"Error listing logs for {exp_path}: {e}")
        
        return sorted(all_logs, key=extract_number)
    
    def get_raw_data_files(self, exp_path):
        """
        List all raw data files (.raw.h5) in an experiment directory.
        
        Args:
            exp_path: Experiment path
            
        Returns:
            list: List of raw data file paths
        """
        if exp_path.startswith('s3://'):
            if wr is None:
                return []
            try:
                raw_files = wr.list_objects(exp_path, suffix='.raw.h5')
                return sorted(raw_files, key=extract_number)
            except Exception as e:
                print(f"Error listing raw data files for {exp_path}: {e}")
                return []
        else:
            try:
                files = os.listdir(exp_path)
                raw_files = [os.path.join(exp_path, f) for f in files if f.endswith('.raw.h5')]
                return sorted(raw_files, key=extract_number)
            except Exception as e:
                print(f"Error listing raw data files for {exp_path}: {e}")
                return []
    
    def get_log_file_for_sub_experiment(self, exp_path, sub_exp, log_suffixes=None):
        """
        Get the log file for a specific sub-experiment.
        
        Args:
            exp_path: Experiment path
            sub_exp: Sub-experiment identifier
            log_suffixes: List of log file suffixes to check
            
        Returns:
            str or None: Path to the log file or None if not found
        """
        if log_suffixes is None:
            log_suffixes = LOG_SUFFIXES
        
        for suffix in log_suffixes:
            log_path = f"{exp_path}{sub_exp}{suffix}"
            
            if log_path.startswith('s3://'):
                bucket, key = self.parse_s3_path(log_path)
                s3 = self.get_s3_client(bucket)
                try:
                    s3.head_object(Bucket=bucket, Key=key)
                    return log_path
                except:
                    continue
            else:
                if os.path.exists(log_path):
                    return log_path
        
        return None


class CatalogGenerator:
    """
    Generates a catalog of Braindance experiments from S3 storage.
    
    Scans configured S3 paths, discovers experiments, extracts metadata,
    and produces a DataFrame catalog compatible with RecordingCatalog.
    
    Features:
        - Incremental updates (skip existing entries)
        - Sub-experiment expansion
        - Chip-level unit count (one S3 read per chip)
        - Progress reporting
    
    Usage:
        generator = CatalogGenerator()
        catalog = generator.generate(verbose=True)
        catalog.to_csv('catalog.csv', index=False)
    """
    
    def __init__(self, s3_bases=None, existing_catalog_path=None):
        """
        Initialize the catalog generator.
        
        Args:
            s3_bases: List of S3 base paths to search. Defaults to DEFAULT_S3_BASES.
            existing_catalog_path: Path to existing catalog for incremental updates.
        """
        self.s3_bases = s3_bases or DEFAULT_S3_BASES
        self.path_manager = DataPathManager(self.s3_bases)
        self.existing_catalog = None
        
        if existing_catalog_path and os.path.exists(existing_catalog_path):
            try:
                self.existing_catalog = pd.read_csv(existing_catalog_path)
                print(f"Loaded existing catalog with {len(self.existing_catalog)} entries")
            except Exception as e:
                print(f"Error loading existing catalog: {e}")
    
    def is_in_catalog(self, base_path, chip, experiment):
        """Check if an experiment is already in the catalog."""
        if self.existing_catalog is None:
            return False
        
        matches = self.existing_catalog[
            (self.existing_catalog['base_path'] == base_path) & 
            (self.existing_catalog['chip'] == chip) & 
            (self.existing_catalog['experiment'] == experiment)
        ]
        
        return len(matches) > 0
    
    def generate(self, chip_filter=None, verbose=True, expand_sub_experiments=True,
                 include_units=True):
        """
        Generate a comprehensive catalog of all experiments.
        
        Args:
            chip_filter: Optional list of chip IDs to filter by
            verbose: Whether to print progress messages
            expand_sub_experiments: Whether to expand sub-experiments (e.g., freqs_cont_N)
            include_units: Whether to include num_units column (requires S3 reads)
            
        Returns:
            pd.DataFrame: Catalog of experiments with columns:
                - base_path, proj, exp, chip, experiment, full_path
                - n_stims, freq, type, n_raw_files, success
                - num_units (if include_units=True)
        """
        data = []
        
        # Start with existing catalog if available
        if self.existing_catalog is not None:
            data = self.existing_catalog.to_dict('records')
        
        # Track unit counts per chip (one read per chip)
        chip_unit_counts = {}
        
        for base_path in self.s3_bases:
            if verbose:
                print(f"\nProcessing base path: {base_path}")
            
            chips = self.path_manager.list_chips(base_path)
            
            if chip_filter:
                chips = [c for c in chips if c in chip_filter]
            
            for chip in chips:
                if verbose:
                    print(f"  Processing chip: {chip}")
                
                # Get unit count for this chip (once per chip)
                if include_units:
                    chip_key = (base_path, chip)
                    if chip_key not in chip_unit_counts:
                        num_units = get_num_units_for_chip(base_path, chip)
                        chip_unit_counts[chip_key] = num_units
                        if verbose and num_units is not None:
                            print(f"    Found {num_units} units for chip")
                
                experiments = self.path_manager.list_experiments(base_path, chip)
                
                for exp in experiments:
                    if verbose:
                        print(f"    Processing experiment: {exp}")
                    
                    try:
                        if expand_sub_experiments:
                            exp_path = self.path_manager.get_experiment_path(base_path, chip, exp)
                            sub_exps = self.path_manager.list_sub_experiments(exp_path)
                            
                            if sub_exps:
                                if verbose:
                                    print(f"      Found {len(sub_exps)} sub-experiments")
                                
                                for sub_exp in sub_exps:
                                    full_exp = f"{exp}/{sub_exp}"
                                    
                                    if self.is_in_catalog(base_path, chip, full_exp):
                                        if verbose:
                                            print(f"        Skipping (already in catalog): {sub_exp}")
                                        continue
                                    
                                    try:
                                        sub_exp_obj = BraindanceExperiment(
                                            base_path, chip, exp, self.path_manager, sub_exp
                                        )
                                        exp_info = sub_exp_obj.get_experiment_info()
                                        
                                        # Add unit count
                                        if include_units:
                                            exp_info['num_units'] = chip_unit_counts.get((base_path, chip))
                                        
                                        data.append(exp_info)
                                        
                                        if verbose:
                                            print(f"        {sub_exp}: type={exp_info['type']}, "
                                                  f"stims={exp_info['n_stims']}, freq={exp_info['freq']}")
                                    except Exception as e:
                                        if verbose:
                                            print(f"        Error processing {sub_exp}: {e}")
                                        data.append(self._create_error_entry(
                                            base_path, chip, full_exp, include_units, chip_unit_counts
                                        ))
                            else:
                                # No sub-experiments, add main experiment
                                if self.is_in_catalog(base_path, chip, exp):
                                    if verbose:
                                        print(f"      Skipping (already in catalog): {exp}")
                                    continue
                                
                                braindance_exp = BraindanceExperiment(
                                    base_path, chip, exp, self.path_manager
                                )
                                exp_info = braindance_exp.get_experiment_info()
                                
                                if include_units:
                                    exp_info['num_units'] = chip_unit_counts.get((base_path, chip))
                                
                                data.append(exp_info)
                                
                                if verbose:
                                    print(f"      type={exp_info['type']}, stims={exp_info['n_stims']}, "
                                          f"freq={exp_info['freq']}")
                        else:
                            # Don't expand sub-experiments
                            if self.is_in_catalog(base_path, chip, exp):
                                continue
                            
                            braindance_exp = BraindanceExperiment(
                                base_path, chip, exp, self.path_manager
                            )
                            exp_info = braindance_exp.get_experiment_info()
                            
                            if include_units:
                                exp_info['num_units'] = chip_unit_counts.get((base_path, chip))
                            
                            data.append(exp_info)
                            
                    except Exception as e:
                        if verbose:
                            print(f"      Error processing experiment: {e}")
                        
                        if not self.is_in_catalog(base_path, chip, exp):
                            data.append(self._create_error_entry(
                                base_path, chip, exp, include_units, chip_unit_counts
                            ))
        
        # Create DataFrame and reorder columns
        catalog = pd.DataFrame(data)
        
        if len(catalog) > 0:
            catalog = self._reorder_columns(catalog)
        
        return catalog
    
    def _create_error_entry(self, base_path, chip, experiment, include_units, chip_unit_counts):
        """Create a catalog entry for a failed experiment."""
        entry = {
            'base_path': base_path,
            'chip': chip,
            'experiment': experiment,
            'full_path': f"{base_path}{chip}/{experiment}/",
            'n_stims': 0,
            'freq': None,
            'type': 'error',
            'n_raw_files': 0,
            'success': False
        }
        if include_units:
            entry['num_units'] = chip_unit_counts.get((base_path, chip))
        return entry
    
    def _reorder_columns(self, catalog):
        """Reorder columns to have proj, exp after base_path."""
        # Add proj and exp columns if not present
        from .validators import extract_proj_from_s3_path, extract_exp_from_experiment_path
        
        if 'proj' not in catalog.columns:
            catalog['proj'] = catalog['base_path'].apply(extract_proj_from_s3_path)
        if 'exp' not in catalog.columns:
            catalog['exp'] = catalog['experiment'].apply(extract_exp_from_experiment_path)
        
        # Define preferred column order
        preferred_order = [
            'base_path', 'proj', 'exp', 'chip', 'experiment', 'full_path',
            'n_stims', 'freq', 'calculated_freq', 'type', 'n_raw_files', 
            'num_units', 'success', 'log_available'
        ]
        
        # Get columns that exist in the DataFrame
        existing_cols = [col for col in preferred_order if col in catalog.columns]
        # Add any remaining columns not in preferred order
        remaining_cols = [col for col in catalog.columns if col not in preferred_order]
        
        return catalog[existing_cols + remaining_cols]
    
    # Alias for backward compatibility
    generate_catalog = generate
