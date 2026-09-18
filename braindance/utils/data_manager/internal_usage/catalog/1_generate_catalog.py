#!/usr/bin/env python
"""
Step 1: Generate Catalog
========================

Scans S3 paths defined in config.py and generates a catalog of experiments.
Uses the optimized catalogging utilities for S3 operations and experiment parsing.

Features:
  - Incremental updates (skips paths already in catalog)
  - Parallel processing with ThreadPoolExecutor
  - Caching of S3 directory listings

Usage:
    python 1_generate_catalog.py                    # Incremental (skip existing paths)
    python 1_generate_catalog.py --force            # Full regeneration of everything
    python 1_generate_catalog.py --refresh s3://... # Refresh specific path(s)
    python 1_generate_catalog.py --workers 8        # Control parallelism

Output:
    braindance_catalog.csv  - Raw catalog (input for step 2)
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import config
from config import S3_BASES, RAW_CATALOG

# Import optimized utilities from catalogging
from utils.catalogging.generator import DataPathManager
from utils.catalogging.experiment import BraindanceExperiment


def process_experiment(args):
    """
    Process a single experiment or sub-experiment.
    
    Args:
        args: Tuple of (base_path, chip, exp, path_manager, sub_exp)
        
    Returns:
        dict: Experiment info or error dict
    """
    base_path, chip, exp, path_manager, sub_exp = args
    exp_path = path_manager.get_experiment_path(base_path, chip, exp)
    
    try:
        experiment = BraindanceExperiment(
            base_path, chip, exp, path_manager, sub_exp
        )
        return experiment.get_experiment_info()
    except Exception as e:
        full_exp = f"{exp}/{sub_exp}" if sub_exp else exp
        return {
            'base_path': base_path,
            'chip': chip,
            'experiment': full_exp,
            'full_path': f"{exp_path}{sub_exp}" if sub_exp else exp_path,
            'n_stims': 0,
            'freq': None,
            'type': 'error',
            'n_raw_files': 0,
            'success': False,
            'error': str(e),
        }


def generate_catalog(s3_bases, existing_df=None, refresh_paths=None, verbose=True, max_workers=8):
    """
    Generate catalog by scanning S3 paths with parallel processing.
    
    Args:
        s3_bases: List of S3 base paths to scan
        existing_df: Optional existing catalog to update incrementally
        refresh_paths: List of paths to refresh (re-scan even if in catalog)
        verbose: Print progress
        max_workers: Number of parallel workers
        
    Returns:
        pd.DataFrame: Generated catalog
    """
    path_manager = DataPathManager(s3_bases)
    refresh_paths = set(refresh_paths or [])
    
    # Get existing entries and paths that are already cataloged
    existing_data = []
    existing_base_paths = set()
    
    if existing_df is not None:
        existing_data = existing_df.to_dict('records')
        existing_base_paths = set(existing_df['base_path'].unique())
        if verbose:
            print(f"Loaded {len(existing_data)} existing entries from {len(existing_base_paths)} paths")
    
    # Determine which paths to scan vs skip
    paths_to_scan = []
    paths_to_keep = []  # Paths we'll keep from existing catalog
    
    for base_path in s3_bases:
        if base_path in refresh_paths:
            # User explicitly wants to refresh this path
            paths_to_scan.append(base_path)
            if verbose:
                print(f"  REFRESH: {base_path}")
        elif base_path in existing_base_paths:
            # Already in catalog, skip scanning
            paths_to_keep.append(base_path)
            if verbose:
                print(f"  SKIP (in catalog): {base_path}")
        else:
            # New path, needs scanning
            paths_to_scan.append(base_path)
            if verbose:
                print(f"  SCAN (new): {base_path}")
    
    # Start with data from paths we're keeping (not refreshing)
    data = [row for row in existing_data if row['base_path'] in paths_to_keep]
    if verbose:
        print(f"\nKeeping {len(data)} entries from {len(paths_to_keep)} existing paths")
        print(f"Scanning {len(paths_to_scan)} paths...")
    
    if not paths_to_scan:
        print("Nothing new to scan!")
        return pd.DataFrame(data)
    
    # Collect all experiments to process
    experiments_to_process = []
    
    for base_path in paths_to_scan:
        print(f"\n{'='*60}")
        print(f"Scanning: {base_path}")
        print('='*60)
        
        chips = path_manager.list_chips(base_path)
        print(f"Found {len(chips)} chips")
        
        for chip in chips:
            experiments = path_manager.list_experiments(base_path, chip)
            
            for exp in experiments:
                exp_path = path_manager.get_experiment_path(base_path, chip, exp)
                sub_exps = path_manager.list_sub_experiments(exp_path)
                
                if sub_exps:
                    for sub_exp in sub_exps:
                        experiments_to_process.append((base_path, chip, exp, path_manager, sub_exp))
                else:
                    experiments_to_process.append((base_path, chip, exp, path_manager, None))
            
            if verbose:
                print(f"  {chip}: {len(experiments)} experiments")
    
    print(f"\nProcessing {len(experiments_to_process)} experiments with {max_workers} workers...")
    
    if not experiments_to_process:
        return pd.DataFrame(data)
    
    # Process experiments in parallel
    processed = 0
    errors = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_experiment, args): args for args in experiments_to_process}
        
        for future in as_completed(futures):
            result = future.result()
            data.append(result)
            processed += 1
            
            if result.get('type') == 'error':
                errors += 1
            
            if verbose and processed % 50 == 0:
                print(f"  Processed {processed}/{len(experiments_to_process)} "
                      f"({errors} errors)")
    
    print(f"Completed: {processed} experiments processed, {errors} errors")
    
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(
        description='Generate BrainDance catalog from S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python 1_generate_catalog.py                       # Incremental (skip existing paths)
    python 1_generate_catalog.py --force               # Full regeneration  
    python 1_generate_catalog.py --refresh s3://...    # Refresh specific path(s)
    python 1_generate_catalog.py --workers 16          # More parallelism
    python 1_generate_catalog.py --quiet               # Less output
        """
    )
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force full regeneration (rescan all paths)')
    parser.add_argument('--refresh', type=str, nargs='+',
                       help='Refresh specific S3 path(s) even if already in catalog')
    parser.add_argument('--workers', '-w', type=int, default=8,
                       help='Number of parallel workers (default: 8)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    print("\n" + "="*60)
    print("BRAINDANCE CATALOG GENERATOR")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Workers: {args.workers}")
    
    # Determine paths to scan
    paths_to_scan = S3_BASES
    print(f"Configured paths: {len(paths_to_scan)}")
    
    # Load existing catalog for incremental updates
    existing_df = None
    refresh_paths = None
    
    if args.force:
        print("Force mode: rescanning all paths")
        refresh_paths = paths_to_scan  # Refresh everything
    elif args.refresh:
        print(f"Refresh mode: rescanning {len(args.refresh)} specific paths")
        refresh_paths = args.refresh
    
    if RAW_CATALOG.exists() and not args.force:
        existing_df = pd.read_csv(RAW_CATALOG)
        print(f"Existing catalog: {len(existing_df)} entries")
    elif args.force:
        print("Starting fresh (force mode)")
    else:
        print("No existing catalog found - starting fresh")
    
    # Generate catalog
    catalog = generate_catalog(
        paths_to_scan, 
        existing_df,
        refresh_paths=refresh_paths,
        verbose=verbose,
        max_workers=args.workers
    )
    
    # Save
    catalog.to_csv(RAW_CATALOG, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total entries: {len(catalog)}")
    print(f"Unique chips: {catalog['chip'].nunique()}")
    print(f"Unique base paths: {catalog['base_path'].nunique()}")
    
    if 'type' in catalog.columns:
        print(f"\nExperiment types:")
        for exp_type, count in catalog['type'].value_counts().items():
            print(f"  {exp_type}: {count}")
    
    if 'success' in catalog.columns:
        success_rate = catalog['success'].mean() * 100
        print(f"\nSuccess rate: {success_rate:.1f}%")
    
    print(f"\nSaved to: {RAW_CATALOG}")
    print("\n→ Next step: python 2_amend_catalog.py")


if __name__ == '__main__':
    main()
