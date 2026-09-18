#!/usr/bin/env python
"""
Calculate Data Statistics
=========================

Determines the total amount of:
  - Raw data (GB) - from S3 .raw.h5 file sizes
  - Processed data (GB) - from S3 .pkl and .pickle file sizes
  - Recording duration (Hours) - from SpikeData .length
  - Unit counts - from SpikeData .N or catalog num_units

Uses parallel processing and leverages BrainDance's S3 caching.
"""

import os
import sys
import argparse
import pandas as pd
import boto3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import urlparse

# Add breadance root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from braindance.utils.data_manager import load_recording
from braindance.utils.data_manager.utils.catalogging.s3_helpers import (
    get_s3_client,
    parse_s3_path,
)
from braindance.config import get_catalog_path


def get_s3_size(s3_path, client_cache):
    """Get size of an S3 object in bytes."""
    if not s3_path or not str(s3_path).startswith("s3://"):
        return 0

    try:
        bucket, key = parse_s3_path(s3_path)
        if bucket not in client_cache:
            client_cache[bucket] = get_s3_client(bucket)

        response = client_cache[bucket].head_object(Bucket=bucket, Key=key)
        return response["ContentLength"]
    except Exception:
        return 0


def process_recording_stats(index, row, client_cache):
    """Extract size, duration, and units for a single recording."""
    stats = {
        "index": index,
        "proj": row.get("proj", "unknown"),
        "raw_size": 0,
        "proc_size": 0,
        "duration_hrs": 0,
        "units": 0,
    }

    # 1. Raw Data Size
    raw_path = row.get("full_path")
    stats["raw_size"] = get_s3_size(raw_path, client_cache)

    # 2. Processed Data Size & Duration/Units
    try:
        # Load recording (this handles S3 path resolution and caching)
        # Note: we use internal logic to find the spike path safely
        rec = load_recording(
            row["proj"], row["chip"], row["experiment"], base_path=row["base_path"]
        )

        # Get spike data path on S3 for size retrieval
        # rec._resolve_paths() is called inside _spikes_path
        # but _spikes_path returns local path. We want the S3 path.
        s3_spike_path = rec._construct_s3_spike_path()
        stats["proc_size"] = get_s3_size(s3_spike_path, client_cache)

        # Trigger spike loading to get duration and N
        # This will download/cache the pickle if not already present
        if rec.spikes is not None:
            stats["duration_hrs"] = rec.spikes.length / (1000 * 60 * 60)  # ms to hours
            stats["units"] = rec.spikes.N
        else:
            # Fallback to catalog num_units if spikes couldn't be loaded
            stats["units"] = (
                row.get("num_units", 0) if pd.notna(row.get("num_units")) else 0
            )

    except Exception as e:
        # print(f"Error processing {row.get('experiment')}: {e}")
        stats["units"] = (
            row.get("num_units", 0) if pd.notna(row.get("num_units")) else 0
        )

    return stats


def main():
    parser = argparse.ArgumentParser(description="Calculate BrainDance data statistics")
    parser.add_argument("--catalog", type=str, help="Path to catalog CSV")
    parser.add_argument("--limit", type=int, help="Limit number of rows for testing")
    parser.add_argument(
        "--workers", type=int, default=10, help="Number of parallel workers"
    )
    args = parser.parse_args()

    # Load catalog
    cat_path = Path(args.catalog) if args.catalog else get_catalog_path()
    if not cat_path.exists():
        print(f"Error: Catalog not found at {cat_path}")
        return

    print(f"Loading catalog: {cat_path}")
    df = pd.read_csv(cat_path)

    # Filter for successful recordings
    if "success" in df.columns:
        df = df[df["success"] == True]

    if args.limit:
        df = df.head(args.limit)

    print(f"Processing {len(df)} recordings using {args.workers} workers...")

    client_cache = {}
    results = []

    start_time = datetime.now()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_recording_stats, i, row, client_cache): i
            for i, row in df.iterrows()
        }

        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 50 == 0:
                print(
                    f" Progress: {completed}/{len(df)} recordings processed...",
                    end="\r",
                )

    print(f"\nFinished in {datetime.now() - start_time}")

    # Aggregate by project
    res_df = pd.DataFrame(results)
    summary = res_df.groupby("proj").agg(
        {"raw_size": "sum", "proc_size": "sum", "duration_hrs": "sum", "units": "sum"}
    )

    # Convert sizes to GB
    summary["raw_gb"] = summary["raw_size"] / (1024**3)
    summary["proc_gb"] = summary["proc_size"] / (1024**3)

    # Total
    totals = summary.sum()

    print("\n" + "=" * 80)
    print(
        f"{'PROJECT':<30} | {'RAW (GB)':>10} | {'PROC (GB)':>10} | {'HOURS':>10} | {'UNITS':>8}"
    )
    print("-" * 80)

    for proj, row in summary.sort_values("raw_gb", ascending=False).iterrows():
        print(
            f"{str(proj)[:30]:<30} | {row['raw_gb']:>10.2f} | {row['proc_gb']:>10.2f} | {row['duration_hrs']:>10.2f} | {int(row['units']):>8}"
        )

    print("-" * 80)
    print(
        f"{'TOTAL':<30} | {totals['raw_gb']:>10.2f} | {totals['proc_gb']:>10.2f} | {totals['duration_hrs']:>10.2f} | {int(totals['units']):>8}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
