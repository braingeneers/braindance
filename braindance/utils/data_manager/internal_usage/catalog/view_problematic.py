#!/usr/bin/env python
"""
View Problematic Recordings
============================

Quick utility to inspect recordings that failed spike sorting.

Usage:
    python view_problematic.py              # Show all problematic recordings
    python view_problematic.py --summary    # Show summary stats only
"""

import argparse
from pathlib import Path

import pandas as pd

from config import FINAL_CATALOG


def main():
    parser = argparse.ArgumentParser(description="View problematic recordings")
    parser.add_argument(
        "--summary", "-s", action="store_true", help="Show summary stats only"
    )
    args = parser.parse_args()

    problematic_path = FINAL_CATALOG.parent / "problematic_recordings.csv"

    if not problematic_path.exists():
        print(f"No problematic recordings file found at: {problematic_path}")
        print("Run 2_amend_catalog.py first to generate it.")
        return

    df = pd.read_csv(problematic_path)

    print("=" * 70)
    print("PROBLEMATIC RECORDINGS (Missing Spike Sorting)")
    print("=" * 70)
    print(f"\nTotal: {len(df)} recordings without num_units\n")

    # Summary by project
    if "proj" in df.columns:
        print("By Project:")
        proj_counts = df["proj"].value_counts()
        for proj, count in proj_counts.items():
            print(f"  {proj}: {count}")

    # Summary by chip
    if "chip" in df.columns:
        print(f"\nBy Chip:")
        chip_counts = df["chip"].value_counts()
        for chip, count in chip_counts.items():
            print(f"  {chip}: {count}")

    # Summary by type
    if "type" in df.columns:
        print(f"\nBy Type:")
        type_counts = df["type"].value_counts()
        for exp_type, count in type_counts.items():
            print(f"  {exp_type}: {count}")

    if not args.summary:
        print("\n" + "=" * 70)
        print("DETAILED LIST")
        print("=" * 70)

        # Show all recordings with key info
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_colwidth", 80)

        display_cols = ["chip", "proj", "experiment", "type", "n_stims"]
        available_cols = [c for c in display_cols if c in df.columns]

        print(df[available_cols].to_string(index=False))


if __name__ == "__main__":
    main()
