#!/usr/bin/env python
"""
Step 2: Amend Catalog
=====================

Applies post-processing to the raw catalog:
  - Merges organoid metadata (chip info, ages)
  - Removes known corrupted files
  - Applies manual fixes
  - Identifies baselines and drugs
  - Adds spike sorting unit counts (incremental)
  - Cleans baseline experiments
  - Filters out problematic recordings without spike sorting

Uses the postprocessing and metadata utilities from catalogging.

Usage:
    python 2_amend_catalog.py           # Process the raw catalog
    python 2_amend_catalog.py --quiet   # Less output

Output:
    all_catalog.csv           - Final catalog with only clean recordings
    problematic_recordings.csv - Recordings without spike sorting (num_units=NaN)
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import config
from config import RAW_CATALOG, FINAL_CATALOG, METADATA_CSV

# Import utilities from catalogging
from utils.catalogging.postprocessing import apply_catalog_fixes
from utils.catalogging.metadata import (
    merge_metadata,
    load_org_metadata,
    add_drug_column,
)
from utils.catalogging import add_units, validate_recording_files


def main():
    parser = argparse.ArgumentParser(
        description="Apply post-processing to BrainDance catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script:
  1. Loads the raw catalog from step 1
  2. Merges organoid metadata (ages, chip info)
  3. Removes corrupted/problematic experiments
  4. Adds derived columns (baseline, drug, etc.)
  5. Adds spike sorting unit counts (incremental)
  6. Saves the final catalog

Output is saved to both:
  - internal_usage/catalog/all_catalog.csv
  - utils/catalogging/all_catalog.csv (for data_manager access)
        """,
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    args = parser.parse_args()

    verbose = not args.quiet

    print("\n" + "=" * 60)
    print("BRAINDANCE CATALOG AMENDMENT")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # -------------------------------------------------------------------------
    # Step 1: Load raw catalog
    # -------------------------------------------------------------------------
    print("\n1. Loading raw catalog...")

    if not RAW_CATALOG.exists():
        print(f"ERROR: Raw catalog not found: {RAW_CATALOG}")
        print("Please run 1_generate_catalog.py first")
        sys.exit(1)

    df = pd.read_csv(RAW_CATALOG)
    print(f"   Loaded {len(df)} entries")

    # -------------------------------------------------------------------------
    # Step 2: Load and merge metadata
    # -------------------------------------------------------------------------
    print("\n2. Merging organoid metadata...")

    if METADATA_CSV.exists():
        meta_df = load_org_metadata(METADATA_CSV)
        df = merge_metadata(df, meta_df)
        print(
            f"   Merged metadata for {df['plated'].notna().sum() if 'plated' in df.columns else 0} entries"
        )
    else:
        print(f"   WARNING: Metadata file not found: {METADATA_CSV}")

    # -------------------------------------------------------------------------
    # Step 3: Add drug column
    # -------------------------------------------------------------------------
    print("\n3. Adding drug column...")
    df = add_drug_column(df)
    drug_count = df["drug"].notna().sum()
    print(f"   Found {drug_count} drug experiments")

    # -------------------------------------------------------------------------
    # Step 4: Apply all catalog fixes
    # -------------------------------------------------------------------------
    print("\n4. Applying catalog fixes...")
    df = apply_catalog_fixes(df, verbose=verbose)

    # -------------------------------------------------------------------------
    # Step 4.5: Fill missing proj values from base_path
    # -------------------------------------------------------------------------
    print("\n4.5. Filling missing proj values...")

    # Count how many are missing before
    missing_before = df["proj"].isna().sum()

    if missing_before > 0:
        # For rows with missing proj, extract from base_path
        mask = df["proj"].isna()

        # Extract last part of base_path (removing trailing slash)
        df.loc[mask, "proj"] = df.loc[mask, "base_path"].apply(
            lambda path: path.rstrip("/").split("/")[-1] if pd.notna(path) else None
        )

        # Count how many were filled
        missing_after = df["proj"].isna().sum()
        filled = missing_before - missing_after

        print(f"   Filled {filled} missing proj values from base_path")

        if filled > 0:
            # Show examples of what was filled
            filled_mask = mask & df["proj"].notna()
            examples = df[filled_mask][
                ["base_path", "proj", "chip", "experiment"]
            ].head(3)
            print(f"   Examples:")
            for _, row in examples.iterrows():
                print(
                    f"     {row['chip']}/{row['proj']}/{row['experiment'].split('/')[0]}"
                )
    else:
        print("   No missing proj values found")

    # -------------------------------------------------------------------------
    # Step 4.6: Manual fixes for specific problematic entries
    # -------------------------------------------------------------------------
    print("\n4.6. Applying manual fixes for specific entries...")

    # Fix entries with full S3 paths that should have proper proj names
    # These were causing double slashes in organoid mapping (e.g., "path//chip")
    manual_fixes = [
        {
            "base_path": "s3://braingeneersdev/asrobbin/braindance_data/24-05-13-causal-mousepaper/",
            "correct_proj": "24-05-13-causal-mousepaper",
        },
        {
            "base_path": "s3://braingeneersdev/asrobbin/braindance_data/24-05-24/",
            "correct_proj": "24-05-24",
        },
    ]

    fixes_applied = 0
    for fix in manual_fixes:
        mask = df["base_path"] == fix["base_path"]
        if mask.any():
            df.loc[mask, "proj"] = fix["correct_proj"]
            count = mask.sum()
            fixes_applied += count
            print(
                f"   Fixed {count} entries: {fix['base_path']} -> proj='{fix['correct_proj']}'"
            )

    if fixes_applied == 0:
        print("   No manual fixes needed")
    else:
        print(f"   Total manual fixes applied: {fixes_applied}")

    # -------------------------------------------------------------------------
    # Step 4.7: Assign unique organoid IDs
    # -------------------------------------------------------------------------
    print("\n4.7. Assigning unique organoid IDs...")

    # Import the utility function
    from catalog_utils import assign_organoid_ids

    try:
        df = assign_organoid_ids(df)
        print(f"   ✓ Successfully assigned organoid IDs")
    except (KeyError, ValueError) as e:
        print(f"   ✗ ERROR: {e}")
        print(f"   Catalog generation failed. Please fix the issue and try again.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 5: Add spike sorting unit counts (incremental)
    # -------------------------------------------------------------------------
    print("\n5. Adding spike sorting unit counts...")

    # If final catalog exists, load existing num_units to avoid re-fetching
    if FINAL_CATALOG.exists():
        print(f"   Loading existing num_units from {FINAL_CATALOG.name}...")
        existing_catalog = pd.read_csv(FINAL_CATALOG)
        if "num_units" in existing_catalog.columns:
            # Create a merge key
            existing_catalog["_merge_key"] = (
                existing_catalog["base_path"]
                + "/"
                + existing_catalog["chip"]
                + "/"
                + existing_catalog["experiment"]
            )
            df["_merge_key"] = (
                df["base_path"] + "/" + df["chip"] + "/" + df["experiment"]
            )

            # Merge existing num_units
            num_units_map = existing_catalog.set_index("_merge_key")[
                "num_units"
            ].to_dict()
            df["num_units"] = df["_merge_key"].map(num_units_map)
            df.drop(columns=["_merge_key"], inplace=True)

            existing_count = df["num_units"].notna().sum()
            print(f"   Loaded {existing_count} existing unit counts")

    # Only fetch units for entries without num_units
    df = add_units(df, verbose=verbose)
    units_count = df["num_units"].notna().sum() if "num_units" in df.columns else 0
    print(f"   Total entries with unit counts: {units_count}")

    # -------------------------------------------------------------------------
    # Step 5.5: Validate per-recording spike files exist on S3
    # -------------------------------------------------------------------------
    print("\n5.5. Validating per-recording spike_data files on S3...")
    print("     (add_units assigns folder-level counts; this checks each recording's file)")

    df = validate_recording_files(df, verbose=verbose)

    units_after_validation = df["num_units"].notna().sum() if "num_units" in df.columns else 0
    invalidated = units_count - units_after_validation
    if invalidated > 0:
        print(f"   Invalidated {invalidated} recordings with missing spike_data files")
    print(f"   Recordings with valid unit counts: {units_after_validation}")

    # -------------------------------------------------------------------------
    # Step 6: Reorder columns for readability
    # -------------------------------------------------------------------------
    preferred_order = [
        "base_path",
        "chip",
        "experiment",
        "full_path",
        "n_stims",
        "freq",
        "calculated_freq",
        "log_available",
        "type",
        "n_raw_files",
        "success",
        "num_units",
        "proj",
        "exp",
        "baseline",
        "inherited_baseline_from",
        "drug",
        "plated",
        "Type",
        "line",
        "Org_Day_0",
        "Org_Age",
        "Date_adhered",
        "Days_on_chip",
    ]

    existing_cols = [c for c in preferred_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in preferred_order]
    df = df[existing_cols + remaining_cols]

    # -------------------------------------------------------------------------
    # Step 6: Filter and save outputs
    # -------------------------------------------------------------------------
    print("\n6. Filtering and saving catalogs...")

    # Ensure parent directory exists
    FINAL_CATALOG.parent.mkdir(parents=True, exist_ok=True)

    # Split into problematic and clean recordings
    if "num_units" in df.columns:
        # Problematic recordings: missing num_units
        problematic_mask = df["num_units"].isnull()
        problematic_df = df[problematic_mask].copy()
        clean_df = df[~problematic_mask].copy()

        # Save problematic recordings
        problematic_path = FINAL_CATALOG.parent / "problematic_recordings.csv"
        problematic_df.to_csv(problematic_path, index=False)
        print(f"   Saved {len(problematic_df)} problematic recordings: {problematic_path}")

        # Save clean catalog
        clean_df.to_csv(FINAL_CATALOG, index=False)
        print(f"   Saved {len(clean_df)} clean recordings: {FINAL_CATALOG}")
    else:
        # No num_units column - save everything to final catalog
        df.to_csv(FINAL_CATALOG, index=False)
        print(f"   Saved: {FINAL_CATALOG} (no num_units column found)")

    # -------------------------------------------------------------------------
    # Step 7: Validate baseline coverage
    # -------------------------------------------------------------------------
    print("\n7. Validating baseline coverage...")

    if "baseline" in df.columns and "proj" in df.columns and "exp" in df.columns:
        # Group by (proj, chip, exp) and check if each group has at least one baseline
        # (either direct or inherited)
        grouped = df.groupby(["proj", "chip", "exp"])
        missing_baselines = []

        for (proj, chip, exp), group in grouped:
            has_direct_baseline = group["baseline"].any()
            has_inherited = (
                group["inherited_baseline_from"].notna().any()
                if "inherited_baseline_from" in df.columns
                else False
            )

            if not (has_direct_baseline or has_inherited):
                # TRULY missing - no baseline found in this or earlier experiments
                num_recordings = len(group)
                missing_baselines.append(
                    {
                        "proj": proj,
                        "chip": chip,
                        "exp": exp,
                        "num_recordings": num_recordings,
                        "experiments": group["experiment"].tolist(),
                    }
                )

        if missing_baselines:
            print(
                f"\n   ⚠️  WARNING: Found {len(missing_baselines)} experiment series WITHOUT baselines:"
            )
            print("   " + "-" * 58)
            for item in missing_baselines:  # Show all
                print(
                    f"   • {item['proj']}/{item['chip']}/{item['exp']} ({item['num_recordings']} recordings)"
                )
                if verbose:
                    for exp in item["experiments"]:  # Show all experiments
                        print(f"      - {exp}")
            print("\n   These experiment series may fail Phase 2 rt_sort processing!")
            print(
                "   Run baseline detection test: python braindance/utils/data_manager/internal_tests/test_baseline_detection.py"
            )
        else:
            print(
                f"   ✓ All {grouped.ngroups} experiment series have baselines (direct or inherited)"
            )
    else:
        print("   Skipping validation (missing required columns)")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Use clean_df if we filtered, otherwise use df
    summary_df = clean_df if "num_units" in df.columns and problematic_mask.any() else df

    print(f"Total entries processed: {len(df)}")
    if "num_units" in df.columns:
        print(f"Clean recordings: {len(clean_df)}")
        print(f"Problematic recordings (no units): {len(problematic_df)}")

    print(f"Unique chips: {summary_df['chip'].nunique()}")
    print(f"Unique projects: {summary_df['proj'].nunique() if 'proj' in summary_df.columns else 'N/A'}")

    if "type" in summary_df.columns:
        print(f"\nExperiment types:")
        for exp_type, count in summary_df["type"].value_counts().items():
            print(f"  {exp_type}: {count}")

    if "baseline" in summary_df.columns:
        print(f"\nBaselines: {summary_df['baseline'].sum()}")

    if "drug" in summary_df.columns:
        print(f"Drug experiments: {summary_df['drug'].notna().sum()}")

    if "Org_Age" in summary_df.columns:
        valid_ages = summary_df["Org_Age"].dropna()
        if len(valid_ages) > 0:
            print(
                f"\nOrganoid ages: {valid_ages.min():.0f} - {valid_ages.max():.0f} days"
            )

    print(f"\n✓ Catalog ready for analysis!")
    print(f"  Use: from braindance.utils.data_manager import RecordingCatalog")


if __name__ == "__main__":
    main()
