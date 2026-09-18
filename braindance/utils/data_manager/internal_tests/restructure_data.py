#!/usr/bin/env python3
"""
Restructure BrainDance data directory to match expected format.
Converts flat chip-level files to hierarchical experiment-level directories.

Usage:
    python restructure_data.py --dry-run  # Preview changes
    python restructure_data.py             # Execute changes
"""

import os
import shutil
from pathlib import Path
import argparse

def restructure_chip_directory(chip_dir, dry_run=True):
    """
    Restructure a chip directory by moving files into experiment subdirectories.

    Args:
        chip_dir: Path to chip directory
        dry_run: If True, only print what would be done

    Returns:
        List of (action, source, destination) tuples
    """
    actions = []
    chip_path = Path(chip_dir)

    # Find all spike_data.pkl files
    spike_files = sorted(chip_path.glob("*_spike_data.pkl"))

    for spike_file in spike_files:
        # Extract experiment name by removing _spike_data.pkl suffix
        filename = spike_file.name
        exp_name = filename.replace("_spike_data.pkl", "")

        # Create experiment subdirectory
        exp_dir = chip_path / exp_name

        # Check for matching log file
        log_file = chip_path / f"{exp_name}_log.csv"

        # Create the experiment directory
        actions.append(("mkdir", str(exp_dir), ""))

        # Move spike data file
        spike_dest = exp_dir / filename
        actions.append(("move", str(spike_file), str(spike_dest)))

        # Move log file if it exists
        if log_file.exists():
            log_dest = exp_dir / log_file.name
            actions.append(("move", str(log_file), str(log_dest)))

    # Check for empty spike_data directory
    spike_data_dir = chip_path / "spike_data"
    if spike_data_dir.exists() and spike_data_dir.is_dir():
        if not any(spike_data_dir.iterdir()):
            actions.append(("rmdir", str(spike_data_dir), ""))

    return actions

def print_actions(actions):
    """Print all planned actions in a readable format."""
    print("\n" + "="*80)
    print("PLANNED ACTIONS:")
    print("="*80)

    for action, src, dst in actions:
        if action == "mkdir":
            print(f"  CREATE:  {src}")
        elif action == "move":
            print(f"  MOVE:    {src}")
            print(f"    ->     {dst}")
        elif action == "rmdir":
            print(f"  REMOVE:  {src}")

    print("="*80 + "\n")

def execute_actions(actions, dry_run=True):
    """Execute the planned actions."""
    for action, src, dst in actions:
        try:
            if action == "mkdir":
                if not dry_run:
                    os.makedirs(src, exist_ok=True)
                print(f"✓ Created: {src}")

            elif action == "move":
                if not dry_run:
                    shutil.move(src, dst)
                print(f"✓ Moved: {src}")
                if not dry_run:
                    print(f"    to: {dst}")

            elif action == "rmdir":
                if not dry_run:
                    os.rmdir(src)
                print(f"✓ Removed: {src}")

        except Exception as e:
            print(f"✗ ERROR with {action} {src}: {e}")
            return False

    return True

def main():
    parser = argparse.ArgumentParser(
        description="Restructure BrainDance data directory for data manager compatibility"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview changes without executing (default: show preview)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually execute the moves (required to make changes)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Volumes/hunter_ssd/busy_bee/test_dir/24-105-10_drug_causal",
        help="Path to the test directory"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1

    print(f"Processing directory: {data_dir}")
    print(f"Mode: {'DRY-RUN (no changes will be made)' if not args.execute else 'EXECUTION (changes will be made)'}")

    # Process each chip directory
    all_actions = []
    for chip_dir in sorted(data_dir.iterdir()):
        if chip_dir.is_dir() and not chip_dir.name.startswith("."):
            print(f"\nAnalyzing chip: {chip_dir.name}")
            actions = restructure_chip_directory(chip_dir, dry_run=True)

            if actions:
                print(f"  Found {len(actions)} actions")
                all_actions.extend(actions)
            else:
                print("  No changes needed (already structured correctly)")

    if not all_actions:
        print("\nNo restructuring needed - data already appears to be in correct format!")
        return 0

    # Show what will be done
    print_actions(all_actions)

    if not args.execute:
        print("DRY-RUN MODE: No changes were made.")
        print("To execute these changes, run:")
        print(f"  python {__file__} --execute\n")
        return 0

    # Execute the changes
    print("EXECUTING CHANGES...")
    success = execute_actions(all_actions, dry_run=False)

    if success:
        print("\n✓ Restructuring completed successfully!")
        return 0
    else:
        print("\n✗ Some operations failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
