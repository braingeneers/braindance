"""
Command-line interface for BrainDance Data Manager setup.

Usage:
    python -m braindance.utils.data_manager setup --data-dir /path/to/data --catalog /path/to/catalog.csv
"""

import argparse
import sys
from pathlib import Path
from braindance.config import set_data_dir, set_catalog_path, set_output_dir, get_data_dir, get_catalog_path, get_output_dir


def setup_command(args):
    """Setup data manager paths"""
    if args.data_dir:
        set_data_dir(args.data_dir)
        print(f"✓ Data directory set to: {Path(args.data_dir).resolve()}")

    if args.catalog:
        set_catalog_path(args.catalog)
        print(f"✓ Catalog path set to: {Path(args.catalog).resolve()}")

    if args.output_dir:
        set_output_dir(args.output_dir)
        print(f"✓ Output directory set to: {Path(args.output_dir).resolve()}")

    config_path = Path.home() / '.braindance' / 'config.json'
    print(f"\n✓ Configuration saved to: {config_path}")

    # Show current configuration
    print("\nCurrent configuration:")
    print(f"  Data directory: {get_data_dir()}")
    print(f"  Catalog path: {get_catalog_path()}")
    print(f"  Output directory: {get_output_dir()}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="BrainDance Data Manager CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup all paths
  python -m braindance.utils.data_manager setup --data-dir /path/to/data --catalog /path/to/catalog.csv --output-dir /path/to/plots

  # Setup just data directory
  python -m braindance.utils.data_manager setup --data-dir /path/to/data

  # Setup just catalog path
  python -m braindance.utils.data_manager setup --catalog /path/to/catalog.csv

  # Setup just output directory
  python -m braindance.utils.data_manager setup --output-dir /path/to/plots
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Configure data manager paths')
    setup_parser.add_argument('--data-dir', type=str, help='Path to data directory')
    setup_parser.add_argument('--catalog', type=str, help='Path to catalog CSV file')
    setup_parser.add_argument('--output-dir', type=str, help='Path to output directory for plots/figures')

    args = parser.parse_args()

    if args.command == 'setup':
        setup_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
