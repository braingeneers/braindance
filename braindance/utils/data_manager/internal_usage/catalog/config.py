"""
Catalog Configuration
=====================

USER-EDITABLE configuration for catalog generation.
This file contains ONLY the things you need to change:
  - S3 paths to scan

Output paths are determined by braindance.config settings.
To change the final catalog location, use:
    from braindance.config import set_catalog_path
    set_catalog_path('/path/to/catalog.csv')

To add a new dataset:
1. Add the S3 base path to S3_BASES list below
2. Run: python 1_generate_catalog.py
3. Run: python 2_amend_catalog.py

"""

from pathlib import Path

# Import catalog path from main braindance config
try:
    from braindance.config import get_catalog_path
    FINAL_CATALOG = get_catalog_path()
except ImportError:
    # Fallback if braindance not installed
    FINAL_CATALOG = Path(__file__).parent / "all_catalog.csv"

# =============================================================================
# S3 BASE PATHS - Add your data sources here
# =============================================================================

S3_BASES = [
    # -------------------------------------------------------------------------
    # Busy Bee Experiments
    # -------------------------------------------------------------------------
    's3://braingeneersdev/asrobbin/braindance_data/24-08-16_busybee/',
    's3://braingeneers/braindance/25-02-25_busybees/',
    's3://braingeneersdev/asrobbin/braindance_data/25-02-25_busybees/',
    's3://braingeneers/braindance/25_04_15_busiest_bee/',
    's3://braingeneers/braindance/busy_bee_10-10-2025/',
    's3://braingeneers/braindance/bee_25-11-26/',
    's3://braingeneers/braindance/bee_ctrl_25-11-26/',
    
    # -------------------------------------------------------------------------
    # Drug Experiments
    # -------------------------------------------------------------------------
    's3://braingeneers/braindance/2025-05-14busy_drugs/',
    's3://braingeneers/braindance/25-05-08_bee_drugs/',
    's3://braingeneersdev/asrobbin/braindance_data/23-05-10_drug_causal/',
    's3://braingeneersdev/asrobbin/braindance_data/24-105-10_drug_causal/',
    's3://braingeneersdev/asrobbin/braindance_data/24-05-13-causal-mousepaper/',
    's3://braingeneers/braindance/drug_causal_2025-09-01/',
    's3://braingeneersdev/asrobbin/braindance_data/2024-05-30_drugs_cartpole/',
    's3://braingeneersdev/asrobbin/braindance_data/2025-03-24_cp_full_drugs/',
    's3://braingeneersdev/asrobbin/braindance_data/25-02-2025_cp_drugs_gpu/',
    
    # -------------------------------------------------------------------------
    # FIFE Experiments
    # -------------------------------------------------------------------------
    's3://braingeneers/braindance/250802_fife/',
    
    # -------------------------------------------------------------------------
    # Cartpole Experiments
    # -------------------------------------------------------------------------
    's3://braingeneersdev/asrobbin/braindance_data/23-11-30_cartpole/',
    's3://braingeneersdev/asrobbin/braindance_data/23-11-22_cartpole/',
    's3://braingeneers/braindance/25-10-22_cartpole_again/',
    's3://braingeneersdev/asrobbin/braindance_data/2024-05-06_butterfly/',
    's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/',
    's3://braingeneersdev/asrobbin/braindance_data/24-03-25_plasticity/',
    's3://braingeneersdev/asrobbin/braindance_data/24-05-18_plasticity/',
    's3://braingeneersdev/asrobbin/braindance_data/24-05-24/',
    's3://braingeneersdev/asrobbin/24-01-07_data/',
]


# =============================================================================
# LOCAL FILE PATHS (for intermediate files only)
# =============================================================================

# Directory where this config lives
_THIS_DIR = Path(__file__).parent

# Organoid metadata CSV (chip info, plating dates, etc.)
METADATA_CSV = _THIS_DIR / "org_metadata.csv"

# Raw catalog output from step 1 (intermediate - stays local)
RAW_CATALOG = _THIS_DIR / "braindance_catalog.csv"
