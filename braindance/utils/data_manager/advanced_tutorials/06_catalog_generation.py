"""
BrainDance Data Manager - Tutorial 06: Generating Catalogs

This tutorial shows how to generate experiment catalogs from S3.

Quick Start:
    from braindance.utils.data_manager.utils.catalogging import (
        generate_catalog, add_metadata, add_units
    )
    
    catalog = generate_catalog()
    catalog = add_metadata(catalog)
    catalog = add_units(catalog, verbose=True)
    catalog.to_csv('my_catalog.csv', index=False)

S3 paths are configured in:
    braindance/utils/data_manager/utils/catalogging/config.py
"""

# =============================================================================
# STEP 1: Generate catalog from S3
# =============================================================================

from braindance.utils.data_manager.utils.catalogging import (
    generate_catalog, 
    add_metadata, 
    add_units,
    DEFAULT_S3_BASES,
)

# See what S3 paths will be scanned
print("Configured S3 paths:")
for path in DEFAULT_S3_BASES[:5]:
    print(f"  {path}")
print(f"  ... and {len(DEFAULT_S3_BASES) - 5} more\n")

# Generate catalog (scans all paths in config.py)
catalog = generate_catalog()

# Or specify custom paths:
# catalog = generate_catalog(s3_paths=['s3://mybucket/myproject/'])


# =============================================================================
# STEP 2: Add metadata
# =============================================================================

# Adds organoid age, org_id, drug conditions, etc.
catalog = add_metadata(catalog)


# =============================================================================
# STEP 3: Add unit counts from spike sorting
# =============================================================================

# This searches S3 for spike sorting files and adds num_units column
# Note: Spike sorting is per EXPERIMENT FOLDER (drug1/, drug3/), not per chip
catalog = add_units(catalog, verbose=True)


# =============================================================================
# STEP 4: Save and use
# =============================================================================

# Save your catalog
catalog.to_csv('my_catalog.csv', index=False)

# Quick look at what we got
print("\n" + "="*50)
print("Catalog Summary")
print("="*50)
print(f"Total experiments: {len(catalog)}")
print(f"Unique chips: {catalog['chip'].nunique()}")
print(f"With unit counts: {catalog['num_units'].notna().sum()}")

if 'type' in catalog.columns:
    print(f"\nExperiment types:")
    for t, count in catalog['type'].value_counts().items():
        print(f"  {t}: {count}")


# =============================================================================
# Using the catalog
# =============================================================================
print("\n" + "="*50)
print("Example: Filter and analyze")
print("="*50)

# Filter to experiments with spike sorting
has_units = catalog[catalog['num_units'].notna()]
print(f"Experiments with spike data: {len(has_units)}")

# Filter by type
if 'type' in catalog.columns:
    cartpole = catalog[catalog['type'] == 'cartpole']
    print(f"Cartpole experiments: {len(cartpole)}")

# Filter by unit count
high_units = catalog[catalog['num_units'] >= 100]
print(f"Experiments with >= 100 units: {len(high_units)}")

# Group by chip
print("\nUnits per chip:")
for chip, group in catalog.groupby('chip'):
    units = group['num_units'].iloc[0] if group['num_units'].notna().any() else None
    print(f"  {chip}: {len(group)} experiments, {units} units")
