"""
Internal Usage - Catalog Generation
====================================

This folder contains scripts for generating and maintaining the BrainDance catalog.

Files:
    config.py               - Configuration: S3 paths, settings, corrupted files
    1_generate_catalog.py   - Step 1: Scan S3 and generate raw catalog
    2_amend_catalog.py      - Step 2: Apply metadata and fixes
    org_metadata.csv        - Organoid metadata for age calculation

Usage:
    cd braindance/utils/data_manager/internal_usage/catalog
    conda activate brain
    
    # Step 1: Generate catalog from S3
    python 1_generate_catalog.py
    
    # Step 2: Merge metadata, apply fixes, and add unit counts
    python 2_amend_catalog.py

Output:
    braindance_catalog.csv  - Raw catalog (intermediate)
    all_catalog.csv         - Final catalog for analysis

To add new data:
    1. Edit config.py and add your S3 path to S3_BASES
    2. Run 1_generate_catalog.py
    3. Run 2_amend_catalog.py
"""
