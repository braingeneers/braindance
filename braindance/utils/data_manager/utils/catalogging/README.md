# BrainDance Catalog Generator

Generate experiment catalogs from S3 storage.

## Quick Start

```python
from braindance.utils.data_manager.utils.catalogging import (
    generate_catalog, add_metadata, add_units
)

# Generate catalog from all configured S3 paths
catalog = generate_catalog()

# Add organoid metadata (age, org_id, etc.)
catalog = add_metadata(catalog)

# Add spike sorting unit counts
catalog = add_units(catalog, verbose=True)

# Save
catalog.to_csv('my_catalog.csv', index=False)
```

## Functions

### `generate_catalog(s3_paths=None, verbose=True)`

Scans S3 paths and generates a catalog DataFrame.

```python
# Use default paths from config.py
catalog = generate_catalog()

# Or specify custom paths
catalog = generate_catalog(s3_paths=['s3://mybucket/myproject/'])
```

### `add_metadata(catalog, metadata_path=None, verbose=True)`

Adds organoid metadata (age, org_id, drug conditions).

```python
catalog = add_metadata(catalog)
```

### `add_units(catalog, verbose=True)`

Adds `num_units` column from spike sorting files.

**Note:** Spike sorting is done per **experiment folder** (e.g., `drug1/`, `drug3/`), not per chip. Different folders on the same chip may have different unit counts.

```python
catalog = add_units(catalog, verbose=True)
```

Output:
```
[1/150] 25178ic/drug1 (25-02-2025_cp_drugs_gpu) → 256 units
[2/150] 25178ic/drug3 (25-02-2025_cp_drugs_gpu) → no spike data
[3/150] 25178ic/drug4 (25-02-2025_cp_drugs_gpu) → 187 units
...
```

## Configuration

S3 paths are configured in `config.py`:

```python
# config.py
DEFAULT_S3_BASES = [
    's3://braingeneersdev/asrobbin/braindance_data/24-08-16_busybee/',
    's3://braingeneers/braindance/25-02-25_busybees/',
    # Add new paths here
]
```

## Catalog Columns

| Column | Description |
|--------|-------------|
| `base_path` | S3 base path |
| `chip` | Chip/organoid ID |
| `experiment` | Experiment path (e.g., `drug1/drug1_cartpole_F`) |
| `type` | Experiment type (freqs, cartpole, causal) |
| `freq` | Stimulation frequency (Hz) |
| `num_units` | Number of sorted units |
| `success` | Experiment completed successfully |
| `org_id` | Organoid identifier |
| `age_days` | Organoid age in days |
| `drug` | Drug condition |

## CLI (Alternative)

There's also a CLI if you prefer:

```bash
# Full pipeline
python -m braindance.utils.data_manager.utils.catalogging full \
  --catalog my_catalog.csv

# Just add units to existing catalog
python -m braindance.utils.data_manager.utils.catalogging add-units \
  --catalog existing.csv \
  --output with_units.csv \
  --verbose
```

## See Also

- [Tutorial 06](../advanced_tutorials/06_catalog_generation.py) - Full tutorial
- [Tutorial 01](../advanced_tutorials/01_working_with_catalog.py) - Using catalogs for analysis

---

## Technical: Module Architecture

```
catalogging/
├── __init__.py      # Simple API: generate_catalog(), add_metadata(), add_units()
├── __main__.py      # CLI entry point
├── cli.py           # CLI implementation (alternative to Python API)
├── config.py        # S3 paths and configuration
├── generator.py     # Core catalog generation logic
├── experiment.py    # Single experiment representation
├── s3_helpers.py    # S3 read/write utilities
├── validators.py    # Experiment classification logic
├── metadata.py      # Organoid metadata merging
├── postprocessing.py # Manual fixes and data cleaning
└── org_metadata.csv # Bundled organoid metadata
```

### `config.py`
Central configuration file containing:
- `DEFAULT_S3_BASES` - List of S3 paths to scan for experiments
- `S3_ENDPOINTS` - Bucket-to-endpoint mapping (e.g., braingeneersdev → s3-west.nrp-nautilus.io)
- `LOG_SUFFIXES` - File patterns for stimulus logs (`_log.csv`, `_game_log.csv`, etc.)
- `EXPERIMENT_BLACKLIST_PATTERNS` - Directories to skip (metadata, configs, etc.)

### `generator.py`
Main catalog generation engine with two classes:

**`DataPathManager`** - Handles S3 filesystem operations:
- Lists chips (organoid directories) under each S3 base path
- Lists experiments under each chip
- Caches results to avoid redundant S3 calls

**`CatalogGenerator`** - Orchestrates catalog creation:
- Iterates through all S3 paths → chips → experiments
- Creates `BraindanceExperiment` objects for each
- Extracts metadata and assembles into DataFrame
- Handles incremental updates (skip existing entries)

### `experiment.py`
**`BraindanceExperiment`** - Represents a single experiment:
- Loads stimulus log from S3 (`_log.csv`)
- Extracts stim count, frequency, experiment type
- Handles different naming conventions (freqs/, exp1/, drug1/, etc.)
- Detects sub-experiments (cartpole sessions within a recording)

### `s3_helpers.py`
Low-level S3 utilities:
- `get_s3_client()` - Creates boto3 client with correct endpoint
- `parse_s3_path()` - Splits `s3://bucket/key` into components
- `construct_log_path()` - Builds path to experiment's log file
- `load_log_from_s3()` - Reads CSV log into DataFrame
- `load_spike_data_from_s3()` - Loads pickled spike sorting objects
- `get_num_units_for_experiment_folder()` - Finds unit count for an experiment folder

### `validators.py`
Experiment classification and validation:
- `is_baseline()` / `is_baseline_with_reason()` - Detects baseline recordings (only BL1 variants)
- `calculate_stim_frequency()` - Computes Hz from inter-stim intervals
- `categorize_frequency()` - Bins frequencies (0, 1, 2, 4, 8, 16, 24, 32 Hz)
- `extract_proj_from_s3_path()` - Gets project name from path
- `extract_exp_from_experiment_path()` - Gets experiment type (freqs, cartpole, etc.)

### `metadata.py`
Organoid metadata handling:
- `load_org_metadata()` - Loads bundled `org_metadata.csv`
- `merge_metadata()` - Joins catalog with organoid info by chip ID
- `calculate_organoid_age()` - Computes age in days from experiment date and org birth date
- `add_drug_column()` - Extracts drug condition from experiment paths

### `postprocessing.py`
Data cleaning and manual fixes:
- `apply_catalog_fixes()` - Applies all corrections in sequence
- `apply_corrupted_file_removal()` - Removes known bad recordings
- Hardcoded lists of corrupted files by chip
- Frequency corrections for mislabeled experiments
- Project name standardization

### `cli.py`
Command-line interface (alternative to Python API):
- `generate` - Scan S3 and create catalog
- `merge-metadata` - Add organoid metadata
- `postprocess` - Apply fixes
- `add-units` - Add spike sorting unit counts
- `full` - Run complete pipeline
