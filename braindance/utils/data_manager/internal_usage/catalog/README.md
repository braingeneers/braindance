# Catalog Generation Scripts

Internal scripts for generating and maintaining the BrainDance experiment catalog.

## Quick Start

```bash
cd braindance/utils/data_manager/internal_usage/catalog
conda activate brain

# Step 1: Generate raw catalog from S3
python 1_generate_catalog.py

# Step 2: Add metadata, fixes, and unit counts
python 2_amend_catalog.py
```

## Files

| File | Description |
|------|-------------|
| `config.py` | S3 paths, corrupted file lists, configuration |
| `1_generate_catalog.py` | Scans S3 and creates raw catalog |
| `2_amend_catalog.py` | Adds metadata, fixes, and unit counts |
| `org_metadata.csv` | Organoid metadata (plating dates, ages) |
| `braindance_catalog.csv` | Raw catalog (intermediate output) |
| `all_catalog.csv` | **Final catalog** ready for analysis |

## Workflow

### Step 1: Generate Raw Catalog

```bash
python 1_generate_catalog.py
```

- Scans all S3 paths defined in `config.py`
- Discovers experiments and extracts metadata
- Outputs: `braindance_catalog.csv`

### Step 2: Amend Catalog

```bash
python 2_amend_catalog.py
```

Applies post-processing:
1. Merges organoid metadata (ages, chip info)
2. Adds drug column from experiment paths
3. Removes corrupted/problematic experiments
4. Applies manual fixes (frequency corrections, etc.)
5. **Adds spike sorting unit counts** (incremental - reuses existing values)
6. Outputs: `all_catalog.csv`

> **Note**: Step 2 is incremental for unit counts. If `all_catalog.csv` exists, it will load existing `num_units` values and only fetch counts for new experiments, saving significant time.

## Adding New Data

1. **Edit `config.py`**: Add your S3 path to `S3_BASES`
2. **Run Step 1**: `python 1_generate_catalog.py`
3. **Run Step 2**: `python 2_amend_catalog.py`

## Output Catalog Columns

| Column | Description |
|--------|-------------|
| `base_path` | S3 base path |
| `chip` | Chip/organoid ID |
| `experiment` | Experiment path (e.g., `drug1/drug1_cartpole_F`) |
| `type` | Experiment type (freqs, cartpole, causal) |
| `freq` | Stimulation frequency (Hz) |
| `num_units` | Number of sorted spike units |
| `success` | Experiment completed successfully |
| `org_id` | Organoid identifier |
| `Org_Age` | Organoid age in days |
| `drug` | Drug condition |
| `baseline` | Is baseline recording |

## Relationship to `utils/catalogging`

These scripts are **wrappers** around the catalogging module API:

- **`utils/catalogging/`** - Reusable Python API (`generate_catalog()`, `add_metadata()`, `add_units()`)
- **`internal_usage/catalog/`** - Convenience scripts that call those functions

Both approaches are valid:
- Use **scripts** for routine catalog generation
- Use **Python API** for custom workflows or programmatic access

## Configuration

### S3 Paths (`config.py`)

```python
S3_BASES = [
    's3://braingeneersdev/asrobbin/braindance_data/24-08-16_busybee/',
    's3://braingeneers/braindance/25-02-25_busybees/',
    # Add new paths here
]
```

### Corrupted Files

Known corrupted experiments are listed in `config.py` and automatically removed during Step 2.

## Troubleshooting

**Q: Step 2 is taking too long?**  
A: The unit count fetching can be slow. If you already have `all_catalog.csv`, Step 2 will reuse existing `num_units` values and only fetch counts for new experiments.

**Q: How do I update just the metadata without re-fetching units?**  
A: Just run Step 2 again - it's incremental for unit counts.

**Q: Where is the final catalog used?**  
A: The catalog is used by `RecordingCatalog` in the data_manager for loading experiments.

## See Also

- [Catalogging Module README](../../utils/catalogging/README.md) - Full API documentation
- [Tutorial 06](../../utils/advanced_tutorials/06_catalog_generation.py) - Programmatic usage
- [Tutorial 01](../../utils/advanced_tutorials/01_working_with_catalog.py) - Using catalogs for analysis
