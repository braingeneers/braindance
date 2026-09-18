# __init__.py

**Path:** `braindance/utils/data_manager/utils/catalogging/__init__.py`
**Module:** `braindance.utils.data_manager.utils.catalogging`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Exposes catalog generation and staged enrichment helpers for metadata and neuron counts. Can validate referenced spike-data objects on S3 and clear invalid unit counts without deleting catalog rows.

## Connections
- **Used by:** `braindance.utils.data_manager.advanced_tutorials.06_catalog_generation` — import consumer hint; not a proven runtime call.
- **Uses:** `DEFAULT_S3_BASES` from `braindance.utils.data_manager.utils.catalogging.config` — imports (static evidence).
- **Uses:** `get_default_catalog_path` from `braindance.utils.data_manager.utils.catalogging.config` — imports (static evidence).
- **Uses:** `BraindanceExperiment` from `braindance.utils.data_manager.utils.catalogging.experiment` — imports (static evidence).
- **Uses:** `CatalogGenerator` from `braindance.utils.data_manager.utils.catalogging.generator` — imports (static evidence).
- **Uses:** `DataPathManager` from `braindance.utils.data_manager.utils.catalogging.generator` — imports (static evidence).
- **Uses:** `add_drug_column` from `braindance.utils.data_manager.utils.catalogging.metadata` — imports (static evidence).
- **Uses:** `calculate_organoid_age` from `braindance.utils.data_manager.utils.catalogging.metadata` — imports (static evidence).
- **Uses:** `load_org_metadata` from `braindance.utils.data_manager.utils.catalogging.metadata` — imports (static evidence).
- **Uses:** `merge_metadata` from `braindance.utils.data_manager.utils.catalogging.metadata` — imports (static evidence).
- **Uses:** `apply_catalog_fixes` from `braindance.utils.data_manager.utils.catalogging.postprocessing` — imports (static evidence).
- **Uses:** `apply_corrupted_file_removal` from `braindance.utils.data_manager.utils.catalogging.postprocessing` — imports (static evidence).
- **Uses:** `check_s3_file_exists` from `braindance.utils.data_manager.utils.catalogging.s3_helpers` — imports (static evidence).
- **Uses:** `construct_log_path` from `braindance.utils.data_manager.utils.catalogging.s3_helpers` — imports (static evidence).
- **Uses:** `construct_spike_data_path` from `braindance.utils.data_manager.utils.catalogging.s3_helpers` — imports (static evidence).
- **Uses:** `get_num_units_for_experiment_folder` from `braindance.utils.data_manager.utils.catalogging.s3_helpers` — imports (static evidence).
- **Uses:** `get_s3_client` from `braindance.utils.data_manager.utils.catalogging.s3_helpers` — imports (static evidence).
- **Uses:** `load_log_from_s3` from `braindance.utils.data_manager.utils.catalogging.s3_helpers` — imports (static evidence).
- **Uses:** `parse_s3_path` from `braindance.utils.data_manager.utils.catalogging.s3_helpers` — imports (static evidence).
- **Uses:** `calculate_stim_frequency` from `braindance.utils.data_manager.utils.catalogging.validators` — imports (static evidence).
- **Uses:** `categorize_frequency` from `braindance.utils.data_manager.utils.catalogging.validators` — imports (static evidence).
- **Uses:** `extract_exp_from_experiment_path` from `braindance.utils.data_manager.utils.catalogging.validators` — imports (static evidence).
- **Uses:** `extract_proj_from_s3_path` from `braindance.utils.data_manager.utils.catalogging.validators` — imports (static evidence).
- **Uses:** `is_baseline` from `braindance.utils.data_manager.utils.catalogging.validators` — imports (static evidence).
- **Uses:** `is_baseline_with_reason` from `braindance.utils.data_manager.utils.catalogging.validators` — imports (static evidence).
- **Shared data:** CatalogGenerator→generate_catalog; merge_metadata/add_drug_column/apply_catalog_fixes→add_metadata; s3_helpers→unit counting and validation.

## Dependencies
- `braindance.utils.data_manager.utils.catalogging.config.DEFAULT_S3_BASES` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.config.get_default_catalog_path` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.experiment.BraindanceExperiment` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.generator.CatalogGenerator` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.generator.DataPathManager` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.metadata.add_drug_column` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.metadata.calculate_organoid_age` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.metadata.load_org_metadata` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.metadata.merge_metadata` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.postprocessing.apply_catalog_fixes` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.postprocessing.apply_corrupted_file_removal` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.s3_helpers.check_s3_file_exists` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.s3_helpers.construct_log_path` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.s3_helpers.construct_spike_data_path` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.s3_helpers.get_num_units_for_experiment_folder` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.s3_helpers.get_s3_client` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.s3_helpers.load_log_from_s3` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.s3_helpers.parse_s3_path` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.validators.calculate_stim_frequency` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.validators.categorize_frequency` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.validators.extract_exp_from_experiment_path` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.validators.extract_proj_from_s3_path` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.validators.is_baseline` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.validators.is_baseline_with_reason` — intra-repo import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `generate_catalog(s3_paths=None, verbose=True)`
> Generate a catalog from S3 paths.
> **Called by:** braindance/utils/data_manager/advanced_tutorials/06_catalog_generation.py:38 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/__init__.py:59`
### `add_metadata(catalog, metadata_path=None, verbose=True)`
> Add organoid metadata (age, org_id, etc.) to a catalog.
> **Called by:** braindance/utils/data_manager/advanced_tutorials/06_catalog_generation.py:49 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/__init__.py:86`
### `add_units(catalog, verbose=True)`
> Add num_units column by reading spike sorting files from S3.
> **Called by:** braindance/utils/data_manager/advanced_tutorials/06_catalog_generation.py:58 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/__init__.py:124`
### `validate_recording_files(catalog, verbose=True)`
> Check that each recording's specific spike_data file exists on S3.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/__init__.py:224`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| generate_catalog(s3_paths=None,verbose=True); add_metadata(metadata_path); add_units and validate_recording_files. |

## Data Shapes
- Input/return pandas catalogs; add_units groups by base_path/chip/experiment folder and populates num_units.

## Notes
- generate_catalog disables unit loading for initial scan; add_metadata also applies dataset-specific exclusions.
- Unit-count reuse assumes a shared count within each experiment folder.
