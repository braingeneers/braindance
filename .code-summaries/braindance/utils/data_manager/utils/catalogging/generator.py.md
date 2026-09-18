# generator.py

**Path:** `braindance/utils/data_manager/utils/catalogging/generator.py`
**Module:** `braindance.utils.data_manager.utils.catalogging.generator`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Scans local or S3 project roots for chip folders, experiment folders, and raw-file sub-experiments. Produces a metadata DataFrame, optionally retains existing rows, and can attach chip-level neuron counts.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.catalogging` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.data_loading.recording` — import consumer hint; not a proven runtime call.
- **Uses:** `DEFAULT_S3_BASES` from `braindance.utils.data_manager.utils.catalogging.config` — imports (static evidence).
- **Uses:** `EXPERIMENT_BLACKLIST_PATTERNS` from `braindance.utils.data_manager.utils.catalogging.config` — imports (static evidence).
- **Uses:** `LOG_SUFFIXES` from `braindance.utils.data_manager.utils.catalogging.config` — imports (static evidence).
- **Uses:** `BraindanceExperiment` from `braindance.utils.data_manager.utils.catalogging.experiment` — imports (static evidence).
- **Uses:** `get_num_units_for_chip` from `braindance.utils.data_manager.utils.catalogging.s3_helpers` — imports (static evidence).
- **Uses:** `get_s3_client` from `braindance.utils.data_manager.utils.catalogging.s3_helpers` — imports (static evidence).
- **Uses:** `count_digits` from `braindance.utils.data_manager.utils.catalogging.validators` — imports (static evidence).
- **Uses:** `extract_exp_from_experiment_path` from `braindance.utils.data_manager.utils.catalogging.validators` — imports (static evidence).
- **Uses:** `extract_number` from `braindance.utils.data_manager.utils.catalogging.validators` — imports (static evidence).
- **Uses:** `extract_proj_from_s3_path` from `braindance.utils.data_manager.utils.catalogging.validators` — imports (static evidence).
- **Shared data:** DataPathManager→BraindanceExperiment.get_experiment_info→catalog rows; Recording.sync_experiment_raw_data uses DataPathManager.get_raw_data_files.

## Dependencies
- `boto3` — external or unresolved local import; source import evidence.
- `botocore.client.Config` — external or unresolved local import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.config.DEFAULT_S3_BASES` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.config.EXPERIMENT_BLACKLIST_PATTERNS` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.config.LOG_SUFFIXES` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.experiment.BraindanceExperiment` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.s3_helpers.get_num_units_for_chip` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.s3_helpers.get_s3_client` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.validators.count_digits` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.validators.extract_exp_from_experiment_path` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.validators.extract_number` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.validators.extract_proj_from_s3_path` — intra-repo import; source import evidence.
- `braingeneers.utils.s3wrangler` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.

## Classes
### DataPathManager()
> List and construct project/chip/experiment paths for local files or S3.
**Source:** `braindance/utils/data_manager/utils/catalogging/generator.py:33`
**Kind:** class. **Instantiated by:** braindance/utils/data_manager/utils/catalogging/generator.py:312 (named-call hint); braindance/utils/data_manager/utils/data_loading/recording.py:1961 (named-call hint)
**Constructor:** `__init__(self, s3_bases=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `chip_cache` | inferred at runtime | `{}` |
| `experiment_cache` | inferred at runtime | `{}` |
| `s3_bases` | inferred at runtime | `s3_bases or DEFAULT_S3_BASES` |
**Methods:**
#### `get_s3_client(self, bucket_name)`
> Get appropriate S3 client based on bucket name.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/generator.py:57`
#### `parse_s3_path(self, s3_path)`
> Parse S3 path into bucket and key.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/generator.py:61`
#### `list_chips(self, base_path)`
> List all chips (UUIDs) under a base path.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/catalogging/generator.py:66`
#### `list_experiments(self, base_path, chip)`
> List all experiments for a specific chip.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/catalogging/generator.py:106`
#### `list_sub_experiments(self, exp_path)`
> List all sub-experiments in an experiment directory.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/generator.py:160`
#### `get_experiment_path(self, base_path, chip, exp)`
> Get the full path to an experiment directory.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/generator.py:183`
#### `list_log_files(self, exp_path, log_suffixes=None)`
> List all log files in an experiment directory.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/generator.py:187`
#### `get_raw_data_files(self, exp_path)`
> List all raw data files (.raw.h5) in an experiment directory.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/generator.py:223`
#### `get_log_file_for_sub_experiment(self, exp_path, sub_exp, log_suffixes=None)`
> Get the log file for a specific sub-experiment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/generator.py:251`
### CatalogGenerator()
> Generates a catalog of Braindance experiments from S3 storage.
**Source:** `braindance/utils/data_manager/utils/catalogging/generator.py:284`
**Kind:** class. **Instantiated by:** braindance/utils/data_manager/utils/catalogging/__init__.py:74 (named-call hint)
**Constructor:** `__init__(self, s3_bases=None, existing_catalog_path=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `existing_catalog` | inferred at runtime | `None` |
| `path_manager` | inferred at runtime | `DataPathManager(self.s3_bases)` |
| `s3_bases` | inferred at runtime | `s3_bases or DEFAULT_S3_BASES` |
**Methods:**
#### `is_in_catalog(self, base_path, chip, experiment)`
> Check if an experiment is already in the catalog.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/generator.py:322`
#### `generate(self, chip_filter=None, verbose=True, expand_sub_experiments=True, include_units=True)`
> Scan roots and append unseen experiment metadata to existing catalog rows.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/generator.py:335`
#### `_create_error_entry(self, base_path, chip, experiment, include_units, chip_unit_counts)`
> Create a catalog entry for a failed experiment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/generator.py:479`
#### `_reorder_columns(self, catalog)`
> Reorder columns to have proj, exp after base_path.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/generator.py:496`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| s3_bases,existing_catalog_path; generate chip_filter/expand_sub_experiments=True/include_units=True. |

## Data Shapes
- Output columns include base_path,proj,exp,chip,experiment,full_path,n_stims,freq,type,n_raw_files,num_units,success; errors become type='error' rows.

## Notes
- Roots are concatenated with chip names and should end with slash.
- S3 listing requires braingeneers.utils.s3wrangler; chip/experiment listings are cached in memory.
- include_units uses first suitable chip-level spike file, unlike add_units' experiment-folder grouping.
