# postprocessing.py

**Path:** `braindance/utils/data_manager/utils/catalogging/postprocessing.py`
**Module:** `braindance.utils.data_manager.utils.catalogging.postprocessing`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Applies curated dataset corrections and exclusions to generated catalogs. Normalizes raw-file paths and baseline metadata, then records inherited baselines from earlier experiments on the same project/chip.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.catalogging` — import consumer hint; not a proven runtime call.
- **Uses:** `compare_exp_names` from `braindance.utils.data_manager.utils.catalogging.validators` — imports (static evidence).
- **Uses:** `extract_exp_from_experiment_path` from `braindance.utils.data_manager.utils.catalogging.validators` — imports (static evidence).
- **Uses:** `extract_proj_from_s3_path` from `braindance.utils.data_manager.utils.catalogging.validators` — imports (static evidence).
- **Uses:** `is_baseline` from `braindance.utils.data_manager.utils.catalogging.validators` — imports (static evidence).
- **Shared data:** Public add_metadata calls apply_catalog_fixes; validators provide baseline classification and natural experiment comparison.

## Dependencies
- `braindance.utils.data_manager.utils.catalogging.validators.compare_exp_names` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.validators.extract_exp_from_experiment_path` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.validators.extract_proj_from_s3_path` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.validators.is_baseline` — intra-repo import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `apply_corrupted_file_removal(df, verbose=True)`
> Remove known corrupted files from the catalog.
> **Called by:** braindance/utils/data_manager/utils/catalogging/postprocessing.py:448 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/postprocessing.py:76`
### `apply_specific_removals(df, verbose=True)`
> Remove specific experiments from the catalog.
> **Called by:** braindance/utils/data_manager/utils/catalogging/postprocessing.py:450 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/postprocessing.py:110`
### `apply_pattern_removals(df, verbose=True)`
> Remove experiments matching certain patterns.
> **Called by:** braindance/utils/data_manager/utils/catalogging/postprocessing.py:449 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/postprocessing.py:138`
### `apply_path_fixes(df, verbose=True)`
> Apply path corrections to the catalog.
> **Called by:** braindance/utils/data_manager/utils/catalogging/postprocessing.py:447 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/postprocessing.py:237`
### `apply_frequency_fixes(df, verbose=True)`
> Apply specific frequency corrections.
> **Called by:** braindance/utils/data_manager/utils/catalogging/postprocessing.py:451 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/postprocessing.py:271`
### `clean_baseline_experiments(df, verbose=True)`
> Clean baseline experiments by removing spurious stimulation data.
> **Called by:** braindance/utils/data_manager/utils/catalogging/postprocessing.py:452 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/postprocessing.py:294`
### `add_derived_columns(df)`
> Add derived columns (proj, exp, baseline) if not present.
> **Called by:** braindance/utils/data_manager/utils/catalogging/postprocessing.py:437 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/postprocessing.py:328`
### `resolve_baseline_inheritance(df)`
> Select most recent earlier baseline-bearing experiment per project/chip.
> **Called by:** braindance/utils/data_manager/utils/catalogging/postprocessing.py:455 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/postprocessing.py:350`
### `apply_catalog_fixes(df, verbose=True)`
> Run derived columns, failure/path fixes, exclusions, frequency/baseline cleanup, and inheritance.
> **Called by:** braindance/utils/data_manager/utils/catalogging/__init__.py:114 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/postprocessing.py:420`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Hardcoded corrupted recording lists, explicit removals, and project/chip patterns. |

## Data Shapes
- DataFrame needs chip/experiment/base_path/full_path; derives proj/exp/baseline and inherited_baseline_from.

## Notes
- Functions mutate some columns and remove rows; exclusions are specific historical datasets.
- Corrupted 25178ic list includes concatenated exp1_cont_24exp1_cont_27 entry; substring regex exclusions can match longer names.
- Baseline inheritance records source experiment name without loading its data.
