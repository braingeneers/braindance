# experiment.py

**Path:** `braindance/utils/data_manager/utils/catalogging/experiment.py`
**Module:** `braindance.utils.data_manager.utils.catalogging.experiment`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Represents one catalog-generation experiment and gathers its logs, raw-file inventory, stimulation count, type, and frequency metadata. It supports both main experiments and nested sub-experiments through a supplied path manager.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.catalogging` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.catalogging.generator` — import consumer hint; not a proven runtime call.
- **Uses:** `open_file` from `braindance.io` — imports (static evidence).
- **Uses:** `construct_log_path` from `braindance.utils.data_manager.utils.catalogging.s3_helpers` — imports (static evidence).
- **Uses:** `load_log_from_s3` from `braindance.utils.data_manager.utils.catalogging.s3_helpers` — imports (static evidence).
- **Uses:** `calculate_freq_from_stim_count` from `braindance.utils.data_manager.utils.catalogging.validators` — imports (static evidence).
- **Uses:** `calculate_stim_frequency` from `braindance.utils.data_manager.utils.catalogging.validators` — imports (static evidence).
- **Uses:** `categorize_frequency` from `braindance.utils.data_manager.utils.catalogging.validators` — imports (static evidence).
- **Shared data:** Uses a DataPathManager-like object for discovery, `open_file` for transparent I/O, and s3 helper/validator functions for frequency extraction.

## Dependencies
- `braindance.io.open_file` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.s3_helpers.construct_log_path` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.s3_helpers.load_log_from_s3` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.validators.calculate_freq_from_stim_count` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.validators.calculate_stim_frequency` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.validators.categorize_frequency` — intra-repo import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.

## Classes
### BraindanceExperiment()
> Encapsulate catalog metadata extraction for one main or nested experiment.
**Source:** `braindance/utils/data_manager/utils/catalogging/experiment.py:17`
**Kind:** class. **Instantiated by:** braindance/utils/data_manager/utils/catalogging/generator.py:407 (named-call hint); braindance/utils/data_manager/utils/catalogging/generator.py:434 (named-call hint); braindance/utils/data_manager/utils/catalogging/generator.py:452 (named-call hint)
**Constructor:** `__init__(self, base_path, chip, exp, path_manager, sub_exp=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `base_path` | inferred at runtime | `base_path` |
| `chip` | inferred at runtime | `chip` |
| `exp` | inferred at runtime | `exp` |
| `exp_path` | inferred at runtime | `path_manager.get_experiment_path(base_path, chip, exp)` |
| `log_files` | inferred at runtime | `None` |
| `path_manager` | inferred at runtime | `path_manager` |
| `raw_files` | inferred at runtime | `None` |
| `sub_exp` | inferred at runtime | `sub_exp` |
**Methods:**
#### `get_logs(self, log_suffixes=None)`
> Discover and load recognized experiment logs and label their types.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/experiment.py:52`
#### `get_raw_data(self)`
> Lazily list and cache raw HDF5 paths.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/catalogging/experiment.py:116`
#### `determine_experiment_type(self)`
> Classify an experiment as stimulated or spontaneous from its log row count.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/experiment.py:127`
#### `count_stimulations(self)`
> Count the number of stimulations in the experiment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/experiment.py:137`
#### `get_experiment_info(self)`
> Assemble the catalog-ready experiment metadata record.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/experiment.py:151`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Default recognized logs end with `_log.csv`, `_game_log.csv`, `_pattern_log.csv`, `_reward_log.csv`, or `_causal_log.csv`. |

## Data Shapes
- `get_logs` returns parallel lists of pandas DataFrames and inferred log-type names.
- `get_experiment_info` returns base/chip/path fields, stimulation count, categorized/raw frequency, availability/type/raw-file count, and success.

## Notes
- Log-reading exceptions are printed and converted to empty results.
- Frequency falls back from log-derived timing to a count-based heuristic.
