# 06_catalog_generation.py

**Path:** `braindance/utils/data_manager/advanced_tutorials/06_catalog_generation.py`
**Module:** `braindance.utils.data_manager.advanced_tutorials.06_catalog_generation`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Demonstrates the staged S3 catalog-generation, metadata-enrichment, and experiment-folder unit-count workflow. Saves my_catalog.csv and prints basic filtering/grouping examples.

## Connections
- **Uses:** `DEFAULT_S3_BASES` from `braindance.utils.data_manager.utils.catalogging` — imports (static evidence).
- **Uses:** `add_metadata` from `braindance.utils.data_manager.utils.catalogging` — imports (static evidence).
- **Uses:** `add_units` from `braindance.utils.data_manager.utils.catalogging` — imports (static evidence).
- **Uses:** `generate_catalog` from `braindance.utils.data_manager.utils.catalogging` — imports (static evidence).
- **Shared data:** generate_catalog→add_metadata (including exclusions)→add_units→to_csv.

## Dependencies
- `braindance.utils.data_manager.utils.catalogging.DEFAULT_S3_BASES` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.add_metadata` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.add_units` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.generate_catalog` — intra-repo import; source import evidence.

## Classes
None

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| DEFAULT_S3_BASES from catalogging config; output 'my_catalog.csv' in current directory. |

## Data Shapes
- Catalog DataFrame columns include chip,experiment,type,num_units; CSV output.

## Notes
- Runs on import and scans configured roots, potentially downloading spike files for counts.
- Type filter 'cartpole' may yield none because current generator labels stim/spontaneous.
