# metadata.py

**Path:** `braindance/utils/data_manager/utils/catalogging/metadata.py`
**Module:** `braindance.utils.data_manager.utils.catalogging.metadata`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Merges organoid metadata into catalogs by chip and derives age from experiment dates and organoid day zero. Detects known drug-name substrings in experiment identifiers.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.catalogging` — import consumer hint; not a proven runtime call.
- **Shared data:** Public add_metadata calls merge_metadata and add_drug_column.

## Dependencies
- `pandas` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `load_org_metadata(metadata_path=None)`
> Load organoid metadata from CSV file.
> **Called by:** braindance/utils/data_manager/utils/catalogging/metadata.py:119 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/metadata.py:14`
### `extract_date_from_path(path)`
> Extract experiment date from S3 path.
> **Called by:** braindance/utils/data_manager/utils/catalogging/metadata.py:95 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/metadata.py:36`
### `parse_date(date_str)`
> Parse a date string in MM/DD/YY format.
> **Called by:** braindance/utils/data_manager/utils/catalogging/metadata.py:94 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/metadata.py:62`
### `calculate_organoid_age(row, org_day_0_col='Org_Day_0', path_col='base_path', existing_age_col='Org_Age')`
> Calculate organoid age at experiment time.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/metadata.py:80`
### `merge_metadata(catalog_df, metadata_df=None, metadata_path=None)`
> Merge catalog with organoid metadata.
> **Called by:** braindance/utils/data_manager/utils/catalogging/__init__.py:112 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/metadata.py:104`
### `extract_drug_from_experiment(experiment, drug_list=None)`
> Extract drug name from experiment string if any known drug is found.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/metadata.py:152`
### `add_drug_column(catalog_df)`
> Add a 'drug' column to the catalog based on experiment names.
> **Called by:** braindance/utils/data_manager/utils/catalogging/__init__.py:113 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/metadata.py:178`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Optional metadata DataFrame/path; default adjacent org_metadata.csv; age columns configurable. |

## Data Shapes
- Catalog chip joins metadata Chip; Org_Day_0 parsed MM/DD/YY; path date YY-MM-DD or YY_MM_DD; output Org_Age and drug.

## Notes
- merge_metadata mutates input chip values for 25245lc alias then maps merged values back; missing metadata returns original catalog.
- Drug matching uses ordered substrings, including nbqx_apv before nbqx/apv.
