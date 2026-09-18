# config.py

**Path:** `braindance/utils/data_manager/utils/catalogging/config.py`
**Module:** `braindance.utils.data_manager.utils.catalogging.config`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Defines default S3 roots, known endpoint mappings, log suffixes, and experiment/frequency conventions for catalog generation. Resolves default catalog and organoid-metadata file paths.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.catalogging` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.catalogging.generator` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.catalogging.s3_helpers` — import consumer hint; not a proven runtime call.
- **Shared data:** generator uses blacklist/log suffixes/default roots; s3_helpers uses endpoint map.

## Dependencies
- `config.S3_BASES` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `get_default_catalog_path()`
> Get the default path for the catalog file.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/config.py:68`
### `get_default_metadata_path()`
> Get the default path for the organoid metadata file.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/config.py:96`
### `get_s3_endpoint(bucket_name)`
> Get the S3 endpoint URL for a given bucket.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/config.py:108`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| DEFAULT_S3_BASES imported from internal_usage/catalog/config when available; otherwise []. |
| Known braingeneers/dev buckets use https://s3-west.nrp-nautilus.io; catalog fallback is cwd/braindance_catalog.csv. |

## Data Shapes
None

## Notes
- Temporarily inserts internal catalog directory in sys.path; failed import can leave entry inserted.
- get_default_catalog_path reads config.json key 'catalog'; metadata default is adjacent org_metadata.csv.
