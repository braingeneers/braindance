# s3_loader.py

**Path:** `braindance/utils/data_manager/utils/data_loading/s3_loader.py`
**Module:** `braindance.utils.data_manager.utils.data_loading.s3_loader`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Wraps boto3 and smart_open for S3 downloads, uploads, CSV reads, and compatibility pickle loading. Optional local pickle caching stores downloaded objects under a basename-derived path.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.catalogging.s3_helpers` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.data_loading.recording` — import consumer hint; not a proven runtime call.
- **Uses:** `load_spike_pickle` from `braindance.spike_data` — imports (static evidence).
- **Shared data:** Recording and catalog helpers use S3Loader; legacy-compatible pickle reads delegate to `load_spike_pickle`.

## Dependencies
- `boto3` — external or unresolved local import; source import evidence.
- `braindance.spike_data.load_spike_pickle` — intra-repo import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.
- `smart_open` — external or unresolved local import; source import evidence.

## Classes
### S3Loader()
> Provide endpoint-aware S3 file transfer and simple local caching.
**Source:** `braindance/utils/data_manager/utils/data_loading/s3_loader.py:23`
**Kind:** class. **Instantiated by:** braindance/utils/data_manager/utils/catalogging/s3_helpers.py:220 (named-call hint); braindance/utils/data_manager/utils/data_loading/recording.py:244 (named-call hint)
**Constructor:** `__init__(self, endpoint_url: str=NRP_S3_ENDPOINT, cache_dir: Optional[Path]=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_s3_client` | inferred at runtime | `boto3.client('s3', endpoint_url=endpoint_url)` |
| `cache_dir` | inferred at runtime | `Path(cache_dir) if cache_dir else None` |
| `endpoint_url` | inferred at runtime | `endpoint_url` |
**Methods:**
#### `download_file(self, s3_path: str, local_path: Path) -> bool`
> Copy a complete S3 object to a local path.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/utils/data_manager/utils/data_loading/s3_loader.py:47`
#### `upload_file(self, local_path: Path, s3_path: str) -> bool`
> Copy a complete local file to S3.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/utils/data_manager/utils/data_loading/s3_loader.py:78`
#### `load_pickle(self, s3_path: str, use_cache: bool=True) -> Optional[Any]`
> Load a trusted pickle from cache or S3 and optionally cache the object locally.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/s3_loader.py:111`
#### `load_csv(self, s3_path: str, use_cache: bool=True)`
> Stream an S3 CSV into a pandas DataFrame.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/s3_loader.py:151`
#### `_cache_file(self, s3_path: str, data: Any)`
> Cache data to local disk.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/utils/data_manager/utils/data_loading/s3_loader.py:174`
#### `_get_cache_path(self, s3_path: str) -> Path`
> Generate cache path from S3 URL.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/s3_loader.py:194`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Default endpoint is `https://s3-west.nrp-nautilus.io`; constructor accepts a different endpoint and optional cache directory. |

## Data Shapes
- CSV loads return pandas DataFrames; pickle loads return arbitrary trusted Python objects, including legacy SpikeData.

## Notes
- Download/upload methods read entire files into memory.
- `load_csv` accepts `use_cache` but does not implement CSV caching.
- Cache write failures are silently ignored and cache filenames use only the S3 basename.
