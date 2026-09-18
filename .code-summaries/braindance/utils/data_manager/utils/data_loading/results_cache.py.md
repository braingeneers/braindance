# results_cache.py

**Path:** `braindance/utils/data_manager/utils/data_loading/results_cache.py`
**Module:** `braindance.utils.data_manager.utils.data_loading.results_cache`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Provides local and S3 caching for derived recording results, including shared S3 clients, parameterized cache keys, zstd wire/local compression, downloads, uploads, and compute-on-miss orchestration.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.data_loading.recording` — import consumer hint; not a proven runtime call.
- **Shared data:** Recording.get_binned_fr/get_binned_isi and burst detection use get_or_compute.
- **Shared data:** S3 transport uses boto3 raw get_object/put_object to avoid smart_open decompression dependencies.

## Dependencies
- `boto3` — external or unresolved local import; source import evidence.
- `botocore.config.Config` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `smart_open` — external or unresolved local import; source import evidence.
- `zstandard` — external or unresolved local import; source import evidence.

## Classes
### ResultsManifest()
> Tracks cached results metadata per-experiment. Stored as manifest.json in the results directory.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:240`
**Kind:** class. **Instantiated by:** braindance/utils/data_manager/utils/data_loading/results_cache.py:355 (named-call hint)
**Constructor:** `__init__(self, path: Path)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_manifest` | Dict | `{'version': '1.0', 'results': {}}` |
| `path` | inferred at runtime | `path / 'manifest.json'` |
**Methods:**
#### `_load(self)`
> Load manifest from disk if exists.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:252`
#### `save(self)`
> Save manifest to disk.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:261`
#### `add_result(self, name: str, params: Dict[str, Any], filename: str, shape: Optional[Tuple]=None, dtype: Optional[str]=None)`
> Add a result entry to the manifest. Args: name: Result type (e.g., 'binned_fr') params:.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:267`
#### `get_result(self, name: str, params: Dict[str, Any]) -> Optional[Dict]`
> Get result entry from manifest.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:296`
#### `has_result(self, name: str, params: Dict[str, Any]) -> bool`
> Check if result exists in manifest.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:301`
#### `list_results(self) -> List[Dict]`
> List all cached results.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:306`
### ResultsCache()
> Manages local-first derived-result caches with optional S3 synchronization and parameterized filenames.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:311`
**Kind:** class. **Instantiated by:** braindance/utils/data_manager/utils/data_loading/recording.py:2084 (named-call hint)
**Constructor:** `__init__(self, local_path: Path, s3_path: Optional[str]=None, auto_upload: bool=False, auto_download: bool=True, endpoint_url: str=NRP_S3_ENDPOINT)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_manifest` | inferred at runtime | `ResultsManifest(self.local_path)` |
| `_s3_client` | inferred at runtime | `None` |
| `auto_download` | inferred at runtime | `auto_download` |
| `auto_upload` | inferred at runtime | `auto_upload` |
| `endpoint_url` | inferred at runtime | `endpoint_url` |
| `local_path` | inferred at runtime | `Path(local_path)` |
| `s3_path` | inferred at runtime | `s3_path.rstrip('/') if s3_path else None` |
**Methods:**
#### `_get_s3_client(self)`
> Get or create S3 client.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:360`
#### `_filename(self, name: str, params: Dict[str, Any], ext: str='.npz') -> str`
> Generate filename from name and params.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:372`
#### `_local_file_path(self, name: str, params: Dict[str, Any], ext: str='.npz') -> Path`
> Get local file path for result.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:376`
#### `_s3_file_path(self, name: str, params: Dict[str, Any], ext: str='.npz') -> Optional[str]`
> Get S3 path for result.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:380`
#### `_local_zst_path(self, name: str, params: Dict[str, Any]) -> Path`
> Local path for the zstd-compressed form (`<name><key>.npz.zst`).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:386`
#### `_existing_local(self, name: str, params: Dict[str, Any]) -> Optional[Path]`
> Return the on-disk cache file (compressed preferred), or None.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:390`
#### `exists_local(self, name: str, params: Dict[str, Any]) -> bool`
> Check if result exists locally (compressed `.npz.zst` or `.npz`).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:398`
#### `evict_local(self, name: str, params: Dict[str, Any]) -> bool`
> Delete the on-disk cache file for (name, params), if present. For batch jobs that stream through many recordings once each (no reuse), the downloaded/computed file otherwise accumulates on disk forever — `Recording.clear_cache()` only drops the in-memory `DataContext`, not this file. Returns True if a file was removed.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:402`
#### `_head_s3(self, s3_path: str) -> bool`
> True if the given s3:// object exists.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:416`
#### `_put_s3(self, s3_path: str, data: bytes)`
> Upload bytes to s3:// via boto3 put_object. Uses boto3 directly rather than `smart_open.open(...,'wb')` — the latter pulls in a `backports` module absent from the container image and fails every write. boto3 (already used for head/download) has no such dep.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:426`
#### `_get_s3(self, s3_path: str) -> bytes`
> Fetches raw S3 object bytes so cache-specific decompression handles compressed objects safely.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:436`
#### `exists_s3(self, name: str, params: Dict[str, Any]) -> bool`
> Check if result exists on S3 (compressed `.npz.zst` or legacy `.npz`).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:449`
#### `load_local(self, name: str, params: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]`
> Load result from local disk (transparently handles `.npz.zst`).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:459`
#### `download_from_s3(self, name: str, params: Dict[str, Any]) -> bool`
> Download result from S3 to local cache. Returns: True if successful, False otherwise.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:490`
#### `upload_to_s3(self, name: str, params: Dict[str, Any]) -> bool`
> Upload result from local cache to S3. Returns: True if successful, False otherwise.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:551`
#### `save_local(self, name: str, params: Dict[str, Any], data: Dict[str, np.ndarray])`
> Save result to local disk. Args: name: Result type (e.g., 'binned_fr') params:.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:584`
#### `get_or_compute(self, name: str, params: Dict[str, Any], compute_fn: Callable[[], Dict[str, np.ndarray]], force_recompute: bool=False) -> Dict[str, np.ndarray]`
> Get cached result or compute and cache. This is the main entry point for caching. The workflow is: 1. Check local cache 2. If not found and auto_download, try S3 3. If not found, compute using compute_fn 4. Save to local cache 5. If auto_upload, sync to S3 Args: name: Result type (e.g., 'binned_fr') params:.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:620`
#### `sync_manifest_to_s3(self) -> bool`
> Upload manifest.json to S3.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:673`
#### `sync_all_to_s3(self) -> Tuple[int, int]`
> Upload all cached results to S3. Returns: Tuple of (success_count, fail_count).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:687`

## Functions
### `_shared_s3_client(endpoint_url: Optional[str])`
> Process-wide boto3 S3 client, one per endpoint.
> **Called by:** braindance/utils/data_manager/utils/data_loading/results_cache.py:369 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:86`
### `_zstd_disabled() -> bool`
> unclear — see source
> **Called by:** braindance/utils/data_manager/utils/data_loading/results_cache.py:143 (named-call hint); braindance/utils/data_manager/utils/data_loading/results_cache.py:162 (named-call hint); braindance/utils/data_manager/utils/data_loading/results_cache.py:179 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:137`
### `_wire_zstd_eligible(name: str) -> bool`
> Whether this cache name's S3 object should be zstd-compressed on WRITE.
> **Called by:** braindance/utils/data_manager/utils/data_loading/results_cache.py:568 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:141`
### `_wire_zstd_readable(name: str) -> bool`
> Determines whether a cache name should probe for a zstd-compressed S3 object on reads, independent of local zstandard availability.
> **Called by:** braindance/utils/data_manager/utils/data_loading/results_cache.py:454 (named-call hint); braindance/utils/data_manager/utils/data_loading/results_cache.py:502 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:148`
### `_require_zstd(where: str)`
> unclear — see source
> **Called by:** braindance/utils/data_manager/utils/data_loading/results_cache.py:468 (named-call hint); braindance/utils/data_manager/utils/data_loading/results_cache.py:526 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:167`
### `_local_zstd_eligible(name: str) -> bool`
> Whether this cache name's LOCAL file should be stored zstd-compressed.
> **Called by:** braindance/utils/data_manager/utils/data_loading/results_cache.py:503 (named-call hint); braindance/utils/data_manager/utils/data_loading/results_cache.py:601 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:177`
### `_zstd_migrate_enabled() -> bool`
> unclear — see source
> **Called by:** braindance/utils/data_manager/utils/data_loading/results_cache.py:544 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:184`
### `_zstd_compress(blob: bytes, level: int=_ZSTD_LEVEL) -> bytes`
> unclear — see source
> **Called by:** braindance/utils/data_manager/utils/data_loading/results_cache.py:510 (named-call hint); braindance/utils/data_manager/utils/data_loading/results_cache.py:571 (named-call hint); braindance/utils/data_manager/utils/data_loading/results_cache.py:605 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:188`
### `_zstd_decompress(blob: bytes) -> bytes`
> unclear — see source
> **Called by:** braindance/utils/data_manager/utils/data_loading/results_cache.py:471 (named-call hint); braindance/utils/data_manager/utils/data_loading/results_cache.py:533 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:194`
### `_param_key(params: Dict[str, Any]) -> str`
> Generate a parameter key suffix for file naming. Examples: {'bin_ms': 20} → '_20ms' {'bin_ms': 20, 'method': 'gaussian'} → '_20ms_gaussian' {} → '' Simple params get simple suffixes. Complex params use hash.
> **Called by:** braindance/utils/data_manager/utils/data_loading/results_cache.py:285 (named-call hint); braindance/utils/data_manager/utils/data_loading/results_cache.py:298 (named-call hint); braindance/utils/data_manager/utils/data_loading/results_cache.py:303 (named-call hint); braindance/utils/data_manager/utils/data_loading/results_cache.py:374 (named-call hint); braindance/utils/data_manager/utils/data_loading/results_cache.py:646 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:198`
### `get_auto_upload_enabled() -> bool`
> Check if auto-upload is enabled via environment variable. Set BRAINDANCE_AUTO_UPLOAD=1 to enable (for container deployments).
> **Called by:** braindance/utils/data_manager/utils/data_loading/recording.py:183 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/results_cache.py:716`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| BRAINDANCE_AUTO_UPLOAD, BRAINDANCE_DISABLE_ZSTD, and BRAINDANCE_ZSTD_MIGRATE accept 1/true/yes; ResultsCache defaults auto_upload=False, auto_download=True, and NRP_S3_ENDPOINT. |
| Cache payloads are keyed by result name plus sorted parameters; binned_fr/latents_/val_metrics_pred* use zstd wire objects when available, with local compression for configured prefixes. |

## Data Shapes
- Payload dict[str,array/object] saved to .npz or .npz.zst; manifest.json records name, params, filename, shape, dtype, created_at.

## Notes
- Local compression applies binned_fr prefix; wire compression additionally val_metrics_pred, latents_, popfr_latents_.
- Parameter suffix abbreviates values and can collide; complex-only parameters use truncated MD5.
- Reading NPZ permits pickle; caches must be trusted.
- S3 clients are shared per endpoint under a lock to bound connection-pool file descriptors across catalog fan-out.
- Read-side zstd eligibility is independent of the zstandard package so compressed objects are found; missing zstandard raises a named ImportError instead of becoming a cache miss.
