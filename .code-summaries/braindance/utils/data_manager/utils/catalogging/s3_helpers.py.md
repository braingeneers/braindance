# s3_helpers.py

**Path:** `braindance/utils/data_manager/utils/catalogging/s3_helpers.py`
**Module:** `braindance.utils.data_manager.utils.catalogging.s3_helpers`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Provides S3 path, client, existence, CSV, and SpikeData helpers used during catalog construction. It can cache full pickle downloads locally and probe representative spike files to estimate per-chip or per-experiment-folder unit counts.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.catalogging` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.catalogging.experiment` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.catalogging.generator` — import consumer hint; not a proven runtime call.
- **Uses:** `get_data_dir` from `braindance` — imports (static evidence).
- **Uses:** `load_spike_pickle` from `braindance.spike_data` — imports (static evidence).
- **Uses:** `S3_ENDPOINTS` from `braindance.utils.data_manager.utils.catalogging.config` — imports (static evidence).
- **Uses:** `S3Loader` from `braindance.utils.data_manager.utils.data_loading.s3_loader` — imports (static evidence).
- **Shared data:** Uses boto3/botocore, optional braingeneers `s3wrangler` listing, `S3Loader`, and legacy-compatible `load_spike_pickle`.

## Dependencies
- `boto3` — external or unresolved local import; source import evidence.
- `botocore.client.Config` — external or unresolved local import; source import evidence.
- `braindance.get_data_dir` — intra-repo import; source import evidence.
- `braindance.spike_data.load_spike_pickle` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.config.S3_ENDPOINTS` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.data_loading.s3_loader.S3Loader` — intra-repo import; source import evidence.
- `braingeneers.utils.s3wrangler` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `get_s3_client(bucket_name=None, endpoint_url=None)`
> Create a boto3 S3 client using an explicit or bucket-mapped endpoint.
> **Called by:** braindance/utils/data_manager/utils/catalogging/__init__.py:272 (named-call hint); braindance/utils/data_manager/utils/catalogging/generator.py:59 (named-call hint); braindance/utils/data_manager/utils/catalogging/s3_helpers.py:145 (named-call hint); braindance/utils/data_manager/utils/catalogging/s3_helpers.py:82 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/s3_helpers.py:23`
### `parse_s3_path(s3_path)`
> Validate and split an S3 URI into bucket and key.
> **Called by:** braindance/utils/data_manager/utils/catalogging/__init__.py:266 (named-call hint); braindance/utils/data_manager/utils/catalogging/s3_helpers.py:142 (named-call hint); braindance/utils/data_manager/utils/catalogging/s3_helpers.py:215 (named-call hint); braindance/utils/data_manager/utils/catalogging/s3_helpers.py:79 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/s3_helpers.py:44`
### `check_s3_file_exists(s3_path, s3_client=None)`
> Probe an S3 object with `head_object`, returning a boolean.
> **Called by:** braindance/utils/data_manager/utils/catalogging/__init__.py:278 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/s3_helpers.py:67`
### `construct_log_path(base_path, chip, experiment)`
> Build the conventional stimulation-log URI for an experiment.
> **Called by:** braindance/utils/data_manager/utils/catalogging/experiment.py:187 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/s3_helpers.py:90`
### `load_log_from_s3(s3_path, s3_client=None)`
> Fetch and parse a log CSV from S3.
> **Called by:** braindance/utils/data_manager/utils/catalogging/experiment.py:190 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/s3_helpers.py:127`
### `construct_spike_data_path(base_path, chip, experiment)`
> Build the conventional spike-data pickle URI.
> **Called by:** braindance/utils/data_manager/utils/catalogging/__init__.py:253 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/s3_helpers.py:157`
### `construct_rt_sort_path(base_path, chip, experiment)`
> Build the conventional RT-Sort pickle URI.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/s3_helpers.py:176`
### `load_spike_data_from_s3(s3_path, verbose=False)`
> Download a complete pickle to the local cache and compatibility-load it.
> **Called by:** braindance/utils/data_manager/utils/catalogging/s3_helpers.py:281 (named-call hint); braindance/utils/data_manager/utils/catalogging/s3_helpers.py:296 (named-call hint); braindance/utils/data_manager/utils/catalogging/s3_helpers.py:357 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/s3_helpers.py:195`
### `get_num_units_for_chip(base_path, chip, sample_experiment=None, verbose=False)`
> Find a representative spike-data or RT-Sort file under a chip and return its unit count.
> **Called by:** braindance/utils/data_manager/utils/catalogging/generator.py:378 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/s3_helpers.py:248`
### `get_num_units_for_experiment_folder(base_path, chip, experiment_folder, verbose=False)`
> Find a spike-data file within a named experiment folder and return its unit count.
> **Called by:** braindance/utils/data_manager/utils/catalogging/__init__.py:203 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/catalogging/s3_helpers.py:311`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Bucket-specific endpoints come from `S3_ENDPOINTS`; clients force signature version `s3v4`. |
| Cached spike-data downloads use `get_data_dir()/.s3_cache` and key only by the remote basename. |

## Data Shapes
- S3 paths split into `(bucket, key)`; loaded logs are pandas DataFrames.
- Spike-data/RT-Sort pickles are expected to expose an integer-like `N` unit count.

## Notes
- Most network errors are swallowed and returned as `False` or `None`; verbose mode prints diagnostics.
- The cache basename scheme can collide when different S3 keys share a filename.
