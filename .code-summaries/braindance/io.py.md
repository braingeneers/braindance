# io.py

**Path:** `braindance/io.py`
**Module:** `braindance.io`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Provides one smart-open wrapper for local, HTTP, and S3 paths. S3 URLs receive an explicit boto3 client routed to the NRP endpoint unless the caller already supplied a client.

## Connections
- **Used by:** `braindance.analysis.data_loader` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.analysis.mapping` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.catalogging.experiment` — import consumer hint; not a proven runtime call.
- **Shared data:** Delegates to `smart_open.open` and conditionally constructs a boto3 S3 client.

## Dependencies
- `boto3` — external or unresolved local import; source import evidence.
- `smart_open` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `open_file(uri, mode='r', *, transport_params=None, **kwargs)`
> Open local, HTTP, or S3 data while applying BrainDance's default NRP routing only to S3 paths.
> **Called by:** braindance/analysis/data_loader.py:460 (named-call hint); braindance/analysis/data_loader.py:499 (named-call hint); braindance/analysis/data_loader.py:562 (named-call hint); braindance/analysis/data_loader.py:705 (named-call hint); braindance/analysis/mapping.py:167 (named-call hint); braindance/analysis/mapping.py:17 (named-call hint); braindance/utils/data_manager/utils/catalogging/experiment.py:100 (named-call hint); braindance/utils/data_manager/utils/catalogging/experiment.py:74 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/io.py:8`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Environment variable `ENDPOINT` overrides the default `https://s3-west.nrp-nautilus.io` for S3 URLs. |

## Data Shapes
None

## Notes
- The wrapper creates no S3 client for local or HTTP paths; caller transport parameters take precedence.
