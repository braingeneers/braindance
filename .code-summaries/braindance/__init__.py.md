# __init__.py

**Path:** `braindance/__init__.py`
**Module:** `braindance`
**Feature Area:** `Configuration`
**Entry point:** no — library or imported component

## Overview
Exposes package metadata, configuration accessors and paths for bundled data and RT-Sort detection models. Maintains a named resource registry and publishes the configured data directory.

## Connections
- **Used by:** `braindance.core.maxwell.select_electrodes` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_analysis` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_analysis_2` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_ant` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_ant_cpg` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_foodland` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_loop` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_pacman` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_walker2d` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_walker2d_cpg` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases_analysis_3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.spikedetector.model2` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.analysis_sorting` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.session` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.catalogging.s3_helpers` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.data_loading.recording` — import consumer hint; not a proven runtime call.
- **Uses:** `get_auto_extract_spike_info` from `braindance.config` — imports (static evidence).
- **Uses:** `get_catalog_path` from `braindance.config` — imports (static evidence).
- **Uses:** `get_data_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `set_auto_extract_spike_info` from `braindance.config` — imports (static evidence).
- **Uses:** `set_catalog_path` from `braindance.config` — imports (static evidence).
- **Uses:** `set_data_dir` from `braindance.config` — imports (static evidence).
- **Shared data:** Imports data/catalog/automatic-extraction getters and setters from braindance.config.

## Dependencies
- `braindance.config.get_auto_extract_spike_info` — intra-repo import; source import evidence.
- `braindance.config.get_catalog_path` — intra-repo import; source import evidence.
- `braindance.config.get_data_dir` — intra-repo import; source import evidence.
- `braindance.config.set_auto_extract_spike_info` — intra-repo import; source import evidence.
- `braindance.config.set_catalog_path` — intra-repo import; source import evidence.
- `braindance.config.set_data_dir` — intra-repo import; source import evidence.

## Classes
### Config()
> unclear — see source
**Source:** `braindance/__init__.py:67`
**Kind:** class. **Instantiated by:** braindance/__init__.py:72 (named-call hint)
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `data_dir(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/__init__.py:69`

## Functions
### `get_module_path()`
> Returns the path to the braindance module.
> **Called by:** braindance/__init__.py:29 (named-call hint); braindance/__init__.py:38 (named-call hint); braindance/__init__.py:43 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/__init__.py:13`
### `get_rt_sort_path()`
> Returns the path to the RT-Sort detection models.
> **Called by:** braindance/__init__.py:42 (named-call hint); braindance/core/phases_v3/phases3_ant.py:173 (named-call hint); braindance/core/phases_v3/phases3_ant_cpg.py:192 (named-call hint); braindance/core/phases_v3/phases3_foodland.py:248 (named-call hint); braindance/core/phases_v3/phases3_loop.py:589 (named-call hint); braindance/core/phases_v3/phases3_loop.py:93 (named-call hint); braindance/core/phases_v3/phases3_pacman.py:152 (named-call hint); braindance/core/phases_v3/phases3_walker2d.py:226 (named-call hint); braindance/core/phases_v3/phases3_walker2d_cpg.py:206 (named-call hint); braindance/core/phases_v3/phases_analysis_3.py:199 (named-call hint); braindance/core/phases_v3/phases_analysis_3.py:357 (named-call hint); braindance/examples/streaming_workshop/analysis_sorting.py:34 (named-call hint); braindance/examples/streaming_workshop/session.py:374 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/__init__.py:22`
### `get_data_path()`
> Returns the path to the package's data directory.
> **Called by:** braindance/__init__.py:44 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/__init__.py:31`
### `get_resource_path(resource_name)`
> Returns the path to a specific resource in the package.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/__init__.py:47`
### `__getattr__(name)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/__init__.py:75`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| __version__ = 0.1.9 |
| PACKAGE_RESOURCES: rt_sort_detection_models, module_path, data_path |

## Data Shapes
- get_resource_path returns pathlib.Path and rejects unknown keys

## Notes
- Package version is 0.1.9; data_dir is assigned at import time, so normal attribute reads use that snapshot despite module __getattr__.
