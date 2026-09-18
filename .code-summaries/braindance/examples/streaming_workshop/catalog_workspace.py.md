# catalog_workspace.py

**Path:** `braindance/examples/streaming_workshop/catalog_workspace.py`
**Module:** `braindance.examples.streaming_workshop.catalog_workspace`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Combines the configured BrainDance catalog with discovered workshop/tutorial experiments and user-added paths, persisting additions atomically.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.main` — import consumer hint; not a proven runtime call.
- **Uses:** `get_catalog_path` from `braindance.config` — imports (static evidence).
- **Uses:** `get_data_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `discover_experiments` from `braindance.examples.streaming_workshop.discovery` — imports (static evidence).
- **Uses:** `Recording` from `braindance.utils.data_manager` — imports (static evidence).
- **Uses:** `load_catalog` from `braindance.utils.data_manager` — imports (static evidence).
- **Shared data:** Uses config paths, `load_catalog`, `Recording`, and discovery; served by `/api/catalog`.

## Dependencies
- `braindance.config.get_catalog_path` — intra-repo import; source import evidence.
- `braindance.config.get_data_dir` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.discovery.discover_experiments` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.Recording` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.load_catalog` — intra-repo import; source import evidence.

## Classes
### CatalogWorkspace()
> Maintain the local experiment index beside configured catalog rows.
**Source:** `braindance/examples/streaming_workshop/catalog_workspace.py:12`
**Kind:** class. **Instantiated by:** braindance/examples/streaming_workshop/main.py:82 (named-call hint)
**Constructor:** `__init__(self, output_dir)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `index_path` | inferred at runtime | `self.output_dir / 'catalog_experiments.json'` |
| `lock` | inferred at runtime | `threading.RLock()` |
| `output_dir` | inferred at runtime | `Path(output_dir)` |
**Methods:**
#### `control(self, command)`
> Add paths, resolve catalog recordings, discover experiments, and return merged metadata.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/examples/streaming_workshop/catalog_workspace.py:18`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Commands: `refresh` and `add`; additions accept directories, JSON, HDF5, or NPZ. |

## Data Shapes
- Returns entries, warnings, catalog/index paths, and ISO UTC refresh time.

## Notes
- Normalizes catalog IDs to strings and clears each `Recording` cache.
- Remote paths remain unavailable until locally resolved.
