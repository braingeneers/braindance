# data_context.py

**Path:** `braindance/utils/data_manager/utils/data_loading/data_context.py`
**Module:** `braindance.utils.data_manager.utils.data_loading.data_context`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Implements attribute/dict-style storage for derived results with lazy disk loading and automatic serialization by value type. In-memory assignments collect lightweight metadata, while explicit `save` writes values and sidecar provenance files.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.data_loading.catalog` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.data_loading.recording` — import consumer hint; not a proven runtime call.
- **Uses:** `load_spike_pickle` from `braindance.spike_data` — imports (static evidence).
- **Shared data:** Recording uses DataContext for `rec.results`; pickle loading delegates to `load_spike_pickle` for legacy SpikeData compatibility.

## Dependencies
- `braindance.spike_data.load_spike_pickle` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.

## Classes
### DataContext()
> Expose lazily persisted analysis results through attribute and mapping syntax.
**Source:** `braindance/utils/data_manager/utils/data_loading/data_context.py:41`
**Kind:** class. **Instantiated by:** braindance/utils/data_manager/utils/data_loading/recording.py:186 (named-call hint); braindance/utils/data_manager/utils/data_loading/recording.py:2019 (named-call hint); braindance/utils/data_manager/utils/data_loading/recording.py:2533 (named-call hint)
**Constructor:** `__init__(self, path: Optional[Path]=None, overwrite_existing: bool=True)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_data` | Dict[str, Any] | `{}` |
| `_loaded_from_disk` | inferred at runtime | `set()` |
| `_metadata` | Dict[str, Dict] | `{}` |
| `_overwrite_existing` | inferred at runtime | `overwrite_existing` |
| `_path` | inferred at runtime | `Path(path) if path else None` |
**Methods:**
#### `__setattr__(self, name: str, value: Any)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/data_loading/data_context.py:65`
#### `__getattr__(self, name: str) -> Any`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/data_context.py:82`
#### `__setitem__(self, key: str, value: Any)`
> Support dict-style assignment: ctx['key'] = value
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/data_context.py:93`
#### `__getitem__(self, key: str) -> Any`
> Support dict-style access: value = ctx['key']
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/data_context.py:97`
#### `__contains__(self, name: str) -> bool`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/data_context.py:101`
#### `_exists_on_disk(self, key: str) -> bool`
> Check if a key exists as a file on disk.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/data_context.py:109`
#### `_try_load_key(self, key: str) -> bool`
> Attempt to load a single key from disk. Returns True if successful.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/data_loading/data_context.py:118`
#### `get(self, key: str, default: Any=None) -> Any`
> Return a memory or disk-backed value with a default on absence.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/data_context.py:150`
#### `keys(self) -> List[str]`
> List keys found in memory or supported persistence files.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/data_context.py:157`
#### `save(self, path: Optional[Path]=None, keys: Optional[List[str]]=None)`
> Serialize selected or all in-memory values into the configured directory.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/data_context.py:166`
#### `_save_value(self, path: Path, key: str, value: Any)`
> Save a single value to disk with auto-format detection.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/utils/data_manager/utils/data_loading/data_context.py:181`
#### `_save_index(self, path: Path, key: str, value: Any, note: str=None)`
> Save provenance metadata for a single key.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/utils/data_manager/utils/data_loading/data_context.py:221`
#### `load(self, path: Optional[Path]=None, keys: Optional[List[str]]=None)`
> Point the context at a directory and optionally eagerly load selected keys.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/data_loading/data_context.py:245`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `DataContext(path=None, overwrite_existing=True)` selects the persistence directory and overwrite behavior. |

## Data Shapes
- Arrays use `.npy`; all-array dictionaries use `.npz`; simple-index DataFrames use `.csv`; other objects/dicts use trusted `.pkl`.
- Each saved key gets `<key>_meta.json` containing timestamp, type, shape, and optional note.

## Notes
- Failed loads/saves print warnings instead of raising in most paths.
- A key is marked attempted before loading, so a failed lazy load will not be retried until a new context is created.
- Pickle and NumPy object loading assume trusted local files.
