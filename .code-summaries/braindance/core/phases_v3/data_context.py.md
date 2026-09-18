# data_context.py

**Path:** `braindance/core/phases_v3/data_context.py`
**Module:** `braindance.core.phases_v3.data_context`
**Feature Area:** `Experiment Phases`
**Entry point:** no — library or imported component

## Overview
Stores experiment phase data with attribute access, overwrite policy, and explicit typed persistence. Records data metadata/index/history and provides basic validators and phase dependency/access tracking.

## Connections
- **Used by:** `braindance.core.phases_v3.experiment_v3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phase_base_v3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.experiment_spec` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.native_runner` — import consumer hint; not a proven runtime call.
- **Shared data:** Experiment.data stores phase outputs and inputs; DataDependencyTracker records read/write flow.

## Dependencies
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.

## Classes
### DataContext()
> Dynamic data storage with attribute access.
**Source:** `braindance/core/phases_v3/data_context.py:15`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/experiment_v3.py:349 (named-call hint); braindance/core/phases_v3/experiment_v3.py:97 (named-call hint); braindance/examples/streaming_workshop/experiment_spec.py:94 (named-call hint); braindance/examples/streaming_workshop/native_runner.py:164 (named-call hint); braindance/examples/streaming_workshop/native_runner.py:168 (named-call hint); braindance/examples/streaming_workshop/native_runner.py:188 (named-call hint)
**Constructor:** `__init__(self, overwrite_existing: bool=False)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_data` | Dict[str, Any] | `{}` |
| `_metadata` | Dict[str, Dict] | `{}` |
| `_overwrite_existing` | inferred at runtime | `overwrite_existing` |
**Methods:**
#### `__setattr__(self, name: str, value: Any)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/data_context.py:29`
#### `__getattr__(self, name: str) -> Any`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/data_context.py:46`
#### `__contains__(self, name: str) -> bool`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/data_context.py:51`
#### `__repr__(self) -> str`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/data_context.py:54`
#### `keys(self) -> List[str]`
> Get all data keys.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/data_context.py:65`
#### `get(self, key: str, default: Any=None) -> Any`
> Get data with default value.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/data_context.py:69`
#### `has_data(self) -> bool`
> Check if any data is stored.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/data_context.py:73`
#### `data_summary(self) -> Dict[str, str]`
> Get a summary of stored data types and shapes.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/data_context.py:77`
#### `update(self, data: Dict[str, Any], overwrite: bool=None)`
> Update multiple data values at once.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/data_context.py:89`
#### `set_overwrite_policy(self, overwrite_existing: bool)`
> Set the overwrite policy for future assignments.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/data_context.py:109`
#### `get_overwrite_policy(self) -> bool`
> Get the current overwrite policy.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/data_context.py:113`
#### `force_set(self, key: str, value: Any)`
> Replace value despite current overwrite policy.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/data_context.py:117`
#### `clear(self)`
> Clear all data.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/data_context.py:126`
#### `_is_json_safe(value) -> bool`
> Return True if *value* can be losslessly represented in JSON.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/data_context.py:132`
#### `save(self, path: Path, keys: Optional[List[str]]=None, overwrite_files: bool | None=None, strict: bool=False)`
> Persist selected values and merged index; strict mode propagates failures.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/phases_v3/data_context.py:140`
#### `load(self, path: Path, keys: Optional[List[str]]=None)`
> Load data context from disk.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/data_context.py:287`
### DataValidator()
> Validates data types and constraints.
**Source:** `braindance/core/phases_v3/data_context.py:368`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `validate_type(value: Any, expected_type: type) -> bool`
> Check if value matches expected type.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/data_context.py:374`
#### `validate_shape(value: Any, expected_shape: tuple) -> bool`
> Check if array-like value has expected shape.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/data_context.py:379`
#### `validate_range(value: Any, min_val: float=None, max_val: float=None) -> bool`
> Check if numeric value is within range.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/data_context.py:386`
### DataDependencyTracker()
> Track declared producers/consumers and observed key accesses.
**Source:** `braindance/core/phases_v3/data_context.py:403`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/experiment_v3.py:111 (named-call hint)
**Constructor:** `__init__(self)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `data_flow` | List[tuple] | `[]` |
| `dependencies` | Dict[str, List[str]] | `{}` |
| `productions` | Dict[str, List[str]] | `{}` |
**Methods:**
#### `add_phase(self, phase_name: str, requires: List[str], provides: List[str])`
> Register a phase's data dependencies.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/data_context.py:413`
#### `record_access(self, phase_name: str, key: str, access_type: str)`
> Record data access for tracking.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/data_context.py:418`
#### `get_producers(self, key: str) -> List[str]`
> Find which phases produce a given data key.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/data_context.py:422`
#### `get_consumers(self, key: str) -> List[str]`
> Find which phases consume a given data key.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/data_context.py:430`
#### `visualize_flow(self) -> str`
> Generate a simple text visualization of data flow.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/data_context.py:438`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| overwrite_existing=False; save(path,keys,overwrite_files,strict=False). |

## Data Shapes
- JSON-safe values in results.json; ndarray .npy, array-containing dict .npz, DataFrame/objects .pkl; data_metadata.json/data_index.json.

## Notes
- Separate from data_manager.DataContext; this implementation eagerly loads and defaults to retaining old keys.
- Saving RTSort sets live object's model=None; JSON-safe values merge regardless of overwrite_files.
- Pickle and allow_pickle require trusted persisted data.
