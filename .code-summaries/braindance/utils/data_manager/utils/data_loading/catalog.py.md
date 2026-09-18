# catalog.py

**Path:** `braindance/utils/data_manager/utils/data_loading/catalog.py`
**Module:** `braindance.utils.data_manager.utils.data_loading.catalog`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Wraps catalog metadata in a filterable, naturally ordered collection of lazily created Recording objects. Supports metadata masks, grouped catalogs, ordered serial/threaded mapping, and batch access to recording results.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.data_loading.recording` — import consumer hint; not a proven runtime call.
- **Uses:** `get_catalog_path` from `braindance.config` — imports (static evidence).
- **Uses:** `DataContext` from `braindance.utils.data_manager.utils.data_loading.data_context` — imports (static evidence).
- **Uses:** `Recording` from `braindance.utils.data_manager.utils.data_loading.recording` — imports (static evidence).
- **Shared data:** load_catalog resolves configured catalog; RecordingCatalog creates Recording; BatchResults accesses rec.results.

## Dependencies
- `braindance.config.get_catalog_path` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.data_loading.data_context.DataContext` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.data_loading.recording.Recording` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.

## Classes
### BatchResults()
> Collect named results across recordings.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:96`
**Kind:** class. **Instantiated by:** braindance/utils/data_manager/utils/data_loading/catalog.py:250 (named-call hint)
**Constructor:** `__init__(self, catalog: 'RecordingCatalog')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_catalog` | inferred at runtime | `catalog` |
**Methods:**
#### `__getattr__(self, name: str) -> List[Any]`
> Access a result across all recordings.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:107`
#### `__contains__(self, name: str) -> bool`
> Check if any recording has this result.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:115`
#### `available(self) -> Dict[str, int]`
> Get count of how many recordings have each result.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:122`
### RecordingCatalog()
> A collection of recordings with list-like and pandas-like access.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:131`
**Kind:** class. **Instantiated by:** braindance/utils/data_manager/utils/data_loading/catalog.py:194 (named-call hint); braindance/utils/data_manager/utils/data_loading/catalog.py:200 (named-call hint); braindance/utils/data_manager/utils/data_loading/catalog.py:206 (named-call hint); braindance/utils/data_manager/utils/data_loading/catalog.py:286 (named-call hint); braindance/utils/data_manager/utils/data_loading/catalog.py:482 (named-call hint)
**Constructor:** `__init__(self, df: pd.DataFrame, base_path: Optional[Path]=None, legacy_flat: Optional[bool]=None, sort_by_recording: bool=True)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_base_path` | inferred at runtime | `Path(base_path) if base_path else None` |
| `_df` | inferred at runtime | `df.reset_index(drop=True)` |
| `_legacy_flat` | inferred at runtime | `legacy_flat` |
| `_recordings` | Dict[int, Recording] | `{}` |
**Methods:**
#### `__repr__(self) -> str`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:174`
#### `__len__(self) -> int`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:177`
#### `__iter__(self) -> Iterator[Recording]`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:180`
#### `__getitem__(self, idx) -> Union['RecordingCatalog', Recording]`
> Flexible indexing: - int: Return single Recording - slice: Return new RecordingCatalog - boolean array/Series: Pandas-style filtering
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:184`
#### `__getattr__(self, name: str) -> Any`
> Attribute access: - DataFrame columns return pd.Series (for filtering) - 'results' returns BatchResults - Data properties return lists
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:225`
#### `results(self) -> BatchResults`
> Batch access to results across all recordings.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:248`
#### `filter(self, sort_by_recording=True, **kwargs) -> 'RecordingCatalog'`
> Apply column lookups and construct a filtered catalog.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:254`
#### `_apply_filter(self, key: str, value: Any) -> pd.Series`
> Apply a single filter and return boolean mask.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:290`
#### `_sort_by_recording_id(self, df: pd.DataFrame) -> pd.DataFrame`
> Sort DataFrame by experiment name in natural recording order.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:328`
#### `df(self) -> pd.DataFrame`
> Access underlying DataFrame (escape hatch for complex operations).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:350`
#### `columns(self) -> List[str]`
> List of available metadata columns.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:355`
#### `collect(self, attr: str) -> List[Any]`
> Collect an attribute from all recordings.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:361`
#### `apply(self, func: Callable[[Recording], Any], parallel: bool=False, max_workers: Optional[int]=None, on_error: str='raise') -> List[Any]`
> Map a callable over recordings while preserving result order.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:382`
#### `_apply_parallel(self, func: Callable, max_workers: Optional[int], on_error: str) -> List[Any]`
> Parallel execution using ThreadPoolExecutor.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:418`
#### `group_by(self, column: str) -> Dict[Any, 'RecordingCatalog']`
> Group recordings by a column value.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:469`
#### `summary(self) -> pd.Series`
> Get summary statistics about the catalog.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:489`
#### `describe(self)`
> Print a description of the catalog.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:504`
#### `save_all_results(self, keys: Optional[List[str]]=None)`
> Save results for all recordings.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:513`
#### `from_csv(cls, path: Union[str, Path], base_path: Optional[Path]=None) -> 'RecordingCatalog'`
> Load catalog from CSV file.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:521`
#### `from_dataframe(cls, df: pd.DataFrame, base_path: Optional[Path]=None) -> 'RecordingCatalog'`
> Create catalog from existing DataFrame.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:537`

## Functions
### `_experiment_sort_key(exp_name)`
> Sort key for natural experiment ordering.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:41`
### `load_catalog(path: Optional[Union[str, Path]]=None) -> RecordingCatalog`
> Load a recording catalog from CSV.
> **Called by:** braindance/utils/data_manager/utils/data_loading/recording.py:2722 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/catalog.py:545`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| base_path, legacy_flat, sort_by_recording=True; apply parallel/max_workers/on_error. |

## Data Shapes
- Input DataFrame/CSV rows describe recordings; indexing returns Recording or RecordingCatalog; column attributes return Series.

## Notes
- Catalog __getitem__ creates Recording with base_path but does not pass stored legacy_flat.
- Sorting adds recording_num; contains filter uses pandas regex matching.
