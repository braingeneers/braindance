# waveforms.py

**Path:** `braindance/utils/data_manager/utils/data_loading/waveforms.py`
**Module:** `braindance.utils.data_manager.utils.data_loading.waveforms`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Reads per-unit mean spike waveform NPZ products from the ResultsCache and exposes dict-compatible waveform data with derived time, shape, footprint, activity, and summary conveniences.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.data_loading.recording` — import consumer hint; not a proven runtime call.
- **Shared data:** `Recording.get_waveforms()` resolves waveform cache parameters and returns `Waveforms`; `waveform_s3_path()` exposes the corresponding S3 path.

## Dependencies
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.

## Classes
### Waveforms()
> Wraps one recording’s waveform arrays with dict compatibility, scalar access, shape helpers, time axis, activity mask, footprint, and per-unit summary.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:74`
**Kind:** class. **Instantiated by:** braindance/utils/data_manager/utils/data_loading/recording.py:2272 (named-call hint)
**Constructor:** `__init__(self, data: Dict[str, np.ndarray], params: Optional[Dict[str, Any]]=None, identifier: Optional[str]=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_d` | inferred at runtime | `data` |
| `_identifier` | inferred at runtime | `identifier` |
| `_params` | inferred at runtime | `dict(params) if params else dict(WAVEFORM_PARAMS)` |
**Methods:**
#### `__getitem__(self, key: str) -> np.ndarray`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:93`
#### `__contains__(self, key: str) -> bool`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:96`
#### `__iter__(self) -> Iterator[str]`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:99`
#### `__len__(self) -> int`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:102`
#### `keys(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:105`
#### `items(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:108`
#### `get(self, key: str, default=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:111`
#### `to_dict(self) -> Dict[str, np.ndarray]`
> The raw npz dict, unmodified.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:114`
#### `__getattr__(self, name: str) -> Any`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:119`
#### `__dir__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:134`
#### `__repr__(self) -> str`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:137`
#### `params(self) -> Dict[str, Any]`
> The cache params this was loaded with.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:148`
#### `n_units(self) -> int`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:153`
#### `n_footprint_channels(self) -> int`
> K — how many channels per unit were kept. NOT a fixed radius: routed electrodes are sparse, so mask with `footprint_dist_um` if you need one.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:157`
#### `n_samples(self) -> int`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:163`
#### `t_ms(self) -> np.ndarray`
> Time axis in ms, t=0 at the spike. The window is `n_before + n_after + 1` samples — the spike sample itself is the +1, so at the defaults that is 20 + 40 + 1 = 61. Taken from the array's own shape so it cannot drift out of sync with the data.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:167`
#### `peak(self) -> np.ndarray`
> (n_units, n_samples) — the peak-channel waveform of every unit. Footprint columns are ordered by DISTANCE from the peak channel, so column 0 is always the peak channel. They are deliberately not ordered by amplitude: ranking by amplitude pulls in other neurons firing synchronously during network bursts.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:180`
#### `peak_p2p_uv(self) -> np.ndarray`
> (n_units,) — peak-to-peak amplitude on the peak channel, uV.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:191`
#### `live(self) -> np.ndarray`
> Boolean mask of units with at least one averaged spike. ALWAYS gate on this. Zero-spike units are normal and common — RT-Sort fits its sequences on the base `exp1` recording and applies them to every `_cont_*`, so many sequences are simply silent in any one recording. The catalog's `num_units` counts the sequence set, not the active units.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:197`
#### `footprint(self, unit: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
> One unit's footprint as `(waveforms, channel_xy, dist_um)`, shapes `(K, n_samples)`, `(K, 2)`, `(K,)`.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:207`
#### `summary(self, live_only: bool=False) -> 'pd.DataFrame'`
> One row per unit: spike counts, guard counts, peak channel, p2p. `n_spikes` is capped by step1's `max_spikes` (2000 by default); `n_spikes_total` is the uncapped train.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:217`

## Functions
### `waveform_params(ms_before: float=1.0, ms_after: float=2.0, n_footprint: int=64, artifact_guard_ms: float=10.0) -> Dict[str, float]`
> Build the cache params dict from step1's CLI arguments, named as step1 names them. Mirrors `step1_extract_waveforms.py:451`.
> **Called by:** braindance/utils/data_manager/utils/data_loading/recording.py:2207 (named-call hint); braindance/utils/data_manager/utils/data_loading/recording.py:2256 (named-call hint); braindance/utils/data_manager/utils/data_loading/waveforms.py:71 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:46`
### `waveform_s3_path(rec, **kwargs) -> Optional[str]`
> Builds the S3 path for a recording’s parameterized waveform NPZ product.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/waveforms.py:62`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `WAVEFORM_PARAMS`: default extraction key uses msb=1.0, msa=2.0, k=64, guard=10.0. |
| `waveform_params(...)`: overrides before/after window (ms), footprint channel count, and artifact guard (ms). |

## Data Shapes
- All-unit `waveforms` has shape (n_units, K, n_samples); one-unit `footprint(unit)` returns (K, n_samples) waveforms plus (K, 2) channel coordinates and (K,) distances.
- `peak` is (n_units, n_samples), `peak_p2p_uv` and `live` are (n_units,), and `summary()` returns one row per unit. Time is milliseconds with t=0 at the spike.

## Notes
- Waveform files are read from ResultsCache and never computed on a cache miss; absent products return None.
- Dict compatibility preserves raw NPZ arrays while scalar attributes unwrap 0-D arrays. Gate analyses with `live` because zero-spike units are common.
