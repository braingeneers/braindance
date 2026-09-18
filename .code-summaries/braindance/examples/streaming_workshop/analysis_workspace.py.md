# analysis_workspace.py

**Path:** `braindance/examples/streaming_workshop/analysis_workspace.py`
**Module:** `braindance.examples.streaming_workshop.analysis_workspace`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Thread-safe file-backed analysis service that discovers recorded phases, derives HDF5/NPZ capabilities, runs one bounded background job at a time, and adds sorted spike files to the workspace.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.main` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.sorting_workspace` — import consumer hint; not a proven runtime call.
- **Uses:** `get_data_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `analyze_evoked` from `braindance.examples.streaming_workshop.analysis_evoked` — imports (static evidence).
- **Uses:** `run_sorting` from `braindance.examples.streaming_workshop.analysis_sorting` — imports (static evidence).
- **Uses:** `sorting_capabilities` from `braindance.examples.streaming_workshop.analysis_sorting` — imports (static evidence).
- **Uses:** `discover_experiments` from `braindance.examples.streaming_workshop.discovery` — imports (static evidence).
- **Uses:** `recording_files` from `braindance.examples.streaming_workshop.discovery` — imports (static evidence).
- **Uses:** `native_catalog` from `braindance.examples.streaming_workshop.native_catalog` — imports (static evidence).
- **Uses:** `analysis_source_urls` from `braindance.examples.streaming_workshop.source_links` — imports (static evidence).
- **Shared data:** Uses discovery, native catalog, evoked/sorting adapters, `spikelab.SpikeData`, NumPy, h5py, and pandas; served by `/api/analysis`.

## Dependencies
- `braindance.config.get_data_dir` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.analysis_evoked.analyze_evoked` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.analysis_sorting.run_sorting` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.analysis_sorting.sorting_capabilities` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.discovery.discover_experiments` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.discovery.recording_files` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.native_catalog.native_catalog` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.source_links.analysis_source_urls` — intra-repo import; source import evidence.
- `h5py` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.
- `spikelab.SpikeData` — external or unresolved local import; source import evidence.

## Classes
### AnalysisWorkspace()
> Own selected experiment metadata and serialized browser analysis jobs.
**Source:** `braindance/examples/streaming_workshop/analysis_workspace.py:47`
**Kind:** class. **Instantiated by:** braindance/examples/streaming_workshop/main.py:80 (named-call hint)
**Constructor:** `__init__(self, output_dir)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `executor` | inferred at runtime | `ThreadPoolExecutor(max_workers=1, thread_name_prefix='workshop-analysis')` |
| `jobs` | inferred at runtime | `{}` |
| `lock` | inferred at runtime | `threading.RLock()` |
| `output_dir` | inferred at runtime | `Path(output_dir)` |
| `workspace` | inferred at runtime | `dict(path='', name='', phases=[], files=[], warnings=[])` |
**Methods:**
#### `close(self)`
> Cancel queued futures and close the executor.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/analysis_workspace.py:55`
#### `control(self, command)`
> Validate and dispatch browse/load/run/status commands with defensive copies.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/examples/streaming_workshop/analysis_workspace.py:58`
#### `_file(self, path, phase_id, bounds=None)`
> Inspect a recording and derive raw/spike/overlap/connectivity/latency/sorting capabilities.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/analysis_workspace.py:119`
#### `_load(self, location)`
> Build phase/file metadata from logs, attempts, config, and discovered recordings.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/examples/streaming_workshop/analysis_workspace.py:170`
#### `_worker(self, job_id, row, kind, params)`
> Run one analysis and atomically publish completion or failure.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/analysis_workspace.py:248`
#### `_analyze(self, row, kind, params, progress)`
> Read a bounded window and execute raw, raster, STTC, latency, evoked, or sorting analysis.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/analysis_workspace.py:261`

## Functions
### `_number(params, key, default, low=0, high=1000000000.0)`
> unclear — see source
> **Called by:** braindance/examples/streaming_workshop/analysis_workspace.py:263 (named-call hint); braindance/examples/streaming_workshop/analysis_workspace.py:264 (named-call hint); braindance/examples/streaming_workshop/analysis_workspace.py:265 (named-call hint); braindance/examples/streaming_workshop/analysis_workspace.py:274 (named-call hint); braindance/examples/streaming_workshop/analysis_workspace.py:298 (named-call hint); braindance/examples/streaming_workshop/analysis_workspace.py:301 (named-call hint); braindance/examples/streaming_workshop/analysis_workspace.py:377 (named-call hint); braindance/examples/streaming_workshop/analysis_workspace.py:392 (named-call hint); braindance/examples/streaming_workshop/analysis_workspace.py:400 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/analysis_workspace.py:19`
### `_h5_info(path)`
> unclear — see source
> **Called by:** braindance/examples/streaming_workshop/analysis_workspace.py:127 (named-call hint); braindance/examples/streaming_workshop/analysis_workspace.py:137 (named-call hint); braindance/examples/streaming_workshop/sorting_workspace.py:33 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/analysis_workspace.py:26`
### `_raw(path, info, channel, start, stop)`
> unclear — see source
> **Called by:** braindance/examples/streaming_workshop/analysis_workspace.py:280 (named-call hint); braindance/examples/streaming_workshop/analysis_workspace.py:307 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/analysis_workspace.py:38`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Commands: `browse`, `load`, `run`, `status`; tool parameters cover bounded windows, channels/units, STTC, latency, stimulus logs, and RT-Sort. |

## Data Shapes
- Maxwell HDF5 uses raw datasets and structured spike rows with `frameno`/`frame` plus `channel`; NPZ uses `times_ms`, `unit_ids`, `duration_ms`.
- Jobs contain ID, kind, file metadata, status, progress, result, and error; 50 are retained.

## Notes
- A one-worker executor serializes analysis; loading is refused during active work.
- Caps raw previews at 2 million samples, displays at 20,000 events, files at 1,000, units at 256, and latency bins at 2,000.
