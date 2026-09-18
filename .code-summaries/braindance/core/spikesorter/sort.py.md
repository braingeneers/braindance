# sort.py

**Path:** `braindance/core/spikesorter/sort.py`
**Module:** `braindance.core.spikesorter.sort`
**Feature Area:** `Spike Sorting`
**Entry point:** no — library or imported component

## Overview
Defines the minimal base interface and generic offline streaming loop for spike sorters. It normalizes recordings to channel-by-sample traces and repeatedly delegates frame-by-channel chunks to a subclass's `running_sort` method.

## Connections
- **Used by:** `braindance.core.spikesorter.burst` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.spikesorter.thresh` — import consumer hint; not a proven runtime call.
- **Uses:** `load_recording` from `braindance.core.spikesorter.rt_sort` — imports (static evidence).
- **Uses:** `save_traces` from `braindance.core.spikesorter.rt_sort` — imports (static evidence).
- **Shared data:** Uses `rt_sort.load_recording` and `rt_sort.save_traces` to normalize SpikeInterface recordings and file paths.

## Dependencies
- `braindance.core.spikesorter.rt_sort.load_recording` — intra-repo import; source import evidence.
- `braindance.core.spikesorter.rt_sort.save_traces` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `spikeinterface.core.BaseRecording` — external or unresolved local import; source import evidence.
- `tqdm.tqdm` — external or unresolved local import; source import evidence.

## Classes
### Sort()
> Provide shared sorter state and an offline chunking adapter for concrete spike-sort implementations.
**Source:** `braindance/core/spikesorter/sort.py:10`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, num_units, samp_freq)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `latest_frame` | inferred at runtime | `0` |
| `num_units` | inferred at runtime | `num_units` |
| `samp_freq` | inferred at runtime | `samp_freq` |
**Methods:**
#### `reset(self)`
> Reset the accumulated frame clock.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/sort.py:21`
#### `sort_offline(self, recording, buffer_size, inter_path=None, recording_window_ms=None, reset=True, verbose=False)`
> Normalize an input recording and aggregate chunk-level detections into per-unit spike trains.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/sort.py:24`
#### `running_sort(self, obs) -> list`
> Required subclass hook for stateful sorting of one observation chunk.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/sort.py:95`
#### `sort_chunk(self, obs) -> list`
> Required subclass hook for sorting a standalone chunk.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/sort.py:98`
#### `to_spikedata(self)`
> Required subclass hook for exporting sorter results as SpikeData.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/sort.py:101`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `sort_offline(recording, buffer_size, inter_path=None, recording_window_ms=None, reset=True, verbose=False)` controls chunking, temporary trace location, windowing, and state reset. |

## Data Shapes
- Accepts NumPy traces shaped channels x samples; each chunk is transposed to frames x channels for `running_sort`.
- Returns a one-dimensional object array of length `num_units`, with each element an array of spike times.

## Notes
- Temporary scaled traces are written in the current directory and deleted when no intermediate path is supplied.
- `running_sort`, `sort_chunk`, and `to_spikedata` are abstract by convention and raise `NotImplementedError`.
