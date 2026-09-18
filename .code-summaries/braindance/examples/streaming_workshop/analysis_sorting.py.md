# analysis_sorting.py

**Path:** `braindance/examples/streaming_workshop/analysis_sorting.py`
**Module:** `braindance.examples.streaming_workshop.analysis_sorting`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Optional RT-Sort adapter that discovers units from a baseline, applies the sorter to compatible targets, and emits portable NPZ spike tables and a manifest.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.analysis_workspace` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.batch_sorters` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.global_settings` — import consumer hint; not a proven runtime call.
- **Uses:** `detect_sequences` from `braindance.core.spikesorter.rt_sort` — imports (static evidence).
- **Uses:** `load_recording` from `braindance.core.spikesorter.rt_sort` — imports (static evidence).
- **Uses:** `save_traces` from `braindance.core.spikesorter.rt_sort` — imports (static evidence).
- **Uses:** `get_rt_sort_path` from `braindance` — imports (static evidence).
- **Shared data:** Uses `detect_sequences`, `load_recording`, and `save_traces` from `core.spikesorter.rt_sort`; called by the analysis workspace.

## Dependencies
- `braindance.core.spikesorter.rt_sort.detect_sequences` — intra-repo import; source import evidence.
- `braindance.core.spikesorter.rt_sort.load_recording` — intra-repo import; source import evidence.
- `braindance.core.spikesorter.rt_sort.save_traces` — intra-repo import; source import evidence.
- `braindance.get_rt_sort_path` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `sorting_capabilities()`
> Report dependency/model availability without loading the sorting stack.
> **Called by:** braindance/examples/streaming_workshop/analysis_sorting.py:91 (named-call hint); braindance/examples/streaming_workshop/analysis_workspace.py:160 (named-call hint); braindance/examples/streaming_workshop/analysis_workspace.py:184 (named-call hint); braindance/examples/streaming_workshop/batch_sorters.py:14 (named-call hint); braindance/examples/streaming_workshop/global_settings.py:25 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/analysis_sorting.py:28`
### `validate_params(params=None)`
> Normalize and range-check allowlisted sorting options.
> **Called by:** braindance/examples/streaming_workshop/analysis_sorting.py:81 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/analysis_sorting.py:47`
### `run_sorting(baseline_path, recording_paths, output_dir, params=None, progress=None)`
> Fit RT-Sort on a bounded baseline, sort whole targets, and persist portable spike artifacts with progress metadata.
> **Called by:** braindance/examples/streaming_workshop/analysis_workspace.py:334 (named-call hint); braindance/examples/streaming_workshop/batch_sorters.py:47 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/examples/streaming_workshop/analysis_sorting.py:74`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Options cover device, baseline window in ms, 1–32 processes, stringent/loose thresholds, and minimum activity/sequence criteria. |

## Data Shapes
- NPZ fields: sorted `times_ms`, matching `unit_ids`, `all_unit_ids`, scalar `duration_ms`, `sampling_frequency_hz`, `source_path`, and `source`.
- Baseline and targets require identical sampling frequency, channel IDs, and channel locations.

## Notes
- Creates a UUID output directory and sorter pickle; inputs are unchanged.
- CUDA is checked at launch.
