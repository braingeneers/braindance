# analysis_evoked.py

**Path:** `braindance/examples/streaming_workshop/analysis_evoked.py`
**Module:** `braindance.examples.streaming_workshop.analysis_evoked`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Computes bounded, stimulation-aligned response plots and descriptive matrices from raw Maxwell traces or supplied spike trains. Raw mode applies BrainDance's cubic artifact-removal kernel and threshold detection.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.analysis_workspace` — import consumer hint; not a proven runtime call.
- **Uses:** `cubic_fit5` from `braindance.core.artifact_removal` — imports (static evidence).
- **Shared data:** Called by `AnalysisWorkspace._analyze`; raw mode uses `artifact_removal.cubic_fit5` and SciPy `find_peaks`.

## Dependencies
- `braindance.core.artifact_removal.cubic_fit5` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `scipy.signal.find_peaks` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `analyze_evoked(events_ms, patterns, *, spike_times_ms=None, unit_ids=None, selected_units=None, raw_reader=None, sampling_frequency=20000, params=None, progress=None)`
> Align repeated responses to labeled stimuli, blank artifacts, detect raw peaks when needed, and return browser-ready overlap, raster, rate, and matrix data.
> **Called by:** braindance/examples/streaming_workshop/analysis_workspace.py:319 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/analysis_evoked.py:11`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `params`: `pre_ms=20`, `post_ms=200`, `blank_ms=3`, `first_order_ms=10`, `bin_ms=2`, `threshold_sigma=5`, `max_events=100`, `max_units=64`, and raw `channels`/`channel`. |

## Data Shapes
- `events_ms` and `patterns` are matching one-dimensional trial arrays; spike mode takes matching `spike_times_ms` and `unit_ids`.
- Returns plot dictionaries plus pattern × unit short probability, late mean spike, and total probability matrices.

## Notes
- The artifact interval is omitted, not reported as zero activity; matrices are descriptive associations, not causal proof.
- Raw detections are channel events rather than sorted neurons; work is capped at 20 million units.
