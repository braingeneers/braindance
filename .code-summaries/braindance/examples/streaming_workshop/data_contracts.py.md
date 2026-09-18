# data_contracts.py

**Path:** `braindance/examples/streaming_workshop/data_contracts.py`
**Module:** `braindance.examples.streaming_workshop.data_contracts`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Maps known phase-data names to JSON-safe type, shape and semantic descriptions for the UI. Metadata documents expected values and does not perform runtime shape validation.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.experiment_spec` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.native_catalog` — import consumer hint; not a proven runtime call.
- **Shared data:** phase_catalog and native_catalog attach this metadata to canonical phase inputs and outputs for builder/code reference displays.

## Dependencies
None

## Classes
None

## Functions
### `data_metadata(keys)`
> Return documented types/shapes for selected phase-data keys and an explicit runtime-defined fallback for unknown outputs.
> **Called by:** braindance/examples/streaming_workshop/experiment_spec.py:38 (named-call hint); braindance/examples/streaming_workshop/experiment_spec.py:39 (named-call hint); braindance/examples/streaming_workshop/native_catalog.py:139 (named-call hint); braindance/examples/streaming_workshop/native_catalog.py:140 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/data_contracts.py:4`

## Config / CLI
None

## Data Shapes
- recording_baseline_hz: list[float], (n_channels,), mean Hz; channels may represent detected channels or sorted units.
- response_probe_hz: list[list[float]], (2,n_channels), stimulus-minus-sham rate changes; response_probe_trials: list[list[int]], (2,2), stimulated/sham counts.
- environment_episodes is scalar int; environment_reward scalar float; recording_file str path; recording_duration scalar float seconds.
- Returned mapping associates each requested key with type, shape and description; unknown keys use Any, empty shape and a runtime-inspection explanation.

## Notes
- The known metadata table is local to each call; no imports, file I/O or global mutable registry.
