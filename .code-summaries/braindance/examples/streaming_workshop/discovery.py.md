# discovery.py

**Path:** `braindance/examples/streaming_workshop/discovery.py`
**Module:** `braindance.examples.streaming_workshop.discovery`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Discovers experiment roots and recording files without crossing nested experiments. Traversal is deterministic, skips hidden directories, and does not follow symlinks.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.analysis_workspace` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.catalog_workspace` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.playback` — import consumer hint; not a proven runtime call.
- **Shared data:** Used by catalog and analysis workspaces.

## Dependencies
None

## Classes
None

## Functions
### `is_experiment(path)`
> Test for a recognized experiment marker.
> **Called by:** braindance/examples/streaming_workshop/discovery.py:16 (named-call hint); braindance/examples/streaming_workshop/discovery.py:27 (named-call hint); braindance/examples/streaming_workshop/playback.py:27 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/discovery.py:6`
### `discover_experiments(root)`
> Return resolved top-level experiment roots beneath a path.
> **Called by:** braindance/examples/streaming_workshop/analysis_workspace.py:177 (named-call hint); braindance/examples/streaming_workshop/catalog_workspace.py:80 (named-call hint); braindance/examples/streaming_workshop/catalog_workspace.py:86 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/discovery.py:10`
### `recording_files(root)`
> Yield supported recordings while pruning nested experiments.
> **Called by:** braindance/examples/streaming_workshop/analysis_workspace.py:194 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/discovery.py:22`

## Config / CLI
None

## Data Shapes
- Markers: `experiment.json`/`experiment_log.json`; recordings: `.h5`, `.hdf5`, `.npz`.

## Notes
- Stops descending at experiment roots; excludes symlinked recordings.
