# analysis_workspace.js

**Path:** `braindance/examples/streaming_workshop/analysis_workspace.js`
**Module:** `braindance.examples.streaming_workshop.analysis_workspace.js`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Builds the browser Analysis workspace for loading experiments, selecting recorded phases/files, configuring bounded analyses, polling jobs, rendering plots, and downloading JSON results.

## Connections
- **Shared data:** Posts to `/api/analysis` and exposes `window.workshopAnalysis.load` to the catalog.

## Dependencies
None

## Classes
None

## Functions
### `renderTool()`
> Render the selected tool form and capability state.
**Source:** `braindance/examples/streaming_workshop/analysis_workspace.js:41`
### `renderWorkspace(data)`
> Render loaded phases, files, logs, and warnings.
**Source:** `braindance/examples/streaming_workshop/analysis_workspace.js:70`
### `browse(path)`
> Browse server-local folders while rejecting stale responses.
**Source:** `braindance/examples/streaming_workshop/analysis_workspace.js:82`
### `load(source, path)`
> Load an experiment and notify catalog listeners.
**Source:** `braindance/examples/streaming_workshop/analysis_workspace.js:105`
### `renderJob(job)`
> Update a retained job card and its results.
**Source:** `braindance/examples/streaming_workshop/analysis_workspace.js:117`
### `drawPlot(plot)`
> Draw supported JSON plots on an accessible canvas.
**Source:** `braindance/examples/streaming_workshop/analysis_workspace.js:131`
### `refresh()`
> Poll until no queued or running jobs remain.
**Source:** `braindance/examples/streaming_workshop/analysis_workspace.js:155`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Forms expose raw/spike windows, STTC delta, latency bins, evoked windows/blanking, and RT-Sort settings. |

## Data Shapes
- Consumes workspace `phases`/`files` with capabilities; jobs have status/progress/params and results with `summary` and `plots`.
- Plots use `x`, `y`, `z`, or `series` arrays and axis labels.

## Notes
- Polls once per second only while jobs are active.
