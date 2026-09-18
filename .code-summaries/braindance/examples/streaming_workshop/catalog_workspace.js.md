# catalog_workspace.js

**Path:** `braindance/examples/streaming_workshop/catalog_workspace.js`
**Module:** `braindance.examples.streaming_workshop.catalog_workspace.js`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Builds searchable, paginated catalog UI for configured recordings, workshop runs, tutorial experiments, and added paths, then hands selections to Analysis.

## Connections
- **Shared data:** Posts to `/api/catalog`; calls `window.workshopAnalysis.load`.

## Dependencies
None

## Classes
None

## Functions
### `render()`
> Filter, paginate, and render catalog rows.
**Source:** `braindance/examples/streaming_workshop/catalog_workspace.js:22`
### `current()`
> Show active-run analysis availability.
**Source:** `braindance/examples/streaming_workshop/catalog_workspace.js:43`
### `select(source,path)`
> Load a catalog entry into Analysis.
**Source:** `braindance/examples/streaming_workshop/catalog_workspace.js:49`
### `refresh(command={action:'refresh'})`
> Refresh or extend the local index.
**Source:** `braindance/examples/streaming_workshop/catalog_workspace.js:57`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Pages show at most 100 filtered entries. |

## Data Shapes
- Entries carry name, project, chip, kind, source, path, availability, metadata, and reason.

## Notes
- Lazily refreshes when opened and reuses workshop state for current-run status.
