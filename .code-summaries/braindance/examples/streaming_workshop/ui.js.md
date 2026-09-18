# ui.js

**Path:** `braindance/examples/streaming_workshop/ui.js`
**Module:** `braindance.examples.streaming_workshop.ui.js`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Organizes the single-page workshop into accessible views, manages source selection/uploads, themes, participant-code highlighting, navigation, and phase progress.

## Connections
- **Shared data:** Uses global state/control helpers from `watcher.js`; calls upload and Python verification APIs; hosts containers filled by later scripts.

## Dependencies
None

## Classes
None

## Functions
### `updateSourceUI()`
> Show source-specific fields and simulator availability.
**Source:** `braindance/examples/streaming_workshop/ui.js:46`
### `uploadSourceFile(file, kind)`
> Upload one replay/config file and retain previous selection on failure.
**Source:** `braindance/examples/streaming_workshop/ui.js:64`
### `showView(id)`
> Switch views, navigation state, URL hash, and contextual controls.
**Source:** `braindance/examples/streaming_workshop/ui.js:95`
### `setTheme(theme)`
> Apply and persist light/dark theme.
**Source:** `braindance/examples/streaming_workshop/ui.js:112`
### `highlightPython()`
> Refresh safe participant-code highlighting and verification staleness.
**Source:** `braindance/examples/streaming_workshop/ui.js:124`
### `paintPython(element,code)`
> Tokenize a small Python subset into safe DOM text spans.
**Source:** `braindance/examples/streaming_workshop/ui.js:131`
### `updateNavigation()`
> Update controls, native log, and phase progress from session state.
**Source:** `braindance/examples/streaming_workshop/ui.js:173`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Replay accepts H5/HDF5/NPY and live config accepts CFG through authenticated upload endpoints. |

## Data Shapes
- Creates a `views` map for playground, monitor, experiment, sequence, functions, simulation, catalog, analysis, and settings.

## Notes
- Syntax highlighting builds text nodes so participant source is never interpreted as HTML.
- Responsive navigation closes below 750 px.
