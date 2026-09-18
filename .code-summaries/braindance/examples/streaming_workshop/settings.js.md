# settings.js

**Path:** `braindance/examples/streaming_workshop/settings.js`
**Module:** `braindance.examples.streaming_workshop.settings.js`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Builds the global Settings view for data/catalog/output paths, automatic spike-info extraction, and installed-feature status.

## Connections
- **Shared data:** GETs or POSTs `/api/settings` with the workshop token.

## Dependencies
None

## Classes
None

## Functions
### `message(text, failed=false)`
> Update accessible save/load status.
**Source:** `braindance/examples/streaming_workshop/settings.js:41`
### `render(result, updateForm)`
> Render effective settings and feature badges.
**Source:** `braindance/examples/streaming_workshop/settings.js:45`
### `request(save=false, updateForm=true)`
> Load or save global settings while serializing UI access.
**Source:** `braindance/examples/streaming_workshop/settings.js:65`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Fields: `data_dir`, `catalog_path`, `output_dir`, `auto_extract_spike_info`; blank values restore defaults. |

## Data Shapes
- Consumes settings with saved/effective/environment values and feature rows with availability/detail.

## Notes
- Explains that changes require restart and environment variables override saved values.
