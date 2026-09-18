# global_settings.py

**Path:** `braindance/examples/streaming_workshop/global_settings.py`
**Module:** `braindance.examples.streaming_workshop.global_settings`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Reads or saves global BrainDance settings and adds lightweight feature availability for the Settings view.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.main` — import consumer hint; not a proven runtime call.
- **Uses:** `get_global_settings` from `braindance.config` — imports (static evidence).
- **Uses:** `save_global_settings` from `braindance.config` — imports (static evidence).
- **Uses:** `sorting_capabilities` from `braindance.examples.streaming_workshop.analysis_sorting` — imports (static evidence).
- **Shared data:** Uses config APIs and sorting capability probe; served at `/api/settings`.

## Dependencies
- `braindance.config.get_global_settings` — intra-repo import; source import evidence.
- `braindance.config.save_global_settings` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.analysis_sorting.sorting_capabilities` — intra-repo import; source import evidence.

## Classes
None

## Functions
### `settings_payload(updates=None)`
> Return global configuration plus dependency status, optionally saving updates first.
> **Called by:** braindance/examples/streaming_workshop/main.py:119 (named-call hint); braindance/examples/streaming_workshop/main.py:199 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/global_settings.py:8`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Updates go to `save_global_settings`; otherwise current settings are read. |

## Data Shapes
- Adds feature rows with ID, label, availability, and detail.

## Notes
- Probes import visibility/model files, not compatibility, hardware, GPU, or ROMs.
