# profiles.py

**Path:** `braindance/examples/streaming_workshop/profiles.py`
**Module:** `braindance.examples.streaming_workshop.profiles`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Validates participant mapping signatures without executing code and stores named Python profiles with literal setup settings and immutable revision history.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.main` — import consumer hint; not a proven runtime call.
- **Uses:** `resolve_analysis_config` from `braindance.examples.streaming_workshop.custom_analysis` — imports (static evidence).
- **Shared data:** Uses custom-analysis resolution so saved phase contracts match decorators; served by profile APIs in `main.py`.

## Dependencies
- `braindance.examples.streaming_workshop.custom_analysis.resolve_analysis_config` — intra-repo import; source import evidence.

## Classes
### ProfileStore()
> Manage named participant-code profiles and revisions.
**Source:** `braindance/examples/streaming_workshop/profiles.py:34`
**Kind:** class. **Instantiated by:** braindance/examples/streaming_workshop/main.py:58 (named-call hint)
**Constructor:** `__init__(self, directory, template_file)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `directory` | inferred at runtime | `Path(directory).resolve()` |
| `template_file` | inferred at runtime | `Path(template_file)` |
**Methods:**
#### `path(self, name)`
> Validate a profile name and confine it to the profile directory.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/profiles.py:39`
#### `list(self)`
> List saved profile stems.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/profiles.py:47`
#### `template(self)`
> Read the default participant source.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/profiles.py:50`
#### `load(self, name)`
> Parse a profile's literal settings and refresh analysis contracts.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/profiles.py:53`
#### `save(self, name, code, settings)`
> Validate code/settings, replace the SETTINGS assignment, archive revisions, and atomically save.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/examples/streaming_workshop/profiles.py:67`

## Functions
### `verify_python(code)`
> Check syntax and required synchronous four-argument mapping signatures without execution.
> **Called by:** braindance/examples/streaming_workshop/main.py:242 (named-call hint); braindance/examples/streaming_workshop/main.py:260 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/profiles.py:10`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Code is capped at 100,000 characters; profile names use 1–64 letters, numbers, underscores, or hyphens. |

## Data Shapes
- Profiles are Python source containing top-level `encode`, `decode`, optional `train`, and a literal `SETTINGS` dictionary.

## Notes
- Load/save parse settings with AST and never execute profile code.
- Save snapshots the previous and new versions, then atomically replaces the active profile.
