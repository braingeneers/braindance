# playground.py

**Path:** `braindance/examples/streaming_workshop/playground.py`
**Module:** `braindance.examples.streaming_workshop.playground`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Independent, manually stepped game sandbox (never opens acquisition hardware).

## Connections
- **Used by:** `braindance.examples.streaming_workshop.main` — import consumer hint; not a proven runtime call.
- **Uses:** `WorkshopGame` from `braindance.examples.streaming_workshop.environments` — imports (static evidence).

## Dependencies
- `braindance.examples.streaming_workshop.environments.WorkshopGame` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### Playground()
> Playground method `Playground` manages the independent manually stepped game sandbox.
**Source:** `braindance/examples/streaming_workshop/playground.py:9`
**Kind:** class. **Instantiated by:** braindance/examples/streaming_workshop/main.py:79 (named-call hint)
**Constructor:** `__init__(self, game_factory=WorkshopGame)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `done` | inferred at runtime | `False` |
| `game` | inferred at runtime | `None` |
| `game_factory` | inferred at runtime | `game_factory` |
| `lock` | inferred at runtime | `RLock()` |
| `observation` | inferred at runtime | `None` |
| `reward` | inferred at runtime | `0.0` |
| `time` | inferred at runtime | `0.0` |
**Methods:**
#### `control(self, command)`
> Playground method `control` manages the independent manually stepped game sandbox.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/examples/streaming_workshop/playground.py:19`
#### `snapshot(self)`
> Playground method `snapshot` manages the independent manually stepped game sandbox.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/playground.py:72`
#### `close(self)`
> Playground method `close` manages the independent manually stepped game sandbox.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/examples/streaming_workshop/playground.py:79`

## Functions
None

## Config / CLI
None

## Data Shapes
None

## Notes
None
