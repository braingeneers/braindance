# maxwell_utils.py

**Path:** `braindance/core/maxwell/maxwell_utils.py`
**Module:** `braindance.core.maxwell.maxwell_utils`
**Feature Area:** `Acquisition`
**Entry point:** no — library or imported component

## Overview
Checks whether a TCP connection to the local Maxwell status port succeeds. Returns a boolean instead of querying a hardware state payload.

## Connections
- **Used by:** `braindance.gui.experiment_launcher` — import consumer hint; not a proven runtime call.

## Dependencies
- `maxlab` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `get_maxwell_status()`
> Get the status of the maxwell system. True if initialized, False otherwise.
> **Called by:** braindance/gui/experiment_launcher.py:485 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell/maxwell_utils.py:8`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| MXW_BASE_PORT plus 15; falls back to 7215 |

## Data Shapes
None

## Notes
- Connection success only indicates a listening service; socket is not explicitly closed.
