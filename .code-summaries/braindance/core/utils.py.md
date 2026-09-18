# utils.py

**Path:** `braindance/core/utils.py`
**Module:** `braindance.core.utils`
**Feature Area:** `Acquisition`
**Entry point:** no — library or imported component

## Overview
Controls a smart plug through Braingeneers messaging or a local GPIO pin and launches the Maxwell server. Power-off kills mxwserver and disables the selected power control.

## Connections
- **Used by:** `braindance.experiments.cartpole_continuous` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.cartpole_fast` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.cartpole_schedule` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.gui.experiment_launcher` — import consumer hint; not a proven runtime call.
- **Shared data:** MessageBroker publishes ON/OFF to smartplug/telemetry/{name}/cmnd/Power.

## Dependencies
- `board` — external or unresolved local import; source import evidence.
- `braingeneers.iot.*` — external or unresolved local import; source import evidence.
- `digitalio` — external or unresolved local import; source import evidence.

## Classes
### SmartPlug()
> unclear — see source
**Source:** `braindance/core/utils.py:16`
**Kind:** class. **Instantiated by:** braindance/experiments/cartpole_continuous.py:49 (named-call hint); braindance/experiments/cartpole_fast.py:28 (named-call hint); braindance/experiments/cartpole_schedule.py:28 (named-call hint); braindance/gui/experiment_launcher.py:1512 (named-call hint)
**Constructor:** `__init__(self, smartplug='smartplug_1', verbose=False)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `drew` | inferred at runtime | `None` |
| `is_on` | inferred at runtime | `False` |
| `mb` | inferred at runtime | `None` |
| `smartplug` | inferred at runtime | `smartplug.lower()` |
| `verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `turn_on(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/utils.py:46`
#### `turn_off(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/utils.py:64`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| smartplug names include drew and none; default smartplug_1 |

## Data Shapes
None

## Notes
- Uses Linux Maxwell install paths and shell process commands; turn_on sleeps twice for 12 seconds.
- Optional imports are caught but corresponding features still require those dependencies.
