# base_env.py

**Path:** `braindance/core/base_env.py`
**Module:** `braindance.core.base_env`
**Feature Area:** `Acquisition`
**Entry point:** no — library or imported component

## Overview
Defines the original acquisition environment interface and wall-clock timing helpers. Subclasses implement acquisition, cleanup, reset, rendering, and close while inheriting maximum-duration termination.

## Connections
- **Used by:** `braindance.core.maxwell_env` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.open_ephys_env` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases2` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases2_loop` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_env` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3` — import consumer hint; not a proven runtime call.

## Dependencies
None

## Classes
### BaseEnv()
> unclear — see source
**Source:** `braindance/core/base_env.py:5`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, max_time_sec=60, verbose=1)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `cur_time` | inferred at runtime | `time.perf_counter()` |
| `max_time_sec` | inferred at runtime | `max_time_sec` |
| `start_time` | inferred at runtime | `time.perf_counter()` |
| `verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `_init_time_management(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/base_env.py:11`
#### `time_elapsed(self)`
> Returns time since initialization of the environment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/base_env.py:14`
#### `dt(self)`
> Returns time since the last step.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/base_env.py:19`
#### `_check_if_done(self)`
> Run subclass cleanup after the configured elapsed-time limit.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/base_env.py:23`
#### `_cleanup(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/base_env.py:34`
#### `step(self, action)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/base_env.py:37`
#### `reset(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/base_env.py:40`
#### `render(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/base_env.py:43`
#### `close(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/base_env.py:45`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| max_time_sec=60; verbose=1 |

## Data Shapes
None

## Notes
- Constructor does not initialize clocks; subclasses must call _init_time_management.
