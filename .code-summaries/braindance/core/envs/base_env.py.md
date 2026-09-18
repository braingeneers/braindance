# base_env.py

**Path:** `braindance/core/envs/base_env.py`
**Module:** `braindance.core.envs.base_env`
**Feature Area:** `Acquisition`
**Entry point:** no — library or imported component

## Overview
Defines an alternate abstract acquisition base with initialized wall-clock timing. Abstract step, reset, and close methods constrain concrete subclasses while elapsed-time termination delegates to cleanup.

## Connections
None

## Dependencies
None

## Classes
### BaseEnv(ABC)
> unclear — see source
**Source:** `braindance/core/envs/base_env.py:10`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, max_time_sec: float=60, verbose: int=1)`
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
**Source:** `braindance/core/envs/base_env.py:16`
#### `time_elapsed(self) -> float`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/envs/base_env.py:19`
#### `dt(self) -> float`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/envs/base_env.py:23`
#### `_check_if_done(self) -> bool`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/envs/base_env.py:26`
#### `step(self, action=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/envs/base_env.py:35`
#### `reset(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/envs/base_env.py:39`
#### `close(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/envs/base_env.py:43`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| max_time_sec=60; verbose=1 |

## Data Shapes
None

## Notes
- _check_if_done calls _cleanup, which this base does not define.
