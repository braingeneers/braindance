# command_runner.py

**Path:** `braindance/gui/command_runner.py`
**Module:** `braindance.gui.command_runner`
**Feature Area:** `Configuration`
**Entry point:** no — library or imported component

## Overview
Runs shell commands on a Qt worker thread and streams combined output to signals. Stop escalates process-group interruption from SIGINT to SIGTERM and SIGKILL.

## Connections
- **Used by:** `braindance.gui.experiment_launcher` — import consumer hint; not a proven runtime call.
- **Shared data:** Experiment launcher uses CommandRunner for launch/upload commands.

## Dependencies
- `PyQt5.QtCore.QThread` — external or unresolved local import; source import evidence.
- `PyQt5.QtCore.pyqtSignal` — external or unresolved local import; source import evidence.

## Classes
### CommandRunner(QThread)
> unclear — see source
**Source:** `braindance/gui/command_runner.py:6`
**Kind:** class. **Instantiated by:** braindance/gui/experiment_launcher.py:1081 (named-call hint); braindance/gui/experiment_launcher.py:1628 (named-call hint); braindance/gui/experiment_launcher.py:723 (named-call hint); braindance/gui/experiment_launcher.py:797 (named-call hint)
**Constructor:** `__init__(self, command)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `command` | inferred at runtime | `command` |
| `process` | inferred at runtime | `None` |
**Methods:**
#### `run(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/gui/command_runner.py:16`
#### `stop(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/gui/command_runner.py:47`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| PYTHONUNBUFFERED=1; shutdown waits 5 s then 2 s |

## Data Shapes
- Command string -> output(str),error(str),finished() Qt signals

## Notes
- Uses shell=True and POSIX os.setsid/killpg, so not directly portable to Windows.
- stop assumes process already exists.
