# query_electrodes.py

**Path:** `braindance/core/maxwell/query_electrodes.py`
**Module:** `braindance.core.maxwell.query_electrodes`
**Feature Area:** `Stimulation`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Queries Maxwell stimulation-unit assignments for selected electrodes or a saved configuration. Can keep only the first electrode per valid unit and persist that selection into the supplied experiment JSON.

## Connections
None

## Dependencies
- `maxlab` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `query_electrode_list(e)`
> unclear — see source
> **Called by:** braindance/core/maxwell/query_electrodes.py:114 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell/query_electrodes.py:8`
### `query_config(config, electrodes=None, remove_duplicates=False)`
> Connect and disconnect each electrode to inspect its stimulation unit, optionally retaining unique unit assignments.
> **Called by:** braindance/core/maxwell/query_electrodes.py:103 (named-call hint); braindance/core/maxwell/query_electrodes.py:106 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/maxwell/query_electrodes.py:22`
### `main()`
> unclear — see source
> **Called by:** braindance/core/maxwell/query_electrodes.py:119 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/core/maxwell/query_electrodes.py:67`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| CLI --stim_electrodes/-s, --json/-j, --remove_duplicates/-r |

## Data Shapes
- JSON keys: config, stim_electrodes or selected_electrodes; writes valid_stim_electrodes and stim_electrodes.

## Notes
- Requires maxlab and modifies array routing; duplicate removal overwrites supplied JSON.
