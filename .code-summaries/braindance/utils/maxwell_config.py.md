# maxwell_config.py

**Path:** `braindance/utils/maxwell_config.py`
**Module:** `braindance.utils.maxwell_config`
**Feature Area:** `Configuration`
**Entry point:** no — library or imported component

## Overview
Parses Maxwell routing configuration text into channel, electrode, and coordinate mappings. Maintains a validated list of stimulation electrodes and exposes mapping lookups and a pandas table.

## Connections
None

## Dependencies
- `pandas` — external or unresolved local import; source import evidence.

## Classes
### Config()
> Parse routing entries and track selected stimulation electrodes.
**Source:** `braindance/utils/maxwell_config.py:3`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, filename, is_content=False)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `config` | inferred at runtime | `[]` |
| `df` | inferred at runtime | `None` |
| `mappings` | inferred at runtime | `[]` |
| `stim_electrodes` | inferred at runtime | `[]` |
**Methods:**
#### `get_channels(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/maxwell_config.py:30`
#### `get_electrodes(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/maxwell_config.py:33`
#### `get_channels_for_electrodes(self, electrodes)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/maxwell_config.py:36`
#### `get_num_channels(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/maxwell_config.py:39`
#### `add_stim_electrodes(self, stim_electrodes)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/maxwell_config.py:42`
#### `set_df(self)`
> Set channel, electrode, x, y dataframe
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/maxwell_config.py:54`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| filename accepts a path or a file-like object when is_content=True. |

## Data Shapes
- df columns: channel:int, electrode:int, x:float, y:float.

## Notes
- filename=None exits before df initialization.
