# dummy_maxlab.py

**Path:** `braindance/core/dummy_maxlab.py`
**Module:** `braindance.core.dummy_maxlab`
**Feature Area:** `Replay`
**Entry point:** no — library or imported component

## Overview
Provides in-memory stand-ins for the maxlab sequence, chip, saving, system, and utility interfaces. Dummy calls return strings, values, or chainable unit objects so local replay can construct commands without hardware.

## Connections
- **Used by:** `braindance.core.maxwell_env` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.sample_manual_sequence` — import consumer hint; not a proven runtime call.
- **Shared data:** MaxwellEnv explicitly selects this module for replay even if real maxlab is installed.

## Dependencies
None

## Classes
### Sequence()
> unclear — see source
**Source:** `braindance/core/dummy_maxlab.py:3`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `sequence` | inferred at runtime | `[]` |
| `token` | inferred at runtime | `self.sequence` |
**Methods:**
#### `append(self, item)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:10`
#### `__getitem__(self, key)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:13`
#### `__len__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:16`
#### `__iter__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:19`
#### `__str__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:22`
#### `send(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:25`
### Core()
> unclear — see source
**Source:** `braindance/core/dummy_maxlab.py:30`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `enable_stimulation_power(self, bool_value)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:31`
### Amplifier()
> unclear — see source
**Source:** `braindance/core/dummy_maxlab.py:36`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `set_gain(self, gain)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:37`
### unit()
> unclear — see source
**Source:** `braindance/core/dummy_maxlab.py:42`
**Kind:** class. **Instantiated by:** braindance/core/dummy_maxlab.py:80 (named-call hint)
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `power_up(self, do_power_up)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:43`
#### `connect(self, do_connect)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:48`
#### `set_voltage_mode(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:53`
#### `dac_source(self, source)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:58`
### chip()
> unclear — see source
**Source:** `braindance/core/dummy_maxlab.py:63`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `DAC(channel, value)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:65`
#### `DelaySamples(value)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:71`
#### `StimulationUnit(int_value)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:77`
#### `send(value)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:131`
#### `send_raw(value)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:137`
### saving()
> unclear — see source
**Source:** `braindance/core/dummy_maxlab.py:143`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
None
### system()
> unclear — see source
**Source:** `braindance/core/dummy_maxlab.py:186`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `DelaySamples(value)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:188`
#### `Event(*args)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:194`
### util()
> unclear — see source
**Source:** `braindance/core/dummy_maxlab.py:199`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `offset()`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:200`
#### `initialize()`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:205`
#### `set_gain(gain)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:210`
#### `hpf()`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:215`

## Functions
### `send(value)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:220`
### `send_raw(value)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/dummy_maxlab.py:225`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| DEBUG=False |

## Data Shapes
None

## Notes
- Does not acquire or save real recordings; Array always reports stimulation unit 1.
