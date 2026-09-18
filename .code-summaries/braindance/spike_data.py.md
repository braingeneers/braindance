# spike_data.py

**Path:** `braindance/spike_data.py`
**Module:** `braindance.spike_data`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Normalizes supported spike-train containers into SpikeLab SpikeData and provides compatibility unpickling for historical module paths. The compatibility loader preserves extra attached state while converting reconstructed legacy instances into the native SpikeLab class.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.catalogging.s3_helpers` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.data_loading.data_context` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.data_loading.recording` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.data_loading.s3_loader` — import consumer hint; not a proven runtime call.
- **Shared data:** Used by data-manager local and S3 loaders to read both current and legacy SpikeData pickles.

## Dependencies
- `spikelab.SpikeData` — external or unresolved local import; source import evidence.

## Classes
### _LegacySpikeData()
> Temporary pickle reconstruction target; instances become native SpikeLab.
**Source:** `braindance/spike_data.py:34`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `__class__` | inferred at runtime | `SpikeData` |
**Methods:**
#### `__setstate__(self, state)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/spike_data.py:37`
### _SpikeDataUnpickler(pickle.Unpickler)
> unclear — see source
**Source:** `braindance/spike_data.py:45`
**Kind:** class. **Instantiated by:** braindance/spike_data.py:57 (named-call hint)
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `find_class(self, module, name)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/spike_data.py:46`

## Functions
### `as_spike_data(data)`
> Convert recognized dictionaries and lists to SpikeLab SpikeData while retaining acquisition metadata.
> **Called by:** braindance/spike_data.py:38 (named-call hint); braindance/utils/data_manager/utils/data_loading/recording.py:1001 (named-call hint); braindance/utils/data_manager/utils/data_loading/recording.py:859 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/spike_data.py:14`
### `load_spike_pickle(file)`
> Load a trusted pickle and redirect historical SpikeData class references even when nested inside containers.
> **Called by:** braindance/utils/data_manager/utils/catalogging/s3_helpers.py:238 (named-call hint); braindance/utils/data_manager/utils/data_loading/data_context.py:134 (named-call hint); braindance/utils/data_manager/utils/data_loading/recording.py:859 (named-call hint); braindance/utils/data_manager/utils/data_loading/s3_loader.py:131 (named-call hint); braindance/utils/data_manager/utils/data_loading/s3_loader.py:140 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/spike_data.py:55`

## Config / CLI
None

## Data Shapes
- Dictionary input may contain `train` or `spike_trains`, plus `N`, `length`, `start_time`, metadata, neuron attributes, raw data, and raw time.
- List input is interpreted as one spike-time sequence per neuron.

## Notes
- This is ordinary pickle loading and must only be used with trusted files.
- Unknown containers pass through unchanged rather than raising.
