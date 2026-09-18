# mapping.py

**Path:** `braindance/analysis/mapping.py`
**Module:** `braindance.analysis.mapping`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Provides a DataFrame-backed channel/electrode mapping facade for Maxwell recordings and CSV mappings. It supports selection, identifier conversion, physical-neighbor lookup, position lookup, and CSV persistence.

## Connections
- **Used by:** `braindance.analysis.causal_connectivity` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.experiment_v3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.paper_cartpole.game` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.data_loading.recording` — import consumer hint; not a proven runtime call.
- **Uses:** `data_loader` from `braindance.analysis` — imports (static evidence).
- **Uses:** `open_file` from `braindance.io` — imports (static evidence).
- **Shared data:** `from_maxwell` uses `analysis.data_loader.load_mapping_maxwell`; `save` uses `braindance.io.open_file` so paths may be local or connector-backed.

## Dependencies
- `braindance.analysis.data_loader` — intra-repo import; source import evidence.
- `braindance.io.open_file` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.

## Classes
### Mapping()
> Wrap a mapping DataFrame and keep routed channels, physical electrodes, selections, and geometry conversions together.
**Source:** `braindance/analysis/mapping.py:10`
**Kind:** class. **Instantiated by:** braindance/core/phases_v3/experiment_v3.py:516 (named-call hint)
**Constructor:** `__init__(self, filepath=None, df=None, from_csv=False, channels=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `channels` | inferred at runtime | `mapping['channel'].values` |
| `electrodes` | inferred at runtime | `mapping['electrode'].values` |
| `mapping` | inferred at runtime | `None` |
| `selected_channels` | inferred at runtime | `[]` |
| `selected_electrodes` | inferred at runtime | `[]` |
**Methods:**
#### `from_csv(cls, filepath)`
> Load a mapping CSV through the repository file abstraction.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/mapping.py:27`
#### `from_df(cls, df)`
> Construct from an existing mapping DataFrame.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/mapping.py:31`
#### `from_maxwell(cls, filepath, channels=None)`
> Load a mapping from a Maxwell raw recording, optionally restricted to routed channels.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/mapping.py:35`
#### `select_electrodes(self, electrodes)`
> Record selected electrodes and derive their routed channel IDs.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/mapping.py:38`
#### `select_channels(self, channels)`
> Record selected routed channels and derive their electrode IDs.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/mapping.py:43`
#### `set_mapping(self, mapping)`
> Attach a mapping DataFrame and cache its integer routed-channel and electrode vectors, or clear the state.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/mapping.py:48`
#### `get_electrodes(self, channels=None, orig_channels=None)`
> Translate routed or original channel identifiers to physical electrode identifiers.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/mapping.py:61`
#### `get_channels(self, electrodes=None, orig_channels=None)`
> Translate electrodes or original channel identifiers to routed channel identifiers.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/mapping.py:78`
#### `get_orig_channels(self, channels=None, electrodes=None)`
> Translate routed channels or electrodes back to original hardware channel identifiers.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/mapping.py:96`
#### `get_nearest(self, channel=None, electrode=None, n=None, distance=None)`
> Return nearest mapped channels or electrodes by Euclidean x/y distance, with optional count and radius limits.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/mapping.py:114`
#### `get_positions(self, channels=None, electrodes=None)`
> Return x/y coordinates for all or selected channels/electrodes.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/mapping.py:154`
#### `save(self, filepath)`
> Persist the mapping DataFrame as CSV through `open_file`.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/analysis/mapping.py:166`

## Functions
None

## Config / CLI
None

## Data Shapes
- The mapping DataFrame is expected to contain `channel`, `electrode`, `x`, and `y`; original-channel conversion additionally requires `orig_channel`.
- Position results are N x 2 arrays; nearest-neighbor results are integer channel or electrode lists.

## Notes
- Lookups take the first matching row and therefore raise `IndexError` for missing identifiers.
- `get_orig_channels` has a likely erroneous both-arguments branch returning undefined `self.orig_channels`.
- The imported `os` module is unused.
