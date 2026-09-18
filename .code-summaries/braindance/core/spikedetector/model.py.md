# model.py

**Path:** `braindance/core/spikedetector/model.py`
**Module:** `braindance.core.spikedetector.model`
**Feature Area:** `Spike Detection`
**Entry point:** no — library or imported component

## Overview
Re-exports detector model implementations from model2 for compatibility. Existing imports of ModelSpikeSorter through model continue to resolve to the canonical implementation.

## Connections
- **Used by:** `braindance.analysis.intermediate_rt_sorting.rt_sort` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.spikedetector.spikedetector` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.spikedetector.train` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.spikesorter.rt_sort` — import consumer hint; not a proven runtime call.
- **Uses:** `*` from `braindance.core.spikedetector.model2` — imports (static evidence).
- **Shared data:** Wildcard import from braindance.core.spikedetector.model2.

## Dependencies
- `braindance.core.spikedetector.model2.*` — intra-repo import; source import evidence.

## Classes
None

## Functions
None

## Config / CLI
None

## Data Shapes
None

## Notes
None
