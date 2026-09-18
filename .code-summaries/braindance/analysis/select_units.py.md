# select_units.py

**Path:** `braindance/analysis/select_units.py`
**Module:** `braindance.analysis.select_units`
**Feature Area:** `Neural Analysis`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Builds unit pairs constrained by spike-time tiling coefficient and electrode distance. Also matches sorted spike events with a two-pointer tolerance scan.

## Connections
- **Used by:** `braindance.analysis.intermediate_rt_sorting.rt_sort` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.spikesorter.rt_sort` — import consumer hint; not a proven runtime call.
- **Shared data:** Filter closures call calc_sttc or calc_dist on Unit attributes.

## Dependencies
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### Unit()
> unclear — see source
**Source:** `braindance/analysis/select_units.py:6`
**Kind:** class. **Instantiated by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:1081 (named-call hint); braindance/analysis/select_units.py:134 (named-call hint); braindance/analysis/select_units.py:135 (named-call hint); braindance/core/spikesorter/rt_sort.py:1249 (named-call hint)
**Constructor:** `__init__(self, chan, spike_train, idx=-1) -> None`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `chan` | inferred at runtime | `chan` |
| `idx` | inferred at runtime | `idx` |
| `spike_train` | inferred at runtime | `spike_train` |
**Methods:**
None
### Filters()
> Filters for get_unit_pais()
**Source:** `braindance/analysis/select_units.py:85`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `sttc(recording_duration, min_sttc=0.35, max_sttc=0.5, delta_t=0.2)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/select_units.py:95`
#### `dist(elec_locs, min_dist=50, max_dist=300)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/select_units.py:99`

## Functions
### `calc_sttc(train_a, train_b, recording_duration, delta_t=20)`
> Calculate the spike time tiling coefficient between two spike trains
> **Called by:** braindance/analysis/select_units.py:96 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/select_units.py:14`
### `get_unit_pairs(units, filters=None)`
> Get all pairs of units
> **Called by:** braindance/analysis/select_units.py:137 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/select_units.py:57`
### `calc_dist(x1, y1, x2, y2)`
> unclear — see source
> **Called by:** braindance/analysis/select_units.py:100 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/select_units.py:81`
### `get_matching_events(times1, times2, delta=0.4)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/select_units.py:103`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Filters.sttc bounds default .35â€“.5 with delta_t=.2; Filters.dist bounds 50â€“300 |

## Data Shapes
- Unit stores chan, spike_train and idx
- get_matching_events returns matching_times1, matching_times2, unmatching_times1, unmatching_times2

## Notes
- calc_sttc requires nonempty trains and consistent duration/time units.
- get_unit_pairs iterates filters directly, so default None is not usable.
