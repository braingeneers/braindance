# data_helper.py

**Path:** `braindance/analysis/data_helper.py`
**Module:** `braindance.analysis.data_helper`
**Feature Area:** `Neural Analysis`
**Entry point:** no — library or imported component

## Overview
Runs footprint extraction for selected electrodes on an existing analysis object. Returns selected footprint channels, waveform arrays and electrode mapping.

## Connections
- **Uses:** `FootprintPhase` from `braindance.core.phases_analysis` — imports (static evidence).
- **Shared data:** Calls analysis.select_electrodes then core.phases_analysis.FootprintPhase.run.

## Dependencies
- `braindance.core.phases_analysis.FootprintPhase` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `get_footprints(analysis, selected_electrodes=None, file_path=None)`
> Return the footprint channels and waveforms for the given electrodes.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_helper.py:5`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| FootprintPhase uses rms_mult=1, wind=60, num_channel_thresh=120, similarity_thresh=.65 |

## Data Shapes
- Analysis object and electrode IDs -> (footprint_chs, footprint_waves, mapping)

## Notes
- Mutates selection on the supplied analysis object; file_path is checked but not passed into FootprintPhase.
