# select_electrodes.py

**Path:** `braindance/core/maxwell/select_electrodes.py`
**Module:** `braindance.core.maxwell.select_electrodes`
**Feature Area:** `Stimulation`
**Entry point:** no — library or imported component

## Overview
Prints precomputed stimulation-buffer assignments for requested Maxwell electrodes. Loads electrode IDs from CLI or experiment JSON and indexes the packaged stim_buffers array.

## Connections
- **Shared data:** Reads core/maxwell/stim_buffers.npy; optional mapping_file_path is loaded and printed.

## Dependencies
- `braindance` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| CLI --stim_electrodes/-s and --json/-j |

## Data Shapes
- Electrode grid reshapes 26400 IDs into 120 rows by 220 columns.

## Notes
- Argument parsing and file loads execute at import time.
