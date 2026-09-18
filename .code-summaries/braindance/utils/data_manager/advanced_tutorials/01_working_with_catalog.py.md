# 01_working_with_catalog.py

**Path:** `braindance/utils/data_manager/advanced_tutorials/01_working_with_catalog.py`
**Module:** `braindance.utils.data_manager.advanced_tutorials.01_working_with_catalog`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Demonstrates catalog filtering/grouping and compares evoked-pair counts across stimulation frequencies for one chip. Persists latency results and plots pair counts and pair-count-per-neuron ratios.

## Connections
- **Uses:** `Styler` from `braindance.utils.data_manager` — imports (static evidence).
- **Uses:** `load_catalog` from `braindance.utils.data_manager` — imports (static evidence).
- **Shared data:** catalog.filter/group_by→Recording.calculate_latencies→clear_cache→Styler comparison.

## Dependencies
- `braindance.utils.data_manager.Styler` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.load_catalog` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Configured load_catalog(); chip='25123ic'; baseline=False; compares 1, 2, and 4 Hz recordings. |

## Data Shapes
- Catalog groups produce per-frequency recording/neuron/stimulus/pair counts; plots are a two-panel Matplotlib figure.

## Notes
- Runs at import time and uses obsolete onset_latency instead of current peak_latency.
- Ratio counts electrode-neuron pairs, so it is not necessarily a percentage of unique responsive neurons.
