# 01_quick_start.py

**Path:** `braindance/utils/data_manager/tutorials/01_quick_start.py`
**Module:** `braindance.utils.data_manager.tutorials.01_quick_start`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Executable quick-start tutorial that loads one hardcoded recording and demonstrates spike access, firing rates, raster/STTC plots, stimulation logs, latency analysis, evoked plots, and result persistence. Its top-level statements run immediately when the file is executed or imported.

## Connections
- **Uses:** `calculate_latencies` from `braindance.utils.data_manager` — imports (static evidence).
- **Uses:** `load_recording` from `braindance.utils.data_manager` — imports (static evidence).
- **Shared data:** Uses public data-manager loading/latency APIs and the Recording plot accessor for raster, STTC, evoked raster, and PSTH views.

## Dependencies
- `braindance.utils.data_manager.calculate_latencies` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.load_recording` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Hardcoded identifiers are project `24-105-10_drug_causal`, chip `20247`, experiment `BL1_causal`. |
| Examples use 20 ms STTC distance, 0-60 second plotting windows, latency ratio >=1.5, p <=0.0001, and -100..0/0..100 ms baseline/response windows. |

## Data Shapes
- SpikeData exposes `train[neuron]` as millisecond spike times and `rates(unit='Hz')` as one rate per neuron.
- Latency results are keyed like `electrode_X_neuron_Y` and contain onset latency, response ratio, p-value, and plot support data.

## Notes
- The script assumes latency variable `evoked_pairs` exists in the later visualization condition, which may be undefined when no stimulation log is present.
- It writes firing rates and evoked results through `rec.results` and `rec.save_results`.
