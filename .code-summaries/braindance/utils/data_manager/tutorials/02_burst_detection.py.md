# 02_burst_detection.py

**Path:** `braindance/utils/data_manager/tutorials/02_burst_detection.py`
**Module:** `braindance.utils.data_manager.tutorials.02_burst_detection`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Demonstrates population-burst detection, backbone-neuron statistics, and stimulus-evoked burst latency analysis. Saves population overlays, peri-stimulus burst probability, and trial heatmaps plus cached latency results.

## Connections
- **Uses:** `BurstDetector` from `braindance.utils.data_manager` — imports (static evidence).
- **Uses:** `BurstLatencyAnalyzer` from `braindance.utils.data_manager` — imports (static evidence).
- **Uses:** `Styler` from `braindance.utils.data_manager` — imports (static evidence).
- **Uses:** `load_recording` from `braindance.utils.data_manager` — imports (static evidence).
- **Shared data:** BurstDetector/cache→BurstLatencyAnalyzer→Styler plots→rec.results.burst_latency_results.

## Dependencies
- `braindance.utils.data_manager.BurstDetector` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.BurstLatencyAnalyzer` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.Styler` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.load_recording` — intra-repo import; source import evidence.
- `matplotlib.colors.ListedColormap` — external or unresolved local import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Top-level project/chip/experiment; save_dir='/Volumes/hunter_ssd/busy_bee/test_plots'; burst bin=1ms and analysis window=500ms. |

## Data Shapes
- SpikeData and stim_log→BurstResults and electrode result dict; PNG/SVG plots; burst_matrix(trials,time bins).

## Notes
- Runs on import; hardcoded output directory requires adaptation.
- Assumes bursts and stimulation data exist; visualization aligns to time while analyzer defaults time_mod.
