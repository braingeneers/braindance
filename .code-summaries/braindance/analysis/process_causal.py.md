# process_causal.py

**Path:** `braindance/analysis/process_causal.py`
**Module:** `braindance.analysis.process_causal`
**Feature Area:** `Neural Analysis`
**Entry point:** no — library or imported component

## Overview
Command-line legacy causal-analysis script that loads a Maxwell recording, mapping, spikes, and an adjusted stimulation log. Execution intentionally stops at a guard exception before constructing or running `CausalAnalysis`.

## Connections
- **Uses:** `analysis_helper` from `braindance.analysis` — imports (static evidence).
- **Uses:** `data_loader` from `braindance.analysis` — imports (static evidence).
- **Uses:** `CausalAnalysis` from `braindance.core.phases_analysis` — imports (static evidence).
- **Shared data:** Builds `AnalysisDAO` from `data_loader`, calls `adjust_stim_times2`, and was intended to pass the object into legacy `core.phases_analysis.CausalAnalysis`.

## Dependencies
- `braindance.analysis.analysis_helper` — intra-repo import; source import evidence.
- `braindance.analysis.data_loader` — intra-repo import; source import evidence.
- `braindance.core.phases_analysis.CausalAnalysis` — intra-repo import; source import evidence.
- `braingeneers.analysis.SpikeData` — external or unresolved local import; source import evidence.
- `braingeneers.utils.s3wrangler` — external or unresolved local import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| CLI `--path` defaults to `./` and names the `.raw.h5` input. |
| CLI `--save` uses `type=bool` and defaults to true; `--tag` defaults to `causal`. |
| The prepared causal-analysis call uses `wind_ms=150`, while stimulation alignment uses a 10 ms offset. |

## Data Shapes
- Loads routed channel/electrode vectors and a stimulation-log DataFrame with list-valued `stim_electrodes`.
- Unreachable output code would save per-stimulation-electrode clean voltage and reactive-spike-time NumPy arrays.

## Notes
- Argument parsing and file I/O occur at import time; this is a script, not an import-safe library module.
- A deliberate `ValueError('Check the code below')` makes all causal analysis and saves unreachable.
- Several imports and variables are unused remnants.
