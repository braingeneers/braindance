# neural_config_analysis.py

**Path:** `braindance/experiments/neural_config_analysis.py`
**Module:** `braindance.experiments.neural_config_analysis`
**Feature Area:** `Experiment Phases`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Records three minutes and selects neural footprints with heatmap and footprint analysis phases. Plots selected waveforms and saves the resulting analysis parameters.

## Connections
- **Uses:** `AnalysisDAO` from `braindance.analysis.data_loader` — imports (static evidence).
- **Uses:** `plot_footprints` from `braindance.analysis.plot_helper` — imports (static evidence).
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `maxwell_params` from `braindance.core.params` — imports (static evidence).
- **Uses:** `NeuralSweepPhase` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `PhaseManager` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `RecordPhase` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `FootprintPhase` from `braindance.core.phases_analysis` — imports (static evidence).
- **Uses:** `HeatmapPhase` from `braindance.core.phases_analysis` — imports (static evidence).

## Dependencies
- `braindance.analysis.data_loader.AnalysisDAO` — intra-repo import; source import evidence.
- `braindance.analysis.plot_helper.plot_footprints` — intra-repo import; source import evidence.
- `braindance.core.maxwell_env.MaxwellEnv` — intra-repo import; source import evidence.
- `braindance.core.params.maxwell_params` — intra-repo import; source import evidence.
- `braindance.core.phases.NeuralSweepPhase` — intra-repo import; source import evidence.
- `braindance.core.phases.PhaseManager` — intra-repo import; source import evidence.
- `braindance.core.phases.RecordPhase` — intra-repo import; source import evidence.
- `braindance.core.phases_analysis.FootprintPhase` — intra-repo import; source import evidence.
- `braindance.core.phases_analysis.HeatmapPhase` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| FootprintPhase rms_mult=1,wind=100,num_channel_thresh=120,similarity_thresh=.65 |

## Data Shapes
- AnalysisDAO selected_footprint_chans, selected_footprint_waves, mapping and selected_channels feed plot_footprints

## Notes
- Executes hardware experiment at import and mutates shared core.params.maxwell_params; edit site-specific config/electrode/save paths before use.
