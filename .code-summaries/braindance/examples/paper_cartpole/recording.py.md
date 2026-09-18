# recording.py

**Path:** `braindance/examples/paper_cartpole/recording.py`
**Module:** `braindance.examples.paper_cartpole.recording`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Preserved GUI recording stage from proj/cartpole_v1/neural_config_analysis.py. Requires live Maxwell and the historical analysis dependencies.

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
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `main()`
> Function `main` performs the module operation described by its surrounding code.
> **Called by:** braindance/examples/paper_cartpole/recording.py:87 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/examples/paper_cartpole/recording.py:6`

## Config / CLI
None

## Data Shapes
None

## Notes
None
