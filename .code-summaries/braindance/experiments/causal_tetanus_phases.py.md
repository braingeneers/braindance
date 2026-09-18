# causal_tetanus_phases.py

**Path:** `braindance/experiments/causal_tetanus_phases.py`
**Module:** `braindance.experiments.causal_tetanus_phases`
**Feature Area:** `Experiment Phases`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Composes baseline recording, repeated causal sweeps and tetanus periods into a multi-hour protocol. Reuses phase objects across repeated groups to compare stimulation blocks with intrinsic recording.

## Connections
- **Uses:** `AnalysisDAO` from `braindance.analysis.data_loader` — imports (static evidence).
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `maxwell_params` from `braindance.core.params` — imports (static evidence).
- **Uses:** `FrequencyStimPhase` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `NeuralSweepPhase` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `PhaseManager` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `RecordPhase` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `FootprintPhase` from `braindance.core.phases_analysis` — imports (static evidence).
- **Uses:** `HeatmapPhase` from `braindance.core.phases_analysis` — imports (static evidence).
- **Uses:** `generate_tetanus_pattern` from `braindance.core.trainer` — imports (static evidence).

## Dependencies
- `braindance.analysis.data_loader.AnalysisDAO` — intra-repo import; source import evidence.
- `braindance.core.maxwell_env.MaxwellEnv` — intra-repo import; source import evidence.
- `braindance.core.params.maxwell_params` — intra-repo import; source import evidence.
- `braindance.core.phases.FrequencyStimPhase` — intra-repo import; source import evidence.
- `braindance.core.phases.NeuralSweepPhase` — intra-repo import; source import evidence.
- `braindance.core.phases.PhaseManager` — intra-repo import; source import evidence.
- `braindance.core.phases.RecordPhase` — intra-repo import; source import evidence.
- `braindance.core.phases_analysis.FootprintPhase` — intra-repo import; source import evidence.
- `braindance.core.phases_analysis.HeatmapPhase` — intra-repo import; source import evidence.
- `braindance.core.trainer.generate_tetanus_pattern` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Causal 1 Hz,30 replicates,400 mV; tetanus 10 Hz,60 s,5 ms inter-stim delay |

## Data Shapes
- Phase groups combine RecordPhase, NeuralSweepPhase and FrequencyStimPhase instances

## Notes
- Executes hardware experiment at import and mutates shared core.params.maxwell_params; edit site-specific config/electrode/save paths before use.
