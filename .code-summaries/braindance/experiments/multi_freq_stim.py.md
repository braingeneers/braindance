# multi_freq_stim.py

**Path:** `braindance/experiments/multi_freq_stim.py`
**Module:** `braindance.experiments.multi_freq_stim`
**Feature Area:** `Stimulation`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Demonstrates grouped-electrode stimulation commands on a dummy sine Maxwell stream. Executes the generated command sequence at 2 Hz and its first command at 5 Hz.

## Connections
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `maxwell_params` from `braindance.core.params` — imports (static evidence).
- **Uses:** `FrequencyStimPhase` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `PhaseManager` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `generate_stimulations` from `braindance.core.stim_commands` — imports (static evidence).
- **Shared data:** core.stim_commands.generate_stimulations -> FrequencyStimPhase -> PhaseManager.

## Dependencies
- `braindance.core.maxwell_env.MaxwellEnv` — intra-repo import; source import evidence.
- `braindance.core.params.maxwell_params` — intra-repo import; source import evidence.
- `braindance.core.phases.FrequencyStimPhase` — intra-repo import; source import evidence.
- `braindance.core.phases.PhaseManager` — intra-repo import; source import evidence.
- `braindance.core.stim_commands.generate_stimulations` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| dummy=sine, config=None; amp=400, phase_width=200 |

## Data Shapes
- electrode_inds mixes lists and scalar indices; generate_stimulations returns commands

## Notes
- Executes at import and mutates shared maxwell_params.
