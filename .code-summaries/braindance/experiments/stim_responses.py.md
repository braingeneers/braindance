# stim_responses.py

**Path:** `braindance/experiments/stim_responses.py`
**Module:** `braindance.experiments.stim_responses`
**Feature Area:** `Artifact Removal`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Displays two channels of streaming artifact-cleaned responses while rotating stimulation electrodes. Marks detected spikes and refreshes traces during acquisition.

## Connections
- **Uses:** `ArtifactRemoval` from `braindance.core.artifact_removal` — imports (static evidence).
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `maxwell_params` from `braindance.core.params` — imports (static evidence).
- **Shared data:** MaxwellEnv raw samples -> two ArtifactRemoval instances -> Matplotlib lines/scatters.

## Dependencies
- `braindance.core.artifact_removal.ArtifactRemoval` — intra-repo import; source import evidence.
- `braindance.core.maxwell_env.MaxwellEnv` — intra-repo import; source import evidence.
- `braindance.core.params.maxwell_params` — intra-repo import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| config_st.cfg; read channels [772,514]; N=20; spike_thresh=[-3.5,-20] |
| 400 mV stimulation every .5 s; electrode rotation every 10 s |

## Data Shapes
- data_raw/data_clean shape (80000,2); ArtifactRemoval.fit_step -> clean,artifact,spike

## Notes
- Requires configured Maxwell hardware and interactive Matplotlib; exits status 1 in finally.
