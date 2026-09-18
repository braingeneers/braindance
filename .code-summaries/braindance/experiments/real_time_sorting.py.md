# real_time_sorting.py

**Path:** `braindance/experiments/real_time_sorting.py`
**Module:** `braindance.experiments.real_time_sorting`
**Feature Area:** `Experiment Phases`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Demonstrates streaming RT-Sort after linear artifact removal on a dummy Maxwell recording. Pre-detects sequences from a 20-second recording window, then processes 120-frame batches and plots cleaned traces.

## Connections
- **Uses:** `get_data_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `get_output_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `maxwell_params` from `braindance.core.params` — imports (static evidence).
- **Uses:** `ModelSpikeSorter` from `braindance.core.spikedetector.model2` — imports (static evidence).
- **Uses:** `detect_sequences` from `braindance.core.spikesorter.rt_sort` — imports (static evidence).
- **Uses:** `LinearArtifactRemoval` from `braindance.utils.rt_linear_art_removal` — imports (static evidence).

## Dependencies
- `braindance.config.get_data_dir` — intra-repo import; source import evidence.
- `braindance.config.get_output_dir` — intra-repo import; source import evidence.
- `braindance.core.maxwell_env.MaxwellEnv` — intra-repo import; source import evidence.
- `braindance.core.params.maxwell_params` — intra-repo import; source import evidence.
- `braindance.core.spikedetector.model2.ModelSpikeSorter` — intra-repo import; source import evidence.
- `braindance.core.spikesorter.rt_sort.detect_sequences` — intra-repo import; source import evidence.
- `braindance.utils.rt_linear_art_removal.LinearArtifactRemoval` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `spikeinterface.extractors.MaxwellRecordingExtractor` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `main(recording_path, max_time_sec=60, stim_electrodes=None, recording_window_ms=(120000, 140000), buffer_size=120)`
> Function `main` performs the module operation described by its surrounding code.
> **Called by:** braindance/experiments/real_time_sorting.py:72 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/experiments/real_time_sorting.py:13`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Uses BrainDance configured output directory for generated results. |
| n_channels=951; buffer_size=120; recording_window_ms=(120000,140000) |
| Site-specific recording, model and project helper paths |

## Data Shapes
- Raw observation batches transposed to channels × frames for LinearArtifactRemoval, then transposed into running_sort

## Notes
- Loads model/recording and creates figure at import; unguarded trailing code references main-only variables and exits 1.
