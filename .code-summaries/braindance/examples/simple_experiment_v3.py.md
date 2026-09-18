# simple_experiment_v3.py

**Path:** `braindance/examples/simple_experiment_v3.py`
**Module:** `braindance.examples.simple_experiment_v3`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Demonstrates V3 method chaining, amplitude sweeps and grouped frequency stimulation between baseline/post recordings. A second function shows dependency validation and default phase configuration.

## Connections
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `PhaseGroup` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `FrequencyStimPhaseV3` from `braindance.core.phases_v3.phases3` — imports (static evidence).
- **Uses:** `NeuralSweepPhaseV3` from `braindance.core.phases_v3.phases3` — imports (static evidence).
- **Uses:** `RecordPhaseV3` from `braindance.core.phases_v3.phases3` — imports (static evidence).
- **Shared data:** Experiment.add_phase/add_phase_group -> RecordPhaseV3,NeuralSweepPhaseV3,FrequencyStimPhaseV3; saves data and summary.

## Dependencies
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.PhaseGroup` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3.FrequencyStimPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3.NeuralSweepPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3.RecordPhaseV3` — intra-repo import; source import evidence.

## Classes
None

## Functions
### `main()`
> Run a simple recording and stimulation experiment.
> **Called by:** braindance/examples/simple_experiment_v3.py:128 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/simple_experiment_v3.py:15`
### `advanced_example()`
> More advanced example with custom configuration.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/examples/simple_experiment_v3.py:99`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| config.cfg; stim_electrodes [1001,1002,1003,1004]; 300 s recordings |
| Sweep amp_bounds=(100,400,10),.5 Hz,5 replicates |

## Data Shapes
- Reads recording_file,sweep_results,stim_count from experiment data

## Notes
- Example electrode/config values require site setup.
