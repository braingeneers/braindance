# replay_experiment_v3.py

**Path:** `braindance/examples/replay_experiment_v3.py`
**Module:** `braindance.examples.replay_experiment_v3`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Runs a short V3 recording followed by simulated frequency stimulation from a Maxwell H5 replay source. Uses configured data/output roots to locate the fixture and save the experiment.

## Connections
- **Uses:** `get_data_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `get_output_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `FrequencyStimPhaseV3` from `braindance.core.phases_v3.phases3` — imports (static evidence).
- **Uses:** `RecordPhaseV3` from `braindance.core.phases_v3.phases3` — imports (static evidence).
- **Shared data:** Experiment runs RecordPhaseV3 then FrequencyStimPhaseV3 at 5 Hz using replay backend.

## Dependencies
- `braindance.config.get_data_dir` — intra-repo import; source import evidence.
- `braindance.config.get_output_dir` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3.FrequencyStimPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3.RecordPhaseV3` — intra-repo import; source import evidence.

## Classes
None

## Functions
### `main(source=None, record_seconds=0.5, stim_seconds=0.5, stim_electrode=0, speed=1.0)`
> unclear — see source
> **Called by:** braindance/examples/replay_experiment_v3.py:48 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/replay_experiment_v3.py:8`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| source=None,record_seconds=.5,stim_seconds=.5,stim_electrode=0,speed=1 |
| Default fixture replay_fixtures/maxwell_replay_smoke.raw.h5 |

## Data Shapes
- params.maxwell_env.replay contains source,speed,loop=False,write_output=True

## Notes
- overwrite_existing=True permits replacing replay_example output.
