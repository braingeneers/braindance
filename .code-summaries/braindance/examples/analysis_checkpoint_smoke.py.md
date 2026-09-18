# analysis_checkpoint_smoke.py

**Path:** `braindance/examples/analysis_checkpoint_smoke.py`
**Module:** `braindance.examples.analysis_checkpoint_smoke`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Demonstrates hardware-free V3 phase data passing and checkpoint resume. Stops after producing values, reconstructs the experiment and verifies the resumed sum equals 12.

## Connections
- **Uses:** `get_output_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `AnalysisPhaseV3` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Shared data:** AnalysisPhaseV3 provides/requires declarations drive Experiment.run(stop_at=1) then run(resume=True).

## Dependencies
- `braindance.config.get_output_dir` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.AnalysisPhaseV3` — intra-repo import; source import evidence.

## Classes
### SourceValues(AnalysisPhaseV3)
> unclear — see source
**Source:** `braindance/examples/analysis_checkpoint_smoke.py:11`
**Kind:** class. **Instantiated by:** braindance/examples/analysis_checkpoint_smoke.py:32 (named-call hint); braindance/examples/analysis_checkpoint_smoke.py:36 (named-call hint)
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `run(self, experiment)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/analysis_checkpoint_smoke.py:14`
### SumValues(AnalysisPhaseV3)
> unclear — see source
**Source:** `braindance/examples/analysis_checkpoint_smoke.py:18`
**Kind:** class. **Instantiated by:** braindance/examples/analysis_checkpoint_smoke.py:32 (named-call hint); braindance/examples/analysis_checkpoint_smoke.py:36 (named-call hint)
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `run(self, experiment)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/examples/analysis_checkpoint_smoke.py:22`

## Functions
### `main(output_dir=None)`
> unclear — see source
> **Called by:** braindance/examples/analysis_checkpoint_smoke.py:47 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/analysis_checkpoint_smoke.py:26`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| CLI --output-dir; default get_output_dir()/analysis_checkpoint_smoke |

## Data Shapes
- SourceValues -> {'values':[2,4,6]}; SumValues -> {'total':12}

## Notes
- Refuses an output directory already containing experiment_log.json.
