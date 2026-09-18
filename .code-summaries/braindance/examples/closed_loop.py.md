# closed_loop.py

**Path:** `braindance/examples/closed_loop.py`
**Module:** `braindance.examples.closed_loop`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Runs a resumable closed-loop plasticity pipeline from baseline recording through post-stimulation recording. A decorated selection phase samples neuron pairs with connectivity strictly between .3 and .6.

## Connections
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `phase` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `RecordPhaseV3` from `braindance.core.phases_v3.phases3` — imports (static evidence).
- **Uses:** `ClosedLoopPhaseV3` from `braindance.core.phases_v3.phases3_loop` — imports (static evidence).
- **Uses:** `ConnectivityPhaseV3` from `braindance.core.phases_v3.phases_analysis_3` — imports (static evidence).
- **Uses:** `RTSortPhaseV3` from `braindance.core.phases_v3.phases_analysis_3` — imports (static evidence).
- **Shared data:** RecordPhaseV3 -> RTSortPhaseV3 -> ConnectivityPhaseV3 -> select_random_pair -> ClosedLoopPhaseV3 -> RecordPhaseV3.

## Dependencies
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.phase` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3.RecordPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3_loop.ClosedLoopPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases_analysis_3.ConnectivityPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases_analysis_3.RTSortPhaseV3` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `select_random_pair(exp)`
> Pick a neuron pair whose connectivity falls in a target range.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/examples/closed_loop.py:25`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| CLI --config,--record_duration,--project_id,--chip_id,--experiment_name,--resume |
| Selection searches first 20 neurons for up to 500 attempts |

## Data Shapes
- exp.data.connectivity_matrix -> {'selected_pair': two-index ndarray}

## Notes
- Pair sampling can choose identical indices; raises when no acceptable sample found.
