# 2_rapid_pairing.py

**Path:** `braindance/examples/2_rapid_pairing.py`
**Module:** `braindance.examples.2_rapid_pairing`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Closed-loop plasticity experiment using Phase V3. Phases: 0. Record baseline spontaneous activity 1. RT-Sort to detect neurons 2. Compute connectivity matrix 3. Select a neuron pair with moderate connectivity 4. Closed-loop stimulation 5. Record post-stimulation activity Use ``--resume`` to pick up from the last successful phase if a previous run failed partway through.

## Connections
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `phase` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `RecordPhaseV3` from `braindance.core.phases_v3.phases3` — imports (static evidence).
- **Uses:** `ClosedLoopPhaseV3` from `braindance.core.phases_v3.phases3_loop` — imports (static evidence).
- **Uses:** `ConnectivityPhaseV3` from `braindance.core.phases_v3.phases_analysis_3` — imports (static evidence).
- **Uses:** `RTSortPhaseV3` from `braindance.core.phases_v3.phases_analysis_3` — imports (static evidence).
- **Shared data:** Builds an Experiment pipeline: RecordPhaseV3 → RTSortPhaseV3 → ConnectivityPhaseV3 → select_random_pair → ClosedLoopPhaseV3 → RecordPhaseV3.

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
> Samples a pair from the first 20 connectivity-matrix neurons whose directed connectivity lies strictly between .3 and .6; raises ValueError after 500 misses.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/examples/2_rapid_pairing.py:19`
### `main()`
> Parses the closed-loop experiment options, constructs the six-phase V3 pipeline, and runs it with optional resume.
> **Called by:** braindance/examples/2_rapid_pairing.py:100 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/2_rapid_pairing.py:37`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `--config` Maxwell config (default None); `--record_duration` seconds (default 300); `--project_id` closed_loop; `--chip_id` default_chip; `--experiment_name` closed_loop_plasticity; `--resume` resumes the last successful phase. |

## Data Shapes
- `select_random_pair` reads `exp.data.connectivity_matrix`, samples two indices among at most 20 neurons for up to 500 attempts, and returns `{"selected_pair": numpy_array}` when the value is strictly between .3 and .6.

## Notes
- Creates and runs a Phase V3 experiment; hardware/config requirements are delegated to the selected phase implementations.
