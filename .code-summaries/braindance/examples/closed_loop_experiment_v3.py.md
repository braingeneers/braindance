# closed_loop_experiment_v3.py

**Path:** `braindance/examples/closed_loop_experiment_v3.py`
**Module:** `braindance.examples.closed_loop_experiment_v3`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Builds grouped baseline and post-stimulation analysis around neuron-pair selection and closed-loop stimulation. Reports selected-pair connectivity changes and diagnoses failed phases, with an alternate shorter pipeline.

## Connections
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `PhaseGroup` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `RecordPhaseV3` from `braindance.core.phases_v3.phases3` — imports (static evidence).
- **Uses:** `ClosedLoopPhaseV3` from `braindance.core.phases_v3.phases3_loop` — imports (static evidence).
- **Uses:** `SelectionPhaseV3` from `braindance.core.phases_v3.phases3_selection` — imports (static evidence).
- **Uses:** `ConnectivityPhaseV3` from `braindance.core.phases_v3.phases_analysis_3` — imports (static evidence).
- **Uses:** `FootprintPhaseV3` from `braindance.core.phases_v3.phases_analysis_3` — imports (static evidence).
- **Uses:** `RTSortPhaseV3` from `braindance.core.phases_v3.phases_analysis_3` — imports (static evidence).
- **Shared data:** SelectionPhaseV3 joins RTSort/Connectivity results to ClosedLoopPhaseV3; save_data/save_summary persist outputs.

## Dependencies
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.PhaseGroup` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3.RecordPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3_loop.ClosedLoopPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases3_selection.SelectionPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases_analysis_3.ConnectivityPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases_analysis_3.FootprintPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phases_analysis_3.RTSortPhaseV3` — intra-repo import; source import evidence.

## Classes
None

## Functions
### `run_closed_loop_plasticity_experiment(config)`
> Run a complete closed-loop plasticity experiment.
> **Called by:** braindance/examples/closed_loop_experiment_v3.py:222 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/closed_loop_experiment_v3.py:22`
### `analyze_plasticity_results(exp)`
> Analyze the plasticity induced by closed-loop stimulation.
> **Called by:** braindance/examples/closed_loop_experiment_v3.py:82 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/closed_loop_experiment_v3.py:98`
### `check_failure_point(exp)`
> Diagnose where the experiment failed.
> **Called by:** braindance/examples/closed_loop_experiment_v3.py:93 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/closed_loop_experiment_v3.py:153`
### `run_simple_test(config)`
> Run a simpler test version with shorter durations.
> **Called by:** braindance/examples/closed_loop_experiment_v3.py:220 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/closed_loop_experiment_v3.py:178`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| CLI --config,--stim_electrodes,--verbose,--record_duration,--simple_test |
| Pair connectivity .3â€“.6,min_distance=100; closed loop 600 s,400 mV,delay_ms=1 |

## Data Shapes
- Reads selected_neuron_pair,connectivity_matrix,neurons,spike_count,stim_count and named phase results

## Notes
- Calls Experiment(...,config=config), whose compatibility must be checked against current Experiment API.
- Labels >10% change significant without statistical test; baseline matrix reads current shared data.
