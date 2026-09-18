# 1_cartpole.py

**Path:** `braindance/examples/1_cartpole.py`
**Module:** `braindance.examples.1_cartpole`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Paper CartPole: recording → causal screening → ranked pairs → game.

## Connections
- **Uses:** `main` from `braindance.examples.paper_cartpole.game` — imports (static evidence).
- **Uses:** `find_connectivity_patterns` from `braindance.examples.paper_cartpole.ranking` — imports (static evidence).
- **Shared data:** The recording and causal stages spawn `braindance.examples.paper_cartpole.{recording,causal}`; rank uses `find_connectivity_patterns`; run calls the CartPole game stage.

## Dependencies
- `braindance.examples.paper_cartpole.game.main` — intra-repo import; source import evidence.
- `braindance.examples.paper_cartpole.ranking.find_connectivity_patterns` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `main()`
> CLI orchestrator for the four paper CartPole stages: dispatches recording/causal subprocesses, validates and ranks a derived connectivity matrix, or runs the selected game session.
> **Called by:** braindance/examples/1_cartpole.py:68 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/examples/1_cartpole.py:9`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Positional `stage`: recording, causal, rank, or run. |
| `--json/-j` is required and names the working experiment JSON. |
| `--derived-dir` overrides JSON `derived_dir`; `--order` is first or multi (default multi); `--select-rank` stores a 1-based ranked pair; `--run-index` defaults to 0 for C1/C2 trainer cycling. |

## Data Shapes
- Ranking loads `causal_connectivity_{order}.npy`, a finite square matrix matching `valid_stim_electrodes`; selected output writes sensory_electrodes, motor_electrodes, and pair_selection provenance to the JSON.

## Notes
None
