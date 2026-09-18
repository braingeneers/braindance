# load_context.py

**Path:** `braindance/examples/load_context.py`
**Module:** `braindance.examples.load_context`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Loads saved experiment data and opens an interactive IPython session. The requested experiment name is combined with a fixed saved-run directory.

## Connections
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Shared data:** Uses phases_v3.Experiment then IPython.embed.

## Dependencies
- `IPython` — external or unresolved local import; source import evidence.
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.

## Classes
None

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Required CLI --experiment; save_dir=./test_experiments/spike_sort_test |

## Data Shapes
- Experiment.load_data populates exp.data

## Notes
- Parses arguments and opens interactive shell at import.
