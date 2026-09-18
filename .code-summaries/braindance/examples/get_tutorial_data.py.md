# get_tutorial_data.py

**Path:** `braindance/examples/get_tutorial_data.py`
**Module:** `braindance.examples.get_tutorial_data`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Command-line entry point that downloads a named managed tutorial dataset, or delegates to the frozen scientific validator. It prints and returns the resolved tutorial directory.

## Connections
- **Uses:** `main` from `braindance.examples.validate_tutorial_data` — imports (static evidence).
- **Uses:** `get_test_data` from `braindance.tutorial` — imports (static evidence).
- **Shared data:** Dispatches to `braindance.tutorial.get_test_data` or `braindance.examples.validate_tutorial_data.main`.

## Dependencies
- `braindance.examples.validate_tutorial_data.main` — intra-repo import; source import evidence.
- `braindance.tutorial.get_test_data` — intra-repo import; source import evidence.

## Classes
None

## Functions
### `main(name='closed-loop-small', version=None, cache_dir=None, offline=False, update=False, manifest_path=None, validate=False)`
> Download the requested tutorial version with cache/offline/update controls, optionally running full reference validation.
> **Called by:** braindance/examples/get_tutorial_data.py:35 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/get_tutorial_data.py:8`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| CLI: `--name`, `--version`, `--cache-dir`, `--manifest-path`, `--offline`, `--update`, and `--validate`; `name` defaults to `closed-loop-small`. |

## Data Shapes
- Returns a filesystem path for the managed tutorial folder; validation mode returns the validator's result.

## Notes
- Imports download or validation code lazily.
