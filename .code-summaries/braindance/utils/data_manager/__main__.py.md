# __main__.py

**Path:** `braindance/utils/data_manager/__main__.py`
**Module:** `braindance.utils.data_manager.__main__`
**Feature Area:** `Configuration`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Implements the data-manager setup command for persistent input, catalog, and plot-output paths. Writes configuration through braindance.config setters and prints the resulting settings.

## Connections
- **Uses:** `get_catalog_path` from `braindance.config` — imports (static evidence).
- **Uses:** `get_data_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `get_output_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `set_catalog_path` from `braindance.config` — imports (static evidence).
- **Uses:** `set_data_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `set_output_dir` from `braindance.config` — imports (static evidence).
- **Shared data:** setup_command calls braindance.config path setters/getters.

## Dependencies
- `braindance.config.get_catalog_path` — intra-repo import; source import evidence.
- `braindance.config.get_data_dir` — intra-repo import; source import evidence.
- `braindance.config.get_output_dir` — intra-repo import; source import evidence.
- `braindance.config.set_catalog_path` — intra-repo import; source import evidence.
- `braindance.config.set_data_dir` — intra-repo import; source import evidence.
- `braindance.config.set_output_dir` — intra-repo import; source import evidence.

## Classes
None

## Functions
### `setup_command(args)`
> Setup data manager paths
> **Called by:** braindance/utils/data_manager/__main__.py:73 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/__main__.py:14`
### `main()`
> unclear — see source
> **Called by:** braindance/utils/data_manager/__main__.py:79 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/__main__.py:38`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| python -m braindance.utils.data_manager setup --data-dir PATH --catalog PATH --output-dir PATH. |

## Data Shapes
None

## Notes
- Writes ~/.braindance/config.json; reconfigures standard stream encoding when supported.
