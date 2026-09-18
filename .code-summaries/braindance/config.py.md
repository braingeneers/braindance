# config.py

**Path:** `braindance/config.py`
**Module:** `braindance.config`
**Feature Area:** `Configuration`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Resolves BrainDance data, catalog, output, and automatic spike-extraction settings from environment variables, a user JSON file, or defaults. It also exposes validated atomic bulk updates, legacy individual setters, and a small data-directory CLI.

## Connections
- **Used by:** `braindance` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.cli.replay` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.experiment_v3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.replay` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.3_busybee` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.analysis_checkpoint_smoke` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.ant_cpg_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.labyrinth` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.paper_cartpole.game` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.replay_experiment_v3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.analysis_workspace` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.catalog_workspace` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.global_settings` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.main` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.walker2d_cpg_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.walker2d_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.pong` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.real_time_sorting` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.tutorial` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.__main__` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.data_loading.catalog` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.data_loading.recording` — import consumer hint; not a proven runtime call.
- **Shared data:** Data-manager and analysis code call these getters to locate user data, the catalog, outputs, and the default RT-Sort extraction behavior.

## Dependencies
None

## Classes
None

## Functions
### `get_global_settings()`
> Report saved and effective settings while identifying active environment overrides.
> **Called by:** braindance/config.py:65 (named-call hint); braindance/examples/streaming_workshop/global_settings.py:9 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/config.py:7`
### `save_global_settings(updates)`
> Validate a partial settings update and atomically persist it, treating null values as deletion/reset.
> **Called by:** braindance/examples/streaming_workshop/global_settings.py:9 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/config.py:23`
### `get_data_dir()`
> Resolve the data directory by environment, config file, then home-directory default.
> **Called by:** braindance/__init__.py:70 (named-call hint); braindance/cli/replay.py:43 (named-call hint); braindance/config.py:131 (named-call hint); braindance/config.py:222 (named-call hint); braindance/config.py:252 (named-call hint); braindance/config.py:254 (named-call hint); braindance/core/phases_v3/experiment_v3.py:79 (named-call hint); braindance/core/phases_v3/experiment_v3.py:86 (named-call hint); braindance/core/replay.py:108 (named-call hint); braindance/examples/ant_cpg_example.py:29 (named-call hint); braindance/examples/replay_experiment_v3.py:16 (named-call hint); braindance/examples/streaming_workshop/analysis_workspace.py:62 (named-call hint); braindance/examples/streaming_workshop/catalog_workspace.py:44 (named-call hint); braindance/examples/streaming_workshop/catalog_workspace.py:79 (named-call hint); braindance/examples/walker2d_cpg_example.py:32 (named-call hint); braindance/examples/walker2d_reinforcement_example.py:35 (named-call hint); braindance/experiments/real_time_sorting.py:17 (named-call hint); braindance/tutorial.py:271 (named-call hint); braindance/utils/data_manager/__init__.py:121 (named-call hint); braindance/utils/data_manager/__main__.py:33 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/config.py:67`
### `set_data_dir(path)`
> Legacy setter that stores an absolute data-directory path.
> **Called by:** braindance/config.py:250 (named-call hint); braindance/utils/data_manager/__main__.py:17 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/config.py:91`
### `get_catalog_path()`
> Resolve the catalog CSV path, defaulting under the effective data directory.
> **Called by:** braindance/examples/streaming_workshop/catalog_workspace.py:39 (named-call hint); braindance/utils/data_manager/__init__.py:116 (named-call hint); braindance/utils/data_manager/__init__.py:83 (named-call hint); braindance/utils/data_manager/__main__.py:34 (named-call hint); braindance/utils/data_manager/utils/data_loading/catalog.py:565 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/config.py:111`
### `set_catalog_path(path)`
> Legacy setter that stores an absolute catalog path.
> **Called by:** braindance/utils/data_manager/__init__.py:89 (named-call hint); braindance/utils/data_manager/__main__.py:21 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/config.py:133`
### `get_auto_extract_spike_info()`
> Resolve whether RT-Sort spike-info extraction is automatically enabled.
> **Called by:** braindance/utils/data_manager/utils/data_loading/recording.py:1628 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/config.py:152`
### `set_auto_extract_spike_info(enabled: bool)`
> Persist the automatic extraction flag.
> **Called by:** unclear — see source. **Side effects:** write call (static hint).
**Source:** `braindance/config.py:178`
### `get_output_dir()`
> Resolve the plot/results directory, defaulting under the effective data directory.
> **Called by:** braindance/cli/replay.py:200 (named-call hint); braindance/examples/3_busybee.py:27 (named-call hint); braindance/examples/analysis_checkpoint_smoke.py:27 (named-call hint); braindance/examples/labyrinth.py:21 (named-call hint); braindance/examples/paper_cartpole/game.py:59 (named-call hint); braindance/examples/replay_experiment_v3.py:34 (named-call hint); braindance/examples/streaming_workshop/main.py:57 (named-call hint); braindance/experiments/pong.py:58 (named-call hint); braindance/experiments/real_time_sorting.py:37 (named-call hint); braindance/utils/data_manager/__main__.py:35 (named-call hint); braindance/utils/data_manager/utils/data_loading/recording.py:772 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/config.py:202`
### `set_output_dir(path)`
> Legacy setter that stores an absolute output-directory path.
> **Called by:** braindance/utils/data_manager/__main__.py:25 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/config.py:224`
### `main(argv=None)`
> Implement the module CLI for getting or setting the data directory.
> **Called by:** braindance/config.py:259 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/config.py:243`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Settings file: `~/.braindance/config.json` with keys `data_dir`, `catalog_path`, `output_dir`, and `auto_extract_spike_info`. |
| Environment overrides: `BRAINDANCE_DATA_DIR`, `BRAINDANCE_CATALOG_PATH`, `BRAINDANCE_OUTPUT_DIR`, and `BRAINDANCE_AUTO_EXTRACT_SPIKE_INFO`. |
| CLI accepts `--set_data_dir PATH` and `--get_data_dir`; other settings are managed through Python APIs. |

## Data Shapes
- `get_global_settings` returns `{config_path, settings}` where each setting has `saved`, `effective`, and active `environment` fields.
- The persisted configuration must be a JSON object; path settings are serialized as strings and the automatic-extraction flag as a boolean.

## Notes
- Environment variables take precedence over saved values; defaults are `~/braindance_data`, `<data_dir>/catalog.csv`, and `<data_dir>/outputs`.
- `save_global_settings` validates path kind when the target exists and uses a same-directory temporary file plus replace for atomic persistence.
- Legacy individual setters do less validation and write the config file non-atomically.
