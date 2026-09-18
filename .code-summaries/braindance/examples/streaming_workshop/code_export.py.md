# code_export.py

**Path:** `braindance/examples/streaming_workshop/code_export.py`
**Module:** `braindance.examples.streaming_workshop.code_export`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Generates readable standalone Python experiment bundles from workshop settings and exact participant code. Preview is read-only; validated exports create unique new directories and retain a builder snapshot.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.main` — import consumer hint; not a proven runtime call.
- **Uses:** `resolve_analysis_config` from `braindance.examples.streaming_workshop.custom_analysis` — imports (static evidence).
- **Uses:** `verify_analysis_code` from `braindance.examples.streaming_workshop.custom_analysis` — imports (static evidence).
- **Uses:** `normalize_phases` from `braindance.examples.streaming_workshop.experiment_spec` — imports (static evidence).
- **Uses:** `verify_spec` from `braindance.examples.streaming_workshop.experiment_spec` — imports (static evidence).
- **Shared data:** custom_analysis.resolve_analysis_config and verify_analysis_code establish analysis contract consistency.
- **Shared data:** experiment_spec.normalize_phases supplies default phases; verify_spec gates export.
- **Shared data:** main.py serves these functions through /api/code-preview and /api/code-export; generated experiment.py uses WorkshopSession/NativeSession and get_output_dir.

## Dependencies
- `braindance.examples.streaming_workshop.custom_analysis.resolve_analysis_config` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.custom_analysis.verify_analysis_code` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.experiment_spec.normalize_phases` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.experiment_spec.verify_spec` — intra-repo import; source import evidence.

## Classes
None

## Functions
### `code_bundle(config, participant_code)`
> Construct editable setup, runner and participant source files while collecting contract and Python syntax errors.
> **Called by:** braindance/examples/streaming_workshop/code_export.py:133 (named-call hint); braindance/examples/streaming_workshop/main.py:228 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/code_export.py:9`
### `export_bundle(output_dir, config, participant_code)`
> Require valid bundle/specification then create a unique export directory, write files and preserve the resolved builder configuration.
> **Called by:** braindance/examples/streaming_workshop/main.py:229 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/examples/streaming_workshop/code_export.py:131`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Participant code must be a string of at most 100000 characters; configuration must serialize as JSON without NaN. |
| Export directories are output_dir/exports/<sanitized experiment_name>_<time_ns>; experiment names are truncated to 64 characters. |
| Generated experiment CLI supports --verify and --output-dir; default run output is get_output_dir()/streaming_workshop. |

## Data Shapes
- code_bundle returns {files: {filename: text}, errors: [message]}; bundle files are experiment.py, environment_setup.py, participant_functions.py and README.md.
- Generated PHASES hold ordered phase specs; SETTINGS retain shared source/acquisition settings after phases/functions_file/headless are removed.
- export_bundle writes builder_snapshot.json and returns {directory,files} mapping exported bundle names to paths.

## Notes
- compile checks generated Python syntax but does not execute participant code during bundle construction.
- Custom-analysis decorators are resolved before export and checked against phase declarations; invalid bundles/specs are not written.
- Fresh directories use exist_ok=False so previous hand-edited exports are never overwritten.
- Generated scripts select NativeSession or WorkshopSession using uses_native_runner; participant callbacks are not automatically wired to native trainers.
