# main.py

**Path:** `braindance/examples/streaming_workshop/main.py`
**Module:** `braindance.examples.streaming_workshop.main`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Runs the workshop as a headless experiment or loopback HTTP application. Routes profiles, phase validation, code export, uploads, acquisition controls, native sessions, playback, playground, sorting, catalog and analysis workspaces.

## Connections
- **Uses:** `get_output_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `AnalysisWorkspace` from `braindance.examples.streaming_workshop.analysis_workspace` — imports (static evidence).
- **Uses:** `CatalogWorkspace` from `braindance.examples.streaming_workshop.catalog_workspace` — imports (static evidence).
- **Uses:** `code_bundle` from `braindance.examples.streaming_workshop.code_export` — imports (static evidence).
- **Uses:** `export_bundle` from `braindance.examples.streaming_workshop.code_export` — imports (static evidence).
- **Uses:** `analysis_contracts` from `braindance.examples.streaming_workshop.custom_analysis` — imports (static evidence).
- **Uses:** `verify_analysis_code` from `braindance.examples.streaming_workshop.custom_analysis` — imports (static evidence).
- **Uses:** `phase_catalog` from `braindance.examples.streaming_workshop.experiment_spec` — imports (static evidence).
- **Uses:** `uses_native_runner` from `braindance.examples.streaming_workshop.experiment_spec` — imports (static evidence).
- **Uses:** `verify_spec` from `braindance.examples.streaming_workshop.experiment_spec` — imports (static evidence).
- **Uses:** `settings_payload` from `braindance.examples.streaming_workshop.global_settings` — imports (static evidence).
- **Uses:** `native_catalog` from `braindance.examples.streaming_workshop.native_catalog` — imports (static evidence).
- **Uses:** `NativeSession` from `braindance.examples.streaming_workshop.native_runner` — imports (static evidence).
- **Uses:** `PlaybackSession` from `braindance.examples.streaming_workshop.playback` — imports (static evidence).
- **Uses:** `inspect_playback` from `braindance.examples.streaming_workshop.playback` — imports (static evidence).
- **Uses:** `Playground` from `braindance.examples.streaming_workshop.playground` — imports (static evidence).
- **Uses:** `ProfileStore` from `braindance.examples.streaming_workshop.profiles` — imports (static evidence).
- **Uses:** `verify_python` from `braindance.examples.streaming_workshop.profiles` — imports (static evidence).
- **Uses:** `WorkshopSession` from `braindance.examples.streaming_workshop.session` — imports (static evidence).
- **Uses:** `SortingWorkspace` from `braindance.examples.streaming_workshop.sorting_workspace` — imports (static evidence).
- **Shared data:** Imports WorkshopSession and NativeSession and selects via experiment_spec.uses_native_runner for both initial and started experiments.
- **Shared data:** Delegates contracts to custom_analysis, syntax checks to profiles.verify_python, preflight to session runners, and exports to code_export.
- **Shared data:** Playground, AnalysisWorkspace, SortingWorkspace, CatalogWorkspace and PlaybackSession handle their respective API routes.
- **Shared data:** Serves watcher.html with an injected token and an explicit allowlist of local JS/CSS files; browser requests use their API routes.

## Dependencies
- `braindance.config.get_output_dir` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.analysis_workspace.AnalysisWorkspace` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.catalog_workspace.CatalogWorkspace` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.code_export.code_bundle` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.code_export.export_bundle` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.custom_analysis.analysis_contracts` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.custom_analysis.verify_analysis_code` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.experiment_spec.phase_catalog` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.experiment_spec.uses_native_runner` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.experiment_spec.verify_spec` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.global_settings.settings_payload` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.native_catalog.native_catalog` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.native_runner.NativeSession` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.playback.PlaybackSession` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.playback.inspect_playback` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.playground.Playground` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.profiles.ProfileStore` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.profiles.verify_python` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.session.WorkshopSession` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.sorting_workspace.SortingWorkspace` — intra-repo import; source import evidence.

## Classes
None

## Functions
### `main(environment='cartpole', source=None, live_config=None, channels=400, seed=7, record_seconds=3.0, environment_seconds=120.0, causal_repeats=4, speed=1.0, detection='threshold', threshold_uv=-30.0, write_output=True, output_dir=None, port=8765, headless=False, skip=False, functions_file=None, calibration=None, loop=False, open_browser=True, stim_electrodes=None, left_channels=None, right_channels=None, sorter_path=None, episode_seconds=10.0, profiles_dir=None, profile=None, num_neurons=40)`
> Build configuration and runner; execute headless or serve the local workshop with coordinated resource shutdown.
> **Called by:** braindance/examples/streaming_workshop/main.py:361 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/examples/streaming_workshop/main.py:26`
### `cli(argv=None)`
> Parse environment/acquisition/profile/UI flags and pass them to main.
> **Called by:** braindance/examples/streaming_workshop/main.py:365 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/main.py:330`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| CLI defaults: --environment cartpole (also foodland/ant), --channels 400, --num-neurons 40, --seed 7, --record-seconds 3, --environment-seconds 120, --episode-seconds 10, --causal-repeats 4. |
| Source controls: --source, --live-config, --loop, --speed (1 or max), --detection events/threshold/rt-sort (threshold default), --sorter-path, --threshold-uv -30; --write-output defaults on and --no-write-output disables it. |
| Run/UI controls: --output-dir, --port 8765, --headless, --skip, --no-browser, --functions-file, --calibration, --profiles-dir, --profile. |
| Routing CLI: --stim-electrodes accepts exactly two integers; --left-channels and --right-channels accept integer lists. |
| Configuration initializes amplitude_mv=100, phase_width_us=100 and grid_shape=[20,20] for 400 channels; output defaults to get_output_dir()/streaming_workshop. |

## Data Shapes
- JSON control commands are limited to 200000 bytes; participant/settings/specification dictionaries flow to delegated services.
- Replay/sorting uploads are streamed in 1 MiB chunks with a 64 GiB bound; .cfg uploads are limited to 10 MiB; upload names use random tokens.
- /api/phase-catalog merges binned/custom-analysis phase definitions with the curated native catalog.
- /api/state returns session snapshot plus execution source, timing, engine, bin_ms and hard_realtime=False; native timing has no fixed bin_ms.
- Profiles store path/name/settings selection; start requests merge base config, selected profile settings and allowlisted setup overrides.

## Notes
- Binds only 127.0.0.1; validates localhost Host headers and a per-launch random X-Workshop-Token for POST controls; download uses a token query parameter.
- Nested Handler closes over mutable session and workspace instances; native phases reject cooperative pause/restart commands, playback accepts pause/step/stop.
- Preflight writes unsaved code in a temporary directory, runs the selected session verify_only, and removes temporary data; this is trusted local Python execution.
- Dependency installation blocks acquisition and conflicting analysis activity and requires server restart before subsequent runs.
- Shutdown closes workspaces/playground, requests acquisition stop, joins up to ten seconds and closes the HTTP server.
