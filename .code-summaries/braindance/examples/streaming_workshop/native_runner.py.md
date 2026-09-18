# native_runner.py

**Path:** `braindance/examples/streaming_workshop/native_runner.py`
**Module:** `braindance.examples.streaming_workshop.native_runner`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Validates and runs scientific V3 sequences in an isolated Python subprocess with retained experiment files, console logs and atomic progress snapshots. Observes supported game phases externally so CartPole, FoodLand and Ant can publish browser geometry without transport code inside phases.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.experiment_spec` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.main` — import consumer hint; not a proven runtime call.
- **Uses:** `DataContext` from `braindance.core.phases_v3.data_context` — imports (static evidence).
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `PhaseValidator` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `NeuralSimulationSource` from `braindance.core.simulation` — imports (static evidence).
- **Uses:** `CustomAnalysisPhase` from `braindance.examples.streaming_workshop.custom_analysis` — imports (static evidence).
- **Uses:** `analysis_definition` from `braindance.examples.streaming_workshop.custom_analysis` — imports (static evidence).
- **Uses:** `resolve_analysis_config` from `braindance.examples.streaming_workshop.custom_analysis` — imports (static evidence).
- **Uses:** `verify_analysis_code` from `braindance.examples.streaming_workshop.custom_analysis` — imports (static evidence).
- **Uses:** `instantiate_native` from `braindance.examples.streaming_workshop.native_catalog` — imports (static evidence).
- **Uses:** `native_catalog` from `braindance.examples.streaming_workshop.native_catalog` — imports (static evidence).
- **Uses:** `validate_native_parameters` from `braindance.examples.streaming_workshop.native_catalog` — imports (static evidence).
- **Uses:** `ObservedGame` from `braindance.examples.streaming_workshop.native_visualization` — imports (static evidence).
- **Shared data:** native_catalog performs AST schema checks and selected-class construction; custom_analysis resolves participant declarations and executes custom analysis phases.
- **Shared data:** Experiment, PhaseValidator and DataContext provide V3 lifecycle, dependency checks and data persistence.
- **Shared data:** NativeExperiment wraps phase.game_env with native_visualization.ObservedGame during supported game phases and restores it afterward.

## Dependencies
- `braindance.core.phases_v3.data_context.DataContext` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.PhaseValidator` — intra-repo import; source import evidence.
- `braindance.core.simulation.NeuralSimulationSource` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.custom_analysis.CustomAnalysisPhase` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.custom_analysis.analysis_definition` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.custom_analysis.resolve_analysis_config` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.custom_analysis.verify_analysis_code` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.native_catalog.instantiate_native` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.native_catalog.native_catalog` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.native_catalog.validate_native_parameters` — intra-repo import; source import evidence.
- `braindance.examples.streaming_workshop.native_visualization.ObservedGame` — intra-repo import; source import evidence.

## Classes
### NativeSession()
> Expose threaded start/stop-event control and state snapshots around a subprocess-based V3 run.
**Source:** `braindance/examples/streaming_workshop/native_runner.py:68`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, output_dir, config)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `error` | inferred at runtime | `tail if self.status == 'error' else ''` |
| `process` | inferred at runtime | `process` |
| `run_dir` | inferred at runtime | `run_dir` |
| `snapshot` | inferred at runtime | `dict(status='ready', native=True, restart_supported=False)` |
| `status` | inferred at runtime | `'running'` |
| `stop_event` | inferred at runtime | `threading.Event()` |
| `thread` | inferred at runtime | `None` |
**Methods:**
#### `start(self, overrides=None, skip=False)`
> Apply overrides and start a daemon worker, rejecting concurrent runs.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/examples/streaming_workshop/native_runner.py:76`
#### `run(self, verify_only=False)`
> Run the child lifecycle, capture exceptions and terminate remaining child processes on failure.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/examples/streaming_workshop/native_runner.py:84`
#### `_run(self, verify_only=False)`
> Create attempt files, spawn the native worker, relay progress and retain completion/error logs.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/examples/streaming_workshop/native_runner.py:97`

## Functions
### `verify_native(config, code=None)`
> Resolve code-owned analysis contracts and validate phase IDs, constructor parameters, mapping counts and ordered data dependencies.
> **Called by:** braindance/examples/streaming_workshop/experiment_spec.py:85 (named-call hint); braindance/examples/streaming_workshop/native_runner.py:148 (named-call hint); braindance/examples/streaming_workshop/native_runner.py:98 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/streaming_workshop/native_runner.py:14`
### `main(spec_path, verify=False)`
> Construct and validate phases, optionally stop after constructor verification, or execute them in a scoped NativeExperiment.
> **Called by:** braindance/examples/streaming_workshop/native_runner.py:241 (named-call hint). **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/examples/streaming_workshop/native_runner.py:142`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| CLI: python -m braindance.examples.streaming_workshop.native_runner SPEC_PATH [--verify]. |
| Config keys include phases, experiment_name, native_settings, initial_data, functions_file, source/live_config, channels, seed, speed, loop, adjacency and optional spatial simulation settings. |
| Constructor verification has a 45-second timeout; parent polls progress every 0.15 seconds. |

## Data Shapes
- Phase specs contain id, type, params and optional per-phase settings; validation returns ok/errors/caveats and rows with canonical input origins and outputs.
- Each run creates native_<time_ns>/experiment.json, progress.json, console.log, attempt.json plus V3 outputs/checkpoints.
- Session snapshot contains status/native/restart_supported, phase IDs, output path and worker progress; game progress additionally carries phase_kind, scene, reward and episodes.

## Notes
- Child environment copies os.environ and prepends the repository root to PYTHONPATH; execution uses the current Python interpreter.
- Scientific and interactive phases cannot mix in one sequence. Custom-analysis-only sequences use this runner without acquisition.
- Browser construction sets supported render_mode arguments to None; scientific execution still needs the relevant phase dependencies.
- Each acquisition phase receives a fresh simulation/replay source because preceding phase cleanup closes its source. Simulated chunks target 20 ms.
- Per-phase settings temporarily overlay experiment params/data and are removed after execution unless returned as declared outputs; scoped validation occurs before experiment.run(validate=False).
