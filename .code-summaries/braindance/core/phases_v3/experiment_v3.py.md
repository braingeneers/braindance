# experiment_v3.py

**Path:** `braindance/core/phases_v3/experiment_v3.py`
**Module:** `braindance.core.phases_v3.experiment_v3`
**Feature Area:** `Experiment Phases`
**Entry point:** no — library or imported component

## Overview
Runs ordered V3 phases and shared-environment groups with canonical input/output dependency tracking, acquisition lifecycle management and numbered recording directories. Persists phase outputs before atomically logging verified checkpoint boundaries for fail-closed resume.

## Connections
- **Used by:** `braindance.cli.replay` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_ant` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_ant_cpg` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_foodland` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_loop` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_pacman` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_walker2d` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_walker2d_cpg` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases_analysis_3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.2_rapid_pairing` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.analysis_checkpoint_smoke` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.ant_cpg_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.ant_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.cartpole_ml` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.cartpole_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.closed_loop` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.closed_loop_experiment_v3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.foodland_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.labyrinth` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.load_context` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.mspacman_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.replay_experiment_v3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.simple_experiment_v3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.native_runner` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.session` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.walker2d_cpg_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.walker2d_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Uses:** `Mapping` from `braindance.analysis.mapping` — imports (static evidence).
- **Uses:** `get_data_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `MaxwellEnv` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `maxwell_params` from `braindance.core.params` — imports (static evidence).
- **Uses:** `DataContext` from `braindance.core.phases_v3.data_context` — imports (static evidence).
- **Uses:** `DataDependencyTracker` from `braindance.core.phases_v3.data_context` — imports (static evidence).
- **Uses:** `AnalysisPhaseV3` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `PhaseGroup` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `PhaseV3` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `PhaseValidator` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `ValidationError` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Shared data:** Uses PhaseValidator and DataDependencyTracker with canonical phase.inputs/phase.outputs.
- **Shared data:** Creates MaxwellEnv lazily for hardware phases and allows analysis-only pipelines without opening acquisition.
- **Shared data:** Delegates format-aware persistence to DataContext and electrode mapping loading to braindance.analysis.mapping.Mapping.
- **Shared data:** WorkshopExperiment overrides _execute_phase for retries and runtime injection; this hook runs inside the normal setup/cleanup/checkpoint sequence.

## Dependencies
- `braindance.analysis.mapping.Mapping` — intra-repo import; source import evidence.
- `braindance.config.get_data_dir` — intra-repo import; source import evidence.
- `braindance.core.maxwell_env.MaxwellEnv` — intra-repo import; source import evidence.
- `braindance.core.params.maxwell_params` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.data_context.DataContext` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.data_context.DataDependencyTracker` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.AnalysisPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.PhaseGroup` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.PhaseV3` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.PhaseValidator` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.ValidationError` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### Experiment()
> Streamlined experiment manager with automatic data passing and smart params.
**Source:** `braindance/core/phases_v3/experiment_v3.py:22`
**Kind:** class. **Instantiated by:** braindance/cli/replay.py:223 (named-call hint); braindance/examples/2_rapid_pairing.py:65 (named-call hint); braindance/examples/analysis_checkpoint_smoke.py:30 (named-call hint); braindance/examples/analysis_checkpoint_smoke.py:34 (named-call hint); braindance/examples/ant_cpg_example.py:37 (named-call hint); braindance/examples/ant_cpg_example.py:80 (named-call hint); braindance/examples/ant_reinforcement_example.py:30 (named-call hint); braindance/examples/cartpole_ml.py:41 (named-call hint); braindance/examples/cartpole_reinforcement_example.py:26 (named-call hint); braindance/examples/closed_loop.py:64 (named-call hint); braindance/examples/closed_loop_experiment_v3.py:182 (named-call hint); braindance/examples/closed_loop_experiment_v3.py:32 (named-call hint); braindance/examples/foodland_reinforcement_example.py:20 (named-call hint); braindance/examples/labyrinth.py:19 (named-call hint); braindance/examples/load_context.py:12 (named-call hint); braindance/examples/mspacman_reinforcement_example.py:16 (named-call hint); braindance/examples/replay_experiment_v3.py:31 (named-call hint); braindance/examples/simple_experiment_v3.py:103 (named-call hint); braindance/examples/simple_experiment_v3.py:27 (named-call hint); braindance/examples/walker2d_cpg_example.py:40 (named-call hint); braindance/examples/walker2d_cpg_example.py:94 (named-call hint); braindance/examples/walker2d_reinforcement_example.py:43 (named-call hint); braindance/examples/walker2d_reinforcement_example.py:99 (named-call hint)
**Constructor:** `__init__(self, name: str, params: Union[str, dict]=None, save_dir: str=None, project_id: str=None, chip_id: str=None, base_dir: str=None, results_dir_name: str='results', auto_load_data: bool=True, overwrite_existing: bool=False)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_auto_loaded_keys` | inferred at runtime | `set()` |
| `_phase_output_keys` | inferred at runtime | `set()` |
| `chip_dir` | Optional[Path] | `None` |
| `chip_id` | inferred at runtime | `chip_id` |
| `current_env` | inferred at runtime | `None` |
| `current_phase_idx` | inferred at runtime | `0` |
| `current_recording_count` | Optional[str] | `None` |
| `current_recording_dir` | Optional[Path] | `None` |
| `data` | inferred at runtime | `DataContext(overwrite_existing=overwrite_existing)` |
| `data_tracker` | inferred at runtime | `DataDependencyTracker()` |
| `mapping` | inferred at runtime | `None` |
| `metadata` | inferred at runtime | `{'created': datetime.datetime.now().isoformat(), 'name': name, 'version': '3.0', 'project_id': project_id, 'chip_id': chip_id, 'c…` |
| `name` | inferred at runtime | `name` |
| `params` | inferred at runtime | `{}` |
| `phases` | List[Union[PhaseV3, PhaseGroup]] | `[]` |
| `project_id` | inferred at runtime | `project_id` |
| `results` | List[Dict] | `[]` |
| `results_dir_name` | inferred at runtime | `results_dir_name` |
| `save_dir` | inferred at runtime | `Path(save_dir)` |
| `start_time` | inferred at runtime | `None` |
| `verbose` | inferred at runtime | `True` |
**Methods:**
#### `_auto_load_existing_data(self)`
> Automatically load existing data from the most recent recording's results directory, or fall back to the legacy ``save_dir/data`` path.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/experiment_v3.py:134`
#### `_try_load_from_dir(self, data_dir: Path)`
> Attempt to load DataContext data from a directory.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:163`
#### `_get_checkpoint(self) -> int`
> Locate the first uncompleted phase while verifying identities, group boundaries and all committed output artifacts.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:189`
#### `_output_digest(path: Path, file_format: str, key: str) -> str`
> Hash a keyed JSON value or full serialized artifact for checkpoint verification.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:267`
#### `_persist_phase_outputs(self, result: Dict) -> List[Dict]`
> Strictly persist this phase result and return verified format/path/hash descriptors.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/phases_v3/experiment_v3.py:278`
#### `_read_checkpoint_output(self, output: Dict) -> Any`
> Deserialize an artifact only after checking its exact current hash.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:297`
#### `_load_all_recording_data(self, completed_before: Optional[int]=None)`
> Restore committed checkpoint outputs or load accumulated recording results for an explicit non-checkpoint load.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/experiment_v3.py:335`
#### `_list_recording_dirs(self) -> List[Path]`
> Return existing recording subdirectories sorted by name.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:393`
#### `_get_next_recording_id(self, tag: str='rec') -> str`
> Return the next auto-incremented recording directory name for *tag*.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:404`
#### `_create_recording_dir(self, tag: str='rec') -> Path`
> Create the next recording subdirectory and set it as current.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/experiment_v3.py:422`
#### `_save_chip_metadata(self)`
> Write a stub ``chip_metadata.json`` at the chip directory level.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/phases_v3/experiment_v3.py:440`
#### `_save_recording_metadata(self, phase: PhaseV3, recording_dir: Path, duration: float=None)`
> Write ``[count]_metadata.json`` inside a recording directory.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/phases_v3/experiment_v3.py:463`
#### `load_params(self, params: Union[str, dict, Path]) -> 'Experiment'`
> Load configuration and expose non-environment values as shared phase data.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/experiment_v3.py:484`
#### `set_param(self, key: str, value: Any) -> 'Experiment'`
> Set a param value. Returns self for chaining.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/experiment_v3.py:500`
#### `get_param(self, key: str, default: Any=None) -> Any`
> Get a param value.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:505`
#### `load_mapping(self, mapping_file: Union[str, Path, None]=None) -> 'Experiment'`
> Load physical electrode mapping from an explicit mapping path or the current recording file.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/experiment_v3.py:509`
#### `add_phase(self, phase: Union[PhaseV3, PhaseGroup]) -> 'Experiment'`
> Register a phase or group and track its canonical contracts.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:523`
#### `add_phase_group(self, phases: List[PhaseV3], name: str=None) -> 'Experiment'`
> Add a group of phases that share an environment. Returns self for chaining.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:536`
#### `add_phases(self, *phases: Union[PhaseV3, PhaseGroup]) -> 'Experiment'`
> Add multiple phases at once. Returns self for chaining.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:541`
#### `run(self, start_from: int=0, stop_at: int=None, validate: bool=True, resume: bool=False) -> bool`
> Run experiment phases.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/experiment_v3.py:549`
#### `_execute_phase(self, phase)`
> Run scientific phase work through an overridable runner hook.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:656`
#### `_run_single_phase(self, phase: PhaseV3, phase_idx: int, checkpoint_boundary: bool=True) -> bool`
> Configure, execute and clean up a phase, persist its outputs, then log success or failure.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:660`
#### `_run_phase_group(self, group: PhaseGroup, group_idx: int) -> bool`
> Mark group start, execute children on a shared environment and publish only a completed group boundary.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:701`
#### `_setup_phase(self, phase: PhaseV3)`
> Bind experiment/data/environment references and track declared data reads.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:737`
#### `_cleanup_phase(self, phase: PhaseV3)`
> Invoke phase cleanup and close acquisition when requested.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:757`
#### `_cleanup_experiment(self)`
> Final experiment cleanup.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:766`
#### `_create_environment_for_phase(self, phase: PhaseV3)`
> Create Maxwell environment for phase.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/experiment_v3.py:773`
#### `_close_environment(self)`
> Close current Maxwell environment.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/experiment_v3.py:819`
#### `_summarize_result(result: Optional[Dict]) -> Dict`
> Create a JSON-safe summary of a phase result dict.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:830`
#### `_record_phase_completion(self, phase: PhaseV3, phase_idx: int, success: bool, duration: float, result: Dict, checkpoint_boundary: bool=True, persisted_outputs: Optional[List[Dict]]=None, event: str='completed')`
> Record phase completion and update the experiment log on disk.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:852`
#### `_handle_phase_failure(self, phase: PhaseV3, phase_idx: int, error: Exception, checkpoint_boundary: bool=True)`
> Handle phase failure.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:890`
#### `_save_experiment_log(self)`
> Atomically replace the durable checkpoint log after flushing the temporary file.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/phases_v3/experiment_v3.py:917`
#### `get_result(self, phase_name: str=None, phase_idx: int=None) -> Optional[Dict]`
> Get result from a specific phase.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:954`
#### `get_last_result(self, key: str=None) -> Any`
> Get the last result or specific key from it.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:963`
#### `save_summary(self, path: Optional[Path]=None) -> Path`
> Save a JSON experiment summary at the experiment level.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/phases_v3/experiment_v3.py:975`
#### `save_data(self, path: Optional[Path]=None, overwrite_files: bool | None=None) -> Path`
> Save experiment data context.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/phases_v3/experiment_v3.py:1044`
#### `load_data(self, path: Optional[Path]=None, keys: Optional[List[str]]=None)`
> Load experiment data context.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:1063`
#### `load_data_from_experiment(self, experiment_name: str, experiment_dir: str=None, keys: Optional[List[str]]=None)`
> Load data from another experiment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:1080`
#### `get_data_index_info(self) -> Optional[Dict]`
> Get information about the data index file.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:1113`
#### `print_data_summary(self)`
> Print a summary of current data and saved data.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:1140`
#### `__repr__(self) -> str`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/experiment_v3.py:1176`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Constructor accepts params dict or JSON path, explicit save_dir or base_dir/project_id/chip_id/name hierarchy, results_dir_name, auto_load_data and overwrite_existing. |
| Default base_dir comes from braindance.config.get_data_dir(); nested maxwell_env configuration is kept out of shared phase data. |
| run accepts start_from, exclusive stop_at, validate and resume; a phase recording_tag creates a numbered recording directory. |
| Acquisition construction uses Maxwell defaults plus experiment maxwell_env settings, config, stim_electrodes and phase customization; default raw acquisition maximum is 3600 seconds. |

## Data Shapes
- Shared DataContext is keyed by string phase inputs/outputs; phase run results are dictionaries.
- Checkpoint schema version 1 logs phase identity, success, boundary/event, provided_keys and persisted_outputs descriptors containing key, format, relative path and SHA-256.
- Outputs persist as keyed results.json values, .npy arrays, .npz dictionaries or .pkl artifacts according to DataContext format metadata.
- Recording directories are NNN_tag; metadata and phase summaries store type/shape descriptions for non-JSON values.

## Notes
- Resume rejects legacy/corrupt logs, changed phase identities, incomplete groups, noncontiguous successes and missing/modified output artifacts.
- Resume restores committed artifacts directly and verifies the bytes used for deserialization; explicit inputs survive removal of uncommitted auto-loaded values.
- _execute_phase is the external-runner extension hook for controls and observation; core execution remains phase.run(experiment).
- PhaseGroup children share an acquisition environment; source notes that per-child differing stimulation configuration is not independently reconfigured.
- get_last_result(key) consults a result field, while current completion records store result_summary; use shared data for actual phase output values.
