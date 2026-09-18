# phases_analysis_3.py

**Path:** `braindance/core/phases_v3/phases_analysis_3.py`
**Module:** `braindance.core.phases_v3.phases_analysis_3`
**Feature Area:** `Neural Analysis`
**Entry point:** no — library or imported component

## Overview
Defines V3 analysis phases for activity, RT-Sort initialization/offline sorting, footprint calculation, spike-time-tiling connectivity, and placeholder causal analysis. RTSortPhaseV3 is the functional integration point that builds or loads an RT-Sort object, derives channel/electrode/position metadata, saves sorter state, and emits `SpikeData` artifacts.

## Connections
- **Used by:** `braindance.examples.2_rapid_pairing` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.ant_cpg_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.ant_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.cartpole_ml` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.cartpole_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.closed_loop` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.closed_loop_experiment_v3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.foodland_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.mspacman_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.walker2d_cpg_example` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.walker2d_reinforcement_example` — import consumer hint; not a proven runtime call.
- **Uses:** `data_loader` from `braindance.analysis` — imports (static evidence).
- **Uses:** `load_data_maxwell` from `braindance.analysis.data_loader` — imports (static evidence).
- **Uses:** `Experiment` from `braindance.core.phases_v3.experiment_v3` — imports (static evidence).
- **Uses:** `AnalysisPhaseV3` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Uses:** `RunKilosort` from `braindance.core.spikesorter.kilosort2` — imports (static evidence).
- **Uses:** `RTSort` from `braindance.core.spikesorter.rt_sort` — imports (static evidence).
- **Uses:** `detect_sequences` from `braindance.core.spikesorter.rt_sort` — imports (static evidence).
- **Uses:** `get_rt_sort_path` from `braindance` — imports (static evidence).
- **Shared data:** Extends `AnalysisPhaseV3` and consumes/provides fields through `Experiment` data context; RT-Sort integrates `spikesorter.rt_sort`, SpikeInterface's Maxwell extractor, data-loader mapping, and `SpikeData`.
- **Shared data:** Connectivity calls `SpikeData.spike_time_tilings`; causal analysis converts a prior connectivity matrix into a NetworkX directed graph using random asymmetry.

## Dependencies
- `braindance.analysis.data_loader` — intra-repo import; source import evidence.
- `braindance.analysis.data_loader.load_data_maxwell` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.experiment_v3.Experiment` — intra-repo import; source import evidence.
- `braindance.core.phases_v3.phase_base_v3.AnalysisPhaseV3` — intra-repo import; source import evidence.
- `braindance.core.spikesorter.kilosort2.RunKilosort` — intra-repo import; source import evidence.
- `braindance.core.spikesorter.rt_sort.RTSort` — intra-repo import; source import evidence.
- `braindance.core.spikesorter.rt_sort.detect_sequences` — intra-repo import; source import evidence.
- `braindance.get_rt_sort_path` — intra-repo import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `networkx` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.
- `spikeinterface.extractors.MaxwellRecordingExtractor` — external or unresolved local import; source import evidence.
- `spikelab.SpikeData` — external or unresolved local import; source import evidence.

## Classes
### ActivityPhaseV3(AnalysisPhaseV3)
> Intended to summarize per-electrode activity above a threshold, but currently contains placeholder/random and incompatible data-loading logic.
**Source:** `braindance/core/phases_v3/phases_analysis_3.py:17`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, threshold: float=5.0, name: str=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `recording_file` | inferred at runtime | `None` |
| `threshold` | inferred at runtime | `threshold` |
**Methods:**
#### `run(self, experiment) -> Dict[str, Any]`
> Attempt to load a recording, construct an activity table, and return electrodes above threshold.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_analysis_3.py:37`
### RTSortPhaseV3(AnalysisPhaseV3)
> Initialize or reuse RT-Sort, perform offline sorting, expose neuron metadata and `SpikeData`, and persist sorting artifacts.
**Source:** `braindance/core/phases_v3/phases_analysis_3.py:66`
**Kind:** class. **Instantiated by:** braindance/examples/2_rapid_pairing.py:74 (named-call hint); braindance/examples/ant_cpg_example.py:114 (named-call hint); braindance/examples/ant_reinforcement_example.py:33 (named-call hint); braindance/examples/cartpole_ml.py:46 (named-call hint); braindance/examples/cartpole_reinforcement_example.py:31 (named-call hint); braindance/examples/closed_loop.py:73 (named-call hint); braindance/examples/closed_loop_experiment_v3.py:184 (named-call hint); braindance/examples/closed_loop_experiment_v3.py:37 (named-call hint); braindance/examples/foodland_reinforcement_example.py:23 (named-call hint); braindance/examples/mspacman_reinforcement_example.py:19 (named-call hint); braindance/examples/walker2d_cpg_example.py:127 (named-call hint); braindance/examples/walker2d_reinforcement_example.py:129 (named-call hint)
**Constructor:** `__init__(self, sorter: str='rt_sort', min_spikes: int=100, detection_model_path: str=None, inter_path: str=None, recording_window_ms: Tuple[int, int]=(0, 60000), artifact_removal_params: Dict=None, art_rem_N: int=60, force_redo: bool=True, make_plots: bool=False, save_plots: bool=False, delete_inter: bool=True, save_rt_sort: bool=True, verbose: bool=True, name: str=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `art_rem_N` | inferred at runtime | `art_rem_N` |
| `artifact_removal_params` | inferred at runtime | `artifact_removal_params` |
| `delete_inter` | inferred at runtime | `delete_inter` |
| `detection_model_path` | inferred at runtime | `detection_model_path` |
| `force_redo` | inferred at runtime | `force_redo` |
| `inter_path` | inferred at runtime | `inter_path` |
| `make_plots` | inferred at runtime | `make_plots` |
| `mapping` | inferred at runtime | `experiment.mapping` |
| `min_spikes` | inferred at runtime | `min_spikes` |
| `recording_file` | inferred at runtime | `None` |
| `recording_window_ms` | inferred at runtime | `recording_window_ms` |
| `rt_sort` | inferred at runtime | `None` |
| `rt_sort_path` | inferred at runtime | `experiment.rt_sort_path` |
| `save_plots` | inferred at runtime | `save_plots` |
| `save_rt_sort` | inferred at runtime | `save_rt_sort` |
| `sorter` | inferred at runtime | `sorter` |
| `verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `configure_from_experiment(self, experiment: Experiment)`
> Adopt experiment mapping/path state, optionally load an existing sorter, and choose the intermediate directory.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases_analysis_3.py:122`
#### `run(self, experiment) -> Dict[str, Any]`
> Dispatch the configured sorter workflow and save returned `SpikeData` at recording and experiment levels.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/phases_v3/phases_analysis_3.py:144`
#### `_run_rt_sort_init(self, experiment, spike_data_dir: Path) -> Dict[str, Any]`
> Detect sequences, derive channel/electrode/position metadata, optionally plot, save the RT-Sort object, and return reusable sorter state.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/core/phases_v3/phases_analysis_3.py:190`
#### `_run_rt_sort(self, experiment, spike_data_dir: Path) -> Dict[str, Any]`
> Reload an RT-Sort model, sort a Maxwell recording offline, remove the model for serialization, and return `SpikeData`.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases_analysis_3.py:349`
#### `_run_kilosort2(self, experiment, spike_data_dir: Path) -> Dict[str, Any]`
> Placeholder Kilosort2 entry point that currently delegates to mock sorting.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_analysis_3.py:384`
#### `_run_mock_sort(self, experiment, spike_data_dir: Path) -> Dict[str, Any]`
> Generate random neuron assignments, spike trains, templates, and a pickle artifact for test/demo fallback.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/phases_v3/phases_analysis_3.py:397`
### FootprintPhaseV3(AnalysisPhaseV3)
> Compute placeholder spatial footprint summaries from templates and electrode assignments.
**Source:** `braindance/core/phases_v3/phases_analysis_3.py:458`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, threshold: float=0.1, name: str=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `threshold` | inferred at runtime | `threshold` |
**Methods:**
#### `run(self, experiment) -> Dict[str, Any]`
> Build per-neuron footprint dictionaries and a metrics DataFrame.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_analysis_3.py:479`
### ConnectivityPhaseV3(AnalysisPhaseV3)
> Compute a connectivity matrix from `SpikeData` using the spike-time-tiling method.
**Source:** `braindance/core/phases_v3/phases_analysis_3.py:512`
**Kind:** class. **Instantiated by:** braindance/examples/2_rapid_pairing.py:80 (named-call hint); braindance/examples/closed_loop.py:79 (named-call hint); braindance/examples/closed_loop_experiment_v3.py:185 (named-call hint); braindance/examples/closed_loop_experiment_v3.py:38 (named-call hint); braindance/examples/closed_loop_experiment_v3.py:63 (named-call hint)
**Constructor:** `__init__(self, method: str='spike_time_tiling', window_ms: float=20.0, significance_threshold: float=0.1, name: str=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `method` | inferred at runtime | `method` |
| `neurons` | inferred at runtime | `None` |
| `significance_threshold` | inferred at runtime | `significance_threshold` |
| `spike_data` | SpikeData | None | `None` |
| `window_ms` | inferred at runtime | `window_ms` |
**Methods:**
#### `configure_from_experiment(self, experiment: Experiment)`
> Auto-configure from experiment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_analysis_3.py:538`
#### `run(self, experiment) -> Dict[str, Any]`
> Dispatch supported connectivity analysis and return its matrix.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_analysis_3.py:544`
### CausalAnalysisV3(AnalysisPhaseV3)
> Create a placeholder directed graph by randomly introducing asymmetry into a prior connectivity matrix.
**Source:** `braindance/core/phases_v3/phases_analysis_3.py:559`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, method: str='granger', max_lag_ms: float=100.0, name: str=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `max_lag_ms` | inferred at runtime | `max_lag_ms` |
| `method` | inferred at runtime | `method` |
**Methods:**
#### `run(self, experiment) -> Dict[str, Any]`
> Randomly rescale matrix directions and add graph edges above a fixed 0.1 threshold.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases_analysis_3.py:583`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| RT-Sort defaults: minimum 100 spikes, recording window `(0, 60000)` ms, artifact window 60, force redo true, delete intermediates true, save sorter true. |
| RT-Sort intermediates default to `<experiment.save_dir>/rt_sort_inter`; spike data are written under the recording directory and as `<count>_rt_spike_data.pkl` at experiment level. |
| Connectivity supports only `method='spike_time_tiling'` with default 20 ms window; causal defaults to nominal `granger` and 100 ms lag but uses neither algorithm parameter. |

## Data Shapes
- RT-Sort outputs dictionaries keyed by neuron ID for component channels, electrodes, and lists of x/y positions; offline spikes become a `SpikeData` object.
- Footprints contain electrode lists, per-channel max-absolute template amplitudes, and mean-electrode center; metrics are returned as a neuron-indexed DataFrame.
- Connectivity and causal outputs are N x N matrices; causal graph nodes are neuron IDs and edges carry weights.

## Notes
- `ActivityPhaseV3.run` treats the ndarray returned by `load_data_maxwell` as a mapping-bearing dict and is therefore currently inconsistent/broken; its spike rates are random placeholders.
- Kilosort2 immediately falls back to mock sorting, and mock/footprint/causal results use random or placeholder behavior.
- When `save_rt_sort` is false in `_run_rt_sort_init`, `rt_sort_path` is never assigned before `self.rt_sort_path = rt_sort_path`.
- Broad exception handling falls back from mapping conversion to treating channels as electrodes; plotting and pickle saves are side effects.
