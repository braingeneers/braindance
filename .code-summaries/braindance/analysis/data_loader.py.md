# data_loader.py

**Path:** `braindance/analysis/data_loader.py`
**Module:** `braindance.analysis.data_loader`
**Feature Area:** `Data Catalog`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Legacy Maxwell HDF5 and stimulation-log loading utilities plus the deprecated AnalysisDAO selection container. It normalizes routed channel indices, scales ADC data to millivolts, and aligns logged stimulation times to negative signal peaks.

## Connections
- **Used by:** `braindance.analysis.causal_connectivity` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.analysis.mapping` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.analysis.process_causal` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.cli.replay` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_analysis` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_analysis_2` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_ant` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_ant_cpg` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_foodland` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_loop` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_pacman` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_walker2d` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases3_walker2d_cpg` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases_analysis_3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.paper_cartpole.causal` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.paper_cartpole.recording` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.causal_freq_sort_intrinsic` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.causal_tetanus_phases` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.neural_config_analysis` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.gui.experiment_launcher` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.data_loading.recording` — import consumer hint; not a proven runtime call.
- **Uses:** `open_file` from `braindance.io` — imports (static evidence).
- **Shared data:** `Mapping.from_maxwell` delegates mapping reads to `load_mapping_maxwell`; legacy analysis scripts construct `AnalysisDAO` from these loaders.
- **Shared data:** Stim-time adjustment combines the Maxwell mapping, raw voltage windows, and CSV stimulation logs, then persists aligned `time_mod` values.

## Dependencies
- `braindance.io.open_file` — intra-repo import; source import evidence.
- `h5py` — external or unresolved local import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `numpy.ndarray` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.

## Classes
### AnalysisDAO()
> Deprecated mutable container for a recording's spike table, electrode mapping, selections, stimulation metadata, and derived legacy analysis state.
**Source:** `braindance/analysis/data_loader.py:17`
**Kind:** class. **Instantiated by:** braindance/gui/experiment_launcher.py:1230 (named-call hint); braindance/gui/experiment_launcher.py:1245 (named-call hint)
**Constructor:** `__init__(self, spikes=None, file_path=None, mapping=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `channels` | inferred at runtime | `mapping['channel'].values` |
| `channels_of_interest` | inferred at runtime | `None` |
| `clean_data` | inferred at runtime | `clean_data` |
| `electrodes` | inferred at runtime | `mapping['electrode'].values` |
| `file_path` | inferred at runtime | `file_path` |
| `mapping` | inferred at runtime | `mapping` |
| `mapping_file` | inferred at runtime | `None` |
| `name` | inferred at runtime | `file_path.split('/')[-1].split('.')[0]` |
| `num_spikes` | inferred at runtime | `len(spikes)` |
| `py_file` | inferred at runtime | `None` |
| `reactivity` | inferred at runtime | `None` |
| `reactivity_times` | inferred at runtime | `reactivity_times` |
| `selected_channels` | inferred at runtime | `None` |
| `selected_electrodes` | inferred at runtime | `None` |
| `selected_footprint_chans` | inferred at runtime | `None` |
| `selected_footprint_pk_to_pks` | inferred at runtime | `None` |
| `selected_footprint_waves` | inferred at runtime | `None` |
| `spikes` | inferred at runtime | `spikes` |
| `stim_electrodes` | inferred at runtime | `None` |
| `stim_log` | inferred at runtime | `None` |
**Methods:**
#### `from_file_path(cls, file_path, get_spikes=True)`
> Construct a legacy analysis object by loading a recording mapping and optionally its hardware spike table.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:78`
#### `__repr__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:97`
#### `set_spikes(self, spikes)`
> Replace the stored spike table and update its cached row count, accepting null to clear it.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/data_loader.py:100`
#### `set_footprints(self, footprint_chans, footprint_waves, footprint_pk_to_pks)`
> Attach selected footprint channel, waveform, and peak-to-peak arrays to the legacy analysis state.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/data_loader.py:108`
#### `select_electrodes(self, electrodes)`
> Selects electrodes.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/data_loader.py:113`
#### `select_channels(self, channels)`
> Selects channels.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/data_loader.py:124`
#### `set_file_path(self, file_path, get_mapping=True)`
> Change the recording path and optionally reload its Maxwell mapping.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/data_loader.py:135`
#### `set_mapping(self, mapping)`
> Attach a mapping DataFrame and cache integer channel/electrode lists, or clear mapping state.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/data_loader.py:140`
#### `set_stim_electrodes(self, stim_electrodes)`
> Attach the stimulation-electrode selection to the mutable analysis state.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/data_loader.py:157`
#### `set_stim_log(self, stim_log)`
> Attach a stimulation-log DataFrame to the mutable analysis state.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/data_loader.py:160`
#### `set_channels_of_interest(self, channels_of_interest)`
> Attach the channel subset used by downstream legacy analyses.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/data_loader.py:163`
#### `set_reactivity(self, reactivity)`
> Attach computed reactivity values to the analysis state.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/data_loader.py:166`
#### `set_reactivity_times(self, reactivity_times)`
> Attach computed reactive spike times to the analysis state.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/data_loader.py:169`
#### `set_clean_data(self, clean_data)`
> Attach artifact-cleaned voltage data to the analysis state.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/data_loader.py:172`
#### `get_electrodes(self, channels=None)`
> Get electrodes from the AnalysisDAO.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:175`
#### `get_channels(self, electrodes=None)`
> Get channels from the AnalysisDAO.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:190`
#### `get_orig_channels(self, channels=None, electrodes=None)`
> Get original channels from the AnalysisDAO, converting either channels or electrodes to original channels.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:205`
#### `get_nearest_channels(self, channels=None, electrodes=None, n=1)`
> Get the nearest channels to the given channels or electrodes.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:225`
#### `get_positions(self, channels=None, electrodes=None)`
> Get the x and y coordinates of the given channels or electrodes.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:264`
#### `get_spikes(self, channels=None, amplitudes=None, frame_bounds=None)`
> Filter the stored spike DataFrame by routed channels, frame bounds, and optional amplitudes.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:286`
#### `nbytes(self)`
> Get the number of bytes of the AnalysisDAO. This includes the dataframes for spikes and mapping.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:312`
#### `save_params(self, file_path=None)`
> Write current selection metadata as a Python parameter file and mapping CSV beside a recording.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/analysis/data_loader.py:321`
#### `update_json(self, json_file_path, overwrite=False)`
> Merge selected electrodes and generated artifact paths into an experiment JSON file.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/analysis/data_loader.py:353`
#### `save(self, file_path)`
> Saves the AnalysisDAO to a file.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/analysis/data_loader.py:388`
#### `load(cls, file_path)`
> Loads the AnalysisDAO from a file.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:403`
### NumpyEncoder(json.JSONEncoder)
> Serialize NumPy scalar and array values into ordinary JSON-compatible Python values.
**Source:** `braindance/analysis/data_loader.py:424`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `default(self, obj)`
> Convert NumPy integer, float, and ndarray objects before delegating unsupported values to the base JSON encoder.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:425`

## Functions
### `load_info_maxwell(filepath)`
> Read shape, acquisition start time/frame, and hardware scaling metadata from old or newer Maxwell HDF5 layouts.
> **Called by:** braindance/analysis/data_loader.py:673 (named-call hint); braindance/analysis/data_loader.py:838 (named-call hint); braindance/analysis/data_loader.py:935 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:439`
### `load_mapping_maxwell(filepath, channels=None)`
> Load old/new Maxwell mapping datasets and assign contiguous routed channel IDs while preserving hardware channel IDs.
> **Called by:** braindance/analysis/data_loader.py:138 (named-call hint); braindance/analysis/data_loader.py:58 (named-call hint); braindance/analysis/data_loader.py:582 (named-call hint); braindance/analysis/data_loader.py:818 (named-call hint); braindance/analysis/data_loader.py:915 (named-call hint); braindance/analysis/data_loader.py:92 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:478`
### `load_data_maxwell(filepath, channels=None, start=0, length=-1, spikes=False, dtype=np.float32, suffix=None, verbose=False)`
> Load routed raw voltage or recorded hardware spikes from a Maxwell HDF5 file, with channel reordering and voltage scaling.
> **Called by:** braindance/analysis/data_loader.py:52 (named-call hint); braindance/analysis/data_loader.py:683 (named-call hint); braindance/analysis/data_loader.py:865 (named-call hint); braindance/analysis/data_loader.py:89 (named-call hint); braindance/analysis/data_loader.py:962 (named-call hint); braindance/cli/replay.py:148 (named-call hint); braindance/core/phases_v3/phases_analysis_3.py:44 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:523`
### `convert_uint16_maxwell(data)`
> Convert Maxwell unsigned ADC counts to float32 millivolts using legacy fixed scale constants.
> **Called by:** braindance/core/phases_v3/phases3_ant.py:527 (named-call hint); braindance/core/phases_v3/phases3_ant.py:674 (named-call hint); braindance/core/phases_v3/phases3_ant_cpg.py:555 (named-call hint); braindance/core/phases_v3/phases3_ant_cpg.py:720 (named-call hint); braindance/core/phases_v3/phases3_foodland.py:679 (named-call hint); braindance/core/phases_v3/phases3_foodland.py:891 (named-call hint); braindance/core/phases_v3/phases3_loop.py:1054 (named-call hint); braindance/core/phases_v3/phases3_loop.py:270 (named-call hint); braindance/core/phases_v3/phases3_loop.py:301 (named-call hint); braindance/core/phases_v3/phases3_loop.py:989 (named-call hint); braindance/core/phases_v3/phases3_pacman.py:528 (named-call hint); braindance/core/phases_v3/phases3_pacman.py:682 (named-call hint); braindance/core/phases_v3/phases3_walker2d.py:595 (named-call hint); braindance/core/phases_v3/phases3_walker2d.py:760 (named-call hint); braindance/core/phases_v3/phases3_walker2d_cpg.py:602 (named-call hint); braindance/core/phases_v3/phases3_walker2d_cpg.py:758 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:642`
### `load_windows_maxwell(filepath, starts, window_sz=2000, channels=None, dtype=np.float32)`
> Read multiple fixed-size voltage windows, substituting zero windows after per-window failures.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:657`
### `load_stim_log(filepath, adjust=False, suffix='_log.csv')`
> Read a stimulation CSV and parse literal list columns.
> **Called by:** braindance/analysis/data_loader.py:824 (named-call hint); braindance/analysis/data_loader.py:921 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:692`
### `apply_literal_eval_stim_log(stim_log)`
> Parse string representations in stimulation-electrode and optional pattern columns in place.
> **Called by:** braindance/analysis/data_loader.py:828 (named-call hint); braindance/analysis/data_loader.py:925 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:717`
### `get_stim_electrodes(stim_log)`
> Get the unique stim electrodes from the stim log
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:724`
### `set_stim_patterns(stim_log)`
> Get the unique stim patterns from the stim log
> **Called by:** braindance/analysis/data_loader.py:882 (named-call hint); braindance/analysis/data_loader.py:979 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:733`
### `adjust_stim_times(data, stim_df, window_size_ms=200, fs=20000, ch_choice=0, stim_offset_ms=10, plot_check=False)`
> Align in-memory stimulation timestamps to local negative peaks in a chosen voltage channel.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/data_loader.py:747`
### `adjust_stim_times2(file_path, stim_log=None, stim_log_path=None, window_size_ms=200, fs=20000, ch_choice=None, stim_offset_ms=10, plot_check=False, tag=None, save_adjustments=True, force_adjustment=False, verbose=False)`
> Align a recording's stimulation log to negative peaks on each stimulated electrode's routed channel and optionally save it.
> **Called by:** unclear — see source. **Side effects:** write call (static hint).
**Source:** `braindance/analysis/data_loader.py:795`
### `adjust_stim_times3(file_path, stim_log=None, stim_log_path=None, window_size_ms=200, fs=20000, ch_choice=None, stim_offset_ms=10, plot_check=False, tag=None, save_adjustments=True, force_adjustment=False, verbose=False)`
> Duplicate file-based stimulation alignment routine retained under a second API name.
> **Called by:** unclear — see source. **Side effects:** write call (static hint).
**Source:** `braindance/analysis/data_loader.py:892`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `load_data_maxwell(start=0, length=-1, spikes=False, dtype=np.float32, suffix=None)`: selects frames/channels or the spike table; float output uses HDF5 scaling with legacy defaults. |
| Stim-time adjusters default to `fs=20000`, `window_size_ms=200`, `stim_offset_ms=10`, and optionally overwrite `<recording>_log.csv`. |
| `AnalysisDAO` is deprecated and emits `FutureWarning`; retained for legacy selection workflows. |

## Data Shapes
- Maxwell voltage arrays are channels x frames in HDF5 and returned in that orientation; `load_windows_maxwell` returns windows x channels x frames.
- Spike DataFrames use `frame`, `channel`, and `amplitude`; channel IDs are remapped from `orig_channel` to contiguous routed indices.
- Mapping DataFrames contain `channel`, `orig_channel`, `electrode`, `x`, and `y`; stimulation logs contain seconds in `time`, list-valued `stim_electrodes`, and optional `stim_pattern`/`time_mod`.

## Notes
- `load_data_maxwell(length=-1)` computes `frame_end=start-1`, so normal Python slicing excludes the final frame rather than expressing an unlimited read.
- `adjust_stim_times2` and `adjust_stim_times3` are duplicate implementations and load all channels for every stimulation window.
- Several APIs write pickle, Python, CSV, or JSON artifacts directly beside the source recording; `AnalysisDAO.load` unpickles trusted input.
