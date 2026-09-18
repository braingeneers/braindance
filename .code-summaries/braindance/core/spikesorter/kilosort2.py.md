# kilosort2.py

**Path:** `braindance/core/spikesorter/kilosort2.py`
**Module:** `braindance.core.spikesorter.kilosort2`
**Feature Area:** `Spike Sorting`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Implements a legacy end-to-end Kilosort2 pipeline: recording conversion, MATLAB launch, Phy-result loading, waveform extraction, two-stage unit curation, result compilation, and diagnostic plots. The public `run_kilosort2` entry point materializes its large argument surface into module globals consumed by the pipeline.

## Connections
- **Used by:** `braindance.core.phases_v3.phases_analysis_3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.spikedetector.data` — import consumer hint; not a proven runtime call.
- **Uses:** `save_traces` from `braindance.core.spikesorter.rt_sort` — imports (static evidence).
- **Shared data:** Uses SpikeInterface extractors/preprocessing for Maxwell/NWB loading, scaling, filtering, binary conversion, and recording slicing.
- **Shared data:** Calls `save_traces` from `rt_sort` when deep-learning output is requested.
- **Shared data:** Feeds KilosortSortingExtractor into WaveformExtractor, Curation, Compiler, and plot helper classes.

## Dependencies
- `braindance.core.spikesorter.rt_sort.save_traces` — intra-repo import; source import evidence.
- `h5py` — external or unresolved local import; source import evidence.
- `joblib` — external or unresolved local import; source import evidence.
- `matplotlib` — external or unresolved local import; source import evidence.
- `matplotlib.axes._axes` — external or unresolved local import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `natsort.natsorted` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.
- `scipy.io.savemat` — external or unresolved local import; source import evidence.
- `six.exec_` — external or unresolved local import; source import evidence.
- `spikeinterface.core.BaseRecording` — external or unresolved local import; source import evidence.
- `spikeinterface.core.BinaryRecordingExtractor` — external or unresolved local import; source import evidence.
- `spikeinterface.core.segmentutils` — external or unresolved local import; source import evidence.
- `spikeinterface.extractors.MaxwellRecordingExtractor` — external or unresolved local import; source import evidence.
- `spikeinterface.extractors.NwbRecordingExtractor` — external or unresolved local import; source import evidence.
- `spikeinterface.preprocessing.ScaleRecording` — external or unresolved local import; source import evidence.
- `spikeinterface.preprocessing.bandpass_filter` — external or unresolved local import; source import evidence.
- `tqdm.tqdm` — external or unresolved local import; source import evidence.

## Classes
### RunKilosort()
> Generate Kilosort2 MATLAB inputs, launch MATLAB, and expose the resulting Phy folder as a sorting object.
**Source:** `braindance/core/spikesorter/kilosort2.py:57`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:3792 (named-call hint)
**Constructor:** `__init__(self)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `path` | inferred at runtime | `self.set_kilosort_path(KILOSORT_PATH)` |
**Methods:**
#### `run(self, recording, recording_dat_path, output_folder)`
> Prepare MATLAB scripts and recording metadata, execute Kilosort2, and return an extractor for the output folder.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:72`
#### `setup_recording_files(self, recording, recording_dat_path, output_folder)`
> Write the Kilosort MATLAB driver, configuration, channel map, and bundled NPY writer scripts into an output folder.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/spikesorter/kilosort2.py:82`
#### `start_sorting(self, output_folder, raise_error, verbose)`
> Time the MATLAB invocation, collect its log, and optionally convert a failed run into an exception.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:371`
#### `execute_kilosort_file(output_folder, verbose)`
> Create and launch the platform-specific MATLAB shell command and reject a nonzero process exit status.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:408`
#### `check_if_installed(self)`
> Check the configured Kilosort tree for either supported MATLAB entry script.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:439`
#### `set_kilosort_path(kilosort_path)`
> Resolve Kilosort from an argument or environment variable and publish the resolved path to child processes.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:446`
#### `format_params()`
> Normalize batch length to a multiple of 32 and encode common-average referencing as the numeric value MATLAB expects.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:463`
#### `get_result_from_folder(cls, output_folder)`
> Construct a KilosortSortingExtractor over an existing Kilosort/Phy output directory.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:475`
### KilosortSortingExtractor()
> Load Kilosort/Phy arrays and cluster metadata and provide per-unit spike trains and template/channel information.
**Source:** `braindance/core/spikesorter/kilosort2.py:479`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:476 (named-call hint)
**Constructor:** `__init__(self, folder_path, exclude_cluster_groups=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `folder` | inferred at runtime | `phy_folder.absolute()` |
| `sampling_frequency` | inferred at runtime | `params['sample_rate']` |
| `spike_clusters` | inferred at runtime | `np.atleast_1d(np.load(str(phy_folder / 'spike_clusters.npy')).flatten())` |
| `spike_times` | inferred at runtime | `np.atleast_1d(np.load(str(phy_folder / 'spike_times.npy')).astype(int).flatten())` |
| `unit_ids` | inferred at runtime | `[]` |
**Methods:**
#### `get_num_segments()`
> Report the single segment assumed by this Kilosort result adapter.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:554`
#### `get_unit_spike_train(self, unit_id, segment_index: Union[int, None]=None, start_frame: Union[int, None]=None, end_frame: Union[int, None]=None)`
> Return a copy of one unit’s sample indices, optionally clipped to a half-open frame interval.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:558`
#### `get_templates_all(self)`
> Memory-map the full Kilosort template tensor from templates.npy.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:572`
#### `get_channel_map(self)`
> Memory-map and flatten Kilosort’s internal-to-recording channel mapping.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:576`
#### `get_chans_max(self)`
> Get the max channel of each unit based on Kilosort2's template and whether to use (min/argmin or max/argmax) for computing peak values
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:580`
#### `get_templates_half_windows_sizes(self, chans_max_kilosort, window_size_scale=0.75)`
> Get the half window sizes that will be used to recenter the spike times on the peak
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:617`
#### `ms_to_samples(self, ms)`
> Convert milliseconds to the nearest sample using the sorting sampling frequency.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:646`
### ShellScript()
> Write, run, monitor, stop, and clean up a shell script used to launch MATLAB.
**Source:** `braindance/core/spikesorter/kilosort2.py:650`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:431 (named-call hint)
**Constructor:** `__init__(self, script: str, script_path: Optional[PathType]=None, log_path: Optional[PathType]=None, keep_temp_files: bool=False, verbose: bool=False)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_dirs_to_remove` | List[str] | `[]` |
| `_files_to_remove` | List[str] | `[]` |
| `_keep_temp_files` | inferred at runtime | `keep_temp_files` |
| `_log_path` | inferred at runtime | `log_path` |
| `_process` | Optional[subprocess.Popen] | `None` |
| `_script` | inferred at runtime | `'\n'.join(lines)` |
| `_script_path` | inferred at runtime | `script_path` |
| `_start_time` | Optional[float] | `None` |
| `_verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `__del__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:680`
#### `substitute(self, old: str, new: Any) -> None`
> Replace a literal placeholder throughout the in-memory script text.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/kilosort2.py:683`
#### `write(self, script_path: Optional[str]=None) -> None`
> Write the script text to disk and mark it executable.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/spikesorter/kilosort2.py:686`
#### `start(self) -> None`
> Materialize the script, launch it as a subprocess, and synchronously stream combined output into its log and optionally stdout.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/core/spikesorter/kilosort2.py:695`
#### `wait(self, timeout=None) -> Optional[int]`
> Return the exit status when the process completes within the optional timeout, or None on timeout.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:730`
#### `cleanup(self) -> None`
> Remove temporary script directories unless retention was requested.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:740`
#### `stop(self) -> None`
> Escalate repeated interrupt, terminate, and kill signals until the subprocess exits.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:746`
#### `kill(self) -> None`
> Send SIGKILL to a running subprocess and wait briefly for termination.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:761`
#### `stopWithSignal(self, sig, timeout) -> bool`
> Send one requested signal and report whether the subprocess exits before the timeout.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:773`
#### `elapsedTimeSinceStart(self) -> Optional[float]`
> Return wall-clock seconds since launch, or None before launch.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:785`
#### `isRunning(self) -> bool`
> Poll the subprocess and report whether it still lacks an exit code.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:791`
#### `isFinished(self) -> bool`
> Report whether a launched subprocess has stopped.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:799`
#### `returnCode(self) -> Optional[int]`
> Return the completed subprocess status, rejecting calls made before completion.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:804`
#### `scriptPath(self) -> Optional[str]`
> Return the configured script path, which may be None for a temporary script.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:810`
#### `_remove_initial_blank_lines(self, lines: List[str]) -> List[str]`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:813`
#### `_get_num_initial_spaces(self, line: str) -> int`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:819`
#### `_rmdir_with_retries(dirname, num_retries, delay_between_tries=1)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:826`
### WaveformExtractor()
> Persist centered per-unit waveforms, compute templates, and create filtered curation views over a sorting.
**Source:** `braindance/core/spikesorter/kilosort2.py:845`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, recording, sorting, root_folder, folder)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_waveforms` | inferred at runtime | `{}` |
| `chans_max_all` | inferred at runtime | `None` |
| `chans_max_folder` | inferred at runtime | `root_folder / 'channels_max'` |
| `chans_max_kilosort` | inferred at runtime | `None` |
| `dtype` | inferred at runtime | `parameters['dtype']` |
| `folder` | inferred at runtime | `Path(folder)` |
| `nafter` | inferred at runtime | `self.ms_to_samples(parameters['ms_after']) + 1` |
| `nbefore` | inferred at runtime | `self.ms_to_samples(parameters['ms_before'])` |
| `nsamples` | inferred at runtime | `self.nbefore + self.nafter` |
| `peak_ind` | inferred at runtime | `parameters['peak_ind']` |
| `recording` | inferred at runtime | `recording` |
| `return_scaled` | inferred at runtime | `False` |
| `root_folder` | inferred at runtime | `root_folder` |
| `sampling_frequency` | inferred at runtime | `parameters['sampling_frequency']` |
| `sorting` | inferred at runtime | `sorting` |
| `template_cache` | inferred at runtime | `{}` |
| `templates_half_windows_sizes` | inferred at runtime | `self.sorting.get_templates_half_windows_sizes(self.chans_max_kilosort)` |
| `use_pos_peak` | inferred at runtime | `None` |
**Methods:**
#### `create_initial(cls, recording_path, recording, sorting, root_folder, initial_folder)`
> Initialize waveform storage, persist extraction parameters and Kilosort spike arrays, and cache each template’s peak polarity and recording channel.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/spikesorter/kilosort2.py:903`
#### `load_from_folder(cls, recording, sorting, root_folder, folder, use_pos_peak=None, chans_max_kilosort=None, chans_max_all=None)`
> Reopen waveform, template, channel-peak, and curated-unit arrays from an existing extraction folder.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:940`
#### `ms_to_samples(self, ms)`
> Convert milliseconds to an integer sample count using the recording rate.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:964`
#### `run_extract_waveforms(self, **job_kwargs)`
> Allocate per-unit waveform files, extract centered snippets by recording chunk, and rewrite Kilosort spike_times.npy with centered times while preserving the original.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/core/spikesorter/kilosort2.py:970`
#### `sample_spikes(self)`
> Uniform random selection of spikes per unit and save to .npy
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/spikesorter/kilosort2.py:1024`
#### `select_random_spikes_uniformly(self)`
> Uniform random selection of spikes per unit.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1056`
#### `_waveform_extractor_chunk(segment_index, start_frame, end_frame, worker_ctx)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1103`
#### `_init_worker_waveform_extractor(recording, sorting, waveform_extractor, wfs_memmap, selected_spikes, selected_spike_times, nbefore, nafter, return_scaled)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1192`
#### `get_waveforms(self, unit_id, with_index=False, cache=False, memmap=True)`
> Return waveforms for the specified unit id.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/kilosort2.py:1221`
#### `get_sampled_indices(self, unit_id)`
> Return sampled spike indices of extracted waveforms (which waveforms correspond to which spikes if "max_spikes_per_unit" is not None)
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1265`
#### `get_computed_template(self, unit_id, mode)`
> Return template (average waveform).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1291`
#### `compute_templates(self, modes=('average', 'std'), unit_ids=None, folder=None)`
> Compute all template for different "modes": * average * std * median
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/core/spikesorter/kilosort2.py:1326`
#### `load_units(self)`
> Replace the attached sorting’s unit IDs, spike times, and cluster assignments with arrays from this curation folder.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/kilosort2.py:1389`
#### `get_curation_history(self)`
> Load this curation folder’s JSON history when present.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1394`
#### `save_curated_units(self, unit_ids, waveforms_root_folder, curated_folder, curation_history)`
> Filters units by storing curated unit ids in a new folder.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/spikesorter/kilosort2.py:1405`
### ChunkRecordingExecutor()
> Apply waveform extraction work over recording chunks serially or through the embedded process pool.
**Source:** `braindance/core/spikesorter/kilosort2.py:1473`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:1006 (named-call hint)
**Constructor:** `__init__(self, recording, func, init_func, init_args, verbose=True, progress_bar=False, handle_returns=False, n_jobs=1, total_memory=None, chunk_size=None, chunk_memory=None, job_name='')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `chunk_size` | inferred at runtime | `Utils.ensure_chunk_size(recording, total_memory=total_memory, chunk_size=chunk_size, chunk_memory=chunk_memory, n_jobs=self.n_job…` |
| `func` | inferred at runtime | `func` |
| `handle_returns` | inferred at runtime | `handle_returns` |
| `init_args` | inferred at runtime | `init_args` |
| `init_func` | inferred at runtime | `init_func` |
| `job_name` | inferred at runtime | `job_name` |
| `n_jobs` | inferred at runtime | `Utils.ensure_n_jobs(recording, n_jobs=n_jobs)` |
| `progress_bar` | inferred at runtime | `progress_bar` |
| `recording` | inferred at runtime | `recording` |
| `verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `run(self)`
> Runs the defined jobs.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/kilosort2.py:1543`
#### `function_wrapper(args)`
> Invoke the process-global chunk function with the worker context initialized for that process.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1595`
#### `divide_recording_into_chunks(recording, chunk_size)`
> Enumerate segment index and half-open frame bounds for every recording chunk.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1602`
#### `divide_segment_into_chunks(num_frames, chunk_size)`
> Partition one segment into contiguous half-open frame ranges, retaining a shorter final range.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1611`
#### `worker_initializer(func, init_func, init_args)`
> Initialize each worker’s reusable context and chunk callable in process globals.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1632`
### _ThreadWakeup()
> unclear — see source
**Source:** `braindance/core/spikesorter/kilosort2.py:1723`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:2223 (named-call hint)
**Constructor:** `__init__(self)`
**Key attributes:**
None
**Methods:**
#### `close(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1727`
#### `wakeup(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1731`
#### `clear(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1734`
### _RemoteTraceback(Exception)
> unclear — see source
**Source:** `braindance/core/spikesorter/kilosort2.py:1764`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:1784 (named-call hint); braindance/core/spikesorter/kilosort2.py:1821 (named-call hint); braindance/core/spikesorter/kilosort2.py:2042 (named-call hint)
**Constructor:** `__init__(self, tb)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `tb` | inferred at runtime | `tb` |
**Methods:**
#### `__str__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1768`
### _ExceptionWithTraceback()
> unclear — see source
**Source:** `braindance/core/spikesorter/kilosort2.py:1772`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:1859 (named-call hint); braindance/core/spikesorter/kilosort2.py:1893 (named-call hint)
**Constructor:** `__init__(self, exc, tb)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `exc` | inferred at runtime | `exc` |
| `tb` | inferred at runtime | `'\n"""\n%s"""' % tb` |
**Methods:**
#### `__reduce__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1779`
### _WorkItem(object)
> unclear — see source
**Source:** `braindance/core/spikesorter/kilosort2.py:1788`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:2290 (named-call hint)
**Constructor:** `__init__(self, future, fn, args, kwargs)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `args` | inferred at runtime | `args` |
| `fn` | inferred at runtime | `fn` |
| `future` | inferred at runtime | `future` |
| `kwargs` | inferred at runtime | `kwargs` |
**Methods:**
None
### _ResultItem(object)
> unclear — see source
**Source:** `braindance/core/spikesorter/kilosort2.py:1796`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:1856 (named-call hint); braindance/core/spikesorter/kilosort2.py:1860 (named-call hint)
**Constructor:** `__init__(self, work_id, exception=None, result=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `exception` | inferred at runtime | `exception` |
| `result` | inferred at runtime | `result` |
| `work_id` | inferred at runtime | `work_id` |
**Methods:**
None
### _CallItem(object)
> unclear — see source
**Source:** `braindance/core/spikesorter/kilosort2.py:1803`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:1932 (named-call hint)
**Constructor:** `__init__(self, work_id, fn, args, kwargs)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `args` | inferred at runtime | `args` |
| `fn` | inferred at runtime | `fn` |
| `kwargs` | inferred at runtime | `kwargs` |
| `work_id` | inferred at runtime | `work_id` |
**Methods:**
None
### _SafeQueue(Queue)
> Safe Queue set exception to the future object linked to a job
**Source:** `braindance/core/spikesorter/kilosort2.py:1811`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:2207 (named-call hint)
**Constructor:** `__init__(self, max_size=0, *, ctx, pending_work_items)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `pending_work_items` | inferred at runtime | `pending_work_items` |
**Methods:**
#### `_on_queue_feeder_error(self, e, obj)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1818`
### BrokenProcessPool(_base.BrokenExecutor)
> Raised when a process in a ProcessPoolExecutor terminated abruptly while a future was in the running state.
**Source:** `braindance/core/spikesorter/kilosort2.py:2142`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:2038 (named-call hint); braindance/core/spikesorter/kilosort2.py:2282 (named-call hint)
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
None
### ProcessPoolExecutor(_base.Executor)
> Vendored process-pool executor used by chunked waveform extraction, with worker initialization and broken-pool handling.
**Source:** `braindance/core/spikesorter/kilosort2.py:2149`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:1576 (named-call hint)
**Constructor:** `__init__(self, max_workers=None, mp_context=None, initializer=None, initargs=())`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_broken` | inferred at runtime | `False` |
| `_call_queue` | inferred at runtime | `_SafeQueue(max_size=queue_size, ctx=self._mp_context, pending_work_items=self._pending_work_items)` |
| `_initargs` | inferred at runtime | `initargs` |
| `_initializer` | inferred at runtime | `initializer` |
| `_max_workers` | inferred at runtime | `os.cpu_count() or 1` |
| `_mp_context` | inferred at runtime | `mp_context` |
| `_pending_work_items` | inferred at runtime | `{}` |
| `_processes` | inferred at runtime | `{}` |
| `_queue_count` | inferred at runtime | `0` |
| `_queue_management_thread` | inferred at runtime | `None` |
| `_queue_management_thread_wakeup` | inferred at runtime | `_ThreadWakeup()` |
| `_result_queue` | inferred at runtime | `mp_context.SimpleQueue()` |
| `_shutdown_lock` | inferred at runtime | `threading.Lock()` |
| `_shutdown_thread` | inferred at runtime | `False` |
| `_work_ids` | inferred at runtime | `queue.Queue()` |
**Methods:**
#### `_start_queue_management_thread(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/kilosort2.py:2225`
#### `_adjust_process_count(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/kilosort2.py:2253`
#### `submit(*args, **kwargs)`
> Queue one callable as a work item, wake the queue-management thread, and return its Future.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/kilosort2.py:2264`
#### `map(self, fn, *iterables, timeout=None, chunksize=1)`
> Returns an iterator equivalent to map(fn, iter).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:2304`
#### `shutdown(self, wait=True)`
> Stop accepting work, optionally join the management thread, and close queues and wakeup descriptors.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/kilosort2.py:2333`
### Curation()
> Compute firing-rate, spike-count, ISI, waveform-consistency, and SNR filters for units.
**Source:** `braindance/core/spikesorter/kilosort2.py:2367`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `update_history(curation_history, key, curated, failed, metrics)`
> Append a curation stage name and store its passing units, failing units, and per-unit metrics in the shared history.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:2369`
#### `spikes_min(sorting, spikes_min)`
> Curate units in sorting.unit_ids based on minimum number of spikes per unit Units with greater spikes than spikes_min are curated
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:2376`
#### `firing_rate(recording, sorting)`
> Curate units in sorting.unit_ids based on firing rate Units with firing rate greater than FR_MIN are curated
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:2414`
#### `isi_violation(recording, sorting, isi_threshold_ms=1.5, min_isi_ms=0)`
> Calculate Inter-Spike Interval (ISI) violations for a spike train.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:2455`
#### `std_norm_max(waveform_extractor)`
> Curate units in waveform_extractor.sorting.unit_ids based on maximum normalized standard deviation Maximum normalized standard deviation is based on standard deviation at peak or over entire waveform window (determined by user parameters) divided (normalized) by amplitude since higher amplitude units will have greater absolute std
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:2541`
#### `snr(waveform_extractor)`
> Curate units in waveform_extractor.sorting.unit_ids based on ratio of peak amplitude to noise of the channel that defines the peak amplitude
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:2596`
#### `get_noise_levels(recording, return_scaled=True, **random_chunk_kwargs)`
> Estimate noise for each channel using MAD methods.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:2643`
#### `get_random_data_chunks(recording, return_scaled=False, num_chunks=20, chunk_size=10000, seed=0)`
> Extract random chunks from recording
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:2661`
### Utils()
> Utility functions implemented by spikeinterface
**Source:** `braindance/core/spikesorter/kilosort2.py:2703`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `read_python(path)`
> Parses python scripts in a dictionary
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:2709`
#### `ensure_n_jobs(recording, n_jobs=1)`
> Normalize requested worker count, expand -1 to CPU count, and fall back to serial execution on unsupported Python versions.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:2736`
#### `ensure_chunk_size(recording, total_memory=None, chunk_size=None, chunk_memory=None, n_jobs=1, **other_kwargs)`
> 'chunk_size' is the traces.shape[0] for each worker.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:2763`
#### `_mem_to_int(mem)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:2812`
### Stopwatch()
> Measure elapsed wall time for verbose pipeline-stage logging.
**Source:** `braindance/core/spikesorter/kilosort2.py:2826`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:1351 (named-call hint); braindance/core/spikesorter/kilosort2.py:1426 (named-call hint); braindance/core/spikesorter/kilosort2.py:2650 (named-call hint); braindance/core/spikesorter/kilosort2.py:3570 (named-call hint); braindance/core/spikesorter/kilosort2.py:3769 (named-call hint); braindance/core/spikesorter/kilosort2.py:3790 (named-call hint); braindance/core/spikesorter/kilosort2.py:3847 (named-call hint); braindance/core/spikesorter/kilosort2.py:3891 (named-call hint); braindance/core/spikesorter/kilosort2.py:3918 (named-call hint); braindance/core/spikesorter/kilosort2.py:3928 (named-call hint); braindance/core/spikesorter/kilosort2.py:3938 (named-call hint); braindance/core/spikesorter/kilosort2.py:3948 (named-call hint); braindance/core/spikesorter/kilosort2.py:3982 (named-call hint); braindance/core/spikesorter/kilosort2.py:4007 (named-call hint); braindance/core/spikesorter/kilosort2.py:4017 (named-call hint); braindance/core/spikesorter/kilosort2.py:4046 (named-call hint); braindance/core/spikesorter/kilosort2.py:4083 (named-call hint); braindance/core/spikesorter/kilosort2.py:4104 (named-call hint); braindance/core/spikesorter/kilosort2.py:4547 (named-call hint)
**Constructor:** `__init__(self, start_msg=None, use_print_stage=True)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_time_start` | inferred at runtime | `time.time()` |
**Methods:**
#### `log_time(self, text=None)`
> Print elapsed seconds with an optional completion label.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:2836`
### Compiler()
> Combine curated units and recording metadata into NPZ/MAT outputs, waveform files, and optional figures.
**Source:** `braindance/core/spikesorter/kilosort2.py:2843`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:4123 (named-call hint); braindance/core/spikesorter/kilosort2.py:4129 (named-call hint); braindance/core/spikesorter/kilosort2.py:4505 (named-call hint)
**Constructor:** `__init__(self)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `bar_plot` | inferred at runtime | `BarPlot()` |
| `compile_to_mat` | inferred at runtime | `COMPILE_TO_MAT` |
| `compile_to_npz` | inferred at runtime | `COMPILE_TO_NPZ` |
| `compile_waveforms` | inferred at runtime | `COMPILE_WAVEFORMS` |
| `create_figures` | inferred at runtime | `CREATE_FIGURES` |
| `create_std_scatter_plot` | inferred at runtime | `CURATE_SECOND and SPIKES_MIN_SECOND is not None and (STD_NORM_MAX is not None)` |
| `halves` | inferred at runtime | `(self.neg, self.pos)` |
| `neg` | inferred at runtime | `HalfCompiler(False)` |
| `pos` | inferred at runtime | `HalfCompiler(True)` |
| `rec_channel_locations` | inferred at runtime | `dict()` |
| `rec_fs` | inferred at runtime | `dict()` |
| `rec_n_samples` | inferred at runtime | `dict()` |
| `rec_names` | inferred at runtime | `[]` |
| `rec_spike_clusters` | inferred at runtime | `dict()` |
| `rec_spike_times` | inferred at runtime | `dict()` |
| `recs_cache` | inferred at runtime | `[]` |
| `save_electrodes` | inferred at runtime | `SAVE_ELECTRODES` |
| `std_scatter` | inferred at runtime | `StdScatterPlot()` |
| `templates_plot` | inferred at runtime | `TemplatesPlot()` |
**Methods:**
#### `add_recording(self, rec_name, w_e)`
> Add recording to self.recs_cache to be added to compiler all together when saving results
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:2870`
#### `_add_recording(self, rec_name, w_e)`
> Add units from sorted recording
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/kilosort2.py:2884`
#### `save_results(self, folder)`
> Save compiled results to folder
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/core/spikesorter/kilosort2.py:2948`
#### `concatenate_spike_times(self)`
> Increment spike times as if the recordings added to the compiler are one long recording in the order that the recordings were added
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:3076`
#### `concatenate_spike_clusters(self)`
> Concatenate the spike_clusters.npy of each recording as if the recordings added to the compiler are one long recording in the order that the recordings were added
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:3092`
### HalfCompiler()
> Class to compile half of the compiled results Functions with class Compiler I.e. One HalfCompiler for negative peak data and another for positive peak data
**Source:** `braindance/core/spikesorter/kilosort2.py:3104`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:2864 (named-call hint); braindance/core/spikesorter/kilosort2.py:2865 (named-call hint)
**Constructor:** `__init__(self, has_pos_peak)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `has_pos_peak` | inferred at runtime | `has_pos_peak` |
| `units` | inferred at runtime | `[]` |
**Methods:**
#### `add_unit(self, unit)`
> Append unit to self._units_all
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:3122`
### SortedUnit()
> Package a unit's centered template, channel/amplitude statistics, waveform window, and compiled metadata.
**Source:** `braindance/core/spikesorter/kilosort2.py:3134`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:2929 (named-call hint)
**Constructor:** `__init__(self, unit_id, rec_name, w_e, is_curated)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `amplitude_max` | inferred at runtime | `self.amplitudes[self.chan_max]` |
| `amplitudes` | inferred at runtime | `template_abs[peak_ind, range(peak_ind.size)]` |
| `chan_max` | inferred at runtime | `w_e.chans_max_all[unit_id]` |
| `chan_max_id` | inferred at runtime | `w_e.recording.get_channel_ids()[self.chan_max]` |
| `electrode` | inferred at runtime | `None` |
| `is_curated` | inferred at runtime | `is_curated` |
| `nafter` | inferred at runtime | `w_e.ms_to_samples(COMPILED_WAVEFORMS_MS_AFTER) + 1` |
| `nbefore` | inferred at runtime | `w_e.ms_to_samples(COMPILED_WAVEFORMS_MS_BEFORE)` |
| `peak_ind` | inferred at runtime | `peak_ind` |
| `rec_name` | inferred at runtime | `rec_name` |
| `sampling_frequency` | inferred at runtime | `w_e.sampling_frequency` |
| `sorted_index` | inferred at runtime | `None` |
| `spike_train` | inferred at runtime | `None` |
| `std_norms` | inferred at runtime | `stds / self.amplitudes` |
| `template` | inferred at runtime | `template_mean[w_e.peak_ind - self.nbefore:w_e.peak_ind + self.nafter, :]` |
| `template_full` | inferred at runtime | `template_mean` |
| `template_full_peak` | inferred at runtime | `w_e.peak_ind` |
| `unit_id` | inferred at runtime | `unit_id` |
| `waveforms` | inferred at runtime | `waveforms[:, w_e.peak_ind - self.nbefore:w_e.peak_ind + self.nafter, :]` |
| `x_max` | inferred at runtime | `None` |
| `y_max` | inferred at runtime | `None` |
**Methods:**
#### `sort_units(units)`
> Sort units based on amplitude in descending order
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:3208`
### Figure()
> Base class for creating 3 figures
**Source:** `braindance/core/spikesorter/kilosort2.py:3220`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, **subplots_kwargs)`
**Key attributes:**
None
**Methods:**
#### `save(self, save_path, verbose=True)`
> Save figure
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/spikesorter/kilosort2.py:3235`
### BarPlot(Figure)
> Class for creating bar plot showing curation
**Source:** `braindance/core/spikesorter/kilosort2.py:3252`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:2964 (named-call hint)
**Constructor:** `__init__(self, **subplots_kwargs)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `df` | inferred at runtime | `pd.DataFrame({self.total_label: [], self.selected_label: []}, index=[])` |
| `label_rotation` | inferred at runtime | `BAR_LABEL_ROTATION` |
| `selected_label` | inferred at runtime | `BAR_SELECTED_LABEL` |
| `total_label` | inferred at runtime | `BAR_TOTAL_LABEL` |
| `x_label` | inferred at runtime | `BAR_X_LABEL` |
| `y_label` | inferred at runtime | `BAR_Y_LABEL` |
**Methods:**
#### `add_recording(self, rec_name, n_total, n_selected)`
> Add recording to plot
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/kilosort2.py:3276`
#### `save(self, save_path, verbose=True)`
> Save figure
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/spikesorter/kilosort2.py:3291`
### StdScatterPlot(Figure)
> Class for creating std scatter plot
**Source:** `braindance/core/spikesorter/kilosort2.py:3312`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:2966 (named-call hint)
**Constructor:** `__init__(self, **subplots_kwargs)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `alpha` | inferred at runtime | `SCATTER_RECORDING_ALPHA` |
| `loc_off_plot` | inferred at runtime | `(-10000.0, -10000.0)` |
| `n_spikes_max` | inferred at runtime | `-np.inf` |
| `n_spikes_thresh` | inferred at runtime | `SPIKES_MIN_SECOND` |
| `rec_colors` | inferred at runtime | `dict()` |
| `rec_n_spikes_data` | inferred at runtime | `dict()` |
| `rec_std_data` | inferred at runtime | `dict()` |
| `std_max` | inferred at runtime | `-np.inf` |
| `std_thresh` | inferred at runtime | `STD_NORM_MAX` |
| `thresh_line_kwargs` | inferred at runtime | `{'linestyle': 'dotted', 'linewidth': 1, 'c': '#000000'}` |
| `unused_colors` | inferred at runtime | `SCATTER_RECORDING_COLORS[:]` |
| `x_max_buffer` | inferred at runtime | `SCATTER_X_MAX_BUFFER` |
| `y_max_buffer` | inferred at runtime | `SCATTER_Y_MAX_BUFFER` |
**Methods:**
#### `add_recording(self, rec_name, std_data, n_spikes_data)`
> Add recording to scatter plot
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/kilosort2.py:3347`
#### `add_unit(self, unit)`
> Add unit to plot
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/kilosort2.py:3370`
#### `save(self, save_path, verbose=True)`
> Save figure
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/spikesorter/kilosort2.py:3388`
### TemplatesPlot(Figure)
> Collect positive- and negative-peak units and render vertically offset template columns with curation colors and timing guides.
**Source:** `braindance/core/spikesorter/kilosort2.py:3421`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:2969 (named-call hint)
**Constructor:** `__init__(self, **subplots_kwargs)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `color_curated` | inferred at runtime | `ALL_TEMPLATES_COLOR_CURATED` |
| `color_failed` | inferred at runtime | `ALL_TEMPLATES_COLOR_FAILED` |
| `line_after` | inferred at runtime | `ALL_TEMPLATES_LINE_MS_AFTER_PEAK` |
| `line_before` | inferred at runtime | `ALL_TEMPLATES_LINE_MS_BEFORE_PEAK` |
| `line_kwargs` | inferred at runtime | `{'color': 'black', 'linestyle': 'dotted'}` |
| `n_templates_per_col` | inferred at runtime | `ALL_TEMPLATES_PER_COLUMN` |
| `subplots_kwargs` | inferred at runtime | `subplots_kwargs` |
| `units_neg` | inferred at runtime | `[]` |
| `units_pos` | inferred at runtime | `[]` |
| `window` | inferred at runtime | `[-ALL_TEMPLATES_WINDOW_MS_BEFORE_PEAK, ALL_TEMPLATES_WINDOW_MS_AFTER_PEAK]` |
| `xlabel` | inferred at runtime | `ALL_TEMPLATES_X_LABEL` |
| `y_lim_buffer` | inferred at runtime | `ALL_TEMPLATES_Y_LIM_BUFFER` |
| `y_spacing` | inferred at runtime | `ALL_TEMPLATES_Y_SPACING` |
**Methods:**
#### `add_unit(self, unit, has_pos_peak)`
> Add unit to plot
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:3448`
#### `save(self, save_path, verbose=True, units_are_sorted=True)`
> Save figure
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/core/spikesorter/kilosort2.py:3464`
### Tee()
> Context-manage stdout so console output is duplicated to a pipeline log and exceptions include their traceback there.
**Source:** `braindance/core/spikesorter/kilosort2.py:4139`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/kilosort2.py:4045 (named-call hint); braindance/core/spikesorter/kilosort2.py:4546 (named-call hint)
**Constructor:** `__init__(self, file_path, file_mode)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_file` | inferred at runtime | `_file` |
**Methods:**
#### `__enter__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:4147`
#### `__exit__(self, exc_type, exc_val, exc_tb)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/kilosort2.py:4151`
#### `_write(self, s)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:4161`

## Functions
### `print_stage(text)`
> Print a centered, fixed-width banner used to separate stages in long sorting logs.
> **Called by:** braindance/core/spikesorter/kilosort2.py:1349 (named-call hint); braindance/core/spikesorter/kilosort2.py:1425 (named-call hint); braindance/core/spikesorter/kilosort2.py:2830 (named-call hint); braindance/core/spikesorter/kilosort2.py:3568 (named-call hint); braindance/core/spikesorter/kilosort2.py:3720 (named-call hint); braindance/core/spikesorter/kilosort2.py:3789 (named-call hint); braindance/core/spikesorter/kilosort2.py:3846 (named-call hint); braindance/core/spikesorter/kilosort2.py:3890 (named-call hint); braindance/core/spikesorter/kilosort2.py:3956 (named-call hint); braindance/core/spikesorter/kilosort2.py:3981 (named-call hint); braindance/core/spikesorter/kilosort2.py:4025 (named-call hint); braindance/core/spikesorter/kilosort2.py:4054 (named-call hint); braindance/core/spikesorter/kilosort2.py:4087 (named-call hint); braindance/core/spikesorter/kilosort2.py:4550 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:5`
### `_python_exit()`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1739`
### `_rebuild_exc(exc, tb)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1783`
### `_get_chunks(*iterables, chunksize)`
> Iterates over zip()ed iterables in chunks.
> **Called by:** braindance/core/spikesorter/kilosort2.py:2329 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1831`
### `_process_chunk(fn, chunk)`
> Processes a chunk of an iterable passed to map.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1841`
### `_sendback_result(result_queue, work_id, result=None, exception=None)`
> Safely send back the given result or exception
> **Called by:** braindance/core/spikesorter/kilosort2.py:1894 (named-call hint); braindance/core/spikesorter/kilosort2.py:1896 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1853`
### `_process_worker(call_queue, result_queue, initializer, initargs)`
> Evaluates calls from call_queue and places the results in result_queue.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1863`
### `_add_call_item_to_queue(pending_work_items, work_ids, call_queue)`
> Fills call_queue with _WorkItems from pending_work_items.
> **Called by:** braindance/core/spikesorter/kilosort2.py:2004 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1904`
### `_queue_management_worker(executor_reference, processes, pending_work_items, work_ids_queue, call_queue, result_queue, thread_wakeup)`
> Manages the communication between this process and the worker processes.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:1942`
### `_check_system_limits()`
> unclear — see source
> **Called by:** braindance/core/spikesorter/kilosort2.py:2163 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:2106`
### `_chain_from_iterable_of_lists(iterable)`
> Specialized implementation of itertools.chain.from_iterable. Each item in *iterable* should be a list. This function is careful not to keep references to yielded objects.
> **Called by:** braindance/core/spikesorter/kilosort2.py:2331 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:2130`
### `create_folder(folder, parents=True)`
> Create a missing directory tree and log the created path.
> **Called by:** braindance/core/spikesorter/kilosort2.py:1381 (named-call hint); braindance/core/spikesorter/kilosort2.py:1428 (named-call hint); braindance/core/spikesorter/kilosort2.py:2962 (named-call hint); braindance/core/spikesorter/kilosort2.py:2989 (named-call hint); braindance/core/spikesorter/kilosort2.py:2990 (named-call hint); braindance/core/spikesorter/kilosort2.py:3070 (named-call hint); braindance/core/spikesorter/kilosort2.py:3764 (named-call hint); braindance/core/spikesorter/kilosort2.py:3805 (named-call hint); braindance/core/spikesorter/kilosort2.py:4044 (named-call hint); braindance/core/spikesorter/kilosort2.py:4506 (named-call hint); braindance/core/spikesorter/kilosort2.py:867 (named-call hint); braindance/core/spikesorter/kilosort2.py:906 (named-call hint); braindance/core/spikesorter/kilosort2.py:927 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:3549`
### `delete_folder(folder)`
> Delete an existing directory tree or file and log the removed path.
> **Called by:** braindance/core/spikesorter/kilosort2.py:3757 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:3556`
### `load_recording(rec_path)`
> Load, optionally slice/concatenate, coordinate-transform, scale, and bandpass a supported recording.
> **Called by:** braindance/core/spikesorter/kilosort2.py:4059 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:3567`
### `load_single_recording(rec_path)`
> Open a SpikeInterface, Maxwell HDF5, or NWB recording, enforce one segment, scale to microvolts, and apply the configured bandpass filter.
> **Called by:** braindance/core/spikesorter/kilosort2.py:3571 (named-call hint); braindance/core/spikesorter/kilosort2.py:3697 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:3624`
### `concatenate_recordings(rec_path)`
> Natural-sort and concatenate supported recordings in a directory while recording source frame boundaries for later split compilation.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:3687`
### `get_paths(rec_path, inter_path, results_path)`
> Derive all intermediate and result paths, apply recomputation deletion flags, and force recompilation when upstream artifacts were removed.
> **Called by:** braindance/core/spikesorter/kilosort2.py:4049 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:3719`
### `write_recording(recording_filtered, recording_dat_path, verbose=True)`
> Create Kilosort’s int16 binary input with configured parallelism unless a cached .dat file already exists.
> **Called by:** braindance/core/spikesorter/kilosort2.py:3800 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:3768`
### `spike_sort(rec_cache, rec_path, recording_dat_path, output_folder)`
> Reuse or run Kilosort2 for a prepared recording and return its sorting extractor.
> **Called by:** braindance/core/spikesorter/kilosort2.py:4066 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:3788`
### `extract_waveforms(recording_path, recording, sorting, root_folder, initial_folder, **job_kwargs)`
> Reuse or compute persistent waveforms and average/std templates.
> **Called by:** braindance/core/spikesorter/kilosort2.py:4072 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:3822`
### `curate(we_raw, waveforms_root_folder, curation_first_folder, curation_second_folder)`
> Apply configured first- and second-stage curation and persist each selected unit set.
> **Called by:** braindance/core/spikesorter/kilosort2.py:4076 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:3863`
### `curate_first(we_raw, curation_history, waveforms_root_folder, curation_first_folder)`
> Apply enabled firing-rate, ISI, SNR, and minimum-spike filters, intersect their passing unit sets, and persist the first curation view and history.
> **Called by:** braindance/core/spikesorter/kilosort2.py:3884 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:3889`
### `curate_second(we_raw, curation_history, waveforms_root_folder, curation_second_folder)`
> Apply enabled minimum-spike and normalized-waveform-variability filters to first-stage survivors and persist the second curation view.
> **Called by:** braindance/core/spikesorter/kilosort2.py:3885 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:3980`
### `process_recording(rec_name, rec_path, inter_path, results_path, rec_loaded=None)`
> Execute the complete sorting, waveform, curation, compilation, and optional trace-export workflow for one recording.
> **Called by:** braindance/core/spikesorter/kilosort2.py:4532 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:4043`
### `copy_script(path)`
> Archive a timestamped copy of the running Kilosort pipeline module beside intermediate results.
> **Called by:** braindance/core/spikesorter/kilosort2.py:4055 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:4094`
### `compile_results(rec_name, rec_path, results_path, w_e)`
> Compile one recording, or each configured frame chunk, into result artifacts unless an existing result may be reused.
> **Called by:** braindance/core/spikesorter/kilosort2.py:4079 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:4103`
### `run_kilosort2(recording_files: list, intermediate_folders=None, results_folders=None, compiled_results_folder=None, out_file='run_kilosort2.out', kilosort_path=None, kilosort_params={'detect_threshold': 6, 'projection_threshold': [10, 4], 'preclust_threshold': 8, 'car': True, 'minFR': 0.1, 'minfr_goodchannels': 0.1, 'freq_min': 150, 'sigmaMask': 30, 'nPCs': 3, 'ntbuff': 64, 'nfilt_factor': 4, 'NT': None, 'keep_good_only': False}, recompute_recording=True, recompute_sorting=False, reextract_waveforms=False, recurate_first=False, recurate_second=False, recompile_single_recording=False, recompile_all_recordings=False, delete_inter=True, save_script=False, n_jobs=8, total_memory='16G', use_parallel_processing_for_raw_conversion=True, first_n_mins=None, mea_y_max=None, gain_to_uv=None, offset_to_uv=None, rec_chunks=[], freq_min=300, freq_max=6000, waveforms_ms_before=2, waveforms_ms_after=2, pos_peak_thresh=2, max_waveforms_per_unit=300, curate_first=True, curate_second=True, fr_min=0.05, isi_viol_max=1, snr_min=5, spikes_min_first=30, spikes_min_second=50, std_norm_max=1, std_at_peak=True, std_over_window_ms_before=0.5, std_over_window_ms_after=1.5, save_electrodes=True, save_spike_times=True, save_dl_data=True, compile_single_recording=True, compile_to_mat=False, compile_to_npz=True, compile_waveforms=False, compile_all_recordings=False, compiled_waveforms_ms_before=2, compiled_waveforms_ms_after=2, scale_compiled_waveforms=True, create_figures=False, figures_dpi=None, figures_font_size=12, bar_x_label='Recording', bar_y_label='Number of Units', bar_label_rotation=0, bar_total_label='First Curation', bar_selected_label='Selected Curation', scatter_std_max_units_per_recording=None, scatter_recording_colors=['#f74343', '#fccd56', '#74fc56', '#56fcf6', '#1e1efa', '#fa1ed2'], scatter_recording_alpha=1, scatter_x_label='Number of Spikes', scatter_y_label='avg. STD / amplitude', scatter_x_max_buffer=300, scatter_y_max_buffer=0.2, all_templates_color_curated='#000000', all_templates_color_failed='#FF0000', all_templates_per_column=50, all_templates_y_spacing=50, all_templates_y_lim_buffer=10, all_templates_window_ms_before_peak=5, all_templates_window_ms_after_peak=5, all_templates_line_ms_before_peak=1, all_templates_line_ms_after_peak=4, all_templates_x_label='Time Rel. to Peak (ms)')`
> Configure global pipeline state and process one or more recordings, optionally compiling across recordings.
> **Called by:** braindance/core/spikedetector/data.py:1077 (named-call hint); braindance/core/spikesorter/kilosort2.py:4559 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/kilosort2.py:4167`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `run_kilosort2` exposes paths, recomputation flags, Kilosort thresholds, CPU/RAM limits, recording slices, filter bands, waveform windows, curation thresholds, export formats, and plot styling; defaults include 300-6000 Hz filtering, 2 ms waveform sides, FR >= 0.05 Hz, ISI violations <= 1%, SNR >= 5, and NPZ output. |
| `KILOSORT_PATH` may come from the `KILOSORT_PATH` environment variable; MATLAB and a Kilosort2 checkout are required. |
| Global parameters are assigned inside `run_kilosort2` and read throughout the module, so concurrent calls are not isolated. |

## Data Shapes
- Kilosort/Phy arrays include one-dimensional `spike_times`, `spike_clusters`, and `unit_ids`, plus `templates.npy` shaped templates x samples x sorter-channels.
- Extracted waveform files are shaped spikes x samples x recording-channels; cached templates are units-indexed arrays shaped `(max_unit_id + 1, samples, channels)`.
- Compiled unit records can include spike trains, XY location, samples x channels templates, per-channel amplitudes/std norms, channel identifiers, peak indices, and optional waveform arrays.

## Notes
- The module embeds a modified process-pool executor implementation and performs destructive intermediate-directory cleanup under recompute/delete flags.
- `Utils.read_python` executes a Phy `params.py` file to recover metadata.
- Kilosort runs as an external MATLAB subprocess and writes generated `.m`, `.dat`, `.npy`, JSON, MAT, NPZ, plot, and log artifacts.
