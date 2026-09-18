# causal_connectivity.py

**Path:** `braindance/analysis/causal_connectivity.py`
**Module:** `braindance.analysis.causal_connectivity`
**Feature Area:** `Neural Analysis`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Extracts stimulation-aligned recording windows, removes artifacts and detects response spikes for causal connectivity analysis. Computes short-latency response probabilities, longer-latency spike counts and burst fractions, with optional RT-Sort and lab-specific batch transfer workflows.

## Connections
- **Used by:** `braindance.core.phases_analysis_2` — import consumer hint; not a proven runtime call.
- **Uses:** `data_loader` from `braindance.analysis` — imports (static evidence).
- **Uses:** `mapping` from `braindance.analysis` — imports (static evidence).
- **Uses:** `ArtifactRemoval` from `braindance.core.artifact_removal` — imports (static evidence).
- **Uses:** `cubic_fit5` from `braindance.core.artifact_removal` — imports (static evidence).
- **Uses:** `ModelSpikeSorter` from `braindance.core.spikedetector.model2` — imports (static evidence).
- **Uses:** `RTSort` from `braindance.core.spikesorter.rt_sort` — imports (static evidence).
- **Uses:** `detect_sequences` from `braindance.core.spikesorter.rt_sort` — imports (static evidence).
- **Uses:** `sort_intrinsic` from `braindance.experiments.causal_freq_sort_intrinsic` — imports (static evidence).
- **Shared data:** data_loader adjusts stim times and loads windows -> cubic_fit5 cleanup -> threshold_data or RTSort.sort_offline -> causal_connectivity_metrics.
- **Shared data:** main_rt_sort loads/detects sorter, runs causal analysis and optionally experiments.causal_freq_sort_intrinsic.sort_intrinsic.

## Dependencies
- `braindance.analysis.data_loader` — intra-repo import; source import evidence.
- `braindance.analysis.mapping` — intra-repo import; source import evidence.
- `braindance.core.artifact_removal.ArtifactRemoval` — intra-repo import; source import evidence.
- `braindance.core.artifact_removal.cubic_fit5` — intra-repo import; source import evidence.
- `braindance.core.spikedetector.model2.ModelSpikeSorter` — intra-repo import; source import evidence.
- `braindance.core.spikesorter.rt_sort.RTSort` — intra-repo import; source import evidence.
- `braindance.core.spikesorter.rt_sort.detect_sequences` — intra-repo import; source import evidence.
- `braindance.experiments.causal_freq_sort_intrinsic.sort_intrinsic` — intra-repo import; source import evidence.
- `h5py` — external or unresolved local import; source import evidence.
- `numba.njit` — external or unresolved local import; source import evidence.
- `numba.prange` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `psutil` — external or unresolved local import; source import evidence.
- `scipy.signal.find_peaks` — external or unresolved local import; source import evidence.
- `spikeinterface.extractors.MaxwellRecordingExtractor` — external or unresolved local import; source import evidence.
- `tqdm.contrib.concurrent.process_map` — external or unresolved local import; source import evidence.
- `tqdm.tqdm` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `_deprecated(function)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/causal_connectivity.py:25`
### `_clean_stim_response(data)`
> Processes response of one stimulus Removes artifacts, thresholds, and returns the processed data
> **Called by:** braindance/analysis/causal_connectivity.py:105 (named-call hint); braindance/analysis/causal_connectivity.py:124 (named-call hint); braindance/analysis/causal_connectivity.py:132 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/causal_connectivity.py:69`
### `_clean_stim_responses(data)`
> Calls _clean_stim_response repeatedly through multiprocessing
> **Called by:** braindance/analysis/causal_connectivity.py:138 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/causal_connectivity.py:94`
### `_clean_stim_responses_all(data)`
> Calls _clean_stim_response repeatedly through multiprocessing. Does this for axis except for the last one.
> **Called by:** braindance/analysis/causal_connectivity.py:144 (named-call hint); braindance/analysis/causal_connectivity.py:296 (named-call hint); braindance/analysis/causal_connectivity.py:299 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/causal_connectivity.py:112`
### `clean_stim_response(data)`
> Processes response of one stimulus Removes artifacts, thresholds, and returns the processed data
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/causal_connectivity.py:130`
### `clean_stim_responses(data)`
> Calls clean_stim_response repeatedly through multiprocessing
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/causal_connectivity.py:136`
### `clean_stim_responses_all(data)`
> Calls clean_stim_response repeatedly through multiprocessing. Does this for axis except for the last one.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/causal_connectivity.py:142`
### `causal_reactions(data_path, react_channels, wind_ms=200, stim_log=None, save_dir=None, tag='causal', save_clean_data=False, stim_patterns=None, remove_start_frames=60, ms_before_stim=None, verbose=False)`
> For a file, stim_log, stim indices, and react indices: 1. Clean data # 2. Threshold data # 3. Calculate separate matrices # 4. Save matrices
> **Called by:** braindance/analysis/causal_connectivity.py:739 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/analysis/causal_connectivity.py:148`
### `get_spikes(data, sorter='thresh', sorter_params={'thresh': -9}, save_dir=None, spikes_offset_ms=0)`
> Takes data of shape (reps, ..., time_window), and returns a sparse listing of spike times, where it is a np object of shape (...) with each element being a list of spike times.
> **Called by:** braindance/analysis/causal_connectivity.py:766 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/analysis/causal_connectivity.py:362`
### `threshold_data(data, thresholds, fs_ms=20)`
> Takes data of shape (n, time_window), and thresholds it based on the given thresholds.
> **Called by:** braindance/analysis/causal_connectivity.py:409 (named-call hint); braindance/analysis/causal_connectivity.py:420 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/causal_connectivity.py:484`
### `causal_connectivity_metrics(spike_data, first_order_ms=10, multi_order_ms=(10, 200), burst_mad_thresh=3, save_dir=None, count_bursts=True)`
> Parameters: spike_data : np.ndarray (reps, stim_electrodes, react_electrodes, ) type=obj Array of lists of spike times on the react electrode from the stimulations on the stim electrode. first_order_ms : int Time window to consider as first order
> **Called by:** braindance/analysis/causal_connectivity.py:781 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/analysis/causal_connectivity.py:520`
### `main(data_path=None, save_dir='none', tag='causal', react_inds='all', stim='infer', remove_start_frames=40, count_bursts=False)`
> Params: data_path If None, use command line args for parameters Else, use arguments to this function
> **Called by:** braindance/analysis/causal_connectivity.py:846 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/causal_connectivity.py:611`
### `main_rt_sort(recording_path, recording_window_ms=(2 * 60 * 1000, 6 * 60 * 1000), include_spikes_ms_before_stim=0, tag='Causal', phases=None, verbose=True)`
> Run main() with RT-Sort
> **Called by:** braindance/analysis/causal_connectivity.py:1015 (named-call hint); braindance/analysis/causal_connectivity.py:992 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/analysis/causal_connectivity.py:797`
### `remove_duplicate_chans_elecs(rec_path)`
> unclear — see source
> **Called by:** braindance/analysis/causal_connectivity.py:1008 (named-call hint); braindance/analysis/causal_connectivity.py:990 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/causal_connectivity.py:862`
### `main_kosik(recording_paths, recording_window_ms=(2 * 60 * 1000, 6 * 60 * 1000), ind_24h=[], include_spikes_ms_before_stim=0, tag='Causal')`
> For Kosik Lab, UCSB, setup: 1. Download recording and log file from server onto GPU computer 2. Process data on GPU computer 3. Upload processed data to server 4. Delete data from GPU computer If there are too many stimulation patterns and not enough disk space, a solution is to split up the log file into multiple different ones and handle, upload, and delete a fraction of all .npy files at a time (they are independent of each other)
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/causal_connectivity.py:930`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| window_ms=200; remove_start_frames; optional ms_before_stim; tag; save_clean_data |
| sorter thresh\\|std_thresh\\|RT-Sort; global sorter_params; first_order_ms=10,multi_order_ms=(10,200),burst_mad_thresh=3 |
| CLI data_path,--save_dir,--tag/-t,--react/-r,--stim |

## Data Shapes
- Clean data (repetitions,stim_patterns,react_channels,time_frames), optionally split per pattern on disk
- Threshold spike object array (repetitions,stim_patterns,react_channels) contains frame indexes; RT-Sort detections are milliseconds
- Connectivity matrices (stim_patterns,react_channels); burst_percent per stimulation pattern

## Notes
- This module is deprecated and emits FutureWarning through compatibility wrappers; evaluate SpikeLab or another stimulation-timed replacement before adding consumers.
- Most timing assumes 20 kHz; main skips metrics for RT-Sort because units/output integration is unfinished.
- Memory-based split caches are reused by file existence; current return can contain only last split buffer.
- remove_duplicate_chans_elecs rewrites H5 mapping in place; main_kosik transfers via SSH/rsync and can recursively remove downloaded parent directories.
- main function references args.stim in one branch even when called programmatically.
