# rt_sort.py

**Path:** `braindance/core/spikesorter/rt_sort.py`
**Module:** `braindance.core.spikesorter.rt_sort`
**Feature Area:** `Spike Sorting`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Implements RT-Sort sequence discovery and real-time/offline spike assignment around a neural detection model. It saves centered traces/model outputs, forms and statistically splits co-detection clusters, reassigns spikes, merges compatible sequences, and builds a tensorized RTSort runtime.

## Connections
- **Used by:** `braindance.analysis.causal_connectivity` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_analysis_2` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases_v3.phases_analysis_3` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.spikedetector.data` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.spikesorter.burst` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.spikesorter.kilosort2` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.spikesorter.sort` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.analysis_sorting` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.streaming_workshop.session` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.real_time_sorting` — import consumer hint; not a proven runtime call.
- **Uses:** `Unit` from `braindance.analysis.select_units` — imports (static evidence).
- **Uses:** `Config` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `ModelSpikeSorter` from `braindance.core.spikedetector.model` — imports (static evidence).
- **Shared data:** Uses `ModelSpikeSorter` for compiled inference and SpikeInterface for recordings and optional sorting output.
- **Shared data:** `detect_sequences` flows through `save_traces`, `run_detection_model`, `form_all_clusters`, `setup_coc_clusters_parallel`, `reassign_spikes_to_clusters`, `intra_merge`, and `inter_merge` before constructing `RTSort`.
- **Shared data:** The Maxwell environment process loads an RTSort and communicates through shared-memory observation/stimulation buffers.

## Dependencies
- `braindance.analysis.select_units.Unit` — intra-repo import; source import evidence.
- `braindance.core.maxwell_env.Config` — intra-repo import; source import evidence.
- `braindance.core.spikedetector.model.ModelSpikeSorter` — intra-repo import; source import evidence.
- `diptest.diptest` — external or unresolved local import; source import evidence.
- `h5py` — external or unresolved local import; source import evidence.
- `numba.jit` — external or unresolved local import; source import evidence.
- `numba.prange` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pynvml` — external or unresolved local import; source import evidence.
- `scipy` — external or unresolved local import; source import evidence.
- `sklearn.mixture.GaussianMixture` — external or unresolved local import; source import evidence.
- `spikeinterface.core.BaseRecording` — external or unresolved local import; source import evidence.
- `spikeinterface.extractors.MaxwellRecordingExtractor` — external or unresolved local import; source import evidence.
- `spikeinterface.extractors.NumpySorting` — external or unresolved local import; source import evidence.
- `spikeinterface.extractors.NwbRecordingExtractor` — external or unresolved local import; source import evidence.
- `spikelab.SpikeData` — external or unresolved local import; source import evidence.
- `threadpoolctl.threadpool_limits` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.
- `tqdm.tqdm` — external or unresolved local import; source import evidence.

## Classes
### RTSort()
> Tensorize learned sequence footprints and assign detected peaks to sequences online or offline using spatial, latency, and amplitude gates.
**Source:** `braindance/core/spikesorter/rt_sort.py:421`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/rt_sort.py:2314 (named-call hint); braindance/core/spikesorter/rt_sort.py:401 (named-call hint)
**Constructor:** `__init__(self, sequences, model, params: dict, buffer_size=100, device='cuda', dtype=torch.float16)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_seq_no_overlap_mask_np` | inferred at runtime | `self.seq_no_overlap_mask.cpu().numpy()` |
| `_seq_root_elecs` | inferred at runtime | `[seq.root_elec for seq in sequences]` |
| `buffer_size` | inferred at runtime | `buffer_size` |
| `chan_ids` | inferred at runtime | `params.get('chan_ids', None)` |
| `clip_amp_median_diff` | inferred at runtime | `max_amp_median_diff_spikes * clip_amp_median_diff_factor` |
| `clip_latency_diff` | inferred at runtime | `max_latency_diff_spikes * clip_latency_diff_factor` |
| `comp_elecs` | inferred at runtime | `torch.tensor(all_comp_elecs, dtype=torch.long, device=device)[:, None]` |
| `comp_elecs_flattened` | inferred at runtime | `self.comp_elecs.flatten()` |
| `cur_num_pre_median_frames` | inferred at runtime | `0` |
| `device` | inferred at runtime | `device` |
| `dtype` | inferred at runtime | `dtype` |
| `end_buffer` | inferred at runtime | `model.buffer_end_sample` |
| `front_buffer` | inferred at runtime | `model.buffer_front_sample` |
| `full_end_buffer` | inferred at runtime | `self.end_buffer + self.seq_n_after` |
| `full_front_buffer` | inferred at runtime | `self.front_buffer + self.seq_n_before` |
| `ignore_spikes_before_minuend` | inferred at runtime | `self.input_size - self.end_buffer - self.seq_n_after` |
| `input_scale` | inferred at runtime | `model.input_scale` |
| `input_size` | inferred at runtime | `model.sample_size` |
| `last_detections` | inferred at runtime | `torch.full((num_seqs,), -pre_median_frames, dtype=torch.int64, device=device)` |
| `last_pre_median_output_frame` | inferred at runtime | `-np.inf` |
| `latest_frame` | inferred at runtime | `0` |
| `loose_thresh_logit` | inferred at runtime | `sigmoid_inverse(loose_thresh)` |
| `max_amp_median_diff_spikes` | inferred at runtime | `max_amp_median_diff_spikes` |
| `max_chan_id` | inferred at runtime | `max(self.chan_ids) if self.chan_ids is not None else None` |
| `max_latency_diff_spikes` | inferred at runtime | `max_latency_diff_spikes` |
| `max_pool` | inferred at runtime | `torch.nn.MaxPool1d(seq_n_before + seq_n_after + 1, return_indices=True)` |
| `max_root_amp_median_std_spikes` | inferred at runtime | `max_root_amp_median_std_spikes` |
| `min_elecs_for_array_noise` | inferred at runtime | `min_elecs_for_array_noise` |
| `min_inner_loose_detections` | inferred at runtime | `min_inner_loose_detections` |
| `model` | inferred at runtime | `model.model.conv.to(device)` |
| `model_num_output_locs` | inferred at runtime | `model.num_output_locs` |
| `num_elecs` | inferred at runtime | `elec_locs.shape[0]` |
| `num_seqs` | inferred at runtime | `len(sequences)` |
| `overlap` | inferred at runtime | `round(repeated_detection_overlap_time * samp_freq)` |
| `pre_median_frames` | inferred at runtime | `torch.full((elec_locs.shape[0], pre_median_frames), torch.nan, dtype=dtype, device=device)` |
| `pre_medians` | inferred at runtime | `None` |
| `samp_freq` | inferred at runtime | `params['samp_freq']` |
| `seq_comp_elecs` | inferred at runtime | `[seq.comp_elecs for seq in sequences]` |
| `seq_locs` | inferred at runtime | `np.array([elec_locs[seq.root_elec] for seq in sequences])` |
| `seq_n_after` | inferred at runtime | `seq_n_after` |
| `seq_n_before` | inferred at runtime | `seq_n_before` |
| `seq_no_overlap_mask` | inferred at runtime | `seq_no_overlap_mask` |
| `seq_spike_trains` | inferred at runtime | `[seq.spike_train for seq in sequences]` |
| `seqs_amp_weights` | inferred at runtime | `torch.vstack(seqs_amp_weights)` |
| `seqs_amps` | inferred at runtime | `torch.vstack(seqs_amps)` |
| `seqs_inner_loose_elecs` | inferred at runtime | `torch.vstack(seqs_inner_loose_elecs)` |
| `seqs_latencies` | inferred at runtime | `torch.vstack(seqs_latencies)` |
| `seqs_latency_weights` | inferred at runtime | `torch.vstack(seqs_latency_weights)` |
| `seqs_loose_elecs` | inferred at runtime | `torch.vstack(seqs_loose_elecs)` |
| `seqs_min_loose_elecs` | inferred at runtime | `torch.tensor(seqs_min_loose_elecs, dtype=torch.int16, device=device)` |
| `seqs_root_amp_means` | inferred at runtime | `torch.tensor(seqs_root_amp_means, dtype=dtype, device=device)` |
| `seqs_root_amp_stds` | inferred at runtime | `torch.tensor(seqs_root_amp_stds, dtype=dtype, device=device)` |
| `seqs_root_elecs` | inferred at runtime | `list(seqs_root_elecs)` |
| `seqs_root_elecs_rel_comp_elecs` | inferred at runtime | `seqs_root_elecs_rel_comp_elecs` |
| `spike_arange` | inferred at runtime | `torch.arange(0, seq_n_before + seq_n_after + 1, device=device)` |
| `stringent_thresh_logit` | inferred at runtime | `sigmoid_inverse(stringent_thresh)` |
| `total_num_pre_median_frames` | inferred at runtime | `pre_median_frames` |
**Methods:**
#### `reset(self)`
> Resets the internal state to prepare for a new experiment.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/rt_sort.py:621`
#### `running_sort(self, obs, model_chunk=None, latest_frame=None, use_numba=False, remove_median=True)`
> Update rolling trace normalization and sort the newest frames while maintaining the recording clock.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/rt_sort.py:641`
#### `sort_offline(self, recording, inter_path=None, recording_window_ms=None, model_outputs=None, reset=True, return_spikeinterface_sorter=False, verbose=False)`
> Stream a recording or saved traces through online sorting and collect per-sequence spike trains.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:707`
#### `sort_chunk(self, chunk, torch_window=None, spike_times_frame_offset=0, ignore_spikes_before=0)`
> Score model-output peaks against every sequence footprint and suppress repeated spatially overlapping assignments.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:817`
#### `sort_chunk_numba(self, chunk, torch_window=None, spike_times_frame_offset=0, ignore_spikes_before=0)`
> Numba-optimized version of sort_chunk_faster. Keeps GPU computation for the heavy parallel work, uses Numba for the sequential loop.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1015`
#### `select_seqs(self, seq_ind: list)`
> Create a reduced sorter containing requested sequences plus nearby sequences needed for overlap suppression.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1110`
#### `calc_pre_medians(self, pre_median_frames: torch.Tensor)`
> Estimate per-comparison-electrode noise from pre-event absolute medians, convert MAD to sigma, and clamp the scale away from zero.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1173`
#### `set_model(self, model, num_elecs=None, input_size=None)`
> Load if necessary and compile a ModelSpikeSorter for this sorter’s electrode count, input window, device, and runtime use.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/rt_sort.py:1182`
#### `to_tensor(self, data)`
> Copy array-like data to a tensor using the sorter’s configured dtype and device.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1196`
#### `save(self, pickle_path)`
> Serialize sorter state without the compiled model so it can be reattached when loaded.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/rt_sort.py:1199`
#### `save_seq_data(self, save_path)`
> As numpy array, save each sequence's (spike_train, root_elecrode) This is mainly for testing
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/core/spikesorter/rt_sort.py:1209`
#### `get_seq_root_elecs(self)`
> self.seq_root_elecs is not in the correct order because this is only used for sorting
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1219`
#### `get_units(self)`
> Reserved unit-object conversion hook; it currently raises NotImplementedError before its unreachable legacy conversion code.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1231`
#### `to_spikedata(self, all_seq_detections=None)`
> Params all_seq_detections If None, use self.seq_spike_trains (sequences' spikes after second merging but before final spike reassignment) to create SpikeData object If not None, should be the return value of method self.sort_offline
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1237`
#### `load_from_file(pickle_path, model=None)`
> Need to specify model because cannot save model in .pickle
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1254`
### CocCluster()
> Represent a preliminary propagation sequence and its splitting electrodes and spike train.
**Source:** `braindance/core/spikesorter/rt_sort.py:1698`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/rt_sort.py:1752 (named-call hint); braindance/core/spikesorter/rt_sort.py:1930 (named-call hint)
**Constructor:** `__init__(self, root_elec, split_elecs, spike_train)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_spike_train` | inferred at runtime | `spike_train` |
| `root_elec` | inferred at runtime | `root_elec` |
| `root_elecs` | inferred at runtime | `[root_elec]` |
| `split_elecs` | inferred at runtime | `split_elecs` |
**Methods:**
#### `split(self, split_elec, spike_train)`
> Create a child cluster that adds one splitting electrode and uses the supplied subset of co-occurrence times.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1707`
#### `spike_train(self)`
> Expose the cluster’s candidate spike times in sorted order.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1712`
### Patience()
> Track how many distinct electrode-distance rings have been searched without extending a co-detection branch.
**Source:** `braindance/core/spikesorter/rt_sort.py:2157`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/rt_sort.py:1948 (named-call hint)
**Constructor:** `__init__(self, root_elec, patience_end, elec_locs)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `counter` | inferred at runtime | `0` |
| `elec_locs` | inferred at runtime | `elec_locs` |
| `last_dist` | inferred at runtime | `0` |
| `patience_end` | inferred at runtime | `patience_end` |
| `root_elec` | inferred at runtime | `root_elec` |
**Methods:**
#### `reset(self)`
> Reset the unsuccessful-distance-ring counter while retaining its last observed distance.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/rt_sort.py:2167`
#### `end(self, comp_elec) -> bool`
> Increment counter and check if to end
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikesorter/rt_sort.py:2170`
#### `verbose(self)`
> Print the current branch-search counter and stopping threshold.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:2185`
### Merge()
> Merge two compatible clusters while deduplicating spikes and recomputing electrode statistics.
**Source:** `braindance/core/spikesorter/rt_sort.py:2649`
**Kind:** class. **Instantiated by:** braindance/core/spikesorter/rt_sort.py:3038 (named-call hint)
**Constructor:** `__init__(self, cluster_i, cluster_j)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `cluster_i` | inferred at runtime | `cluster_i` |
| `cluster_j` | inferred at runtime | `cluster_j` |
**Methods:**
#### `merge(self, params)`
> Merge the selected cluster pair’s sorted detections without near-duplicates, align latencies to the retained root, recompute footprint statistics, and return the absorbed cluster.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:2655`

## Functions
### `detect_sequences(recording, inter_path, detection_model=None, recording_window_ms=None, stringent_thresh=0.275, loose_thresh=0.1, inference_scaling_numerator=12.6, ms_before=0.5, ms_after=0.5, pre_median_ms=50, inner_radius=50, outer_radius=100, min_elecs_for_array_noise_n=100, min_elecs_for_array_noise_f=0.1, min_elecs_for_seq_noise_n=50, min_elecs_for_seq_noise_f=0.05, min_activity_root_cocs=2, min_activity_hz=0.05, max_n_components_latency=4, min_coc_n=10, min_coc_p=10, min_extend_comp_p=50, elec_patience=6, split_coc_clusters_amps=True, min_amp_dist_p=0.1, max_n_components_amp=4, min_loose_elec_prob=0.03, min_inner_loose_detections=3, min_loose_detections_n=4, min_loose_detections_r_spikes=1 / 3, min_loose_detections_r_sequences=1 / 3, max_latency_diff_spikes=3.5, max_latency_diff_sequences=3.5, clip_latency_diff_factor=2, max_amp_median_diff_spikes=0.65, max_amp_median_diff_sequences=0.65, clip_amp_median_diff_factor=2, max_root_amp_median_std_spikes=2.5, max_root_amp_median_std_sequences=np.inf, repeated_detection_overlap_time=0.2, min_seq_spikes_n=10, min_seq_spikes_hz=0.05, relocate_root_min_amp=0.8, relocate_root_max_latency=-2, return_spikes=False, delete_inter=False, device='cuda', num_processes=None, ignore_warnings=True, verbose=True, debug=False)`
> Discover propagation sequences from a recording and return a configured RTSort or final reassigned spike trains.
> **Called by:** braindance/analysis/causal_connectivity.py:825 (named-call hint); braindance/core/phases_analysis_2.py:1101 (named-call hint); braindance/core/phases_v3/phases_analysis_3.py:208 (named-call hint); braindance/core/spikesorter/rt_sort.py:3289 (named-call hint); braindance/examples/streaming_workshop/analysis_sorting.py:121 (named-call hint); braindance/experiments/real_time_sorting.py:41 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/core/spikesorter/rt_sort.py:53`
### `process_spikes_numba(spike_scores_np, peak_ind_flat_np, seq_no_overlap_mask_np, seq_n_before_front, spike_times_frame_offset, samp_freq_inv, ignore_spikes_before, overlap, num_seqs)`
> Numba-compiled spike processing loop. Returns list of (seq_idx, spike_time) tuples.
> **Called by:** braindance/core/spikesorter/rt_sort.py:1110 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1274`
### `rt_sort_maxwell_env_process(process_ready, obs_ready, stim_ready, env_done, obs_shm_name, stim_shm_name, start_seq_ind, current_time_name, rt_sort_path, config_path, save_path=None, max_stim_hz=60)`
> Process for sorting with RT-Sort and deciding whether to stim in parallel
> **Called by:** unclear — see source. **Side effects:** write call (static hint).
**Source:** `braindance/core/spikesorter/rt_sort.py:1321`
### `save_traces(recording, inter_path, start_ms=0, end_ms=None, num_processes=None, dtype='float16', verbose=True)`
> Persist scaled electrode-by-frame traces, selecting optimized Maxwell or generic SpikeInterface extraction.
> **Called by:** braindance/core/spikesorter/kilosort2.py:4084 (named-call hint); braindance/core/spikesorter/rt_sort.py:279 (named-call hint); braindance/core/spikesorter/rt_sort.py:778 (named-call hint); braindance/core/spikesorter/sort.py:61 (named-call hint); braindance/examples/streaming_workshop/analysis_sorting.py:133 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1434`
### `save_traces_si(recording: BaseRecording, scaled_traces_path, start_ms=0, end_ms=None, num_processes=16, dtype='float16', verbose=True)`
> Preallocate an electrode-by-frame NPY file and fill each channel in parallel from scaled SpikeInterface traces.
> **Called by:** braindance/core/spikesorter/rt_sort.py:1499 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/core/spikesorter/rt_sort.py:1467`
### `_save_traces_si(task)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1504`
### `save_traces_mea(rec_path, save_path, start_ms=0, end_ms=None, samp_freq=20, default_gain=1, chunk_size=100000, num_processes=2, dtype='float16', verbose=True)`
> Can't save traces with spikeinterface get_traces() because it is really slow on MaxWell MEA recordings
> **Called by:** braindance/core/spikesorter/rt_sort.py:1494 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/core/spikesorter/rt_sort.py:1515`
### `_get_traces_mea_old(rec_path)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1578`
### `_get_traces_mea_new(rec_path)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1583`
### `_save_traces_mea(task)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1588`
### `run_detection_model(recording, model, scaled_traces_path, model_traces_path=None, model_outputs_path=None, inference_scaling_numerator=12.6, pre_median_frames=1000, model_inter_path=None, device='cuda', verbose=True)`
> Run the compiled detector over overlapping input windows and persist centered traces and output logits.
> **Called by:** braindance/core/spikesorter/rt_sort.py:289 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/core/spikesorter/rt_sort.py:1596`
### `form_coc_clusters(root_elec, params)`
> Build latency- and amplitude-split co-detection clusters rooted at one electrode.
> **Called by:** braindance/core/spikesorter/rt_sort.py:2023 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1716`
### `_form_all_clusters(task)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1974`
### `form_all_clusters(params)`
> Form all preliminary propagation sequences (essentially, call form_coc_clusters() on all electrodes)
> **Called by:** braindance/core/spikesorter/rt_sort.py:352 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:1987`
### `branch_coc_cluster(root_cluster, comp_elecs, coc_dict, allowed_root_times, patience, params)`
> Recursive function, first called in form_coc_clusters
> **Called by:** braindance/core/spikesorter/rt_sort.py:1946 (named-call hint); braindance/core/spikesorter/rt_sort.py:2189 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:2010`
### `reassign_spikes_to_clusters(all_clusters, model, params)`
> Run RT-Sort over preliminary clusters, optionally in GPU-sized groups, and filter by sequence spike count.
> **Called by:** braindance/core/spikesorter/rt_sort.py:368 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:2192`
### `_setup_coc_clusters_parallel(task)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:2289`
### `setup_coc_clusters_parallel(coc_clusters, params)`
> Run setup_cluster on coc_clusters with parallel processing
> **Called by:** braindance/core/spikesorter/rt_sort.py:2325 (named-call hint); braindance/core/spikesorter/rt_sort.py:353 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:2302`
### `setup_coc_clusters(coc_clusters, params)`
> Populate merge and filtering statistics for each preliminary cluster in place, with optional progress display.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:2326`
### `setup_cluster(cluster, params, n_cocs=None)`
> Derive per-electrode probability, latency, amplitude, and comparison sets for filtering and merging a sequence.
> **Called by:** braindance/core/spikesorter/rt_sort.py:2338 (named-call hint); braindance/core/spikesorter/rt_sort.py:2376 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:2335`
### `setup_elec_stats(cluster, params)`
> Set cluster.loose_elecs, cluster.inner_loose_elecs and the root elec
> **Called by:** braindance/core/spikesorter/rt_sort.py:2638 (named-call hint); braindance/core/spikesorter/rt_sort.py:2821 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:2518`
### `relocate_root(cluster, new_root, params)`
> Relocate cluster's root electrode
> **Called by:** braindance/core/spikesorter/rt_sort.py:2659 (named-call hint); braindance/core/spikesorter/rt_sort.py:2668 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:2584`
### `relocate_root_latency(cluster, params)`
> Change root electrode to most negative-latency electrode with: Mean latency of -2 frames or less Median detection above stringent threhsold Mean amplitude that is 80% or more of current root If no electrode meets requirements, keep current root electrode
> **Called by:** braindance/core/spikesorter/rt_sort.py:3102 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:2597`
### `relocate_root_prob(cluster, params)`
> Change root electrode to electrode with highest median detection score
> **Called by:** braindance/core/spikesorter/rt_sort.py:3151 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:2618`
### `filter_clusters(coc_clusters, params)`
> Return coc_clusters that enough loose and inner loose electrodes But not too many to be considered a noise sequence
> **Called by:** braindance/core/spikesorter/rt_sort.py:3149 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:2627`
### `merge_verbose(merge, update_history=True)`
> Verbose for merge_coc_clusters
> **Called by:** braindance/core/spikesorter/rt_sort.py:3070 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:2783`
### `merge_coc_clusters(coc_clusters, params)`
> Greedily merge the best compatible cluster pair until no candidate passes spatial and footprint constraints.
> **Called by:** braindance/core/spikesorter/rt_sort.py:3100 (named-call hint); braindance/core/spikesorter/rt_sort.py:3146 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:2841`
### `_intra_merge(task)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:3052`
### `intra_merge(all_clusters, params)`
> intra = merge clusters with same root electrode
> **Called by:** braindance/core/spikesorter/rt_sort.py:380 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:3061`
### `inter_merge(intra_merged_clusters, params)`
> inter = merge clusters with different root electrodes
> **Called by:** braindance/core/spikesorter/rt_sort.py:389 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:3093`
### `pickle_dump(obj, path)`
> Serialize an object to a file using pickle’s highest protocol.
> **Called by:** braindance/core/spikesorter/rt_sort.py:1221 (named-call hint); braindance/core/spikesorter/rt_sort.py:2551 (named-call hint); braindance/core/spikesorter/rt_sort.py:342 (named-call hint); braindance/core/spikesorter/rt_sort.py:355 (named-call hint); braindance/core/spikesorter/rt_sort.py:370 (named-call hint); braindance/core/spikesorter/rt_sort.py:398 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/core/spikesorter/rt_sort.py:3120`
### `pickle_load(path)`
> Deserialize and return a trusted pickle file.
> **Called by:** braindance/core/spikesorter/rt_sort.py:1282 (named-call hint); braindance/core/spikesorter/rt_sort.py:350 (named-call hint); braindance/core/spikesorter/rt_sort.py:366 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:3125`
### `load_recording(recording)`
> If recording is a spikeinterface recording (some subclass of BaseRecording), then simply return it Else, recording should be a file path, and it will be loaded as MaxwellRecordingExtractor
> **Called by:** braindance/core/spikesorter/rt_sort.py:1484 (named-call hint); braindance/core/spikesorter/rt_sort.py:241 (named-call hint); braindance/core/spikesorter/sort.py:54 (named-call hint); braindance/examples/streaming_workshop/analysis_sorting.py:102 (named-call hint); braindance/examples/streaming_workshop/analysis_sorting.py:103 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:3130`
### `calc_dist(x1, y1, x2, y2)`
> Compute Euclidean distance between two planar electrode coordinates.
> **Called by:** braindance/core/spikesorter/rt_sort.py:1790 (named-call hint); braindance/core/spikesorter/rt_sort.py:2179 (named-call hint); braindance/core/spikesorter/rt_sort.py:2218 (named-call hint); braindance/core/spikesorter/rt_sort.py:2274 (named-call hint); braindance/core/spikesorter/rt_sort.py:2435 (named-call hint); braindance/core/spikesorter/rt_sort.py:2450 (named-call hint); braindance/core/spikesorter/rt_sort.py:2455 (named-call hint); braindance/core/spikesorter/rt_sort.py:2587 (named-call hint); braindance/core/spikesorter/rt_sort.py:2598 (named-call hint); braindance/core/spikesorter/rt_sort.py:2603 (named-call hint); braindance/core/spikesorter/rt_sort.py:2933 (named-call hint); braindance/core/spikesorter/rt_sort.py:485 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:3148`
### `rec_ms_to_output_frame(ms, samp_freq, front_buffer)`
> Map recording time in milliseconds to a detector-output frame after subtracting the model front buffer.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:3152`
### `sigmoid(x)`
> Convert logits to probabilities after clipping the input range to avoid exponential overflow.
> **Called by:** braindance/core/spikesorter/rt_sort.py:2421 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:3157`
### `sigmoid_inverse(y)`
> Convert probabilities to logits with the inverse logistic transform.
> **Called by:** braindance/core/spikesorter/rt_sort.py:1801 (named-call hint); braindance/core/spikesorter/rt_sort.py:1802 (named-call hint); braindance/core/spikesorter/rt_sort.py:610 (named-call hint); braindance/core/spikesorter/rt_sort.py:611 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:3167`
### `calc_pre_median(frame, traces, pre_median_frames, elecs=slice(None))`
> Get the preceeding mean and median of traces before :time:
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:3171`
### `order_sequences(sequences)`
> Sort sequence objects by root-electrode index and rewrite their sequential idx fields.
> **Called by:** braindance/core/spikesorter/rt_sort.py:3152 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:3193`
### `get_gpu_free_memory()`
> Return free memory on the active CUDA device in GiB through NVML, or zero when CUDA is unavailable.
> **Called by:** braindance/core/spikesorter/rt_sort.py:2258 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:3201`
### `set_warning_level(ignore_warnings=False)`
> Globally switch Python warning filters between ignore and default behavior.
> **Called by:** braindance/core/spikesorter/rt_sort.py:2021 (named-call hint); braindance/core/spikesorter/rt_sort.py:2337 (named-call hint); braindance/core/spikesorter/rt_sort.py:3099 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:3233`
### `main()`
> Run a hard-coded development RT-Sort detection example over a fixed Maxwell recording interval.
> **Called by:** braindance/core/spikesorter/rt_sort.py:3294 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikesorter/rt_sort.py:3241`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `detect_sequences` exposes detection thresholds, millisecond windows, electrode radii, noise/activity gates, GMM split criteria, assignment/merge tolerances, minimum sequence activity, device/process settings, and debug-cache behavior. |
| Default execution uses CUDA and may load the packaged MEA `ModelSpikeSorter`; `neuropixels_params` supplies alternate threshold overrides. |
| Debug mode persists intermediate pickle files, while `delete_inter=True` removes the entire intermediate directory after successful construction. |

## Data Shapes
- Saved traces and model outputs are electrode x frame arrays, usually float16 `.npy` memmaps.
- `RTSort.running_sort` accepts observations shaped frames x electrodes and returns `(sequence_index, spike_time_ms)` pairs.
- Offline sorting returns an object array of length `num_sequences`, each element a sorted millisecond spike train; it can instead produce a SpikeInterface `NumpySorting`.
- Cluster statistics store electrode x co-detection probabilities, latencies in frames, and amplitude-to-noise ratios.

## Notes
- Sampling frequency is represented internally in kHz/frames per millisecond, while SpikeInterface output converts it back to Hz.
- The first pre-median window initializes normalization and produces no online detections.
- GPU memory estimates may partition preliminary sequences before reassignment; multiprocessing is used for trace extraction, sequence formation, setup, and intra-root merging.
