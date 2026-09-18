# rt_sort.py

**Path:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py`
**Module:** `braindance.analysis.intermediate_rt_sorting.rt_sort`
**Feature Area:** `Spike Sorting`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Builds propagation sequence templates from multichannel recordings using neural spike detection, Gaussian mixture splits, reassignment, and spatial/latency/amplitude based merging. Its RTSort copy exports intermediate per-chunk diagnostic data instead of the ordinary spike trains expected by several retained callers.

## Connections
- **Uses:** `Unit` from `braindance.analysis.select_units` — imports (static evidence).
- **Uses:** `Config` from `braindance.core.maxwell_env` — imports (static evidence).
- **Uses:** `ModelSpikeSorter` from `braindance.core.spikedetector.model` — imports (static evidence).
- **Shared data:** detect_sequences saves traces, runs ModelSpikeSorter, forms CocCluster candidates, sets electrode statistics, reassigns spikes, merges within then across roots, and constructs RTSort.
- **Shared data:** form_coc_clusters and branch_coc_cluster split local codetections by latency Gaussian mixtures and amplitude distribution tests; merge_coc_clusters greedily combines compatible templates.
- **Shared data:** RTSort.running_sort maintains noise estimates and rolling buffers before sort_chunk compares peak timing and amplitude to sequence templates.
- **Shared data:** The multiprocessing Maxwell adapter coordinates shared buffers/events and a stimulation pipe; main is a local diagnostic entry point.

## Dependencies
- `braindance.analysis.select_units.Unit` — intra-repo import; source import evidence.
- `braindance.core.maxwell_env.Config` — intra-repo import; source import evidence.
- `braindance.core.spikedetector.model.ModelSpikeSorter` — intra-repo import; source import evidence.
- `diptest.diptest` — external or unresolved local import; source import evidence.
- `h5py` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pynvml` — external or unresolved local import; source import evidence.
- `scipy` — external or unresolved local import; source import evidence.
- `sklearn.mixture.GaussianMixture` — external or unresolved local import; source import evidence.
- `spikeinterface.core.BaseRecording` — external or unresolved local import; source import evidence.
- `spikeinterface.extractors.MaxwellRecordingExtractor` — external or unresolved local import; source import evidence.
- `spikeinterface.extractors.NwbRecordingExtractor` — external or unresolved local import; source import evidence.
- `threadpoolctl.threadpool_limits` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.
- `tqdm.tqdm` — external or unresolved local import; source import evidence.

## Classes
### RTSort()
> Maintain sequence tensors and streaming buffers for diagnostic template matching.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:397`
**Kind:** class. **Instantiated by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:2088 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:375 (named-call hint)
**Constructor:** `__init__(self, sequences, model, params: dict, buffer_size=100, device='cuda', dtype=torch.float16)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
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
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:592`
#### `running_sort(self, obs, model_chunk=None, latest_frame=None)`
> Append one streaming chunk, update noise estimates and return chunk diagnostics after warmup.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:612`
#### `sort_offline(self, recording, inter_path=None, recording_window_ms=None, model_outputs=None, reset=True, verbose=False)`
> Run chunks through the diagnostic sorter and return a chunk-keyed intermediate-data dictionary.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:668`
#### `sort_chunk(self, chunk, torch_window=None, spike_times_frame_offset=0, ignore_spikes_before=0)`
> Score candidate peaks against templates, resolve overlaps and return intermediate matching data.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:777`
#### `calc_pre_medians(self, pre_median_frames: torch.Tensor)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1020`
#### `set_model(self, model, num_elecs=None, input_size=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1029`
#### `to_tensor(self, data)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1043`
#### `save(self, pickle_path)`
> Reset transient state and pickle the sorter while temporarily excluding its model.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1046`
#### `save_seq_data(self, save_path)`
> As numpy array, save each sequence's (spike_train, root_elecrode) This is mainly for testing
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1056`
#### `get_seq_root_elecs(self)`
> self.seq_root_elecs is not in the correct order because this is only used for sorting
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1066`
#### `get_units(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1078`
#### `to_spikedata(self, all_seq_detections=None)`
> Params all_seq_detections If None, use self.seq_spike_trains (sequences' spikes after second merging but before final spike reassignment) to create SpikeData object If not None, should be the return value of method self.sort_offline
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1084`
#### `to(self, device='cpu')`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1097`
#### `load_from_file(pickle_path, model=None)`
> Unpickle a saved sorter and optionally restore a model.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1116`
### CocCluster()
> unclear — see source
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1515`
**Kind:** class. **Instantiated by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:1526 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:1704 (named-call hint)
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
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1524`
#### `spike_train(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1529`
### Patience()
> unclear — see source
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1974`
**Kind:** class. **Instantiated by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:1722 (named-call hint)
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
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1984`
#### `end(self, comp_elec) -> bool`
> Increment counter and check if to end
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1987`
#### `verbose(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2002`
### Merge()
> unclear — see source
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2473`
**Kind:** class. **Instantiated by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:2818 (named-call hint)
**Constructor:** `__init__(self, cluster_i, cluster_j)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `cluster_i` | inferred at runtime | `cluster_i` |
| `cluster_j` | inferred at runtime | `cluster_j` |
**Methods:**
#### `merge(self, params)`
> Combine cluster event statistics while suppressing closely spaced duplicate detections.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2479`

## Functions
### `detect_sequences(recording, inter_path, detection_model=None, recording_window_ms=None, stringent_thresh=0.275, loose_thresh=0.1, inference_scaling_numerator=12.6, ms_before=0.5, ms_after=0.5, pre_median_ms=50, inner_radius=50, outer_radius=100, min_elecs_for_array_noise_n=100, min_elecs_for_array_noise_f=0.1, min_elecs_for_seq_noise_n=50, min_elecs_for_seq_noise_f=0.05, min_activity_root_cocs=2, min_activity_hz=0.05, max_n_components_latency=4, min_coc_n=10, min_coc_p=10, min_extend_comp_p=50, elec_patience=6, split_coc_clusters_amps=True, min_amp_dist_p=0.1, max_n_components_amp=4, min_loose_elec_prob=0.03, min_inner_loose_detections=3, min_loose_detections_n=4, min_loose_detections_r_spikes=1 / 3, min_loose_detections_r_sequences=1 / 3, max_latency_diff_spikes=3.5, max_latency_diff_sequences=3.5, clip_latency_diff_factor=2, max_amp_median_diff_spikes=0.65, max_amp_median_diff_sequences=0.65, clip_amp_median_diff_factor=2, max_root_amp_median_std_spikes=2.5, max_root_amp_median_std_sequences=np.inf, repeated_detection_overlap_time=0.2, min_seq_spikes_n=10, min_seq_spikes_hz=0.05, relocate_root_min_amp=0.8, relocate_root_max_latency=-2, return_spikes=False, delete_inter=False, device='cuda', num_processes=None, ignore_warnings=True, verbose=True, debug=False)`
> Orchestrate cached trace inference, candidate formation/reassignment/merging and sorter construction.
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:3068 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:37`
### `rt_sort_maxwell_env_process(process_ready, obs_ready, stim_ready, env_done, obs_shm_name, stim_shm_name, start_seq_ind, current_time_name, rt_sort_path, config_path, save_path=None, max_stim_hz=60)`
> Process for sorting with RT-Sort and deciding whether to stim in parallel
> **Called by:** unclear — see source. **Side effects:** write call (static hint).
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1134`
### `save_traces(recording, inter_path, start_ms=0, end_ms=None, num_processes=None, dtype='float16', verbose=True)`
> Dispatch extraction to Maxwell HDF5 or SpikeInterface workers and persist scaled traces.
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:257 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:721 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1247`
### `save_traces_si(recording: BaseRecording, scaled_traces_path, start_ms=0, end_ms=None, num_processes=16, dtype='float16', verbose=True)`
> unclear — see source
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:1273 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1280`
### `_save_traces_si(task)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1317`
### `save_traces_mea(rec_path, save_path, start_ms=0, end_ms=None, samp_freq=20, default_gain=1, chunk_size=100000, num_processes=2, dtype='float16', verbose=True)`
> Can't save traces with spikeinterface get_traces() because it is really slow on MaxWell MEA recordings
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:1268 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1328`
### `_get_traces_mea_old(rec_path)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1391`
### `_get_traces_mea_new(rec_path)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1396`
### `_save_traces_mea(task)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1401`
### `run_detection_model(recording, model, scaled_traces_path, model_traces_path=None, model_outputs_path=None, inference_scaling_numerator=12.6, pre_median_frames=1000, model_inter_path=None, device='cuda', verbose=True, compile_model=True)`
> Run sliding neural inference into trace and logit memmaps.
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:267 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1409`
### `form_coc_clusters(root_elec, params)`
> Build and split candidate propagation clusters around a root electrode.
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:1797 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1533`
### `_form_all_clusters(task)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1791`
### `form_all_clusters(params)`
> Form all preliminary propagation sequences (essentially, call form_coc_clusters() on all electrodes)
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:328 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1804`
### `branch_coc_cluster(root_cluster, comp_elecs, coc_dict, allowed_root_times, patience, params)`
> Recursively divide codetections by local latency mixture components.
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:1720 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:1963 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:1827`
### `reassign_spikes_to_clusters(all_clusters, model, params)`
> Partition sequence groups by GPU capacity and attempt offline reassignment using RTSort.
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:344 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2009`
### `_setup_coc_clusters_parallel(task)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2106`
### `setup_coc_clusters_parallel(coc_clusters, params)`
> Run setup_cluster on coc_clusters with parallel processing
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:2098 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:329 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2119`
### `setup_coc_clusters(coc_clusters, params)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2143`
### `setup_cluster(cluster, params, n_cocs=None)`
> Compute electrode probabilities, relative latencies and normalized amplitudes from cached events.
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:2111 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:2149 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2152`
### `setup_elec_stats(cluster, params)`
> Set cluster.loose_elecs, cluster.inner_loose_elecs and the root elec
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:2418 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:2601 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2342`
### `relocate_root(cluster, new_root, params)`
> Relocate cluster's root electrode
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:2439 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:2448 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2408`
### `relocate_root_latency(cluster, params)`
> Change root electrode to most negative-latency electrode with: Mean latency of -2 frames or less Median detection above stringent threhsold Mean amplitude that is 80% or more of current root If no electrode meets requirements, keep current root electrode
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:2882 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2421`
### `relocate_root_prob(cluster, params)`
> Change root electrode to electrode with highest median detection score
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:2931 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2442`
### `filter_clusters(coc_clusters, params)`
> Return coc_clusters that enough loose and inner loose electrodes But not too many to be considered a noise sequence
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:2929 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2451`
### `merge_verbose(merge, update_history=True)`
> Verbose for merge_coc_clusters
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:2850 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2607`
### `merge_coc_clusters(coc_clusters, params)`
> Greedily merge spatially overlapping templates with compatible weighted latency and amplitude statistics.
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:2880 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:2926 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2665`
### `_intra_merge(task)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2876`
### `intra_merge(all_clusters, params)`
> Merge candidates sharing a root electrode in worker processes.
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:356 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2885`
### `inter_merge(intra_merged_clusters, params)`
> Merge across roots, filter sequences, relocate roots and establish stable sequence order.
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:365 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2917`
### `pickle_dump(obj, path)`
> unclear — see source
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:1053 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:2331 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:331 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:346 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2944`
### `pickle_load(path)`
> unclear — see source
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:1125 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:326 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:342 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2949`
### `load_recording(recording)`
> Resolve supported Maxwell/NWB paths or return an existing recording extractor.
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:1258 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:223 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2954`
### `calc_dist(x1, y1, x2, y2)`
> unclear — see source
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:1564 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:1953 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:1992 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:2048 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:2208 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:2223 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:2228 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:2367 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:2378 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:2383 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:2713 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:455 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2972`
### `rec_ms_to_output_frame(ms, samp_freq, front_buffer)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2976`
### `sigmoid(x)`
> unclear — see source
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:2194 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2981`
### `sigmoid_inverse(y)`
> unclear — see source
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:1575 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:1576 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:576 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:577 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2991`
### `calc_pre_median(frame, traces, pre_median_frames, elecs=slice(None))`
> Get the preceeding mean and median of traces before :time:
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:2995`
### `order_sequences(sequences)`
> unclear — see source
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:2932 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:3017`
### `get_gpu_free_memory()`
> unclear — see source
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:2032 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:3025`
### `set_warning_level(ignore_warnings=False)`
> unclear — see source
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:1795 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:2110 (named-call hint); braindance/analysis/intermediate_rt_sorting/rt_sort.py:2879 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:3057`
### `main()`
> unclear — see source
> **Called by:** braindance/analysis/intermediate_rt_sorting/rt_sort.py:3081 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/analysis/intermediate_rt_sorting/rt_sort.py:3065`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| detect_sequences accepts recording/inter_path, model, recording window, detection thresholds, spatial radii, latency/amplitude tolerances, minimum detections/spikes, device, process count, debug cache, return_spikes and delete_inter flags. |
| RTSort requires buffer_size=100; defaults to CUDA and torch.float16, estimates normalization from a 50 ms window, and supports precomputed model outputs. |
| neuropixels_params overrides inference scaling and sequence tolerances; main uses hardcoded local Maxwell recording/intermediate paths. |

## Data Shapes
- Recordings load through SpikeInterface BaseRecording, Maxwell .h5, or .nwb; saved scaled_traces.npy/model_traces.npy/model_outputs.npy are electrode-by-frame memmaps.
- running_sort consumes frame-by-electrode chunks, while sort_offline consumes electrode-by-frame arrays or recording paths.
- sort_chunk returns a dict with stringent_crosses, stringent_crosses_not_noise, seq_spike_data and detections; detections contain 1-based sequence IDs and frame offsets relative to the input-window center.
- sort_offline returns a dict keyed new_chunk_frame_<zero-padded start frame>; seq_spike_data contains 0-based seq_id, spike_time, overlap_score, latencies and amps.
- CocCluster stores spike times in milliseconds and electrode-by-event probability, latency and normalized amplitude arrays; shared-memory adapter expects raw float32 (100,1027) buffers.

## Notes
- Diagnostic dictionary returns bypass ordinary detection/spike-array return code; reassign_spikes_to_clusters and other retained consumers expect the older return format.
- rt_sort_maxwell_env_process passes unsupported chan_ind to running_sort and expects tuple detections, so this adapter does not match the diagnostic sorter API.
- GPU inference/compilation, NVML, multiprocessing, torch backend state, pickle caches and recording dependencies are involved; delete_inter recursively removes the configured intermediate directory.
- get_units and merge_verbose explicitly raise NotImplementedError; to_spikedata refers to SpikeData whose import is commented out.
- A numeric recording_window_ms denotes a duration in minutes in the implementation; a tuple supplies millisecond endpoints.
