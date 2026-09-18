# data.py

**Path:** `braindance/core/spikedetector/data.py`
**Module:** `braindance.core.spikedetector.data`
**Feature Area:** `Spike Detection`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Builds detector training examples by pasting curated spike templates into spike-free recording noise. Supplies waveform filtering, multi-recording datasets, leave-one-recording-out splits, bandpass filtering, and Kilosort preparation.

## Connections
- **Used by:** `braindance.core.spikedetector.train` — import consumer hint; not a proven runtime call.
- **Uses:** `plot` from `braindance.core.spikedetector` — imports (static evidence).
- **Uses:** `run_kilosort2` from `braindance.core.spikesorter.kilosort2` — imports (static evidence).
- **Uses:** `save_traces` from `braindance.core.spikesorter.rt_sort` — imports (static evidence).
- **Shared data:** setup_dl_folders calls run_kilosort2(save_dl_data=True); training consumes MultiRecordingDataset.

## Dependencies
- `braindance.core.spikedetector.plot` — intra-repo import; source import evidence.
- `braindance.core.spikesorter.kilosort2.run_kilosort2` — intra-repo import; source import evidence.
- `braindance.core.spikesorter.rt_sort.save_traces` — intra-repo import; source import evidence.
- `h5py` — external or unresolved local import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `scipy.signal` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.
- `torch.utils.data.ConcatDataset` — external or unresolved local import; source import evidence.
- `torch.utils.data.DataLoader` — external or unresolved local import; source import evidence.
- `torch.utils.data.Dataset` — external or unresolved local import; source import evidence.

## Classes
### Recording()
> Represents a raw ephys recording
**Source:** `braindance/core/spikedetector/data.py:22`
**Kind:** class. **Instantiated by:** braindance/core/spikedetector/data.py:483 (named-call hint)
**Constructor:** `__init__(self, rec_path, sample_size, start, mmap_mode='r', n_before=60, n_after=60)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `n_channels` | inferred at runtime | `self.traces.shape[0]` |
| `n_samples` | inferred at runtime | `len(self.sample_times)` |
| `sample_size` | inferred at runtime | `sample_size` |
| `sample_times` | inferred at runtime | `self.get_sample_times(start, self.traces.shape[1], spike_times, n_before, n_after)` |
| `traces` | inferred at runtime | `np.load(rec_path / 'scaled_traces.npy', mmap_mode=mmap_mode)` |
**Methods:**
#### `get_sample_times(self, start, total_duration, spike_times, n_before, n_after)`
> Exclude every noise-window start that overlaps a known spike or its configured margins.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:60`
#### `get_sample(self, channel=None)`
> Get a random sample from the recording
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:100`
#### `__getitem__(self, idx)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:119`
#### `__len__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:122`
### Waveform()
> Class to represent a single waveform and its properties
**Source:** `braindance/core/spikedetector/data.py:126`
**Kind:** class. **Instantiated by:** braindance/core/spikedetector/data.py:290 (named-call hint); braindance/core/spikedetector/data.py:302 (named-call hint)
**Constructor:** `__init__(self, waveform, peak_idx, alpha, curated)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `alpha` | inferred at runtime | `alpha` |
| `curated` | inferred at runtime | `curated` |
| `len` | inferred at runtime | `waveform.size` |
| `peak_idx` | inferred at runtime | `peak_idx` |
| `waveform` | inferred at runtime | `waveform` |
**Methods:**
#### `unravel(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:135`
### Unit()
> unclear — see source
**Source:** `braindance/core/spikedetector/data.py:139`
**Kind:** class. **Instantiated by:** braindance/core/spikedetector/data.py:294 (named-call hint); braindance/core/spikedetector/data.py:304 (named-call hint)
**Constructor:** `__init__(self, waveforms)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `wfs` | inferred at runtime | `waveforms` |
**Methods:**
#### `plot_stack(self, axis=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:143`
### WaveformDataset(Dataset)
> Curate unit channel templates by amplitude and normalized standard deviation.
**Source:** `braindance/core/spikedetector/data.py:170`
**Kind:** class. **Instantiated by:** braindance/core/spikedetector/data.py:493 (named-call hint)
**Constructor:** `__init__(self, rec_path, thresh_amp, thresh_std, use_positive_peak=False, x_highest=None, ms_before_after=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `num_units` | inferred at runtime | `len(sorted['units'])` |
| `units` | inferred at runtime | `[]` |
| `waveforms` | inferred at runtime | `[]` |
**Methods:**
#### `__getitem__(self, idx)`
> Get a waveform
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:307`
#### `__len__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:328`
#### `plot_waveforms(self, num_rows=4, num_cols=4, max_plots=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:331`
### MultiRecordingDataset(Dataset)
> Dataset that represents 1 or more recordings A wrapper of Recording and WaveformDataset
**Source:** `braindance/core/spikedetector/data.py:387`
**Kind:** class. **Instantiated by:** braindance/core/spikedetector/data.py:1106 (named-call hint); braindance/core/spikedetector/data.py:774 (named-call hint); braindance/core/spikedetector/data.py:930 (named-call hint); braindance/core/spikedetector/data.py:935 (named-call hint)
**Constructor:** `__init__(self, rec_paths, samples_per_waveform=2, front_buffer=40, end_buffer=40, num_wfs_probs=[0.5, 0.3, 0.12, 0.06, 0.02], isi_wf_min=4, isi_wf_max=None, thresh_amp=18.88275, thresh_std=0.6, x_highest=None, use_positive_peaks=False, ms_before_after=None, sample_size=200, start=0, ms_before=3, ms_after=3, device='cuda', dtype=torch.float32, mmap_mode='r')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_len` | inferred at runtime | `self.samples_per_waveform * sum([len(wf) for wf in self.wf_datasets])` |
| `device` | inferred at runtime | `device` |
| `isi_wf_max` | inferred at runtime | `isi_wf_max` |
| `isi_wf_min` | inferred at runtime | `isi_wf_min` |
| `loc_range` | inferred at runtime | `(front_buffer, sample_size - end_buffer)` |
| `n_recs` | inferred at runtime | `len(self.recs)` |
| `n_wf_datasets` | inferred at runtime | `len(self.wf_datasets)` |
| `num_wfs_probs` | inferred at runtime | `num_wfs_probs` |
| `recs` | inferred at runtime | `[]` |
| `samp_freq` | inferred at runtime | `fs` |
| `sample_size` | inferred at runtime | `sample_size` |
| `samples_per_waveform` | inferred at runtime | `samples_per_waveform` |
| `torch_dtype` | inferred at runtime | `dtype` |
| `wf_datasets` | inferred at runtime | `[]` |
| `wfs` | inferred at runtime | `wfs` |
**Methods:**
#### `__len__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:534`
#### `__getitem__(self, idx)`
> Select noise, insert spaced templates, median-center, and return tensors with location labels.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:538`
#### `cat_full(self)`
> Concatenate all samples together
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:631`
#### `plot_sample(self, trace: torch.Tensor, num_wfs: torch.Tensor, wf_locs: torch.Tensor, wf_alphas: torch.Tensor)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:642`
#### `add_wf_to_trace(self, trace, wf, wf_peak_idx, wf_len, wf_loc)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:690`
#### `get_means_and_stds(self, num_samples)`
> Get mean and std of individual samples
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:697`
#### `get_mean_and_std_across(self, num_samples)`
> Get mean and std across samples
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:721`
#### `get_ranges(self, num_samples)`
> Get range (max - min) of samples
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:742`
#### `load_single(path_folder, samples_per_waveform, front_buffer, end_buffer, num_wfs_probs, isi_wf_min, isi_wf_max, sample_size, thresh_amp, thresh_std, gain_to_uv, x_highest=None, use_positive_peaks=False, ms_before_after=None, device='cuda:0', dtype=torch.float32, mmap_mode='r')`
> Return a MultiRecordingDataset object that represents a single recording
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:758`
### SubMultiRecordingDataset(MultiRecordingDataset)
> Class to represent a subset of a MultiRecordingDataset (only some indices)
**Source:** `braindance/core/spikedetector/data.py:784`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, dataset: MultiRecordingDataset, indices: list)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `dataset` | inferred at runtime | `dataset` |
| `indices` | inferred at runtime | `indices` |
| `wf_datasets` | inferred at runtime | `dataset.wf_datasets` |
**Methods:**
#### `__len__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:792`
#### `__getitem__(self, idx)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:795`
### RecordingDataloader(DataLoader, MultiRecordingDataset)
> Same as PyTorch's DataLoader except that this class inherits members from MultiRecordingDataset (important since functions need :method alpha_to_waveform_dict:
**Source:** `braindance/core/spikedetector/data.py:800`
**Kind:** class. **Instantiated by:** braindance/core/spikedetector/data.py:946 (named-call hint); braindance/core/spikedetector/data.py:947 (named-call hint)
**Constructor:** `__init__(self, dataset, *args, **kwargs)`
**Key attributes:**
None
**Methods:**
#### `__getattr__(self, item)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:805`
#### `__setattr__(self, attr, val)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:810`
### TensorDataloader(MultiRecordingDataset)
> Convert concatenated samples as Tensors to dataloader format
**Source:** `braindance/core/spikedetector/data.py:814`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, inputs, labels, dataset)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `inputs` | inferred at runtime | `inputs` |
| `labels` | inferred at runtime | `labels` |
| `stop` | inferred at runtime | `0` |
| `wf_datasets` | inferred at runtime | `dataset.wf_datasets` |
**Methods:**
#### `__iter__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:826`
#### `__next__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikedetector/data.py:829`
### RecordingCrossVal()
> Generate train/validation datasets or loaders by withholding one recording.
**Source:** `braindance/core/spikedetector/data.py:837`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, samples_per_waveform, front_buffer, end_buffer, num_wfs_probs, isi_wf_min, isi_wf_max, rec_paths, thresh_amp, thresh_std, sample_size, start=0, ms_before=3, ms_after=3, device='cuda', dtype=torch.float16, mmap_mode='r', x_highest=None, use_positive_peaks=False, ms_before_after=None, verbose=True, as_datasets=False, **dataloader_kwargs)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `as_datasets` | inferred at runtime | `as_datasets` |
| `dataloader_kwargs` | inferred at runtime | `dataloader_kwargs` |
| `dataset_kwargs` | inferred at runtime | `{'front_buffer': front_buffer, 'end_buffer': end_buffer, 'num_wfs_probs': num_wfs_probs, 'isi_wf_min': isi_wf_min, 'isi_wf_max': …` |
| `rec_i` | inferred at runtime | `-1` |
| `rec_paths` | inferred at runtime | `[Path(p) for p in rec_paths]` |
| `samples_per_waveform_train` | inferred at runtime | `samples_per_waveform[0]` |
| `samples_per_waveform_val` | inferred at runtime | `samples_per_waveform[1]` |
| `verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `__getitem__(self, idx)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:917`
#### `__iter__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikedetector/data.py:951`
#### `__next__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/spikedetector/data.py:955`
#### `__len__(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:964`
#### `name_to_idx(self, name)`
> Convert name of one of cross-val recordings to index in self.recordings
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:967`
#### `summarize(rec, train, val)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:982`
### BandpassFilter()
> From SpikeInterface
**Source:** `braindance/core/spikedetector/data.py:1002`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, band=(300, 6000), sf=20000, btype='bandpass', filter_order=5, ftype='butter', filter_mode='sos', margin_ms=5.0, coeff=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `coeff` | inferred at runtime | `filter_coeff` |
| `filter_mode` | inferred at runtime | `filter_mode` |
| `margin` | inferred at runtime | `margin` |
**Methods:**
#### `__call__(self, trace)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:1059`

## Functions
### `setup_dl_folders(recording_files, dl_folders, **run_kilsort2_kwargs)`
> Set up recordings and necessary files and folders to train DL model
> **Called by:** braindance/core/spikedetector/data.py:1096 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:1068`
### `is_dl_folder(folder)`
> unclear — see source
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:1089`
### `main()`
> unclear — see source
> **Called by:** braindance/core/spikedetector/data.py:1112 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/core/spikedetector/data.py:1095`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Amplitude/std curation thresholds, sample size, placement buffers, waveform count probabilities, ISI bounds |
| Device/dtype and memory mapping; recording spike-exclusion windows in milliseconds |

## Data Shapes
- scaled_traces.npy shape (channels,frames); sorted.npz includes fs, spike_times, units with template/amplitudes/std_norms/peak_ind.
- Dataset item: trace tensor (1,sample_size), waveform count, padded waveform locations, padded waveform IDs; padding=-1.

## Notes
- Global NumPy/Python/PyTorch random generators generate samples; templates and noise come from prepared folders.
- Some legacy helpers have stale contracts: cat_full expects two returned fields, load_single forwards removed gain_to_uv.
