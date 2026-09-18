# phases_analysis_2.py

**Path:** `braindance/core/phases_analysis_2.py`
**Module:** `braindance.core.phases_analysis_2`
**Feature Area:** `Neural Analysis`
**Entry point:** no — library or imported component

## Overview
Provides experiment-oriented activity, footprint, causal connectivity, and RT-Sort analysis phases returning result dictionaries. These phases update experiment electrode selections, connectivity metrics, and sorting metadata for later acquisition or feedback.

## Connections
- **Uses:** `analysis` from `braindance` — imports (static evidence).
- **Uses:** `causal_connectivity` from `braindance.analysis` — imports (static evidence).
- **Uses:** `data_loader` from `braindance.analysis` — imports (static evidence).
- **Uses:** `plot_helper` from `braindance.analysis` — imports (static evidence).
- **Uses:** `plot_causal_connectivity_experiment` from `braindance.analysis.plot_helper` — imports (static evidence).
- **Uses:** `plot_footprints` from `braindance.analysis.plot_helper` — imports (static evidence).
- **Uses:** `plot_reactivity_traces_experiment` from `braindance.analysis.plot_helper` — imports (static evidence).
- **Uses:** `artifact_removal_new2` from `braindance.core` — imports (static evidence).
- **Uses:** `Phase` from `braindance.core.phases` — imports (static evidence).
- **Uses:** `ModelSpikeSorter` from `braindance.core.spikedetector.model2` — imports (static evidence).
- **Uses:** `detect_sequences` from `braindance.core.spikesorter.rt_sort` — imports (static evidence).
- **Shared data:** causal_connectivity.causal_reactions -> get_spikes(std_thresh) -> causal_connectivity_metrics and plots.
- **Shared data:** MaxwellRecordingExtractor and ModelSpikeSorter feed rt_sort.detect_sequences.

## Dependencies
- `braindance` — external or unresolved local import; source import evidence.
- `braindance.analysis` — intra-repo import; source import evidence.
- `braindance.analysis.causal_connectivity` — intra-repo import; source import evidence.
- `braindance.analysis.data_loader` — intra-repo import; source import evidence.
- `braindance.analysis.plot_helper` — intra-repo import; source import evidence.
- `braindance.analysis.plot_helper.plot_causal_connectivity_experiment` — intra-repo import; source import evidence.
- `braindance.analysis.plot_helper.plot_footprints` — intra-repo import; source import evidence.
- `braindance.analysis.plot_helper.plot_reactivity_traces_experiment` — intra-repo import; source import evidence.
- `braindance.core.artifact_removal_new2` — intra-repo import; source import evidence.
- `braindance.core.phases.Phase` — intra-repo import; source import evidence.
- `braindance.core.spikedetector.model2.ModelSpikeSorter` — intra-repo import; source import evidence.
- `braindance.core.spikesorter.rt_sort.detect_sequences` — intra-repo import; source import evidence.
- `matplotlib.animation` — external or unresolved local import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `mpl_toolkits.mplot3d.Axes3D` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `scipy.ndimage.maximum_filter` — external or unresolved local import; source import evidence.
- `scipy.ndimage.minimum_filter` — external or unresolved local import; source import evidence.
- `spikeinterface.extractors.MaxwellRecordingExtractor` — external or unresolved local import; source import evidence.
- `tqdm.tqdm` — external or unresolved local import; source import evidence.

## Classes
### AnalysisPhase(Phase)
> unclear — see source
**Source:** `braindance/core/phases_analysis_2.py:18`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `predicted_time` | inferred at runtime | `0` |
**Methods:**
#### `run(self, filename)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_analysis_2.py:23`
### ActivityPhase(AnalysisPhase)
> Identifies active electrodes in the recording by analyzing spike activity.
**Source:** `braindance/core/phases_analysis_2.py:27`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, name='ActivityPhase', verbose=False, make_plots=True, make_gif=False, save_plots=False, experiment=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `experiment` | inferred at runtime | `experiment` |
| `filename` | inferred at runtime | `None` |
| `make_gif` | inferred at runtime | `make_gif` |
| `make_plots` | inferred at runtime | `make_plots` |
| `name` | inferred at runtime | `name` |
| `predicted_time` | inferred at runtime | `30` |
| `provides` | inferred at runtime | `['basename', 'mapping', 'active_electrodes', 'spikes']` |
| `save_plots` | inferred at runtime | `save_plots` |
| `verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `set_filename(self, filename)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_analysis_2.py:85`
#### `run(self, experiment=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/core/phases_analysis_2.py:88`
#### `info(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_analysis_2.py:377`
### FootprintPhase(AnalysisPhase)
> Analyzes the spatial footprint of neural activity for each active electrode.
**Source:** `braindance/core/phases_analysis_2.py:387`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, wind=100, rms_mult=7, spikes_per_channel=300, num_channel_thresh=60, num_spikes_min=20, remove_bad=True, similarity_thresh=0.7, pk_to_pk_thresh=9, remove_redundant=True, load_whole_recording=False, dao=None, verbose=False, experiment=None, name='FootprintPhase')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `experiment` | inferred at runtime | `experiment` |
| `load_whole_recording` | inferred at runtime | `load_whole_recording` |
| `name` | inferred at runtime | `name` |
| `num_channel_thresh` | inferred at runtime | `num_channel_thresh` |
| `num_spikes_min` | inferred at runtime | `num_spikes_min` |
| `pk_to_pk_thresh` | inferred at runtime | `pk_to_pk_thresh` |
| `provides` | inferred at runtime | `['selected_electrodes']` |
| `remove_bad` | inferred at runtime | `remove_bad` |
| `remove_redundant` | inferred at runtime | `remove_redundant` |
| `rms_mult` | inferred at runtime | `rms_mult` |
| `similarity_thresh` | inferred at runtime | `similarity_thresh` |
| `spikes_per_channel` | inferred at runtime | `spikes_per_channel` |
| `verbose` | inferred at runtime | `verbose` |
| `wind` | inferred at runtime | `wind` |
**Methods:**
#### `run(self, experiment=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/core/phases_analysis_2.py:495`
#### `info(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_analysis_2.py:693`
### CausalAnalysis(AnalysisPhase)
> Analyzes causal connectivity between selected electrodes, seeing the reaction to stimulations.
**Source:** `braindance/core/phases_analysis_2.py:704`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, wind_ms=150, channels_of_interest=None, file_path=None, stim_log_path=None, clean_data_path=None, tag='causal', remove_start_frames=40, art_rem_N=60, save_clean_data=True, verbose=False)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `art_rem_N` | inferred at runtime | `art_rem_N` |
| `channels_of_interest` | inferred at runtime | `channels_of_interest` |
| `clean_data_path` | inferred at runtime | `clean_data_path` |
| `experiment` | inferred at runtime | `experiment` |
| `file_path` | inferred at runtime | `file_path` |
| `remove_start_frames` | inferred at runtime | `remove_start_frames` |
| `save_clean_data` | inferred at runtime | `save_clean_data` |
| `stim_electrodes` | inferred at runtime | `experiment.params['stim_electrodes']` |
| `stim_log_path` | inferred at runtime | `stim_log_path` |
| `tag` | inferred at runtime | `tag` |
| `verbose` | inferred at runtime | `verbose` |
| `wind_ms` | inferred at runtime | `wind_ms` |
**Methods:**
#### `set_from_experiment(self, experiment)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_analysis_2.py:785`
#### `run(self, experiment, file_path=None, stim_electrode_ind=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_analysis_2.py:794`
### RTSortPhase(AnalysisPhase)
> Performs real-time spike sorting on a baseline recording.
**Source:** `braindance/core/phases_analysis_2.py:928`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, name='RTSortPhase', detection_model_path=None, inter_path=None, recording_window_ms=(0, 1 * 60 * 1000), artifact_removal_params=None, art_rem_N=60, force_redo=True, make_plots=True, save_plots=True, delete_inter=True, verbose=True, file_path=None, save_rt_sort=True)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `delete_inter` | inferred at runtime | `delete_inter` |
| `detection_model_path` | inferred at runtime | `braindance.get_rt_sort_path()` |
| `file_path` | inferred at runtime | `file_path` |
| `force_redo` | inferred at runtime | `force_redo` |
| `inter_path` | inferred at runtime | `Path(os.path.dirname(file_path)) / 'rt_sort_tmp'` |
| `make_plots` | inferred at runtime | `make_plots` |
| `name` | inferred at runtime | `name` |
| `predicted_time` | inferred at runtime | `300` |
| `provides` | inferred at runtime | `['rt_sort', 'sequence_spike_trains', 'channels_per_neuron', 'electrodes_per_neuron', 'positions_per_neuron']` |
| `recording_window_ms` | inferred at runtime | `recording_window_ms` |
| `save_plots` | inferred at runtime | `save_plots` |
| `save_rt_sort` | inferred at runtime | `save_rt_sort` |
| `verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `run(self, experiment=None, file_path=None)`
> Run RT-Sort on the baseline recording from experiment or file path
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/core/phases_analysis_2.py:1038`
#### `info(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_analysis_2.py:1232`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Plot/save flags, response window and tag, footprint thresholds |
| RTSortPhase detection_model_path, inter_path, recording_window_ms, delete_inter, save_rt_sort |

## Data Shapes
- Activity returns basename/mapping/active_electrodes/spikes; footprints return selected channels/electrodes and footprint_chans.
- RT-Sort results include rt_sort, sequence_spike_trains, channels/electrodes/positions_per_neuron.
- Causal cleaned traces have repetition/stimulation/reacting-channel/sample axes.

## Notes
- Footprint phase invokes experiment.maxwell_clean_electrodes, so analysis can query hardware routing.
- RTSortPhase default inter_path derives dirname(file_path) during construction; file_path=None needs an explicit inter_path.
