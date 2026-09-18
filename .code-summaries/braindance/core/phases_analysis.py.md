# phases_analysis.py

**Path:** `braindance/core/phases_analysis.py`
**Module:** `braindance.core.phases_analysis`
**Feature Area:** `Neural Analysis`
**Entry point:** no — library or imported component

## Overview
Provides original AnalysisDAO-based electrode activity, waveform footprint, and post-stimulation response phases. Activity selects local maxima, footprint analysis prunes broad or redundant signals, and causal analysis builds latency histograms after cubic cleaning.

## Connections
- **Used by:** `braindance.analysis.data_helper` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.analysis.plot_helper` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.analysis.process_causal` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.core.phases2` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.paper_cartpole.causal` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.paper_cartpole.recording` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.causal_tetanus_phases` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.experiments.neural_config_analysis` — import consumer hint; not a proven runtime call.
- **Uses:** `analysis` from `braindance` — imports (static evidence).
- **Uses:** `data_loader` from `braindance.analysis` — imports (static evidence).
- **Uses:** `plot_helper` from `braindance.analysis` — imports (static evidence).
- **Uses:** `ArtifactRemoval` from `braindance.core.artifact_removal` — imports (static evidence).
- **Uses:** `cubic_fit5` from `braindance.core.artifact_removal` — imports (static evidence).
- **Uses:** `Phase` from `braindance.core.phases` — imports (static evidence).
- **Shared data:** PhaseManager chains returned AnalysisDAO; analysis.data_loader loads traces/windows/stimulation logs; cubic_fit5 and scipy.find_peaks detect responses.

## Dependencies
- `braindance.analysis` — intra-repo import; source import evidence.
- `braindance.analysis.data_loader` — intra-repo import; source import evidence.
- `braindance.analysis.plot_helper` — intra-repo import; source import evidence.
- `braindance.core.artifact_removal.ArtifactRemoval` — intra-repo import; source import evidence.
- `braindance.core.artifact_removal.cubic_fit5` — intra-repo import; source import evidence.
- `braindance.core.phases.Phase` — intra-repo import; source import evidence.
- `matplotlib.animation` — external or unresolved local import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `mpl_toolkits.mplot3d.Axes3D` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.
- `scipy.ndimage.maximum_filter` — external or unresolved local import; source import evidence.
- `scipy.ndimage.minimum_filter` — external or unresolved local import; source import evidence.
- `scipy.signal.find_peaks` — external or unresolved local import; source import evidence.
- `tqdm.tqdm` — external or unresolved local import; source import evidence.

## Classes
### AnalysisPhase(Phase)
> unclear — see source
**Source:** `braindance/core/phases_analysis.py:12`
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
**Source:** `braindance/core/phases_analysis.py:17`
### HeatmapPhase(AnalysisPhase)
> unclear — see source
**Source:** `braindance/core/phases_analysis.py:21`
**Kind:** class. **Instantiated by:** braindance/examples/paper_cartpole/causal.py:57 (named-call hint); braindance/examples/paper_cartpole/recording.py:44 (named-call hint); braindance/experiments/neural_config_analysis.py:26 (named-call hint)
**Constructor:** `__init__(self, name, verbose=False, make_plots=True, make_gif=False, save_plots=False)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `filename` | inferred at runtime | `None` |
| `make_gif` | inferred at runtime | `make_gif` |
| `make_plots` | inferred at runtime | `make_plots` |
| `name` | inferred at runtime | `name` |
| `predicted_time` | inferred at runtime | `120` |
| `save_plots` | inferred at runtime | `save_plots` |
| `verbose` | inferred at runtime | `verbose` |
**Methods:**
#### `run(self, analysis=None, filename=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state; write call (static hint).
**Source:** `braindance/core/phases_analysis.py:34`
#### `info(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_analysis.py:320`
### FootprintPhase(AnalysisPhase)
> unclear — see source
**Source:** `braindance/core/phases_analysis.py:327`
**Kind:** class. **Instantiated by:** braindance/analysis/data_helper.py:16 (named-call hint); braindance/analysis/plot_helper.py:919 (named-call hint); braindance/examples/paper_cartpole/recording.py:45 (named-call hint); braindance/experiments/neural_config_analysis.py:27 (named-call hint)
**Constructor:** `__init__(self, wind=100, rms_mult=7, spikes_per_channel=300, num_channel_thresh=60, num_spikes_min=20, remove_bad=True, similarity_thresh=0.7, pk_to_pk_thresh=9, remove_redundant=True, load_whole_recording=False, dao=None, verbose=False)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `load_whole_recording` | inferred at runtime | `load_whole_recording` |
| `num_channel_thresh` | inferred at runtime | `num_channel_thresh` |
| `num_spikes_min` | inferred at runtime | `num_spikes_min` |
| `pk_to_pk_thresh` | inferred at runtime | `pk_to_pk_thresh` |
| `remove_bad` | inferred at runtime | `remove_bad` |
| `remove_redundant` | inferred at runtime | `remove_redundant` |
| `rms_mult` | inferred at runtime | `rms_mult` |
| `similarity_thresh` | inferred at runtime | `similarity_thresh` |
| `spikes_per_channel` | inferred at runtime | `spikes_per_channel` |
| `verbose` | inferred at runtime | `verbose` |
| `wind` | inferred at runtime | `wind` |
**Methods:**
#### `run(self, analysis)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_analysis.py:374`
### CausalAnalysis(AnalysisPhase)
> unclear — see source
**Source:** `braindance/core/phases_analysis.py:540`
**Kind:** class. **Instantiated by:** braindance/analysis/process_causal.py:79 (named-call hint); braindance/examples/paper_cartpole/causal.py:60 (named-call hint)
**Constructor:** `__init__(self, wind_ms=150, channels_of_interest=None, file_path=None, stim_log_path=None, clean_data_path=None, tag='causal', remove_start_frames=40, art_rem_N=60, verbose=False)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `art_rem_N` | inferred at runtime | `art_rem_N` |
| `channels_of_interest` | inferred at runtime | `channels_of_interest` |
| `clean_data_path` | inferred at runtime | `clean_data_path` |
| `file_path` | inferred at runtime | `file_path` |
| `remove_start_frames` | inferred at runtime | `remove_start_frames` |
| `stim_log_path` | inferred at runtime | `stim_log_path` |
| `tag` | inferred at runtime | `tag` |
| `verbose` | inferred at runtime | `verbose` |
| `wind_ms` | inferred at runtime | `wind_ms` |
**Methods:**
#### `run(self, analysis, stim_electrode_ind=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_analysis.py:561`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Heatmap plotting/GIF/save flags; footprint window, RMS/amplitude thresholds, minimum spikes and overlap threshold |
| Causal wind_ms, remove_start_frames, art_rem_N, channel selection and stimulation tag |

## Data Shapes
- Heatmap mapping gains spike_count, mean_amp, spike_amp_product.
- Causal reactivity shape (stim_electrodes, channels, wind_ms); per-repetition spike times retained in object cells.

## Notes
- Assumes 20 kHz and Maxwell 120x220 electrode grid; footprint RMS samples 50-60 seconds of recording.
- Mutates DAO mapping/selection/results; optional clean data shape (repetitions, stim_electrodes, channels, samples).
