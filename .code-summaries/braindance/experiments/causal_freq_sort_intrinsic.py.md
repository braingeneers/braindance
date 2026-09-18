# causal_freq_sort_intrinsic.py

**Path:** `braindance/experiments/causal_freq_sort_intrinsic.py`
**Module:** `braindance.experiments.causal_freq_sort_intrinsic`
**Feature Area:** `Experiment Phases`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Extracts intrinsic windows from repeated stimulation cycles and sorts each window with RT-Sort. Builds mean unit templates from filtered microvolt traces and saves per-cycle MATLAB and NPZ bundles.

## Connections
- **Used by:** `braindance.analysis.causal_connectivity` — import consumer hint; not a proven runtime call.
- **Uses:** `data_loader` from `braindance.analysis` — imports (static evidence).

## Dependencies
- `braindance.analysis.data_loader` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `scipy.io.savemat` — external or unresolved local import; source import evidence.
- `spikeinterface.extractors.MaxwellRecordingExtractor` — external or unresolved local import; source import evidence.
- `spikeinterface.preprocessing.filter.bandpass_filter` — external or unresolved local import; source import evidence.
- `spikeinterface.preprocessing.scale.scale_to_uV` — external or unresolved local import; source import evidence.
- `tqdm.tqdm` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `_get_unit_dict(task)`
> Function `_get_unit_dict` performs the module operation described by its surrounding code.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/experiments/causal_freq_sort_intrinsic.py:18`
### `sort_intrinsic(recording_path, save_root_path, rt_sort, phases, ms_before=3, ms_after=4, num_waveforms_for_template=300)`
> phases should be a list where each element is a list of [phase_name, num_cycles, intrinsic_start_min, cycle_duration_min] min refers to minutes intrinsic_start_min is when the intrinsic period starts in each cycle NOTE: phases assumes that in each cycle, the beginning period has stimulation and the remaining period is just intrinsic activity phases example: ["pre", 3, 2, 6], ["seq_stim", 14, 3, 6], ["post1", 3, 2, 6], ["intrinsic", 7, 0, 6], ["p…
> **Called by:** braindance/analysis/causal_connectivity.py:851 (named-call hint); braindance/experiments/causal_freq_sort_intrinsic.py:119 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/experiments/causal_freq_sort_intrinsic.py:45`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| ms_before=3, ms_after=4, num_waveforms_for_template=300; Pool(processes=8) |

## Data Shapes
- phases entries: [phase_name,num_cycles,intrinsic_start_min,cycle_duration_min]
- Saved units contain unit_id, spike_train in absolute frames, x_max,y_max,template,electrode; bundle includes locations,fs,cycle_start_frame

## Notes
- Workers depend on module globals, making spawn-based multiprocessing problematic.
- Missing-electrode fallback dereferences None; sample main passes None for required paths/sorter.
