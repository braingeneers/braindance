# phases3_selection.py

**Path:** `braindance/core/phases_v3/phases3_selection.py`
**Module:** `braindance.core.phases_v3.phases3_selection`
**Feature Area:** `Neural Analysis`
**Entry point:** no — library or imported component

## Overview
Selects directed neuron pairs from connectivity ranges, neuron subsets from footprint centers, or units from activity statistics. Returns selections plus candidate scores, simple spatial clusters, or per-neuron firing metrics.

## Connections
- **Used by:** `braindance.examples.closed_loop_experiment_v3` — import consumer hint; not a proven runtime call.
- **Uses:** `AnalysisPhaseV3` from `braindance.core.phases_v3.phase_base_v3` — imports (static evidence).
- **Shared data:** Analysis phases populate shared data; Experiment.configure_from_experiment supplies declared required keys.

## Dependencies
- `braindance.core.phases_v3.phase_base_v3.AnalysisPhaseV3` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.

## Classes
### SelectionPhaseV3(AnalysisPhaseV3)
> Select neuron pairs based on connectivity criteria.
**Source:** `braindance/core/phases_v3/phases3_selection.py:12`
**Kind:** class. **Instantiated by:** braindance/examples/closed_loop_experiment_v3.py:186 (named-call hint); braindance/examples/closed_loop_experiment_v3.py:46 (named-call hint)
**Constructor:** `__init__(self, connectivity_range: Tuple[float, float]=(0.3, 0.6), min_distance: Optional[float]=None, max_distance: Optional[float]=None, selection_method: str='best_in_range', name: str=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `connectivity_matrix` | inferred at runtime | `None` |
| `connectivity_range` | inferred at runtime | `connectivity_range` |
| `electrodes_per_neuron` | inferred at runtime | `None` |
| `max_distance` | inferred at runtime | `max_distance` |
| `min_distance` | inferred at runtime | `min_distance` |
| `neurons` | inferred at runtime | `None` |
| `selection_method` | inferred at runtime | `selection_method` |
**Methods:**
#### `calculate_distance(self, neuron1: int, neuron2: int) -> Optional[float]`
> Calculate spatial distance between neurons if electrode info available.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_selection.py:58`
#### `run(self, experiment) -> Dict[str, Any]`
> Select neuron pairs based on criteria.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_selection.py:72`
### SpatialSelectionPhaseV3(AnalysisPhaseV3)
> Select neurons based on spatial criteria.
**Source:** `braindance/core/phases_v3/phases3_selection.py:141`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, selection_criteria: str='distributed', n_select: Optional[int]=None, min_separation: float=100.0, name: str=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `min_separation` | inferred at runtime | `min_separation` |
| `n_select` | inferred at runtime | `n_select` |
| `selection_criteria` | inferred at runtime | `selection_criteria` |
**Methods:**
#### `run(self, experiment) -> Dict[str, Any]`
> Select neurons based on spatial criteria.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_selection.py:179`
### ActivitySelectionPhaseV3(AnalysisPhaseV3)
> Select neurons based on activity patterns.
**Source:** `braindance/core/phases_v3/phases3_selection.py:256`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** `__init__(self, min_rate: float=0.1, max_rate: float=100.0, regularity_range: Optional[Tuple[float, float]]=None, n_select: Optional[int]=None, name: str=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `max_rate` | inferred at runtime | `max_rate` |
| `min_rate` | inferred at runtime | `min_rate` |
| `n_select` | inferred at runtime | `n_select` |
| `recording_duration` | inferred at runtime | `None` |
| `regularity_range` | inferred at runtime | `regularity_range` |
**Methods:**
#### `configure_from_experiment(self, experiment)`
> Auto-configure from experiment.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/core/phases_v3/phases3_selection.py:296`
#### `calculate_metrics(self, spike_times: np.ndarray, duration: float) -> Dict[str, float]`
> Calculate activity metrics for a neuron.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_selection.py:304`
#### `run(self, experiment) -> Dict[str, Any]`
> Select neurons based on activity.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/core/phases_v3/phases3_selection.py:324`

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Pair connectivity_range=(.3,.6),distance bounds,selection_method best/median/random_in_range; spatial distributed/clustered,n_select,min_separation=100; activity min_rate=.1,max_rate=100,regularity_range. |

## Data Shapes
- Requires connectivity_matrix+neurons, or footprints/electrodes_per_neuron, or spike_trains dict; outputs selected pair/neuron IDs and metadata.

## Notes
- Spatial distances use mean electrode IDs/footprint scalar centers, not physical xy distances.
- Pair phase does not declare electrodes_per_neuron requirement, so optional distance mapping must be assigned separately.
- Activity timing assumes seconds; burst threshold is ISI<.01; distributed selection treats neuron ID 0 as false when choosing candidates.
