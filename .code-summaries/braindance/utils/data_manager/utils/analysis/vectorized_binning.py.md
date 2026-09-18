# vectorized_binning.py

**Path:** `braindance/utils/data_manager/utils/analysis/vectorized_binning.py`
**Module:** `braindance.utils.data_manager.utils.analysis.vectorized_binning`
**Feature Area:** `Neural Analysis`
**Entry point:** no — library or imported component

## Overview
Bins neuron spike trains into count matrices and mean inter-spike-interval features, including a joint implementation that reuses sorted spikes. Also validates spike trains, chunks neurons, and creates next-step prediction windows.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.data_loading.recording` — import consumer hint; not a proven runtime call.
- **Shared data:** Recording.get_binned_fr/get_binned_isi/get_binned_fr_and_isi call these functions.

## Dependencies
- `numpy` — external or unresolved local import; source import evidence.
- `scipy.sparse` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `bin_spike_data_vectorized(spike_trains: List[np.ndarray], bin_size_ms: float=100.0, time_range: Optional[Tuple[float, float]]=None, return_sparse: bool=False, verbose: bool=True) -> Tuple[np.ndarray, np.ndarray]`
> Vectorized binning of multiple spike trains into count matrix.
> **Called by:** braindance/utils/data_manager/utils/analysis/vectorized_binning.py:131 (named-call hint); braindance/utils/data_manager/utils/analysis/vectorized_binning.py:165 (named-call hint); braindance/utils/data_manager/utils/data_loading/recording.py:2162 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/vectorized_binning.py:17`
### `bin_spike_data_chunked(spike_trains: List[np.ndarray], bin_size_ms: float=100.0, time_range: Optional[Tuple[float, float]]=None, chunk_size_neurons: int=1000, return_sparse: bool=False, verbose: bool=True) -> Tuple[np.ndarray, np.ndarray]`
> Memory-efficient chunked binning for very large datasets.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/vectorized_binning.py:105`
### `create_sliding_windows(data: np.ndarray, window_size: int, step_size: int=1, return_sparse: bool=False) -> Tuple[np.ndarray, np.ndarray]`
> Create sliding windows for time series data.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/vectorized_binning.py:188`
### `validate_spike_data(spike_trains: List[np.ndarray], expected_length_ms: Optional[float]=None, verbose: bool=True) -> dict`
> Validate spike train data quality and characteristics.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/vectorized_binning.py:243`
### `compute_binned_isi_vectorized(spike_trains: List[np.ndarray], bin_size_ms: float=20.0, time_range: Optional[Tuple[float, float]]=None, sentinel_value: Optional[float]=None, normalize: bool=True, verbose: bool=True) -> Tuple[np.ndarray, np.ndarray]`
> Compute mean ISI per neuron per time bin using fully vectorized operations.
> **Called by:** braindance/utils/data_manager/utils/data_loading/recording.py:2319 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/vectorized_binning.py:316`
### `compute_binned_fr_isi_vectorized(spike_trains: List[np.ndarray], bin_size_ms: float=20.0, time_range: Optional[Tuple[float, float]]=None, isi_sentinel_value: Optional[float]=None, normalize_isi: bool=True, verbose: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
> Compute BOTH firing rates and mean ISI in a single efficient pass.
> **Called by:** braindance/utils/data_manager/utils/data_loading/recording.py:2373 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/analysis/vectorized_binning.py:439`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| bin_size_ms, time_range, return_sparse; ISI sentinel defaults bin width and normalize=True; window_size/step_size. |

## Data Shapes
- Input list of per-neuron time arrays in ms; counts shape(time bins,neurons), dtype int32; ISI same shape float32; time axis is bin centers in ms.
- Sliding windows shape(windows,window_size,features), targets(windows,features); optional CSR count output.

## Notes
- 'FR' functions return counts, not counts divided by bin duration.
- ISI assigned to bin of second spike and can span bin boundaries; sentinel represents no measured ISI.
- Single and joint binning differ at supplied end_time when end_time does not align with bin edges.
