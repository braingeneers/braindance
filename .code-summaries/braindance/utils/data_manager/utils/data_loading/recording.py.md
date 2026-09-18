# recording.py

**Path:** `braindance/utils/data_manager/utils/data_loading/recording.py`
**Module:** `braindance.utils.data_manager.utils.data_loading.recording`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Defines the central Recording facade over one catalog row, with lazy access to spikes, logs, mappings, locations, raw data, waveforms, plotting, and analysis results. It resolves legacy, standard, and nested experiment/recording local layouts plus S3/ephys-manager sources, then exposes both simple DataContext persistence and parameterized ResultsCache products.

## Connections
- **Used by:** `braindance.utils.data_manager.utils.data_loading.catalog` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.plotting.accessor` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.plotting.evoked_response` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.plotting.population` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.plotting.raster` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.utils.plotting.spatial_latency` — import consumer hint; not a proven runtime call.
- **Uses:** `data_loader` from `braindance.analysis` — imports (static evidence).
- **Uses:** `Mapping` from `braindance.analysis.mapping` — imports (static evidence).
- **Uses:** `get_auto_extract_spike_info` from `braindance.config` — imports (static evidence).
- **Uses:** `get_output_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `get_data_dir` from `braindance` — imports (static evidence).
- **Uses:** `SpikeData` from `braindance.spike_data` — imports (static evidence).
- **Uses:** `as_spike_data` from `braindance.spike_data` — imports (static evidence).
- **Uses:** `load_spike_pickle` from `braindance.spike_data` — imports (static evidence).
- **Uses:** `BurstDetector` from `braindance.utils.data_manager.utils.analysis.burst_detector` — imports (static evidence).
- **Uses:** `calculate_latencies` from `braindance.utils.data_manager.utils.analysis.latency_helper` — imports (static evidence).
- **Uses:** `group_stimulations_by_electrode` from `braindance.utils.data_manager.utils.analysis.latency_helper` — imports (static evidence).
- **Uses:** `bin_spike_data_vectorized` from `braindance.utils.data_manager.utils.analysis.vectorized_binning` — imports (static evidence).
- **Uses:** `compute_binned_fr_isi_vectorized` from `braindance.utils.data_manager.utils.analysis.vectorized_binning` — imports (static evidence).
- **Uses:** `compute_binned_isi_vectorized` from `braindance.utils.data_manager.utils.analysis.vectorized_binning` — imports (static evidence).
- **Uses:** `DataPathManager` from `braindance.utils.data_manager.utils.catalogging.generator` — imports (static evidence).
- **Uses:** `load_catalog` from `braindance.utils.data_manager.utils.data_loading.catalog` — imports (static evidence).
- **Uses:** `DataContext` from `braindance.utils.data_manager.utils.data_loading.data_context` — imports (static evidence).
- **Uses:** `ResultsCache` from `braindance.utils.data_manager.utils.data_loading.results_cache` — imports (static evidence).
- **Uses:** `get_auto_upload_enabled` from `braindance.utils.data_manager.utils.data_loading.results_cache` — imports (static evidence).
- **Uses:** `S3Loader` from `braindance.utils.data_manager.utils.data_loading.s3_loader` — imports (static evidence).
- **Uses:** `WAVEFORM_NAME` from `braindance.utils.data_manager.utils.data_loading.waveforms` — imports (static evidence).
- **Uses:** `Waveforms` from `braindance.utils.data_manager.utils.data_loading.waveforms` — imports (static evidence).
- **Uses:** `waveform_params` from `braindance.utils.data_manager.utils.data_loading.waveforms` — imports (static evidence).
- **Uses:** `PlotAccessor` from `braindance.utils.data_manager.utils.plotting` — imports (static evidence).
- **Shared data:** Uses DataContext for basic results, ResultsCache for named parameterized products, S3Loader for transfers, and SpikeData compatibility loading.
- **Shared data:** Delegates binning, burst detection, latency analysis, waveform wrapping, mapping, raw Maxwell loading, and plotting to focused modules.
- **Shared data:** Classmethod `from_identifiers` may load the catalog to enrich a detached identifier-based Recording.

## Dependencies
- `braindance.analysis.data_loader` — intra-repo import; source import evidence.
- `braindance.analysis.mapping.Mapping` — intra-repo import; source import evidence.
- `braindance.config.get_auto_extract_spike_info` — intra-repo import; source import evidence.
- `braindance.config.get_output_dir` — intra-repo import; source import evidence.
- `braindance.get_data_dir` — intra-repo import; source import evidence.
- `braindance.spike_data.SpikeData` — intra-repo import; source import evidence.
- `braindance.spike_data.as_spike_data` — intra-repo import; source import evidence.
- `braindance.spike_data.load_spike_pickle` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.analysis.burst_detector.BurstDetector` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.analysis.latency_helper.calculate_latencies` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.analysis.latency_helper.group_stimulations_by_electrode` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.analysis.vectorized_binning.bin_spike_data_vectorized` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.analysis.vectorized_binning.compute_binned_fr_isi_vectorized` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.analysis.vectorized_binning.compute_binned_isi_vectorized` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.catalogging.generator.DataPathManager` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.data_loading.catalog.load_catalog` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.data_loading.data_context.DataContext` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.data_loading.results_cache.ResultsCache` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.data_loading.results_cache.get_auto_upload_enabled` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.data_loading.s3_loader.S3Loader` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.data_loading.waveforms.WAVEFORM_NAME` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.data_loading.waveforms.Waveforms` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.data_loading.waveforms.waveform_params` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.PlotAccessor` — intra-repo import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.
- `smart_open` — external or unresolved local import; source import evidence.
- `torch` — external or unresolved local import; source import evidence.
- `utils.data_loading.cache.CacheManager` — external or unresolved local import; source import evidence.

## Classes
### Recording()
> Represent one catalog recording and lazily coordinate its local/S3 data, derived caches, plotting, and analyses.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:102`
**Kind:** class. **Instantiated by:** braindance/utils/data_manager/utils/data_loading/catalog.py:220 (named-call hint)
**Constructor:** `__init__(self, row: pd.Series, base_path: Optional[Path]=None, legacy_flat: Optional[bool]=None, auto_upload: Optional[bool]=None)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `_auto_upload` | inferred at runtime | `auto_upload if auto_upload is not None else get_auto_upload_enabled()` |
| `_base_path` | inferred at runtime | `None` |
| `_data` | inferred at runtime | `DataContext(overwrite_existing=True)` |
| `_detected_format` | Optional[str] | `None` |
| `_legacy_flat` | inferred at runtime | `legacy_flat` |
| `_paths_resolved` | inferred at runtime | `False` |
| `_plot_accessor` | inferred at runtime | `PlotAccessor(self)` |
| `_resolved_paths` | Dict[str, Path] | `{}` |
| `_results` | Optional[DataContext] | `None` |
| `_results_cache` | Optional[ResultsCache] | `None` |
| `_row` | inferred at runtime | `row` |
| `_s3_loader` | Optional['S3Loader'] | `None` |
**Methods:**
#### `__repr__(self) -> str`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:193`
#### `__getattr__(self, name: str) -> Any`
> Attribute access priority: 1. Internal attributes (_*) 2. Data properties (spikes, stim_log, etc.) - lazy loaded 3. Catalog row columns (metadata)
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:198`
#### `__getitem__(self, key: str) -> Any`
> Dict-like access to metadata.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:218`
#### `__contains__(self, key: str) -> bool`
> Check if key exists in metadata or results.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:224`
#### `_get_s3_loader(self)`
> Get or create S3Loader instance.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:230`
#### `_parse_experiment_components(self) -> Tuple[str, str]`
> Parse experiment into exp_base and exp_name components.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:247`
#### `_construct_s3_spike_path(self) -> Optional[str]`
> Construct S3 path for spike data using catalog's base_path.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:281`
#### `_construct_s3_stim_log_path(self) -> Optional[str]`
> Construct S3 path for stim log using catalog's base_path.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:320`
#### `_construct_s3_game_log_path(self) -> Optional[str]`
> Construct S3 path for game log using catalog's base_path.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:347`
#### `_construct_s3_pattern_log_path(self) -> Optional[str]`
> Construct S3 path for pattern log using catalog's base_path.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:374`
#### `_construct_s3_reward_log_path(self) -> Optional[str]`
> Construct S3 path for reward log using catalog's base_path.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:401`
#### `_construct_s3_raw_data_path(self) -> Optional[str]`
> Construct S3 path for raw data (.raw.h5) using catalog fields.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:428`
#### `_construct_s3_experiment_dir(self) -> Optional[str]`
> Construct S3 experiment directory for listing raw data files.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:465`
#### `_download_from_s3(self, s3_path: str, local_path: Path) -> bool`
> Download a file from S3 to local storage.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:492`
#### `_resolve_paths(self)`
> Resolve all data paths from catalog row and base_path.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:512`
#### `_detect_directory_structure(self, base_path: Path, proj: str, chip: str, exp_name: str) -> bool`
> Auto-detect whether data uses legacy flat or standard directory structure.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:667`
#### `_spikes_path(self) -> Optional[Path]`
> Path to spike data file.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:700`
#### `_stim_log_path(self) -> Optional[Path]`
> Path to stimulus log file.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:706`
#### `_mapping_path(self) -> Optional[Path]`
> Path to mapping file.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:712`
#### `_raw_data_path(self) -> Optional[Path]`
> Path to raw data file (.raw.h5).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:718`
#### `_game_log_path(self) -> Optional[Path]`
> Path to game log file.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:724`
#### `_pattern_log_path(self) -> Optional[Path]`
> Path to pattern log file.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:730`
#### `_reward_log_path(self) -> Optional[Path]`
> Path to reward log file.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:736`
#### `_spike_locations_path(self) -> Optional[Path]`
> Path to spike locations file (spike_info.json or spike_info.pkl).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:742`
#### `_results_path(self) -> Path`
> Path to results directory.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:748`
#### `base_output_dir(self) -> Path`
> Base output directory (without project/chip structure).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:754`
#### `catalog_path(self) -> Path`
> Path to the catalog CSV file.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:775`
#### `output_dir(self) -> Path`
> Output directory for this recording's plots and figures.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:813`
#### `_load_data_property(self, name: str) -> Any`
> Load a data property if not already cached.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:839`
#### `_load_pickle_safe(self, file_path: Path) -> Any`
> Load a trusted pickle with legacy SpikeData format conversion.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:853`
#### `_load_spikes_from_autocuration(self) -> Any`
> Load spike data from ephys_manager autocuration zip.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:864`
#### `_load_spikes(self) -> Any`
> Load spike data from .pkl file (local or S3). - Try to load pickle - Handle multiple formats (SpikeData object, dict with 'train', list of arrays) - Convert to SpikeData if needed - Return None on failure
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:960`
#### `_adjust_stim_times(self, stim_log: pd.DataFrame, spike_data: 'SpikeData', adjust_window_ms: float=100.0) -> pd.DataFrame`
> Adjust stimulation times using artifact detection.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:1046`
#### `_load_stim_log(self) -> Optional[pd.DataFrame]`
> Load stimulus log with artifact-based time adjustment.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:1123`
#### `_load_game_log(self) -> Optional[pd.DataFrame]`
> Load game log from CSV (for RL/CartPole experiments).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:1209`
#### `_load_pattern_log(self) -> Optional[pd.DataFrame]`
> Load pattern log from CSV (for RL/CartPole experiments).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:1306`
#### `_load_reward_log(self) -> Optional[pd.DataFrame]`
> Load reward log from CSV (for RL/CartPole experiments).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:1365`
#### `_load_mapping(self) -> Any`
> Load electrode mapping from CSV or pickle and return as Mapping object.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:1432`
#### `_load_spike_locations(self) -> Any`
> Load spike locations (neuron spatial coordinates) from JSON or pickle.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:1504`
#### `_extract_spike_info_from_rt_sort(self, json_path: Path) -> bool`
> Extract spike_info from RT-Sort pickle and save as JSON.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:1690`
#### `_load_raw_data(self, channels: Optional[List[int]]=None, start: int=0, length: int=-1, spikes: bool=False, dtype: np.dtype=np.float32, suffix: Optional[str]=None, verbose: bool=False, sync_if_missing: bool=False) -> Any`
> Load raw data (lazy, returns loader or memmap).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:1825`
#### `sync_raw_data(self, overwrite: bool=False, local_path: Optional[Union[str, Path]]=None) -> Optional[Path]`
> Download this recording's raw file to a chosen local path.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:1906`
#### `sync_experiment_raw_data(self, overwrite: bool=False, legacy_flat: Optional[bool]=None) -> List[Path]`
> List and download all raw HDF5 files in the containing experiment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:1938`
#### `results(self) -> DataContext`
> Provide simple attribute-based derived-data persistence through DataContext.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2004`
#### `cache(self) -> ResultsCache`
> Provide parameterized local/S3 ResultsCache access with source-specific remote prefixes.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2023`
#### `get_binned_fr(self, bin_ms: float=20.0, time_range: Optional[Tuple[float, float]]=None, force_recompute: bool=False) -> Dict[str, np.ndarray]`
> Load or compute cached time-by-neuron spike counts.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2092`
#### `_load_wf(self) -> Optional['Waveforms']`
> Loader behind `rec.wf`. Reads the extracted-waveform npz at the default parameters; see `get_waveforms()` for anything else.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2177`
#### `has_waveforms(self, **kwargs) -> bool`
> Check local and S3 cache tiers for an extracted waveform product.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2193`
#### `get_waveforms(self, ms_before: float=1.0, ms_after: float=2.0, n_footprint: int=64, artifact_guard_ms: float=10.0, download: bool=True) -> Optional['Waveforms']`
> Load a parameter-matched waveform product without computing it on cache miss.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2213`
#### `get_binned_isi(self, bin_ms: float=20.0, time_range: Optional[Tuple[float, float]]=None, force_recompute: bool=False) -> Dict[str, np.ndarray]`
> Load or compute cached normalized time-by-neuron mean ISI.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2274`
#### `get_binned_fr_and_isi(self, bin_ms: float=20.0, time_range: Optional[Tuple[float, float]]=None, force_recompute: bool=False) -> Dict[str, np.ndarray]`
> Compute/cache firing-rate and ISI matrices together to share preprocessing.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2336`
#### `detect_bursts(self, bin_size: float=1.0, smoothing_window: int=50, burst_detection_params: Optional[Dict]=None, burst_edge_params: Optional[Dict]=None, backbone_threshold: float=0.9, compute_all_metrics: bool=True, use_cache: bool=True, check_s3: bool=False, force_recompute: bool=False)`
> Run cached burst detection over this recording's spike data.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2390`
#### `pl(self) -> 'PlotAccessor'`
> Lazily create the plotting accessor bound to this recording.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2465`
#### `metadata(self) -> Dict[str, Any]`
> Get all metadata as a dictionary.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2487`
#### `name(self) -> str`
> Recording name (experiment name or identifier).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2492`
#### `identifier(self) -> str`
> Unique identifier string: proj/chip/experiment.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2497`
#### `save_results(self, keys: Optional[List[str]]=None)`
> Save results to disk.
> **Called by:** unresolved static dispatch. **Side effects:** write call (static hint).
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2506`
#### `save(self)`
> Save all modified data (primarily results).
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2511`
#### `info(self) -> Dict[str, Any]`
> Get summary information about this recording.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2517`
#### `clear_cache(self)`
> Drop lazily loaded primary data so later access reloads it.
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2531`
#### `calculate_latencies(self, min_response_ratio: float=1.5, max_p_value: float=0.0001, baseline_window: Tuple[float, float]=(-100, 0), response_window: Tuple[float, float]=(0, 100), use_time_mod: bool=True, save_results: bool=True, result_key: str='evoked_latencies', verbose: bool=False, use_cache: bool=True, force_recompute: bool=False, **kwargs) -> Dict[str, Dict[str, Any]]`
> Run or reuse stimulus-evoked latency validation and optionally persist the pair dictionary.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2537`
#### `group_stimulations_by_electrode(self, use_time_mod: bool=True) -> Dict[int, np.ndarray]`
> Return this recording's stimulation times grouped by electrode.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2631`
#### `load(cls, path: Union[str, Path], catalog_row: Optional[pd.Series]=None) -> 'Recording'`
> Construct a Recording from a directory and optional catalog row.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2663`
#### `from_identifiers(cls, proj: str, chip: str, experiment: str, base_path: Optional[Path]=None, legacy_flat: Optional[bool]=None, auto_upload: Optional[bool]=None) -> 'Recording'`
> Resolve a catalog-backed or detached Recording from project, chip, and experiment identifiers.
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2686`

## Functions
### `load_recording(proj: str, chip: str, experiment: str, base_path: Optional[Union[str, Path]]=None, legacy_flat: Optional[bool]=None, auto_upload: Optional[bool]=None) -> Recording`
> Convenience entry point for `Recording.from_identifiers`.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/utils/data_loading/recording.py:2754`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `Recording(row, base_path=None, legacy_flat=None, auto_upload=None)` controls path override, layout detection, and derived-result upload behavior; upload defaults through `BRAINDANCE_AUTO_UPLOAD` handling in ResultsCache utilities. |
| Raw/spike path resolution consults catalog columns including `base_path`, `data_path`, `log_path`, `results_path`, `raw_data_path`, `mapping_path`, `full_path`, `source`, UUID fields, and experiment identifiers; relative explicit paths are resolved beneath the base path. |
| Waveform defaults are 1 ms before, 2 ms after, 64 nearest channels, and a 10 ms artifact guard. |

## Data Shapes
- SpikeData holds one millisecond spike train per neuron; ephys-manager autocuration `qm.npz` train values are converted from samples using `fs`.
- Stim/game/pattern/reward logs are pandas DataFrames; stim `time`/`time_mod` are seconds, while grouping and latency APIs return/use milliseconds.
- Binned FR/ISI arrays are time bins x neurons with a one-dimensional millisecond center axis.
- Waveform products expose units x footprint-channels x samples; mapping and location products connect units/channels to electrode coordinates.

## Notes
- Lazy primary data is cleared by `clear_cache`; results contexts and ResultsCache instances persist separately.
- Stim artifact correction may rewrite the local CSV and optionally upload it when a real `time_mod` correction is produced.
- The ephys-manager import path probes repository and `/app/src` layouts at module import time.
- Automatic spike-location extraction can deserialize an RT-Sort object from S3 and upload a generated JSON file.
