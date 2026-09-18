# __init__.py

**Path:** `braindance/utils/data_manager/__init__.py`
**Module:** `braindance.utils.data_manager`
**Feature Area:** `Data Catalog`
**Entry point:** no — library or imported component

## Overview
Exports the public data-manager loading, analysis, and plotting interface. Its load_catalog wrapper supplies configured catalog and local data paths, and import-time fallback can register a bundled catalog.

## Connections
- **Used by:** `braindance.examples.streaming_workshop.catalog_workspace` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.validate_tutorial_data` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.advanced_tutorials.01_working_with_catalog` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.advanced_tutorials.03_population_vectors` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.advanced_tutorials.04_s3_loading_tutorial` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.tutorials.01_quick_start` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.utils.data_manager.tutorials.02_burst_detection` — import consumer hint; not a proven runtime call.
- **Uses:** `get_catalog_path` from `braindance.config` — imports (static evidence).
- **Uses:** `get_data_dir` from `braindance.config` — imports (static evidence).
- **Uses:** `set_catalog_path` from `braindance.config` — imports (static evidence).
- **Uses:** `BatchResults` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `BurstDetector` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `BurstLatencyAnalyzer` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `DataContext` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `JOURNAL_SPECS` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `JournalPageSpec` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `PlotAccessor` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `Recording` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `RecordingCatalog` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `S3Loader` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `Styler` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `UltraOptimizedLatencyHelper` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `WAVEFORM_PARAMS` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `Waveforms` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `bin_spike_data_chunked` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `bin_spike_data_vectorized` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `calculate_latencies` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `create_sliding_windows` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `get_journal_spec` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `group_stimulations_by_electrode` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `load_recording` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `plot_evoked_psth` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `plot_evoked_raster` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `plot_firing_rate_histogram` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `plot_raster_with_pop` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `plot_sttc_matrix` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `spatial_animation` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `spatial_latency` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `spatial_response_map` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `validate_spike_data` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `waveform_params` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Uses:** `waveform_s3_path` from `braindance.utils.data_manager.utils` — imports (static evidence).
- **Shared data:** Reexports utilities; load_catalog calls RecordingCatalog.from_csv.

## Dependencies
- `braindance.config.get_catalog_path` — intra-repo import; source import evidence.
- `braindance.config.get_data_dir` — intra-repo import; source import evidence.
- `braindance.config.set_catalog_path` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.BatchResults` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.BurstDetector` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.BurstLatencyAnalyzer` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.DataContext` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.JOURNAL_SPECS` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.JournalPageSpec` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.PlotAccessor` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.Recording` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.RecordingCatalog` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.S3Loader` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.Styler` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.UltraOptimizedLatencyHelper` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.WAVEFORM_PARAMS` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.Waveforms` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.bin_spike_data_chunked` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.bin_spike_data_vectorized` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.calculate_latencies` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.create_sliding_windows` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.get_journal_spec` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.group_stimulations_by_electrode` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.load_recording` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plot_evoked_psth` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plot_evoked_raster` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plot_firing_rate_histogram` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plot_raster_with_pop` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plot_sttc_matrix` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.spatial_animation` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.spatial_latency` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.spatial_response_map` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.validate_spike_data` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.waveform_params` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.waveform_s3_path` — intra-repo import; source import evidence.

## Classes
None

## Functions
### `load_catalog(path=None, base_path=None)`
> Load a recording catalog from CSV. If no path is provided, uses the catalog path from DataContext configuration. Args: path: Optional path to catalog CSV file base_path: Optional base path for resolving data file paths. If None, uses the configured data_dir from config. Returns: RecordingCatalog Example: >>> catalog = load_catalog() # Uses configured catalog and data_dir >>> catalog = load_catalog('my_catalog.csv') # Custom catalog.
> **Called by:** braindance/examples/streaming_workshop/catalog_workspace.py:45 (named-call hint); braindance/examples/validate_tutorial_data.py:23 (named-call hint); braindance/utils/data_manager/advanced_tutorials/01_working_with_catalog.py:28 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/__init__.py:96`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| load_catalog(path=None, base_path=None) defaults to configured get_catalog_path() and get_data_dir(); import-time bundled all_catalog.csv fallback can call set_catalog_path(). |

## Data Shapes
- load_catalog returns a RecordingCatalog built from catalog CSV metadata and configured data paths.

## Notes
- Import can persist a default catalog path if bundled all_catalog.csv exists and configured path is missing.
