"""
BrainDance Data Manager - Utilities

Core utilities and analysis tools:

Data Management:
    Recording, load_recording    - Single recording with lazy loading
    RecordingCatalog, load_catalog - Multi-recording catalog management
    DataContext                  - Results caching and persistence
    S3Loader                     - S3 file loading with local caching

Latency Analysis:
    calculate_latencies          - Stimulus-evoked response detection
    UltraOptimizedLatencyHelper  - Advanced PSTH-based latency analysis

Burst Detection:
    BurstDetector               - Network burst detection (logISI method)
    BurstLatencyAnalyzer        - Stimulus-evoked burst analysis

Spike Binning:
    bin_spike_data_vectorized   - Fast vectorized binning
    create_sliding_windows      - Overlapping windows for ML
    validate_spike_data         - Format validation

Plotting:
    Styler                      - Publication-ready styling
    plot_raster_with_pop        - Raster + population rate
    plot_sttc_matrix            - Connectivity matrix
    plot_firing_rate_histogram  - Rate distribution
"""

# Core data management
from braindance.utils.data_manager.utils.data_loading import (
    DataContext,
    Recording,
    load_recording,
    RecordingCatalog,
    BatchResults,
    load_catalog,
    S3Loader,
    ResultsCache,
    Waveforms,
    WAVEFORM_PARAMS,
    waveform_params,
    waveform_s3_path,
    ResultsManifest,
    get_auto_upload_enabled
)

# Latency analysis
from braindance.utils.data_manager.utils.analysis import (
    calculate_latencies,
    group_stimulations_by_electrode,
    UltraOptimizedLatencyHelper
)

# Burst detection and analysis
from braindance.utils.data_manager.utils.analysis import (
    BurstDetector,
    BurstLatencyAnalyzer
)

# Vectorized binning utilities
from braindance.utils.data_manager.utils.analysis import (
    bin_spike_data_vectorized,
    bin_spike_data_chunked,
    create_sliding_windows,
    validate_spike_data
)

# Plotting utilities
from braindance.utils.data_manager.utils.plotting import (
    Styler,
    JournalPageSpec,
    JOURNAL_SPECS,
    get_journal_spec,
    PlotAccessor,
    plot_raster_with_pop,
    plot_evoked_raster,
    plot_evoked_psth,
    plot_sttc_matrix,
    plot_firing_rate_histogram,
    spatial_latency,
    spatial_response_map,
    spatial_animation
)

__all__ = [
    # Core data management
    'DataContext',
    'Recording',
    'load_recording',
    'RecordingCatalog',
    'BatchResults',
    'load_catalog',
    'S3Loader',
    'ResultsCache',
    'Waveforms',
    'WAVEFORM_PARAMS',
    'waveform_params',
    'waveform_s3_path',
    'ResultsManifest',
    'get_auto_upload_enabled',
    
    # Latency analysis
    'calculate_latencies',
    'group_stimulations_by_electrode',
    'UltraOptimizedLatencyHelper',
    
    # Burst detection
    'BurstDetector',
    'BurstLatencyAnalyzer',
    
    # Binning utilities
    'bin_spike_data_vectorized',
    'bin_spike_data_chunked',
    'create_sliding_windows',
    'validate_spike_data',
    
    # Plotting
    'Styler',
    'JournalPageSpec',
    'JOURNAL_SPECS',
    'get_journal_spec',
    'PlotAccessor',
    'plot_raster_with_pop',
    'plot_evoked_raster',
    'plot_evoked_psth',
    'plot_sttc_matrix',
    'plot_firing_rate_histogram',
    'spatial_latency',
    'spatial_response_map',
    'spatial_animation'
]
