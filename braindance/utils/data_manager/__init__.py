"""
BrainDance Data Manager

A unified interface for loading, analyzing, and visualizing neural recording data.

Core Components:
    load_recording      - Load a single recording by proj/chip/experiment
    load_catalog        - Load a catalog of recordings from CSV
    Recording           - Single recording with lazy data loading
    RecordingCatalog    - Collection of recordings with filtering/grouping

Analysis Tools:
    calculate_latencies         - Detect stimulus-evoked responses with statistical validation
    UltraOptimizedLatencyHelper - Advanced latency analysis with PSTH computation
    BurstDetector               - Network burst detection and backbone neuron classification
    BurstLatencyAnalyzer        - Stimulus-evoked burst latency analysis

Binning Utilities:
    bin_spike_data_vectorized   - Fast vectorized spike binning
    create_sliding_windows      - Sliding windows for ML feature extraction
    validate_spike_data         - Validate spike data format

Plotting:
    Styler                - Publication-ready plot styling
    plot_raster_with_pop  - Raster plot with population rate overlay
    plot_evoked_raster    - Evoked response raster plot for single neuron
    plot_evoked_psth      - Evoked response PSTH plot for single neuron
    plot_sttc_matrix      - STTC connectivity matrix visualization
    plot_firing_rate_histogram - Firing rate distribution

Example:
    from braindance.utils.data_manager import load_recording, calculate_latencies

    rec = load_recording('project', 'chip', 'experiment')
    spikes = rec.spikes
    evoked = calculate_latencies(spikes, rec.stim_log)
    rec.pl.raster_with_pop(time_window=(0, 60))
"""

from .utils import (
    DataContext,
    Recording,
    load_recording,
    Waveforms,
    WAVEFORM_PARAMS,
    waveform_params,
    waveform_s3_path,
    RecordingCatalog,
    BatchResults,
    S3Loader,
    calculate_latencies,
    group_stimulations_by_electrode,
    UltraOptimizedLatencyHelper,
    BurstDetector,
    BurstLatencyAnalyzer,
    bin_spike_data_vectorized,
    bin_spike_data_chunked,
    create_sliding_windows,
    validate_spike_data,
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


# Auto-configure catalog path if not already set
try:
    import os
    from pathlib import Path
    from braindance.config import get_catalog_path, set_catalog_path

    # Check if catalog_path is already configured
    current_catalog_path = get_catalog_path()

    # Check if the configured path exists, if not try to set the default
    _default_catalog_path = Path(__file__).parent / 'all_catalog.csv'
    if _default_catalog_path.exists() and not current_catalog_path.exists():
        # Set default catalog path in config
        set_catalog_path(_default_catalog_path)
        print(f"  [INFO] Auto-configured catalog_path: {_default_catalog_path}")
except Exception as e:
    # Silent fail if catalog not available or error during configuration
    pass


def load_catalog(path=None, base_path=None):
    """
    Load a recording catalog from CSV.
    
    If no path is provided, uses the catalog path from DataContext configuration.
    
    Args:
        path: Optional path to catalog CSV file
        base_path: Optional base path for resolving data file paths.
                  If None, uses the configured data_dir from config.
    
    Returns:
        RecordingCatalog
    
    Example:
        >>> catalog = load_catalog()  # Uses configured catalog and data_dir
        >>> catalog = load_catalog('my_catalog.csv')  # Custom catalog
    """
    if path is None:
        from braindance.config import get_catalog_path
        path = get_catalog_path()
    
    # If no base_path provided, use configured data_dir
    if base_path is None:
        from braindance.config import get_data_dir
        base_path = get_data_dir()
    
    return RecordingCatalog.from_csv(path, base_path=base_path)



__all__ = [
    # Core data management
    'Recording',
    'Waveforms',
    'WAVEFORM_PARAMS',
    'waveform_params',
    'waveform_s3_path',
    'RecordingCatalog',
    'BatchResults',
    'DataContext',
    'S3Loader',
    'load_catalog',
    'load_recording',
    
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
    'spatial_animation',
]
