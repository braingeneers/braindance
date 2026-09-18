# BrainDance Data Manager - Utilities Reference

A comprehensive toolkit for loading, analyzing, and visualizing neural recordings from the Braingeneers platform with intelligent S3-backed caching.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Analysis Index & Quick Reference](functions_cheatsheet.md)
4. [Data Loading & Caching](#data-loading--caching)
5. [File Reference](#file-reference)
6. [Advanced Usage](#advanced-usage)

---

## Overview

The BrainDance Data Manager provides a unified interface for working with neural recordings stored on S3. Key features:

- **Automatic S3 data loading** with local caching
- **Intelligent results caching** to avoid expensive recomputation
- **Vectorized analysis libraries** optimized for large datasets
- **Network-level analysis**: burst detection, latency analysis, connectivity
- **Publication-ready plotting** with standardized styling

### Architecture

```
User Code
    ↓
Recording (recording.py) ← Main entry point
    ├─→ Catalog (catalog.py) ← S3 data discovery
    ├─→ S3Loader (s3_loader.py) ← Data fetching
    ├─→ ResultsCache (results_cache.py) ← Derived data caching
    └─→ Analysis Libraries
         ├─→ BurstDetector
         ├─→ BurstLatencyAnalyzer
         ├─→ VectorizedBinning
         └─→ Plotting utilities
```

---

## Quick Start

### Loading a Recording

```python
from braindance.utils.data_manager import load_recording

# Load recording with automatic S3 fetching and local caching
rec = load_recording(
    proj='25-02-25_busybees',
    chip='25123ic',
    experiment='exp1_cont_1'
)

# Access data
spikes = rec.spikes        # SpikeData object
stim_log = rec.stim_log    # Stimulation DataFrame
metadata = rec.metadata    # Experiment metadata
```

### Cached Analysis

```python
# Burst detection - automatically cached
results = rec.detect_bursts(
    bin_size=1.0,
    smoothing_window=50
)
print(f"Detected {results.n_bursts} bursts")

# Subsequent runs load from cache instantly!
# [CACHE] ✓ Loaded bursts_... from local cache
```

---

## Data Loading & Caching

### 1. Catalog (`catalog.py`)

**Purpose**: Discovers and organizes recordings stored in S3.

The `Catalog` class builds a hierarchical view of your data:
```
braingeneers/
└── braindance/
    └── proj/
        ├── chip1/
        │   ├── experiment1/
        │   │   ├── data/
        │   │   └── results/
        │   └── experiment2/
        └── chip2/
```

**Key Features**:
- Scans S3 bucket for available recordings
- Validates data completeness (requires spike data + stim log)
- Provides filtering and search capabilities
- Caches catalog locally for fast lookups

**Usage**:
```python
from braindance.utils.data_manager.utils import Catalog

catalog = Catalog()
catalog.build()  # Scans S3 and builds index

# List all recordings
recordings = catalog.list_recordings()

# Filter by project
busy_bee_recordings = catalog.filter_by_project('25-02-25_busybees')

# Get recording info
info = catalog.get_recording_info(proj, chip, experiment)
```

### 2. Recording (`recording.py`)

**Purpose**: Main interface for working with a single recording.

The `Recording` class provides:
- **Data access**: `spikes`, `stim_log`, `metadata`
- **Cached analysis methods**: `get_binned_fr()`, `detect_bursts()`
- **Results management**: Save/load derived data
- **Automatic caching**: All analysis uses `ResultsCache`

**Key Methods**:
*Refer to the [Functions Cheat Sheet](functions_cheatsheet.md) for a full list of analysis methods including `detect_bursts()` and `calculate_latencies()`.*

### 3. S3 Loader (`s3_loader.py`)

**Purpose**: Low-level S3 data fetching with retry logic and error handling.

Handles:
- S3 authentication with NRP endpoints
- Exponential backoff for transient failures
- Streaming large files
- Download progress tracking

**Usually used internally**, but available for custom S3 operations:
```python
from braindance.utils.data_manager.utils import S3Loader

loader = S3Loader()
loader.download_file(
    s3_path='s3://bucket/path/file.h5',
    local_path='local/file.h5'
)
```

### 4. Results Cache (`results_cache.py`)

**Purpose**: S3-backed caching for derived data (binned FR, burst detection, etc.)

#### How Caching Works

**Cache Strategy**: Local-first with optional S3 sync
```
1. Check local cache (fast) → Return if found
2. Check S3 cache (optional, configurable) → Download and return if found
3. Compute result
4. Save to local cache
5. Upload to S3 (optional, for distributed jobs)
```

**File Naming**: Parameter-keyed for automatic cache invalidation
```
bursts_1.0ms_50_25_2.5_200_0.5_0.2_20_500_0.9.npz
       ↑    ↑  ↑  ↑   ↑   ↑   ↑   ↑  ↑   ↑
       │    │  │  │   │   │   │   │  │   └─ backbone_threshold (0.9)
       │    │  │  │   │   │   │   │  └───── max_burst_width (500ms)
       │    │  │  │   │   │   │   └──────── min_burst_width (20ms)
       │    │  │  │   │   │   └──────────── edge_threshold_factor (0.2)
       │    │  │  │   │   └──────────────── peak_prominence (0.5)
       │    │  │  │   └──────────────────── peak_distance (200ms)
       │    │  │  └──────────────────────── peak_threshold_factor (2.5)
       │    │  └─────────────────────────── baseline_percentile (25)
       │    └────────────────────────────── smoothing_window (50ms)
       └─────────────────────────────────── bin_size (1.0ms)
```

**Directory Structure**:
```
{data_dir}/{proj}/{chip}/{exp}/
├── data/                    # Raw data (managed by S3Loader)
│   ├── spikes.h5
│   └── stim_log.csv
└── results/                 # Cached derived data
    ├── manifest.json        # Index of cached results
    ├── binned_fr_20ms.npz   # Binned firing rates
    ├── bursts_....npz       # Burst detection results
    └── burst_latency_...npz # Latency analysis results
```

**Usage**:
```python
from braindance.utils.data_manager.utils import ResultsCache

cache = ResultsCache(
    local_path=Path('results/'),
    s3_path='s3://braingeneers/braindance/proj/chip/exp/results/',
    auto_upload=False,      # Set True in distributed jobs
    auto_download=True      # Try S3 before computing
)

# Get or compute with automatic caching
result = cache.get_or_compute(
    name='binned_fr',
    params={'bin_ms': 20},
    compute_fn=lambda: expensive_computation()
)

# Force recomputation
result = cache.get_or_compute(
    name='binned_fr',
    params={'bin_ms': 20},
    compute_fn=lambda: expensive_computation(),
    force_recompute=True
)

# Manual cache operations
cache.exists_local('binned_fr', {'bin_ms': 20})   # Check local
cache.exists_s3('binned_fr', {'bin_ms': 20})      # Check S3
cache.upload_to_s3('binned_fr', {'bin_ms': 20})   # Manual upload
cache.sync_all_to_s3()                             # Upload everything
```

**Environment Variables**:
```bash
# Enable auto-upload in container jobs
export BRAINDANCE_AUTO_UPLOAD=1

# Results cache will automatically upload after computing
```

**Performance Benefits**:
- **First run**: Compute + save (~2-5 seconds for burst detection)
- **Subsequent runs**: Load from cache (~0.01-0.1 seconds)
- **50-500x speedup** for repeated analysis

---

## Analysis Libraries

### 1. Burst Detector (`burst_detector.py`)

**Purpose**: Detect synchronized network bursts in population activity.

Bursts are periods of elevated population firing that indicate network synchronization - a hallmark of organoid and neural culture activity.

#### Algorithm Overview

```
1. Bin population activity (default: 1ms bins)
2. Smooth with Gaussian (default: 50ms window)
3. Detect peaks above threshold (adaptive baseline)
4. Find burst edges (start/end times)
5. Calculate burst involvement per neuron
6. Classify backbone neurons (rigid vs non-rigid)
```

#### Automatic Caching

`BurstDetector` integrates with `ResultsCache`:
- Cache key includes all detection parameters
- Different parameters = separate cached results
- Loading from cache populates internal state for method access

For detailed usage and parameter descriptions, see the [Functions Cheat Sheet](functions_cheatsheet.md#2-spontaneous-activity-burst-detection).

#### Returned Results (`BurstResults` object)

Access via dot notation:
```python
results.n_bursts                    # Total number of bursts
results.burst_frequency             # Bursts per second
results.smoothed_activity           # Population rate (binned & smoothed)
results.time_bins                   # Time axis for population rate
results.peak_indices                # Burst peak locations (bin indices)
results.burst_widths                # Width of each burst (ms)
results.mean_burst_width            # Mean ± std burst width
results.std_burst_width
results.burst_amplitudes            # Peak height of each burst
results.bic_matrix                  # Burst Involvement Coefficient per neuron
results.backbone_classification     # {'rigid': [...], 'nonrigid': [...]}
```

#### Use Cases

- **Network maturation**: Track burst frequency over development
- **Drug effects**: Compare burst properties before/after drug application
- **Backbone analysis**: Identify neurons that reliably participate in bursts
- **Stimulus response**: Correlate bursts with stimulation events

---

### 2. Burst Latency Analyzer (`burst_latency_analyzer.py`)

**Purpose**: Analyze stimulus-evoked burst latencies with statistical validation.

Identifies electrodes that reliably evoke bursts and measures response timing.

#### Algorithm Overview

```
1. Detect network bursts (uses BurstDetector, cached)
2. Group stimuli by electrode
3. For each electrode:
   a. Find bursts in analysis window after each stimulus
   b. Calculate baseline burst probability (pre-stimulus)
   c. Calculate evoked burst probability (post-stimulus)
   d. Compute mean latency and classify timing (immediate vs late)
   e. Statistical validation (probability increase + p-value)
4. Return validated electrodes with latency statistics
```

#### Automatic Caching

- Caches entire latency analysis results
- Cache key includes analysis window, baseline window, validation params
- Reuses cached burst detection internally

For detailed usage and examples, see the [Functions Cheat Sheet](functions_cheatsheet.md#3-evoked-activity-latency-psth).

#### Electrode Results Structure

Each electrode returns:
```python
{
    'electrode_id': 23493,
    'n_stimuli': 50,
    'n_evoked_bursts': 9,
    'evoked_burst_probability': 0.18,  # 18% of stimuli evoked bursts
    'burst_latencies': [120.5, 145.2, ...],  # Latencies in ms
    'mean_burst_latency': 141.2,
    'std_burst_latency': 120.9,
    'burst_types': {
        'immediate': 0,   # <25ms latency
        'late': 9         # >25ms latency
    },
    'baseline_stats': {
        'baseline_burst_probability': 0.14,
        'mean_baseline_bursts_per_window': 0.7
    },
    'validation': {
        'is_validated': False,
        'p_value': 0.7850,
        'probability_ratio': 1.29  # 1.29x baseline probability
    },
    'validated': False  # Overall validation status
}
```

#### Validation Criteria

An electrode is validated if **ALL** conditions are met:
1. **Probability increase**: `evoked_prob / baseline_prob >= probability_increase_factor`
2. **Statistical significance**: `p_value <= max_p_value`
3. **Sufficient instances**: `n_evoked_bursts >= min_evoked_bursts`
4. **Reasonable timing**: `mean_latency <= max_mean_latency_ms`

#### Use Cases

- **Electrode screening**: Identify which electrodes reliably evoke responses
- **Latency analysis**: Measure response timing for different conditions
- **Network connectivity**: Map stimulus-response pathways
- **Plasticity studies**: Track latency changes over time

---

### 3. Latency Helpers

#### `latency_helper.py`

**Purpose**: Basic utilities for stimulus-response analysis.

Simple helper functions for:
- Grouping stimulations by electrode
- Parsing stimulation logs
- Basic latency calculations

#### `ultra_optimized_latency_helper.py`

**Purpose**: High-performance vectorized latency analysis.

Optimized implementation using:
- NumPy vectorization
- Efficient time window searching
- Batch processing for large datasets

Used internally by `BurstLatencyAnalyzer` for performance-critical operations.

**Key Methods**:
```python
from braindance.utils.data_manager.utils import UltraOptimizedLatencyHelper

helper = UltraOptimizedLatencyHelper()

# Group stimuli by electrode
electrode_groups = helper.group_stimulations_by_electrode(
    stim_log,
    use_time_mod=True
)

# Find events in time windows (vectorized)
events_in_windows = helper.find_events_in_windows(
    event_times=burst_times,
    window_starts=stim_times,
    window_duration=500.0  # ms
)
```

---

### 4. Vectorized Binning (`vectorized_binning.py`)

**Purpose**: Ultra-fast spike data binning for firing rate analysis.

Provides vectorized implementations for:
- Population firing rate binning
- Per-neuron firing rate binning
- Sliding window analysis
- PSTH (Peri-Stimulus Time Histogram) generation

**Key Functions**:

For detailed usage and examples, see the [Functions Cheat Sheet](functions_cheatsheet.md#3-evoked-activity-latency-psth).

**Performance**:
- **100-1000x faster** than loop-based binning
- Handles millions of spikes efficiently
- Memory-optimized for large recordings

---

### 4. Spatial Analysis

#### Electrode Mapping (`Mapping` class from `braindance.analysis.mapping`)

**Purpose**: Access electrode spatial coordinates and channel-electrode mappings.

The `Mapping` class provides spatial information about the electrode array:
- Physical (x, y) positions of electrodes
- Channel-to-electrode ID mappings
- Utilities for finding nearest neighbors

**Access via Recording:**
```python
mapping = rec.mapping  # Lazy-loaded Mapping object

# Get electrode positions
positions = mapping.get_positions(electrodes=[123, 456])
# Returns: array([[x1, y1], [x2, y2]])

# Convert between channels and electrodes
channels = mapping.get_channels(electrodes=[123, 456])
electrodes = mapping.get_electrodes(channels=[5, 10])

# Find nearest electrodes
nearest = mapping.get_nearest(electrode=123, n=5)
```

#### Spike Locations

**Purpose**: Access neuron spatial coordinates for spatial visualization.

Spike locations are loaded from `spike_info.json` or `spike_info.pkl` files and provide (x, y) coordinates for each neuron.

**Access via Recording:**
```python
spike_locs = rec.spike_locations  # Lazy-loaded list of (x, y) arrays

# Example: Get position of neuron 10
neuron_pos = spike_locs[10]
print(f"Neuron 10 at (x={neuron_pos[0]}, y={neuron_pos[1]}) μm")

# Plot neuron distribution
import matplotlib.pyplot as plt
x_coords = [loc[0] for loc in spike_locs]
y_coords = [loc[1] for loc in spike_locs]
plt.scatter(x_coords, y_coords)
plt.xlabel('X Position (μm)')
plt.ylabel('Y Position (μm)')
plt.show()
```

#### Spatial Visualization

**Purpose**: Create spatial plots of neural responses.

The `rec.pl.spatial_latency()` method creates scatter plots showing neurons on the electrode array, color-coded by response latency and sized by response strength.

**Example:**
```python
# Calculate latencies by electrode
from proj.busy_bee_analysis.utils.latency_helper import LatencyHelper
helper = LatencyHelper()
results_by_electrode = helper.calculate_latencies_by_electrode_batched(
    spike_data, stim_log, ...
)

# Create spatial latency plot
fig = rec.pl.spatial_latency(
    electrode_id=23493,
    latency_results=results_by_electrode[23493],
    time_window=(0, 100),  # Latency range for color mapping
    save_path='./spatial_plots',
    show=True
)
```

**Visualization features:**
- **Color**: Onset latency (blue=early, red=late)
- **Size**: Response strength (larger=stronger)
- **Gold star**: Stimulation electrode position
- **Gray dots**: Non-responsive neurons

**Use Cases:**
- Visualize spatial propagation of neural responses
- Identify response patterns across electrode array
- Compare latency distributions for different stimulation sites
- Validate spatial clustering of responsive neurons

---

## Visualization Tools

### 1. Plot Styler (`plot_styler.py`)

**Purpose**: Standardized publication-ready plot styling.

Provides:
- Consistent color palettes
- Predefined figure sizes for different contexts
- Typography settings (fonts, sizes)
- Layout utilities

See the [Functions Cheat Sheet](functions_cheatsheet.md#5-visualization-plotting) for information on styling and color palettes.

### 2. Plotting Library (`plotting.py`)

**Purpose**: High-level plotting functions for common visualizations.

**Available Plots**:

See the [Functions Cheat Sheet](functions_cheatsheet.md#5-visualization-plotting) for the full plotting API and examples.

**Output Formats**:
- PNG (high DPI for presentations)
- SVG (vector graphics for publications)

---

## File Reference

### Core Infrastructure

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `catalog.py` | S3 recording discovery | `Catalog`, `build()`, `list_recordings()` |
| `recording.py` | Main recording interface | `Recording`, `load_recording()` |
| `s3_loader.py` | S3 data fetching | `S3Loader`, `download_file()` |
| `data_context.py` | Metadata & results storage | `DataContext` |
| `results_cache.py` | Derived data caching | `ResultsCache`, `get_or_compute()` |

### Analysis

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `burst_detector.py` | Network burst detection | `BurstDetector`, `BurstResults` |
| `burst_latency_analyzer.py` | Stimulus-evoked latency | `BurstLatencyAnalyzer` |
| `latency_helper.py` | Basic latency utilities | Helper functions |
| `ultra_optimized_latency_helper.py` | Optimized latency analysis | `UltraOptimizedLatencyHelper` |
| `vectorized_binning.py` | Fast spike binning | `bin_spike_data_vectorized()` |

### Visualization

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `plot_styler.py` | Standardized styling | `Styler` |
| `plotting.py` | High-level plot functions | `plot_raster()`, `plot_population_activity()` |

### Configuration

| File | Purpose |
|------|---------|
| `__init__.py` | Package initialization, exports |

---

## Advanced Usage

### Custom Analysis with Caching

Create your own cached analysis:

```python
def my_custom_analysis(spikes, param1, param2):
    """Your expensive computation."""
    # ... complex analysis ...
    return results

# Use cache
cache = rec.cache
results = cache.get_or_compute(
    name='my_analysis',
    params={'param1': param1, 'param2': param2},
    compute_fn=lambda: my_custom_analysis(spikes, param1, param2)
)
```

### Batch Processing with S3 Sync

For distributed jobs (e.g., cluster computing):

```python
import os

# Enable auto-upload in job script
os.environ['BRAINDANCE_AUTO_UPLOAD'] = '1'

# Load recording
rec = load_recording(proj, chip, experiment)

# Analysis automatically uploads to S3 after computing
results = rec.detect_bursts()  # Computes, saves locally, uploads to S3

# Other jobs can now download this result
```

### Manual Cache Management

```python
cache = rec.cache

# Check what's cached
manifest = cache._manifest.list_results()
for entry in manifest:
    print(f"{entry['name']}: {entry['filename']}")

# Force recomputation
results = cache.get_or_compute(
    name='bursts',
    params={'bin_size': 1.0, ...},
    compute_fn=compute_fn,
    force_recompute=True  # Bypass cache
)

# Upload all results to S3
success, fail = cache.sync_all_to_s3()
print(f"Uploaded {success} files, {fail} failed")
```

---

## Performance Tips

1. **Use caching**: Always pass `cache=rec.cache` to analysis methods
2. **Vectorize operations**: Use `vectorized_binning` instead of loops
3. **Optimize parameters**: Different parameters create different cache files
4. **Batch S3 operations**: Use `sync_all_to_s3()` instead of individual uploads
5. **Monitor cache size**: Large recordings can generate GB of cached results
---`1 