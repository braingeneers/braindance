# BrainDance Data Manager

A unified interface for loading, analyzing, and visualizing neural recording data. Provides lazy-loaded data access, analysis tools for latency and burst detection, and publication-ready plotting.

## Installation & Setup

```bash
pip install braindance
```
TODO: Still need to install spikedata and boto3 to get this to work

**One-time configuration** (saves to `~/.braindance/config.json`):

```bash
# Setup all paths at once
python -m braindance.utils.data_manager setup \
  --data-dir /path/to/your/data \
  --catalog /path/to/your/catalog.csv \
  --output-dir /path/to/your/plots

# Example
python -m braindance.utils.data_manager setup \
  --data-dir /media/mxwbio/zebra/braindance_data \
  --catalog /home/mxwbio/work/BrainDance-dev/braindance/utils/data_manager/internal_usage/catalog/all_catalog.csv \
  --output-dir /media/mxwbio/zebra/plots

# Or setup individually (OMIT --catalog if you don't have a catalog)
python -m braindance.utils.data_manager setup --data-dir /path/to/your/data
python -m braindance.utils.data_manager setup --catalog /path/to/your/catalog.csv
python -m braindance.utils.data_manager setup --output-dir /path/to/your/plots


# Or use environment variables:
# export BRAINDANCE_DATA_DIR=/path/to/your/data
# export BRAINDANCE_CATALOG_PATH=/path/to/your/catalog.csv
# export BRAINDANCE_OUTPUT_DIR=/path/to/your/plots
```

## Understanding the Catalog System

BrainDance data_manager supports two loading modes:

### 1. Catalog-Based Loading (Batch Analysis)

**What is a catalog?**
A catalog is a CSV file that indexes multiple recordings with their metadata (experiment conditions, stimulation frequencies, drug treatments, etc.). It enables filtering, batch processing, and systematic analysis across many experiments.

**Functions that require a catalog:**
- `load_catalog()` - Load collection of recordings
- `RecordingCatalog.from_csv()` - Create catalog from CSV
- `add_metadata()` - Add organoid metadata to catalog
- `add_units()` - Add spike sorting unit counts to catalog

**Setup:**
```bash
# One-time configuration
python -m braindance.utils.data_manager setup --catalog /path/to/catalog.csv

# Or generate a new catalog from S3
python -m braindance.utils.data_manager.catalogging generate
```

**Usage:**
```python
from braindance.utils.data_manager import load_catalog

# Load catalog and filter by conditions
catalog = load_catalog()
stim_recs = catalog.filter(freq__gt=0, baseline=False)
# ↑ Automatically sorted by recording ID for correct temporal order

# Batch process in chronological order
for rec in stim_recs:
    spikes = rec.spikes
    # ... analyze
    rec.clear_cache()  # Free memory

# Disable auto-sorting if you need catalog order preserved
unsorted = catalog.filter(freq__gt=0, sort_by_recording=False)
```

**What if you don't have a catalog?**
If you call `load_catalog()` or related functions without setting up a catalog path, you'll get a helpful error:
```
ValueError: No catalog path provided and none configured.
Either pass a path or run 'python -m braindance.utils.data_manager setup'
```

### 2. Direct Loading (Single Recording Analysis)

**No catalog needed!** You can load individual recordings directly using identifiers.

**Functions that work WITHOUT a catalog:**
- `load_recording(proj, chip, experiment, base_path)` - Load single recording
- `Recording` class and all its methods
- All data access properties (`.spikes`, `.stim_log`, `.mapping`, etc.)
- All analysis functions (`calculate_latencies()`, `BurstDetector`, etc.)
- All plotting functions
- `DataContext` and results management

**Usage:**
```python
from braindance.utils.data_manager import load_recording, calculate_latencies

# Load a single recording directly - NO CATALOG REQUIRED
rec = load_recording(
    proj='my_project',
    chip='chip_123',
    experiment='experiment_01',
    base_path='/path/to/data'
)

# Access data and run analyses (all works without catalog)
spikes = rec.spikes
stim_log = rec.stim_log
evoked = calculate_latencies(spikes, stim_log)
rec.pl.raster_with_pop(time_window=(0, 60))
```

**When to use each mode:**
- **Direct Loading**: Analyzing 1-5 specific recordings, exploratory analysis, testing
- **Catalog Loading**: Analyzing dozens/hundreds of recordings, comparing across conditions, systematic batch processing

## Quick Example

```python
from braindance.utils.data_manager import (
    load_recording, calculate_latencies, BurstDetector, Styler
)

# Load a recording
rec = load_recording('my_project', 'chip_123', 'experiment_01', base_path='/path/to/data')

# Access data (lazy-loaded)
spikes = rec.spikes           # SpikeData object
stim_log = rec.stim_log       # Stimulation timing DataFrame
rates = spikes.rates(unit="Hz")  # Firing rates

# Analyze: stimulus-evoked latencies
evoked = calculate_latencies(spikes, stim_log)

# Analyze: network bursts
detector = BurstDetector(spikes)
bursts = detector.detect_bursts()

# Visualize: raster with population rate
rec.pl.raster_with_pop(time_window=(0, 60), time_unit='seconds')

# Save results (persisted to disk)
rec.results.evoked_pairs = evoked
rec.save_results()
```

## Available Modules

| Category | Functions/Classes | Description |
|----------|-------------------|-------------|
| **Core** | `load_recording`, `load_catalog`, `Recording`, `RecordingCatalog` | Load and manage recordings |
| **Latency** | `calculate_latencies`, `UltraOptimizedLatencyHelper` | Stimulus-evoked response detection |
| **Bursts** | `BurstDetector`, `BurstLatencyAnalyzer` | Network burst detection and analysis |
| **Binning** | `bin_spike_data_vectorized`, `create_sliding_windows` | Spike binning for ML/dimensionality reduction |
| **Plotting** | `Styler`, `plot_raster_with_pop`, `plot_sttc_matrix` | Publication-ready visualizations |

## Tutorials

### Basic Tutorials (Single Recording Analysis)
Located in `tutorials/`:

1. **[01_quick_start.py](tutorials/01_quick_start.py)** — Loading data, firing rates, latency analysis, raster plots
2. **[02_burst_detection.py](tutorials/02_burst_detection.py)** — Network burst detection, backbone neurons, burst latency analysis
3. **[03_population_vectors.py](tutorials/03_population_vectors.py)** — Spike binning, PCA, UMAP, sliding windows for ML

### Advanced Tutorials (Multi-Recording Workflows)
Located in `advanced_tutorials/`:

1. **[01_working_with_catalog.py](advanced_tutorials/01_working_with_catalog.py)** — Filtering and finding recordings by experimental conditions
2. **[02_batch_processing.py](advanced_tutorials/02_batch_processing.py)** — Efficient batch processing, memory management, caching

## Common Workflows

```python
# Load catalog and filter
from braindance.utils.data_manager import load_catalog
catalog = load_catalog()
stim_recs = catalog.filter(freq__gt=0)  # Stimulated recordings

# Latency analysis
from braindance.utils.data_manager import calculate_latencies
evoked = calculate_latencies(rec.spikes, rec.stim_log, max_p_value=0.001)

# Burst detection
from braindance.utils.data_manager import BurstDetector
detector = BurstDetector(rec.spikes, min_channels=5)
bursts = detector.detect_bursts()

# Binning for ML
from braindance.utils.data_manager import bin_spike_data_vectorized
binned = bin_spike_data_vectorized(rec.spikes, bin_size_ms=50)

# Plotting with custom style
from braindance.utils.data_manager import Styler, plot_raster_with_pop
styler = Styler()
fig, axes = plot_raster_with_pop(rec.spikes, styler=styler, time_window=(0, 120))
```

## Recording Data Access

Each `Recording` object provides:

| Property | Type | Description |
|----------|------|-------------|
| `rec.spikes` | SpikeData | Spike trains for all neurons |
| `rec.stim_log` | DataFrame | Stimulation times and parameters |
| `rec.mapping` | DataFrame | Electrode mapping |
| `rec.results` | DataContext | Cached analysis results |
| `rec.pl` | PlotAccessor | Plotting interface (`rec.pl.raster_with_pop()`) |
| `rec.base_output_dir` | Path | Base output directory (no automatic subdirectories) |
| `rec.output_dir` | Path | Recording-specific output directory (auto-creates `{output_dir}/{proj}/{chip}/`) |
| `rec.identifier` | str | Unique ID: `proj/chip/experiment` |

## Automatic Caching (Makes Second Runs Instant!)

Results are automatically saved and reloaded from disk:

```python
# First run - compute and save
evoked = rec.calculate_latencies()
rec.results.evoked_latencies = evoked
rec.save_results()  # Saves to disk

# Second run - instant load from cache
try:
    evoked = rec.results.evoked_latencies  # Auto-loads from disk
    print("Loaded from cache!")
except AttributeError:
    evoked = rec.calculate_latencies()     # Compute if not cached
    rec.results.evoked_latencies = evoked
    rec.save_results()
```

**Key points:**
- First access: `rec.results.{key}` loads from disk automatically
- Saves memory: Only loads what you access
- Clear cache: Use `rec.clear_cache()` in batch processing to free memory

## Plotting Interface

```python
# Via Recording accessor
rec.pl.raster_with_pop(time_window=(0, 60))
rec.pl.sttc_matrix(sttc_matrix=my_sttc)
rec.pl.firing_rate_hist()

# Standalone functions
from braindance.utils.data_manager import plot_raster_with_pop, Styler
styler = Styler()
fig, ax = plot_raster_with_pop(spikes, styler=styler, save_path='./figs', filename='raster')
```

## Organized Plot Storage

Two properties provide flexible output directory organization:

```python
from braindance.config import set_output_dir
from braindance.utils.data_manager import load_recording
import matplotlib.pyplot as plt

# Configure output directory (one-time)
set_output_dir('/Volumes/hunter_ssd/busy_bee/plots')

# Load recording
rec = load_recording('25-02-25_busybees', '25123ic', 'exp1/exp1_cont_95')

# Recording-specific output (automatic proj/chip structure)
plt.savefig(rec.output_dir / 'raster.png')
# → /Volumes/hunter_ssd/busy_bee/plots/25-02-25_busybees/25123ic/raster.png

plt.savefig(rec.output_dir / 'pca' / 'trajectory.png')
# → /Volumes/hunter_ssd/busy_bee/plots/25-02-25_busybees/25123ic/pca/trajectory.png

# Base output directory (for cross-project files)
plt.savefig(rec.base_output_dir / 'all_chips_summary.png')
# → /Volumes/hunter_ssd/busy_bee/plots/all_chips_summary.png

# Custom organization from base
analysis_type = 'connectivity'
plt.savefig(rec.base_output_dir / analysis_type / rec.chip / 'sttc.png')
# → /Volumes/hunter_ssd/busy_bee/plots/connectivity/25123ic/sttc.png
```

**Benefits:**
- `rec.output_dir`: Automatic organization by project and chip
- `rec.base_output_dir`: Flexible custom organization
- Both work like standard pathlib Path objects
- Directories created automatically
- No manual path construction needed
