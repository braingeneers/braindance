# Phase V3 Spike Sorting and Data Management

## Overview

The Phase V3 framework has been enhanced to properly handle spike sorting and non-JSON-serializable data. This ensures that complex data like numpy arrays, spike trains, and custom objects are properly saved and can be reloaded.

## Key Features

### 1. Enhanced RTSortPhaseV3

The `RTSortPhaseV3` now supports:

- **Real RT sort integration**: Automatically runs the actual RT sort algorithm when available
- **Graceful fallback**: Falls back to mock data when RT sort is not available
- **Multiple sorters**: Supports 'rt_sort', 'kilosort2', and custom sorters
- **Proper data saving**: Spike data is saved as pickle files with references in JSON

```python
from braindance.core.phases_v3.phases_analysis_3 import RTSortPhaseV3

# Create RT sort phase
sort_phase = RTSortPhaseV3(
    sorter='rt_sort',  # or 'kilosort2'
    min_spikes=100,
    recording_window_ms=(0, 60000),
    verbose=True
)
```

### 2. Smart Data Saving

The experiment framework now intelligently handles different data types:

- **JSON-serializable data**: Saved directly in the summary JSON
- **Numpy arrays**: Saved as `.npy` files
- **Pandas DataFrames**: Saved as pickle files
- **Dict of numpy arrays**: Saved as `.npz` files (e.g., spike_trains)
- **Complex objects**: Saved as pickle files

### 3. File References

Non-JSON-serializable data is saved separately and referenced in the summary:

```json
{
  "spike_trains": {
    "_type": "file_reference",
    "format": "numpy_dict",
    "absolute_path": "/path/to/data/spike_trains.npz",
    "relative_path": "data/spike_trains.npz",
    "data_type": "dict",
    "length": 10
  }
}
```

## Usage Example

```python
from braindance.core.phases_v3.experiment_v3 import SimpleExperiment
from braindance.core.phases_v3.phases3 import RecordPhaseV3
from braindance.core.phases_v3.phases_analysis_3 import RTSortPhaseV3

# Create experiment
exp = SimpleExperiment("my_spike_sorting_exp")

# Add phases
exp.add_phases(
    RecordPhaseV3(duration=300),
    RTSortPhaseV3(
        sorter='rt_sort',
        min_spikes=50,
        verbose=True
    )
)

# Run experiment
exp.run()

# Data is automatically saved
# Access spike trains
spike_trains = exp.data.spike_trains  # Dict of numpy arrays
neurons = exp.data.neurons  # List of neuron IDs

# Save summary (includes file references)
exp.save_summary()

# Load data in a new session
exp2 = SimpleExperiment("reload", save_dir=exp.save_dir)
exp2.load_data()
spike_trains_reloaded = exp2.data.spike_trains
```

## Data Organization

After running an experiment with spike sorting:

```
experiment_dir/
├── my_experiment_summary.json    # Experiment summary with file references
├── data/                         # All data files
│   ├── data_metadata.json       # Metadata about data types
│   ├── data_index.json          # Index of saved files
│   ├── spike_trains.npz         # Spike times for each neuron
│   ├── templates.pkl            # Spike templates
│   ├── rt_sort_object.pkl       # RT sort object
│   └── ...                      # Other data files
└── spike_data/                  # Additional spike-specific files
    ├── rt_sort.pkl              # RT sort object backup
    └── spike_data.pkl           # Combined spike data
```

## RT Sort Integration

The RTSortPhaseV3 integrates with the existing RT sort implementation:

1. **Automatic model detection**: Uses `get_rt_sort_path()` to find the model
2. **Artifact removal**: Supports custom artifact removal parameters
3. **Intermediate files**: Manages intermediate processing files
4. **Sequence extraction**: Extracts neuron sequences from RT sort results

## Benefits

1. **Data Persistence**: All data is properly saved and can be reloaded
2. **Path Tracking**: Both absolute and relative paths are stored
3. **Type Safety**: Data types are preserved through save/load cycles
4. **Graceful Degradation**: Works even without RT sort installed
5. **Extensibility**: Easy to add new sorters or data types

## Testing

Run the test script to verify functionality:

```bash
python test_phases_v3_spike_sorting.py
```

This will:

- Test spike sorting with mock data
- Verify data saving and loading
- Check file reference generation
- Test different sorter options
