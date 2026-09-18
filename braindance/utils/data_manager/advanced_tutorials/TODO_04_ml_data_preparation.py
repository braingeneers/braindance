"""
BrainDance Data Manager - Tutorial 04: ML Data Preparation

This tutorial covers preparing neural population data for machine learning models,
particularly sequence models like LSTMs and Transformers.

Key features:
- Load cached population vectors from Tutorial 03
- Create sliding windows for sequence prediction
- Prepare train/validation/test splits
- Export data in ML-ready formats

Prerequisites:
  - Completed Tutorial 03 (population vectors cached)
  - Python packages: numpy, scikit-learn
  - Optional: torch, tensorflow for ML frameworks
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from braindance.utils.data_manager import (
    load_recording,
    create_sliding_windows
)

# =============================================================================
# 1. Load Recording and Cached Population Vectors
# =============================================================================

proj = '25-02-25_busybees'
chip = '25123ic'
experiment = 'exp1_cont_1'

print(f"Loading recording: {proj}/{chip}/{experiment}")
rec = load_recording(proj, chip, experiment)
spikes = rec.spikes

print(f"✓ Loaded: {spikes.N} neurons, {spikes.length/1000:.1f}s duration")

# Load cached binned spike data from Tutorial 03
print("\nLoading cached population vectors from Tutorial 03...")

bin_size_ms = 100.0
cache_params_binned = {
    'bin_size_ms': float(bin_size_ms),
    'time_range_start': 0.0,
    'time_range_end': float(spikes.length)
}

# Check if cached data exists
cached_binned = rec.cache.load_local('binned_spikes', cache_params_binned)

if cached_binned is None:
    print("Error: No cached population vectors found!")
    print("Please run Tutorial 03 first to generate cached data.")
    exit(1)

binned_data = cached_binned['binned_data']
time_axis = cached_binned['time_axis']

print(f"✓ Loaded cached binned data: {binned_data.shape}")

# Apply log transformation (same as Tutorial 03)
binned_log = np.log1p(binned_data)


# =============================================================================
# 2. Create Sliding Windows for Sequence Models
# =============================================================================

print("\n" + "="*70)
print("SLIDING WINDOWS: Creating Sequence Data for ML")
print("="*70)

window_size = 10  # 10 time bins = 1 second at 100ms resolution
step_size = 1     # Step by 1 bin (overlapping windows)

# Define cache params
cache_params_windows = {
    'bin_size_ms': float(bin_size_ms),
    'window_size': int(window_size),
    'step_size': int(step_size)
}

# Define compute function
def compute_sliding_windows():
    print(f"  Computing sliding windows (window={window_size}, step={step_size})...")
    windows, targets = create_sliding_windows(
        binned_log,
        window_size=window_size,
        step_size=step_size
    )
    return {
        'windows': windows,
        'targets': targets
    }

# Get or compute with cache
cached_windows = rec.cache.get_or_compute(
    'sliding_windows',
    params=cache_params_windows,
    compute_fn=compute_sliding_windows
)

windows = cached_windows['windows']
targets = cached_windows['targets']

print(f"\n✓ Created sliding windows")
print(f"  Window shape: {windows.shape} (n_windows × window_size × n_neurons)")
print(f"  Target shape: {targets.shape} (n_windows × n_neurons)")
print(f"  Window duration: {window_size * bin_size_ms / 1000:.1f}s")
print(f"  Step size: {step_size * bin_size_ms}ms")


# =============================================================================
# 3. Train/Validation/Test Split
# =============================================================================
# STUB: Add train/val/test splitting logic here

print("\n" + "="*70)
print("DATA SPLITTING: Train/Validation/Test")
print("="*70)

# TODO: Implement temporal splitting (not random) to avoid data leakage
# For time series, we want:
#   - Train: first 70% of time
#   - Validation: next 15% of time
#   - Test: last 15% of time

print("STUB: Train/val/test splitting to be implemented")


# =============================================================================
# 4. Export for ML Frameworks (PyTorch, TensorFlow)
# =============================================================================
# STUB: Add export logic for different ML frameworks

print("\n" + "="*70)
print("EXPORT: Preparing Data for ML Frameworks")
print("="*70)

# TODO: Add exports for:
#   - PyTorch tensors (.pt files)
#   - TensorFlow datasets
#   - NumPy arrays (.npz)
#   - HDF5 format

print("STUB: ML framework exports to be implemented")


# =============================================================================
# 5. Save Results
# =============================================================================

rec.results.sliding_windows_shape = windows.shape
rec.results.window_size = window_size
rec.results.step_size = step_size

rec.save_results()
print("\n✓ Results metadata saved to cache")


print("\n" + "="*70)
print("Tutorial 04 complete! (Stubs to be expanded)")
print("="*70)
print("\nNext steps:")
print("  • Implement temporal train/val/test splitting")
print("  • Add ML framework exports (PyTorch, TensorFlow)")
print("  • Add data normalization/scaling options")
print("  • Add batch generator utilities")
print("="*70)
