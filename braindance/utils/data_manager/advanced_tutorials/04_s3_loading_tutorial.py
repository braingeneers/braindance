"""
================================================================================
BrainDance Data Manager - Tutorial 04: S3 Data Loading & Population Vectors
================================================================================

This tutorial demonstrates how to load neural recordings from S3 and compute
population vectors for machine learning pipelines.

Prerequisites:
    1. Run the setup command to configure paths:
       python -m braindance.utils.data_manager setup --data-dir /path/to/data --output-dir /path/to/plots

    2. Install optional dependencies for visualization:
       pip install scikit-learn matplotlib

What You'll Learn:
    1. Loading recordings with automatic S3 fallback
    2. Understanding binned firing rate caching
    3. Computing population vectors for multiple recordings
    4. Visualizing neural dynamics with PCA

Data Source:
    The tutorial uses the BusyBee dataset stored in S3:
    s3://braingeneers/braindance/25-02-25_busybees/25123ic/

Configuration:
    The tutorial uses rec.output_dir for organized plot storage.
    Configure your output directory with:
        python -m braindance.utils.data_manager setup --output-dir /path/to/plots

    Or set environment variable:
        export BRAINDANCE_OUTPUT_DIR=/path/to/plots

Optional Environment Variables:
    BRAINDANCE_AUTO_UPLOAD=1 - Enable S3 upload of cached results

Run with:
    cd braindance/utils/data_manager/advanced_tutorials
    python TODO_04_s3_loading_tutorial.py

================================================================================
"""

import time
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from braindance.utils.data_manager import load_recording, RecordingCatalog
from braindance.utils.data_manager.utils.plotting.plot_styler import Styler


# =============================================================================
# Main Tutorial
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("BrainDance Data Manager - Tutorial 04")
    print("S3 Data Loading & Population Vectors")
    print("="*70)


    # =========================================================================
    # Part 0: Configuration
    # =========================================================================

    # Optional: Enable automatic upload of cached results to S3
    # This is useful when computing expensive operations (like binned firing rates)
    # that you want to share across multiple machines or users
    # Uncomment to enable:
    # os.environ['BRAINDANCE_AUTO_UPLOAD'] = '1'

    # Test data - using BusyBee dataset on S3
    PROJ = "25-02-25_busybees"
    CHIP = "25123ic"

    # Experiments with full paths (as they appear in busy_bee_with_meta.csv)
    TEST_EXPERIMENTS = [
        "exp1/exp1_cont_95",   # 2Hz stim
        "exp1/exp1_cont_96",   # Spontaneous
        "exp1/exp1_cont_97",   # Spontaneous
    ]

    # Binning configuration
    BIN_MS = 20  # 20ms bins for population vectors

    # Initialize plot styler for publication-ready figures
    styler = Styler(journal="draft")
    print("\n✓ Initialized plot styler with scientific formatting")

    # =========================================================================
    # Part 1: Understanding S3 Path Construction
    # =========================================================================

    print("\n" + "="*70)
    print("PART 1: S3 Path Construction")
    print("="*70)

    # The data manager automatically constructs S3 paths from experiment identifiers
    # Experiment format: "exp_base/exp_name" (e.g., "exp1/exp1_cont_95")
    #   - exp_base: "exp1" (the folder containing related experiments)
    #   - exp_name: "exp1_cont_95" (the specific recording)

    experiment = TEST_EXPERIMENTS[0]
    print(f"\nExperiment path from catalog: '{experiment}'")
    print(f"  - exp_base: 'exp1' (the folder)")
    print(f"  - exp_name: 'exp1_cont_95' (the recording)")

    # Create a recording object (no data loaded yet - it's lazy)
    rec = load_recording(
        proj=PROJ,
        chip=CHIP,
        experiment=experiment,
    )

    # Show the S3 paths that would be used
    s3_spike_path = rec._construct_s3_spike_path()
    s3_log_path = rec._construct_s3_stim_log_path()

    print(f"\nConstructed S3 paths:")
    print(f"  Spike data: {s3_spike_path}")
    print(f"  Stim log:   {s3_log_path}")

    # Show local path that will be checked first
    rec._resolve_paths()
    print(f"\nLocal path (checked first):")
    print(f"  {rec._spikes_path}")
    print(f"  Exists locally: {rec._spikes_path.exists() if rec._spikes_path else False}")
    print(f"  Will use: {'Local file' if rec._spikes_path and rec._spikes_path.exists() else 'S3 download'}")

    # =========================================================================
    # Part 2: Load Recording with S3 Fallback
    # =========================================================================

    print("\n" + "="*70)
    print("PART 2: Load Recording with S3 Fallback")
    print("="*70)

    print(f"\nLoading: {PROJ}/{CHIP}/{experiment}")

    # Create recording object (lazy - no data loaded yet)
    start = time.time()
    rec = load_recording(
        proj=PROJ,
        chip=CHIP,
        experiment=experiment,
    )
    create_time = time.time() - start
    print(f"✓ Recording object created in {create_time*1000:.1f}ms (lazy - no data loaded yet)")

    # Access spikes - this triggers the actual load (local file or S3 download)
    print("\nAccessing spikes (triggers lazy load)...")
    start = time.time()
    spikes = rec.spikes
    load_time = time.time() - start

    if spikes is None:
        print("❌ Failed to load spike data!")
        print("   Check S3 connectivity and credentials.")
        print("   Run: python -m braindance.utils.data_manager.utils.setup")
        raise RuntimeError("Failed to load spike data")

    print(f"✓ Loaded spike data in {load_time:.2f}s")
    print(f"  Neurons: {spikes.N}")
    print(f"  Duration: {spikes.length/1000:.1f}s")
    print(f"  Total spikes: {sum(len(t) for t in spikes.train)}")

    # Check stim log
    if rec.stim_log is not None:
        print(f"  Stim events: {len(rec.stim_log)}")
        if 'freq' in rec._row:
            print(f"  Stim frequency: {rec._row.get('freq', 'unknown')} Hz")

    # =========================================================================
    # Part 3: Compute Binned Firing Rates with Caching
    # =========================================================================

    print("\n" + "="*70)
    print("PART 3: Binned Firing Rates with Caching")
    print("="*70)

    print(f"\nComputing binned firing rates (bin_ms={BIN_MS})...")
    print("First run: computes and caches locally")
    print("Subsequent runs: loads from cache instantly")

    # First call - computes and caches
    start = time.time()
    binned = rec.get_binned_fr(bin_ms=BIN_MS)
    first_time = time.time() - start

    if binned is None:
        print("❌ Failed to compute binned FR")
        raise RuntimeError("Failed to compute binned firing rates")

    rates = binned['rates']
    time_axis_ms = binned['time_axis_ms']

    print(f"✓ First call: {first_time:.2f}s")
    print(f"  Shape: {rates.shape} (time_bins × neurons)")
    print(f"  Time range: 0 - {time_axis_ms[-1]/1000:.1f}s")
    print(f"  Mean FR: {np.mean(rates) * 1000/BIN_MS:.2f} Hz")

    # Second call - should be instant from cache
    start = time.time()
    binned2 = rec.get_binned_fr(bin_ms=BIN_MS)
    second_time = time.time() - start

    print(f"✓ Second call (cached): {second_time*1000:.1f}ms")
    print(f"  Speedup: {first_time/second_time:.0f}x")

    # =========================================================================
    # Part 4: Load Multiple Recordings for Comparison
    # =========================================================================

    print("\n" + "="*70)
    print("PART 4: Load Multiple Recordings")
    print("="*70)

    recordings = []
    all_rates = []

    for i, experiment in enumerate(TEST_EXPERIMENTS):
        print(f"\n[{i+1}/{len(TEST_EXPERIMENTS)}] Loading {experiment}...")

        rec = load_recording(
            proj=PROJ,
            chip=CHIP,
            experiment=experiment,
        )

        # Load spikes and compute binned FR
        if rec.spikes is None:
            print(f"  ❌ Failed to load - skipping")
            continue

        binned = rec.get_binned_fr(bin_ms=BIN_MS)
        if binned is None:
            print(f"  ❌ Failed to compute binned FR - skipping")
            continue

        recordings.append(rec)
        all_rates.append(binned['rates'])

        print(f"  ✓ {rec.spikes.N} neurons, {binned['rates'].shape[0]} time bins")

    print(f"\n✓ Successfully loaded {len(recordings)}/{len(TEST_EXPERIMENTS)} recordings")

    # =========================================================================
    # Part 5: Population Vector Analysis
    # =========================================================================

    print("\n" + "="*70)
    print("PART 5: Population Vector Analysis")
    print("="*70)

    if not all_rates:
        print("❌ No data to analyze")
        raise RuntimeError("Failed to load any recordings")

    # Use first recording for demonstration
    rates = all_rates[0]
    print(f"\nAnalyzing population vectors from first recording")
    print(f"  Shape: {rates.shape} (time_bins × neurons)")

    # Log transform for variance stabilization
    rates_log = np.log1p(rates)

    # Standardize
    scaler = StandardScaler()
    rates_scaled = scaler.fit_transform(rates_log)

    # First, fit PCA with all components to determine how many we need for 80% variance
    max_components = min(rates.shape[1], rates.shape[0] - 1)
    pca_full = PCA(n_components=max_components)
    pca_full.fit(rates_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_for_80 = np.argmax(cumvar >= 0.80) + 1

    # Now fit PCA with only the components needed for 80% variance
    pca = PCA(n_components=n_for_80)
    pca_coords = pca.fit_transform(rates_scaled)

    print(f"\n✓ PCA Results:")
    print(f"  Components for 80% variance: {n_for_80}")
    print(f"  PC1 variance: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  PC2 variance: {pca.explained_variance_ratio_[1]:.1%}")
    print(f"  Total variance captured: {cumvar[n_for_80-1]:.1%}")

    # =========================================================================
    # Part 6: Visualization
    # =========================================================================

    print("\n" + "="*70)
    print("PART 6: Visualization")
    print("="*70)

    # if you want to access the base_output_dir
    base_output_dir = rec.base_output_dir
    print(f"Base output dir: {base_output_dir}")

    # Use recording's output directory for organized plot storage that will automatically create the proj/chip subdirectories
    # This creates: {output_dir}/{proj}/{chip}/tutorial_examples/
    print(f"\nPlots will be saved to: {rec.output_dir / 'tutorial_examples'}")

    # Plot 1: Firing rate heatmap (5 minutes)
    # Show first 5 minutes of data as a heatmap of log-transformed firing rates
    five_min_bins = int((5 * 60 * 1000) / BIN_MS)  # 5 minutes in bins
    rates_subset = rates[:five_min_bins, :]
    rates_log_subset = np.log1p(rates_subset)  # log(1+x) transform

    fig, ax = styler.create_figure(size_preset='double')
    im = ax.imshow(
        rates_log_subset.T, 
        aspect='auto', 
        cmap=styler.get_heatmap_cmap('intensity'), 
        interpolation='nearest'
    )
    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron ID')
    ax.set_title(f'Firing Rate Heatmap (5 min, log-transformed): {PROJ}/{CHIP}')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log(1 + spike count)')

    # Add time axis labels in seconds
    time_ticks = np.linspace(0, five_min_bins, 6)
    time_labels = [f'{int(t * BIN_MS / 1000)}s' for t in time_ticks]
    ax.set_xticks(time_ticks)
    ax.set_xticklabels(time_labels)

    # Apply heatmap styling
    styler.style_heatmap(ax)
    save_dir = rec.output_dir / 'tutorial_examples'
    styler.finish_plot(save_plots=True, save_dir=str(save_dir), name='firing_rate_heatmap', show=True)
    print(f"✓ Saved firing rate heatmap to: {save_dir / 'firing_rate_heatmap.png'}")

    # Plot 2: PCA trajectory
    fig, ax = styler.create_figure(size_preset='1.5')
    scatter = ax.scatter(
        pca_coords[:, 0], pca_coords[:, 1],
        c=np.arange(len(pca_coords)), 
        cmap=styler.get_heatmap_cmap('temporal'),
        s=15, alpha=0.7, edgecolors='none'
    )
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance explained)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance explained)')
    ax.set_title(f'Neural State Trajectory (PCA): {PROJ}/{CHIP}')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (bins)')
    ax.grid(True, alpha=0.2, linewidth=0.5)

    styler.finish_plot(save_plots=True, save_dir=str(save_dir), name='pca_trajectory', show=True)
    print(f"✓ Saved PCA trajectory plot to: {save_dir / 'pca_trajectory.png'}")

    # Plot 3: Variance explained
    fig, ax = styler.create_figure(size_preset='1.5')
    ax.plot(
        np.arange(1, len(cumvar)+1), cumvar, 
        color=styler.get_named_color('dark_blue'),
        marker='o', markersize=3, linewidth=1.5,
        label='Cumulative variance'
    )
    ax.axhline(
        0.8, color=styler.get_named_color('red'), 
        linestyle='--', linewidth=1, label='80% threshold'
    )
    ax.axvline(
        n_for_80, color=styler.get_named_color('green'), 
        linestyle='--', linewidth=1, label=f'Components used: {n_for_80}'
    )
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title(f'PCA Variance Explained: {PROJ}/{CHIP}')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # Show up to 1.25x the number of components needed for 80% variance (or all components if fewer)
    x_max = min(int(n_for_80 * 1.25), len(cumvar))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.05)

    styler.finish_plot(save_plots=True, save_dir=str(save_dir), name='variance_explained', show=True)
    print(f"✓ Saved variance explained plot to: {save_dir / 'variance_explained.png'}")

    # =========================================================================
    # Part 7: Using RecordingCatalog with S3 Data
    # =========================================================================

    print("\n" + "="*70)
    print("PART 7: RecordingCatalog with S3 Data")
    print("="*70)

    # Create a mini-catalog DataFrame
    data = {
        'proj': [PROJ] * 3,
        'chip': [CHIP] * 3,
        'experiment': TEST_EXPERIMENTS,
        'type': ['stim', 'spontaneous', 'spontaneous'],
        'freq': [2.0, 0.0, 0.0],
    }
    df = pd.DataFrame(data)

    print("\nMini-catalog:")
    print(df.to_string(index=False))

    # Create catalog - it will use the config system for paths
    catalog = RecordingCatalog(df)

    print(f"\n{catalog}")

    # Filter for stim recordings
    stim_catalog = catalog.filter(type='stim')
    print(f"\nFiltered (type='stim'): {stim_catalog}")

    # Iterate and process
    print("\nProcessing catalog recordings:")
    for rec in catalog[:2]:  # Just first 2 for demo
        print(f"  {rec.identifier}: ", end="")
        if rec.spikes:
            print(f"{rec.spikes.N} neurons")
        else:
            print("failed to load")

    # =========================================================================
    # Summary
    # =========================================================================

    print("\n" + "="*70)
    print("TUTORIAL COMPLETE")
    print("="*70)
    print("""
What you learned:
  1. S3 path construction from experiment identifiers
     - Experiment format: "exp_base/exp_name"
     - Automatic S3 path generation for spike data and stim logs

  2. Automatic S3 fallback when local data doesn't exist
     - Local path checked first
     - Falls back to S3 download if not found
     - Downloaded files are cached locally

  3. Binned firing rate caching for fast recomputation
     - First call computes and caches
     - Subsequent calls load from cache instantly
     - Speedup typically >100x for large recordings

  4. Configuration-driven workflow
     - No need to manually specify paths
     - Config system handles local/S3 resolution
     - Use setup command to configure paths

  5. Organized plot storage with rec.output_dir
     - Automatic {output_dir}/{proj}/{chip}/ structure
     - Works like pathlib Path objects
     - Configure with: python -m braindance.utils.data_manager setup --output-dir /path/to/plots

  6. Using RecordingCatalog for batch processing
     - Filter recordings by metadata
     - Iterate and process multiple recordings
     - Works seamlessly with S3 data

Next steps:
  - Use RecordingCatalog with busy_bee_with_meta.csv for full dataset
  - See 01_working_with_catalog.py for more catalog examples
  - Explore other tutorials in the tutorials/ directory
""")
