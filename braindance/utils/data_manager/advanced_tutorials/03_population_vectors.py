"""
BrainDance Data Manager - Tutorial 03: Population Vectors & Dimensionality Reduction

This tutorial covers creating population activity vectors from spike data and
visualizing neural dynamics using PCA and UMAP. Features automatic caching for
fast re-runs.

Key features:
- Binned spike count matrices with caching
- PCA dimensionality reduction with cached results
- UMAP non-linear embedding with cached results
- Interactive visualization of neural state trajectories

Prerequisites:
  - Completed Tutorials 01 and 02
  - Python packages: numpy, matplotlib, sklearn, umap-learn

Note: For ML data preparation (sliding windows, sequence models), see Tutorial 04.
"""


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

from braindance.utils.data_manager import (
    load_recording,
    bin_spike_data_vectorized,
    validate_spike_data,
    Styler
)


# =============================================================================
# 1. Load Recording Data
# =============================================================================

proj = '25-02-25_busybees'
chip = '25123ic'
experiment = 'exp1_cont_1'

save_dir = '/Volumes/hunter_ssd/busy_bee/test_plots'

print(f"Loading recording: {proj}/{chip}/{experiment}")
rec = load_recording(proj, chip, experiment)
spikes = rec.spikes

print(f"✓ Loaded: {spikes.N} neurons, {spikes.length/1000:.1f}s duration")


# =============================================================================
# 2. Validate Spike Data Quality
# =============================================================================

print("\n" + "="*70)
print("DATA VALIDATION")
print("="*70)

# Get spike trains as list of arrays
spike_trains = [np.array(spikes.train[i]) for i in range(spikes.N)]

# Validate data quality
validation_stats = validate_spike_data(
    spike_trains,
    expected_length_ms=spikes.length,
    verbose=True
)

print(f"\n✓ Data validation complete")
print(f"  Empty neurons: {validation_stats['empty_neurons']}/{validation_stats['n_neurons']}")
print(f"  Issues found: {len(validation_stats['issues'])}")


# =============================================================================
# 3. Create Population Vectors via Binned Spike Counts (WITH CACHING)
# =============================================================================
# Convert spike times to binned count matrix: (time_bins, neurons)

print("\n" + "="*70)
print("POPULATION VECTORS: Binning Spike Data")
print("="*70)

# Bin spikes into 100ms windows
bin_size_ms = 100.0

# Define cache params
cache_params_binned = {
    'bin_size_ms': float(bin_size_ms),
    'time_range_start': 0.0,
    'time_range_end': float(spikes.length)
}

# Define compute function
def compute_binned_spikes():
    print(f"  Computing binned spike data (bin_size={bin_size_ms}ms)...")
    binned_data, time_axis = bin_spike_data_vectorized(
        spike_trains,
        bin_size_ms=bin_size_ms,
        time_range=(0, spikes.length),
        verbose=True
    )
    return {
        'binned_data': binned_data,
        'time_axis': time_axis
    }

# Get or compute with cache
cached_binned = rec.cache.get_or_compute(
    'binned_spikes',
    params=cache_params_binned,
    compute_fn=compute_binned_spikes
)

binned_data = cached_binned['binned_data']
time_axis = cached_binned['time_axis']

print(f"\n✓ Created population vectors")
print(f"  Shape: {binned_data.shape} (time_bins × neurons)")
print(f"  Time resolution: {bin_size_ms}ms per bin")
print(f"  Total duration: {len(time_axis) * bin_size_ms / 1000:.1f}s")

# Basic statistics
total_spikes = np.sum(binned_data)
mean_per_bin = np.mean(binned_data)
print(f"  Total spikes captured: {total_spikes}")
print(f"  Mean spikes per bin: {mean_per_bin:.2f}")


# =============================================================================
# 4. Apply Log Transformation (Variance Stabilization)
# =============================================================================
# Log1p transformation helps with high-variance spike count data

print("\n" + "="*70)
print("PREPROCESSING: Log Transformation")
print("="*70)

# Apply log1p transformation
binned_log = np.log1p(binned_data)

print(f"Raw data: mean={np.mean(binned_data):.2f}, std={np.std(binned_data):.2f}")
print(f"Log data: mean={np.mean(binned_log):.2f}, std={np.std(binned_log):.2f}")


# =============================================================================
# 5. PCA: Reduce Dimensionality (WITH CACHING)
# =============================================================================

print("\n" + "="*70)
print("PCA: Dimensionality Reduction")
print("="*70)

# Define cache params - target 80% variance
cache_params_pca = {
    'bin_size_ms': float(bin_size_ms),
    'target_variance': 0.80,
    'scaled': True
}

# Define compute function
def compute_pca():
    print(f"  Computing PCA to reach 80% explained variance...")
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(binned_log)

    # Fit PCA with maximum possible components first to find how many we need
    max_components = min(binned_log.shape[0], binned_log.shape[1])
    pca_full = PCA(n_components=max_components)
    pca_full.fit(data_scaled)

    # Find how many components we need for 80% variance
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
    if np.any(cumulative_var >= 0.80):
        n_for_80 = np.argmax(cumulative_var >= 0.80) + 1
        print(f"  Found {n_for_80} components needed for 80% variance")
    else:
        # If we can't reach 80%, use all components
        n_for_80 = max_components
        print(f"  Warning: 80% variance not reachable")
        print(f"  Using all {max_components} components (max variance: {cumulative_var[-1]:.1%})")

    # Now fit PCA with the exact number of components we need
    pca = PCA(n_components=n_for_80)
    pca_coords = pca.fit_transform(data_scaled)

    return {
        'pca_coords': pca_coords,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'n_components': n_for_80,
        'n_for_80_var': n_for_80
    }

# Get or compute with cache
cached_pca = rec.cache.get_or_compute(
    'pca',
    params=cache_params_pca,
    compute_fn=compute_pca
)

pca_coords = cached_pca['pca_coords']
explained_variance_ratio = cached_pca['explained_variance_ratio']
cumulative_var = cached_pca['cumulative_variance']
n_components = cached_pca['n_components']
n_for_80 = cached_pca['n_for_80_var']

# For backward compatibility with plotting code
pca_2d = pca_coords[:, :2]

# Print statements
print(f"✓ PCA complete")
print(f"  Components used: {n_components}")
print(f"  Variance explained by PC1: {explained_variance_ratio[0]:.1%}")
print(f"  Variance explained by PC2: {explained_variance_ratio[1]:.1%}")
print(f"  Total variance in first 2 PCs: {cumulative_var[1]:.1%}")
print(f"  Total variance explained: {cumulative_var[-1]:.1%}")


# =============================================================================
# 6. UMAP: Non-linear Embedding (WITH CACHING)
# =============================================================================

print("\n" + "="*70)
print("UMAP: Non-linear Dimensionality Reduction")
print("="*70)

# Use PCA-reduced data for UMAP (faster, often better)
pca_for_umap = pca_coords[:, :n_for_80]

# UMAP parameters
n_neighbors = min(15, len(pca_for_umap) - 1)
min_dist = 0.5  # Increased to spread points more
metric = 'euclidean'

# Define cache params
cache_params_umap = {
    'bin_size_ms': float(bin_size_ms),
    'n_pca_components': int(n_for_80),
    'n_neighbors': int(n_neighbors),
    'min_dist': float(min_dist),
    'metric': metric
}

# Define compute function
def compute_umap():
    print(f"  Computing UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    # Fit UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=42
    )

    umap_coords = reducer.fit_transform(pca_for_umap)

    return {
        'umap_coords': umap_coords
    }

# Get or compute with cache
cached_umap = rec.cache.get_or_compute(
    'umap',
    params=cache_params_umap,
    compute_fn=compute_umap
)

umap_coords = cached_umap['umap_coords']

# Print statements
print(f"✓ UMAP complete")
print(f"  Input dimensions: {pca_for_umap.shape[1]} PCs")
print(f"  Output dimensions: 2")


# =============================================================================
# 7. Visualization
# =============================================================================

print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

# Initialize styler for consistent formatting
styler = Styler()

# -------------------------
# Plot 1: Firing Rate Heatmap (1 minute zoom)
# -------------------------
print("  Creating firing rate heatmap...")

# Calculate firing rates (Hz) from binned data
firing_rates = binned_data * (1000 / bin_size_ms)  # Convert counts to Hz

# Select 1 minute (60 seconds) of data in the middle
duration_s = len(time_axis) * bin_size_ms / 1000
start_s = duration_s / 2 - 30  # Start 30s before middle
end_s = duration_s / 2 + 30    # End 30s after middle

# Convert to indices
start_idx = int(start_s * 1000 / bin_size_ms)
end_idx = int(end_s * 1000 / bin_size_ms)

# Extract the chunk
firing_rates_chunk = firing_rates[start_idx:end_idx, :]
time_chunk = time_axis[start_idx:end_idx]

# Create figure
fig1, ax1 = styler.create_figure(nrows=1, ncols=1, size_preset='single')

# Plot heatmap
im = ax1.imshow(
    firing_rates_chunk.T,  # Transpose so neurons are on y-axis
    aspect='auto',
    cmap='viridis',
    interpolation='nearest',
    extent=[time_chunk[0]/1000, time_chunk[-1]/1000, 0, spikes.N]
)

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Neuron ID')
ax1.set_title('Neural Firing Rate Heatmap (1 minute)')

# Add colorbar
cbar = plt.colorbar(im, ax=ax1)
cbar.set_label('Firing Rate (Hz)')

# Save
Path(save_dir).mkdir(parents=True, exist_ok=True)
save_path_heatmap = Path(save_dir) / 'population_firing_rate_heatmap'
plt.savefig(f"{save_path_heatmap}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{save_path_heatmap}.svg", format='svg', bbox_inches='tight')
print(f"✓ Saved firing rate heatmap to {save_path_heatmap}")
plt.close()

# -------------------------
# Plot 2: PCA Neural State Trajectory
# -------------------------
print("  Creating PCA trajectory plot...")

fig2, ax2 = styler.create_figure(nrows=1, ncols=1, size_preset='single')

# Color by time (viridis colormap)
scatter = ax2.scatter(
    pca_2d[:, 0], pca_2d[:, 1],
    c=np.arange(len(pca_2d)), 
    cmap='viridis',
    s=20,  # Increased marker size
    alpha=0.7,  # Slightly increased alpha
    edgecolors='none'
)

ax2.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} var)')
ax2.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} var)')
ax2.set_title('PCA: Neural State Trajectory')

cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Time (bins)')
ax2.grid(True, alpha=0.3)

# Save
save_path_pca = Path(save_dir) / 'population_pca_trajectory'
plt.savefig(f"{save_path_pca}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{save_path_pca}.svg", format='svg', bbox_inches='tight')
print(f"✓ Saved PCA trajectory to {save_path_pca}")
plt.close()

# -------------------------
# Plot 3: UMAP Neural State Embedding
# -------------------------
print("  Creating UMAP embedding plot...")

fig3, ax3 = styler.create_figure(nrows=1, ncols=1, size_preset='single')

scatter = ax3.scatter(
    umap_coords[:, 0], umap_coords[:, 1],
    c=np.arange(len(umap_coords)), 
    cmap='viridis',
    s=20,  # Increased marker size
    alpha=0.7,  # Slightly increased alpha
    edgecolors='none'
)

ax3.set_xlabel('UMAP 1')
ax3.set_ylabel('UMAP 2')
ax3.set_title('UMAP: Neural State Embedding')

cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Time (bins)')
ax3.grid(True, alpha=0.3)

# Save
save_path_umap = Path(save_dir) / 'population_umap_embedding'
plt.savefig(f"{save_path_umap}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{save_path_umap}.svg", format='svg', bbox_inches='tight')
print(f"✓ Saved UMAP embedding to {save_path_umap}")
plt.close()

print(f"\n✓ All visualizations saved to {save_dir}")


# =============================================================================
# 8. Optional: Export to AnnData Format
# =============================================================================
# AnnData provides a standardized format for downstream analysis with scanpy
# and easy sharing with collaborators

print("\n" + "="*70)
print("OPTIONAL: Exporting to AnnData Format")
print("="*70)

try:
    import anndata as ad
    import pandas as pd

    # Create observation metadata (one row per time bin)
    obs_metadata = pd.DataFrame({
        'time_ms': time_axis,
        'time_s': time_axis / 1000.0,
    })
    obs_metadata.index = [f"timebin_{i}" for i in range(len(time_axis))]

    # Create variable metadata (one row per neuron)
    var_metadata = pd.DataFrame({
        'neuron_id': range(spikes.N)
    })
    var_metadata.index = [f"neuron_{i}" for i in range(spikes.N)]

    # Create AnnData object
    adata = ad.AnnData(
        X=binned_log,  # Log-transformed binned spike counts (time_bins × neurons)
        obs=obs_metadata,
        var=var_metadata
    )

    # Add dimensionality reductions to obsm (standard scanpy convention)
    adata.obsm['X_pca'] = pca_coords
    adata.obsm['X_umap'] = umap_coords

    # Add PCA info to uns
    adata.uns['pca'] = {
        'variance_ratio': explained_variance_ratio.tolist(),
        'cumulative_variance': cumulative_var.tolist(),
        'n_components_80_var': int(n_for_80)
    }

    # Add analysis parameters to uns
    adata.uns['analysis_parameters'] = {
        'bin_size_ms': float(bin_size_ms),
        'n_pca_components': int(n_components),
        'recording_info': {
            'proj': proj,
            'chip': chip,
            'experiment': experiment,
            'n_neurons': int(spikes.N),
            'length_ms': float(spikes.length)
        }
    }

    # Save to file
    adata_path = Path(save_dir) / 'population_vectors.h5ad'
    adata.write_h5ad(adata_path)

    print(f"✓ Exported AnnData to {adata_path}")
    print(f"  Shape: {adata.shape} (time_bins × neurons)")
    print(f"  .X: Log-transformed binned spike counts")
    print(f"  .obsm['X_pca']: PCA coordinates ({n_components} components)")
    print(f"  .obsm['X_umap']: UMAP coordinates (2D)")
    print(f"\nUsage with scanpy:")
    print(f"  import scanpy as sc")
    print(f"  adata = sc.read_h5ad('{adata_path.name}')")
    print(f"  sc.pl.umap(adata, color='time_s')  # Color by time")

except ImportError:
    print("AnnData not installed - skipping export")
    print("Install with: pip install anndata")
except Exception as e:
    print(f"Warning: AnnData export failed: {e}")

print("="*70)


# =============================================================================
# 9. Save Results
# =============================================================================

# Save metadata about cached results
rec.results.binned_data_shape = binned_data.shape
rec.results.bin_size_ms = bin_size_ms
rec.results.pca_variance_explained = explained_variance_ratio[:10].tolist()
rec.results.n_components_80_var = int(n_for_80)
rec.results.umap_computed = True

rec.save_results()
print("\n✓ Results metadata saved to cache")


print("\n" + "=" * 70)
print("Tutorial 03 complete! You've learned:")
print("  • How to validate spike data quality")
print("  • How to bin spikes into population vectors with caching")
print("  • How to apply log transformation for variance stabilization")
print("  • How to use PCA for dimensionality reduction with caching")
print("  • How to use UMAP for non-linear embedding with caching")
print("  • How to visualize neural state trajectories")
print("  • How to export data to AnnData format (optional)")

print("\nNext: See Tutorial 04 for ML data preparation (sliding windows, etc.)")
print("=" * 70)
