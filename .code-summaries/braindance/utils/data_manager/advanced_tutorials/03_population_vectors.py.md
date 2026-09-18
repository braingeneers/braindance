# 03_population_vectors.py

**Path:** `braindance/utils/data_manager/advanced_tutorials/03_population_vectors.py`
**Module:** `braindance.utils.data_manager.advanced_tutorials.03_population_vectors`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Builds cached population spike-count vectors, applies log transformation and standardized PCA, and computes a 2D UMAP embedding. Saves neural-state plots and optionally exports AnnData with embeddings and metadata.

## Connections
- **Uses:** `Styler` from `braindance.utils.data_manager` — imports (static evidence).
- **Uses:** `bin_spike_data_vectorized` from `braindance.utils.data_manager` — imports (static evidence).
- **Uses:** `load_recording` from `braindance.utils.data_manager` — imports (static evidence).
- **Uses:** `validate_spike_data` from `braindance.utils.data_manager` — imports (static evidence).
- **Shared data:** bin_spike_data_vectorized→ResultsCache binned_spikes/pca/umap→Styler plots→DataContext metadata.

## Dependencies
- `anndata` — external or unresolved local import; source import evidence.
- `braindance.utils.data_manager.Styler` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.bin_spike_data_vectorized` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.load_recording` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.validate_spike_data` — intra-repo import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.
- `sklearn.decomposition.PCA` — external or unresolved local import; source import evidence.
- `sklearn.preprocessing.StandardScaler` — external or unresolved local import; source import evidence.
- `umap` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `compute_binned_spikes()`
> Bin full recording and return count/time arrays for cache.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/advanced_tutorials/03_population_vectors.py:97`
### `compute_pca()`
> Select components reaching 80% variance and cache coordinates/statistics.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/advanced_tutorials/03_population_vectors.py:164`
### `compute_umap()`
> Embed retained PCA coordinates into two dimensions.
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/utils/data_manager/advanced_tutorials/03_population_vectors.py:249`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| Project/chip/experiment inline; bin_size_ms=100; PCA target variance=.8; UMAP min_dist=.5,neighbors<=15,random_state=42. |
| save_dir='/Volumes/hunter_ssd/busy_bee/test_plots'. |

## Data Shapes
- Counts(time bins,neurons), PCA(time bins,components), UMAP(time bins,2); optional population_vectors.h5ad.

## Notes
- Runs on import and requires umap/sklearn; anndata optional.
- Hardcoded path and example recording need adaptation; plotting assumes at least two PCs and a usable one-minute window.
