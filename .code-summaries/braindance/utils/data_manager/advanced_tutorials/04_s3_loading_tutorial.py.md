# 04_s3_loading_tutorial.py

**Path:** `braindance/utils/data_manager/advanced_tutorials/04_s3_loading_tutorial.py`
**Module:** `braindance.utils.data_manager.advanced_tutorials.04_s3_loading_tutorial`
**Feature Area:** `Examples and Workshop`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Demonstrates local-first recording retrieval with S3 fallback and parameter-keyed binning cache timing. Loads several example recordings, runs PCA, saves plots, and constructs a small RecordingCatalog.

## Connections
- **Uses:** `RecordingCatalog` from `braindance.utils.data_manager` — imports (static evidence).
- **Uses:** `load_recording` from `braindance.utils.data_manager` — imports (static evidence).
- **Uses:** `Styler` from `braindance.utils.data_manager.utils.plotting.plot_styler` — imports (static evidence).
- **Shared data:** load_recording→private path inspection/lazy spikes→get_binned_fr→PCA/Styler; RecordingCatalog demonstrates filtering.

## Dependencies
- `braindance.utils.data_manager.RecordingCatalog` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.load_recording` — intra-repo import; source import evidence.
- `braindance.utils.data_manager.utils.plotting.plot_styler.Styler` — intra-repo import; source import evidence.
- `matplotlib.pyplot` — external or unresolved local import; source import evidence.
- `numpy` — external or unresolved local import; source import evidence.
- `pandas` — external or unresolved local import; source import evidence.
- `sklearn.decomposition.PCA` — external or unresolved local import; source import evidence.
- `sklearn.preprocessing.StandardScaler` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| PROJ='25-02-25_busybees', CHIP='25123ic', three exp1 recordings, BIN_MS=20; output plots go under rec.output_dir/tutorial_examples. |

## Data Shapes
- Binned rates are time-bin × neuron arrays with a millisecond time axis; PCA produces two component coordinates and plots save PNG/SVG outputs.

## Notes
- Requires configured catalog/data and S3 access for missing files; plotting assumes two PCA components.
- Constructed mini-catalog lacks S3 base_path, so its missing data cannot use catalog-derived fallback automatically.
- Styler is constructed with journal="draft" for this non-paper tutorial.
