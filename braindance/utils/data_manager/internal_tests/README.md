# Data Manager Internal Tests

This directory contains the internal test suite for the BrainDance Data Manager. These tests cover core functionality, data loading, analysis algorithms, caching mechanisms, and integration with S3.

## Test Configuration & Utilities

*   **`conftest.py`**: Pytest fixtures for creating mock data (`mock_spike_data`, `mock_stim_log`, `mock_recording`, etc.) to isolate tests from external dependencies.
*   **`pytest.ini`**: Pytest configuration file defining markers (e.g., `integration`).
*   **`notes.txt`**: Developer notes regarding data setup commands for legacy and new formats.
*   **`restructure_data.py`**: Utility script to restructure flat data directories into the hierarchical format expected by the Data Manager.

## Core Functionality Tests

*   **`test_recording_catalog.py`**: Comprehensive tests for the `Recording`, `RecordingCatalog`, and `DataContext` classes. Covers initialization, filtering, indexing, and batch operations.
*   **`test_catalog_operations.py`**: Specific tests for catalog filtering (Django-style lookups), attribute access, and iteration.
*   **`test_baseline_detection.py`**: Verifies the logic for identifying baseline recordings based on directory naming patterns (e.g., exact matches, `_cont` suffixes).
*   **`test_s3_path_construction.py`**: Tests logic for constructing S3 paths for various bucket configurations (standard, custom) and handling nested experiment paths.

## Analysis & Processing Tests

*   **`test_binned_fr.py`**: Tests vectorized computation of binned firing rates and verifies the caching mechanism for these results.
*   **`test_binned_isi.py`**: Tests vectorized computation of binned Inter-Spike Intervals (ISI), including normalization and edge cases.
*   **`test_burst_detector.py`**: Validates the `BurstDetector` class, checking burst detection accuracy, metric consistency, and BIC matrix calculation.
*   **`test_latency_analysis.py`**: Tests stimulation grouping logic and burst latency analysis, ensuring correct handling of single and multi-electrode stimulations.
*   **`test_population_analysis.py`**: Verifies population rate calculations and plotting logic for STTC matrices (including diagonal masking).

## Data Loading & Caching Tests

*   **`test_results_cache.py`**: Tests the `ResultsCache` system, verifying that results are computed once, saved to disk, and loaded from cache on subsequent calls. Includes S3 sync tests.
*   **`test_s3_caching_flow.py`**: Integration tests for the complete S3 caching workflow, requiring local data and S3 access (marked as integration).
*   **`test_game_log_analysis.py`**: Tests loading of game logs, including fallback mechanisms to download from S3 if local files are missing.
*   **`test_ml_utils.py`**: Tests for ML-specific utilities, including tokenizer robustness and path resolution for ML container environments.
*   **`test_actual_error_reproduction.py`**: A reproduction script for a specific S3 path construction error scenario observed in deployment logs.

## Visualization Tests

*   **`test_plotting.py`**: Smoke tests for plotting functions (`plot_raster_with_pop`, `plot_evoked_raster`, etc.) and the `Recording.pl` accessor, ensuring they generate figures without errors.

## Documentation

*   **`BUGS.md`**: A log of fixed and open bugs related to the Data Manager tests.
*   **`README.md`**: This file.

## Usage

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest test_recording_catalog.py
```

Run integration tests (requires data/S3):
```bash
pytest -m integration
```
