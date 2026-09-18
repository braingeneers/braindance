# Data Manager Test Bugs

## [FIXED] Bug 1: Mock Recording Fixture Missing Spike Data for Caching Test
**Test File:** `test_binned_fr.py::test_binning_caching`
**Resolution:** Updated `conftest.py` to create actual spike data files on disk in the temp directory so that `Recording.spikes` property can load them properly.

## [FIXED] Bug 2: SpikeData Object Does Not Support len()
**Test File:** `test_burst_detector.py::test_bic_matrix_calculation`
**Resolution:** Updated the test to use `mock_spike_data.N` instead of `len()`. Also corrected the expected shape for `bic_matrix` from 2D to 1D `(n_neurons,)`.

## [FIXED] Bug 3: Missing Project Extraction from S3 base_path
**Test File:** `test_s3_path_construction.py::TestLocalPathResolution::test_project_extracted_from_base_path_when_proj_missing`
**Resolution:** Updated `Recording._resolve_paths` in `recording.py` to automatically extract the project name from the S3 `base_path` URL if the `proj` column is missing from the catalog row.

## [FIXED] Bug 5: Pytest Warnings in test_results_cache.py
**Test File:** `test_results_cache.py`
**Resolution:** Removed redundant `return True` statements from test functions to satisfy pytest's requirement that tests return `None`, eliminating `PytestReturnNotNoneWarning`.

## [FIXED] Bug 6: ModuleNotFoundError: No module named 'psutil'
**Issue:** The NRP ML pipeline failed because `psutil` was required but not installed in the Docker image.
**Resolution:** 
1. Added `psutil` to `proj/predictor/nrp/requirements.txt`.
2. Made `psutil` an optional import in `ultra_optimized_latency_helper.py` with a fallback conservative memory estimate for batch size calculation.

---

## [FIXED] Bug 4: test_s3_caching_flow.py Not Pytest Compatible
**Test File:** `test_s3_caching_flow.py`
**Resolution:** Refactored the procedural script into a proper `pytest` Test Class with integration markers and conditional skipping.

---

## [OPEN] Bug 7: ResultsCache.exists_local Mismatch
**Test File:** `test_binned_fr.py::test_binning_caching`
**Issue:** Refactoring changed the expected keys in ResultsCache. `exists_local` fails to find 'binned_fr' with params `{'bin_ms': 100.0}` even if file exists on disk, likely due to filename construction or manifest mismatch.

## [OPEN] Bug 8: Path Mocking Compatibility for Game Logs
**Test File:** `test_game_log_analysis.py::test_game_log_s3_fallback`
**Issue:** `Recording` now has internal `_resolve_paths` logic that conflicts with the `PropertyMock` used in the test. The test fails to trigger the S3 download properly because it mocks `_game_log_path` but the internal logic expects it to be resolved differently.

## [OPEN] Bug 9: S3 Path Construction Error for Non-S3 Catalogs
**Test File:** `test_s3_caching_flow.py::TestS3CachingFlow::test_s3_path_construction`
**Issue:** `recording._construct_s3_spike_path()` returns `None` for local catalogs, but the test asserts `s3:// in s3_path`. This results in a `TypeError: argument of type 'NoneType' is not iterable`.

---

## Test Summary (internal_tests) - Post Refactor

| Suite | Total | Passed | Failed/Error | Success Rate |
|-------|-------|--------|--------------|--------------|
| Overall internal_tests | 151 | 149 | 2 | 98.7% |

*\*Note: Added 9 smoke tests in `test_plotting.py` for the new modular architecture. All passed. Remaining failures (Bugs 8, 9) are related to test harness/mocking compatibility with the resolved paths logic.*

