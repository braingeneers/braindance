"""
Test script for the new ResultsCache caching system.

This tests the caching workflow for binned firing rates:
1. Load recording with legacy flat structure
2. First call computes and caches binned FR
3. Second call loads from cache (instant)
4. Optionally test S3 upload/download

Usage:
    python test_results_cache.py
    
    # With S3 upload enabled
    BRAINDANCE_AUTO_UPLOAD=1 python test_results_cache.py
"""

import sys
from pathlib import Path

import numpy as np
import time
from braindance.utils.data_manager.utils.data_loading.recording import load_recording
from braindance.utils.data_manager.utils.data_loading.results_cache import (
    ResultsCache,
    get_auto_upload_enabled
)


def test_basic_caching():
    """Test basic caching workflow with legacy flat data."""
    print("\n" + "="*60)
    print("TEST 1: Basic Caching with Legacy Flat Structure")
    print("="*60)
    
    # Configuration
    BASE_PATH = Path("/Volumes/hunter_ssd/busy_bee/data")
    PROJ = "25-02-25_busybees"
    CHIP = "25123ic"
    EXP = "exp1_cont_95"  # Just the experiment name (not full path)
    BIN_MS = 20
    
    # Check if data exists
    spike_path = BASE_PATH / PROJ / CHIP / f"{EXP}_spike_data.pkl"
    if not spike_path.exists():
        print(f"⚠️  Test data not found at: {spike_path}")
        print("Skipping test - run with actual data path")
        # TODO: implement s3 caching to test both things at the same time
        import pytest
        pytest.skip("Test data not found")
    
    print(f"\nLoading recording: {PROJ}/{CHIP}/{EXP}")
    print(f"Base path: {BASE_PATH}")
    print(f"Legacy flat: True")
    
    # Load recording
    rec = load_recording(
        proj=PROJ,
        chip=CHIP,
        experiment=EXP,
        base_path=BASE_PATH,
        legacy_flat=True
    )
    
    print(f"✓ Recording loaded: {rec.identifier}")
    
    # First call - should compute
    print(f"\n--- First call (compute) ---")
    start = time.time()
    binned1 = rec.get_binned_fr(bin_ms=BIN_MS)
    time1 = time.time() - start
    
    print(f"Time: {time1:.2f}s")
    print(f"Shape: {binned1['rates'].shape}")
    print(f"Time axis: {binned1['time_axis_ms'][:5]}... to {binned1['time_axis_ms'][-5:]}")
    
    # Second call - should load from cache
    print(f"\n--- Second call (cache) ---")
    start = time.time()
    binned2 = rec.get_binned_fr(bin_ms=BIN_MS)
    time2 = time.time() - start
    
    print(f"Time: {time2:.4f}s")
    print(f"Speedup: {time1/time2:.1f}x")
    
    # Verify data matches
    assert np.array_equal(binned1['rates'], binned2['rates']), "Data mismatch!"
    print("✓ Data verified - cache matches computed")
    
    # Check results directory
    results_path = rec._results_path
    print(f"\nCached files in: {results_path}")
    if results_path.exists():
        for f in results_path.iterdir():
            print(f"  - {f.name}")
    
    return


def test_different_bin_sizes():
    """Test caching with multiple bin sizes."""
    print("\n" + "="*60)
    print("TEST 2: Multiple Bin Sizes")
    print("="*60)
    
    # Configuration
    BASE_PATH = Path("/Volumes/hunter_ssd/busy_bee/data")
    PROJ = "25-02-25_busybees"
    CHIP = "25123ic"
    EXP = "exp1_cont_95"
    
    spike_path = BASE_PATH / PROJ / CHIP / f"{EXP}_spike_data.pkl"
    if not spike_path.exists():
        print(f"⚠️  Test data not found - skipping")
        # TODO: implement s3 caching to test both things at the same time
        import pytest
        pytest.skip("Test data not found")
    
    rec = load_recording(PROJ, CHIP, EXP, base_path=BASE_PATH, legacy_flat=True)
    
    bin_sizes = [10, 20, 50, 100]
    
    for bin_ms in bin_sizes:
        print(f"\n--- Bin size: {bin_ms}ms ---")
        start = time.time()
        binned = rec.get_binned_fr(bin_ms=bin_ms)
        elapsed = time.time() - start
        
        # Check cache file exists
        cache_file = rec._results_path / f"binned_fr_{bin_ms}ms.npz"
        exists = "✓" if cache_file.exists() else "✗"
        print(f"  Shape: {binned['rates'].shape}, Time: {elapsed:.2f}s, Cached: {exists}")
    
    return


def test_s3_sync():
    """Test S3 sync functionality (requires S3 credentials)."""
    print("\n" + "="*60)
    print("TEST 3: S3 Sync (if BRAINDANCE_AUTO_UPLOAD=1)")
    print("="*60)
    
    auto_upload = get_auto_upload_enabled()
    print(f"Auto-upload enabled: {auto_upload}")
    
    if not auto_upload:
        print("Skipping S3 test - set BRAINDANCE_AUTO_UPLOAD=1 to test")
        return
    
    # Configuration
    BASE_PATH = Path("/Volumes/hunter_ssd/busy_bee/data")
    PROJ = "25-02-25_busybees"
    CHIP = "25123ic"
    EXP = "exp1_cont_95"
    
    spike_path = BASE_PATH / PROJ / CHIP / f"{EXP}_spike_data.pkl"
    if not spike_path.exists():
        print(f"⚠️  Test data not found - skipping")
        # TODO: implement s3 caching to test both things at the same time
        import pytest
        pytest.skip("Test data not found")
    
    rec = load_recording(PROJ, CHIP, EXP, base_path=BASE_PATH, legacy_flat=True)
    
    print(f"S3 path: {rec.cache.s3_path}")
    
    # Compute and upload
    binned = rec.get_binned_fr(bin_ms=20)
    
    # Manual sync of all results
    print("\nSyncing all results to S3...")
    success, fail = rec.cache.sync_all_to_s3()
    print(f"Sync complete: {success} uploaded, {fail} failed")
    
    return


def test_results_cache_standalone():
    """Test ResultsCache class directly."""
    print("\n" + "="*60)
    print("TEST 4: Standalone ResultsCache")
    print("="*60)
    
    import tempfile
    
    # Create temp directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResultsCache(
            local_path=Path(tmpdir),
            s3_path=None,  # No S3 for this test
            auto_upload=False
        )
        
        # Test get_or_compute
        def compute_fn():
            return {
                'data': np.random.rand(100, 10),
                'time': np.arange(100)
            }
        
        print("\n--- First compute ---")
        result1 = cache.get_or_compute('test_data', {'n': 100}, compute_fn)
        print(f"Shape: {result1['data'].shape}")
        
        print("\n--- Load from cache ---")
        result2 = cache.get_or_compute('test_data', {'n': 100}, compute_fn)
        print(f"Shape: {result2['data'].shape}")
        
        # Verify
        assert np.array_equal(result1['data'], result2['data']), "Cache mismatch!"
        print("✓ Cache verified")
        
        # Check manifest
        print(f"\nManifest entries: {len(cache._manifest.list_results())}")
        for entry in cache._manifest.list_results():
            print(f"  - {entry['name']}: {entry['filename']}")
    
    return


if __name__ == "__main__":
    print("="*60)
    print("ResultsCache Test Suite")
    print("="*60)
    
    results = {}
    
    # Run tests
    results['standalone'] = test_results_cache_standalone()
    results['basic'] = test_basic_caching()
    results['bin_sizes'] = test_different_bin_sizes()
    results['s3'] = test_s3_sync()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
