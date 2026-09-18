"""
Test case to reproduce the exact error from the container logs.

This test simulates what happens when the dataloader tries to load recordings
and shows the exact paths being constructed.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

from braindance.utils.data_manager.utils.data_loading.recording import Recording


def test_exact_error_scenario():
    """
    Reproduce the exact error from logs:
    
    ERROR: Spike data file not found: /app/data/cache/24-04-18_butterfly/23133/exp1_spike_data.pkl
    [S3] Downloading: /app/data/cache/23133/exp1/spike_data/exp1_cartpole_long_7_spike_data.pkl
    
    The first path (expected local) has the project folder ✓
    The second path (download target) is MISSING the project folder ✗
    """
    row = pd.Series({
        'base_path': 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/',
        'chip': '23133',
        'experiment': 'exp1/exp1_cartpole_long_7',
        'proj': '24-04-18_butterfly'
    })
    rec = Recording(row)
    
    # Step 1: Check what local path would be constructed for spike data
    rec._resolve_paths()
    local_spike_path = rec._spikes_path
    
    print(f"\n=== Path Resolution Test ===")
    print(f"Local spike path: {local_spike_path}")
    print(f"Expected to contain: /app/data/cache/24-04-18_butterfly/23133/")
    
    # This should include the project folder
    assert '24-04-18_butterfly' in str(local_spike_path), \
        f"Local path should include project folder, got: {local_spike_path}"
    assert '23133' in str(local_spike_path), \
        f"Local path should include chip folder, got: {local_spike_path}"
    
    # Step 2: Check what S3 path would be constructed
    s3_spike_path = rec._construct_s3_spike_path()
    
    print(f"\nS3 spike path: {s3_spike_path}")
    print(f"Expected: s3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/23133/exp1/spike_data/exp1_cartpole_long_7_spike_data.pkl")
    
    # This should be an S3 URL
    assert s3_spike_path.startswith('s3://'), \
        f"S3 path should start with s3://, got: {s3_spike_path}"
    assert '24-04-18_butterfly' in s3_spike_path, \
        f"S3 path should include project, got: {s3_spike_path}"
    assert 'exp1/spike_data' in s3_spike_path, \
        f"S3 path should include exp_base/spike_data folder, got: {s3_spike_path}"
    assert 'exp1_cartpole_long_7_spike_data.pkl' in s3_spike_path, \
        f"S3 path should include correct filename, got: {s3_spike_path}"
    
    # Step 3: Simulate what happens when downloading
    # In _load_spikes(), it calls _download_from_s3(s3_path, spike_path)
    # The error shows the S3 path looks like a local path, which means either:
    # A) s3_path is being constructed incorrectly
    # B) The print statement is printing the wrong variable
    
    print(f"\n=== Simulated Download ===")
    print(f"Would download FROM: {s3_spike_path}")
    print(f"Would download TO: {local_spike_path}")
    print(f"\nThese should be DIFFERENT:")
    print(f"  FROM should start with 's3://'")
    print(f"  TO should be a local path with project folder")
    
    # Verify they are different
    assert s3_spike_path != str(local_spike_path), \
        "S3 path and local path should be different!"


def test_binned_fr_loading_paths():
    """
    Test what paths are used when get_binned_fr() is called.
    This triggers spike data loading as a side effect.
    """
    row = pd.Series({
        'base_path': 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/',
        'chip': '23133',
        'experiment': 'exp1/exp1_cartpole_long_7',
        'proj': '24-04-18_butterfly'
    })
    rec = Recording(row)
    
    print(f"\n=== Testing get_binned_fr() Path Resolution ===")
    
    # Trigger path resolution by accessing _spikes_path
    spike_path = rec._spikes_path
    print(f"Spike path resolved to: {spike_path}")
    
    # Check results cache S3 path
    cache = rec.cache
    print(f"Results cache S3 path: {cache.s3_path}")
    print(f"Results cache local path: {cache.local_path}")
    
    # The results cache S3 path should be correctly formed
    assert cache.s3_path.startswith('s3://'), "Cache S3 path should be S3 URL"
    assert '24-04-18_butterfly' in cache.s3_path, "Cache S3 path should have project"
    
    # The local path should NOT be an S3 URL
    assert not str(cache.local_path).startswith('s3://'), "Cache local path should not be S3 URL"
    assert '24-04-18_butterfly' in str(cache.local_path), "Cache local path should have project folder"


if __name__ == '__main__':
    # Run with verbose output to see print statements
    pytest.main([__file__, '-v', '-s', '--tb=short'])
