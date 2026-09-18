"""
Integration tests for S3 caching flow.
These tests require local data on Hunter's SSD and S3 access.
"""
import pytest
import time
from pathlib import Path
import pandas as pd
from braindance.utils.data_manager.utils.data_loading.recording import load_recording
from braindance.utils.data_manager.utils.data_loading.results_cache import get_auto_upload_enabled

# Configuration
LOCAL_DATA_PATH = Path("/Volumes/hunter_ssd/busy_bee/data")
PROJ = "25-02-25_busybees"
CHIP = "25123ic"
TEST_EXPERIMENTS = [
    "exp1_cont_95",
    "exp1_cont_96",
    "exp1_cont_97",
]
BIN_MS = 20

def has_hunter_ssd():
    return (LOCAL_DATA_PATH / PROJ / CHIP).exists()

@pytest.mark.integration
@pytest.mark.skipif(not has_hunter_ssd(), reason="Hunter's SSD not connected or data missing")
class TestS3CachingFlow:
    """
    Validates the complete flow:
    1. Load spike data from local path
    2. Compute binned firing rates
    3. Cache results locally
    4. Upload cached results to S3 (if enabled)
    """

    @pytest.fixture(scope="class")
    def chip_dir(self):
        return LOCAL_DATA_PATH / PROJ / CHIP

    @pytest.fixture(scope="class")
    def valid_experiments(self, chip_dir):
        found = []
        for exp in TEST_EXPERIMENTS:
            spike_file = chip_dir / f"{exp}_spike_data.pkl"
            if spike_file.exists():
                found.append(exp)
        if not found:
            pytest.skip("No test experiments found on local SSD")
        return found

    @pytest.fixture(scope="class")
    def recording(self, valid_experiments):
        exp = valid_experiments[0]
        return load_recording(
            proj=PROJ,
            chip=CHIP,
            experiment=exp,
            base_path=LOCAL_DATA_PATH,
            legacy_flat=True
        )

    def test_spikes_load(self, recording):
        """Test spikes lazy loading."""
        spikes = recording.spikes
        assert spikes is not None
        assert spikes.N > 0
        assert spikes.length > 0

    def test_binned_fr_computation(self, recording):
        """Test binned firing rate computation."""
        binned = recording.get_binned_fr(bin_ms=BIN_MS)
        assert binned is not None
        assert 'rates' in binned
        assert binned['rates'].shape[0] > 0

    def test_local_cache_speedup(self, recording):
        """Test that second call is faster due to local cache."""
        # Clear in-memory cache
        recording.clear_cache()
        
        # First call (from disk cache)
        start = time.time()
        _ = recording.get_binned_fr(bin_ms=BIN_MS)
        time1 = time.time() - start
        
        # Second call (from memory/disk cache again, should be very fast)
        start = time.time()
        _ = recording.get_binned_fr(bin_ms=BIN_MS)
        time2 = time.time() - start
        
        # This is more of a smoke test than a strict assertion
        # because disk I/O can be variable, but time2 should be negligible
        assert time2 < time1 or time2 < 0.1

    def test_s3_path_construction(self, recording):
        """Verify S3 path construction identifier."""
        assert recording.identifier.startswith(f"{PROJ}/{CHIP}")
        # Check that we can construct the S3 path if needed
        s3_path = recording._construct_s3_spike_path()
        assert "s3://" in s3_path
        assert PROJ in s3_path
        assert CHIP in s3_path

    def test_auto_upload_trigger(self, recording):
        """Check if auto-upload is triggered when enabled."""
        auto_upload = get_auto_upload_enabled()
        if not auto_upload:
            pytest.skip("BRAINDANCE_AUTO_UPLOAD not enabled")
        
        cache = recording.cache
        assert cache is not None
        manifest = cache.manifest
        assert len(manifest.list_results()) > 0
