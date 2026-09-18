"""
ML-specific utilities tests for Data Manager.

Contains tests for tokenizers, dataset-related path resolution, and metadata preservation
critical for machine learning workflows.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
import os
import numpy as np

from braindance.utils.data_manager.utils.data_loading.recording import Recording

class TestTokenizerRobustness:
    """Test tokenizer robustness to malformed stimulus data."""
    
    def test_nested_electrode_ids(self):
        """Test that TimeBinTokenizer handles nested electrode ID lists [[123]]."""
        # Note: TimeBinTokenizer should be imported from the predictor package in real usage
        # but here we test the logic if possible or mock the requirement.
        # Since this test file is in braindance, we might need to skip if predictor isn't available
        # or just test the parsing logic in Recording if it was there.
        # Actually, the original test imported from proj.predictor...
        
        try:
            from proj.predictor.nrp.scripts.tokenizers.time_bin_tokenizer import TimeBinTokenizer
            from proj.predictor.nrp.scripts.tokenizers.token_types import NeuralToken, TokenType
        except ImportError:
            pytest.skip("Predictor tokenizers not available")
        
        tokenizer = TimeBinTokenizer(bin_ms=20, max_neurons=10)
        
        # Create a token with a nested electrode ID list
        token = NeuralToken(
            token_type=TokenType.NEURAL_ACTIVITY,
            time_bin=0,
            organoid_id=0,
            population_vector=np.zeros(10),
            stim_event={
                'time_mod': 10.0,
                'electrode_ids': [[123]],  # NESTED LIST (The bug!)
                'amplitude': 400.0
            }
        )
        
        # This should NOT fail now
        encoded = tokenizer.encode_for_model([token])
        
        # Verify the stim feature was captured (dim max_neurons + 1 is electrode_norm)
        # Features are: [stim_present, electrode_norm, time_offset, bins_since_last]
        # input_bins: [seq_len, max_neurons + 4]
        electrode_norm = encoded['input_bins'][0, 10 + 1].item()
        assert electrode_norm == pytest.approx(123 / 26000.0), f"Expected {123 / 26000.0}, got {electrode_norm}"
        
    def test_deeply_nested_electrode_ids(self):
        """Test that TimeBinTokenizer handles deeply nested electrode ID lists [[[123]]]."""
        try:
            from proj.predictor.nrp.scripts.tokenizers.time_bin_tokenizer import TimeBinTokenizer
            from proj.predictor.nrp.scripts.tokenizers.token_types import NeuralToken, TokenType
        except ImportError:
            pytest.skip("Predictor tokenizers not available")
        
        tokenizer = TimeBinTokenizer(bin_ms=20, max_neurons=10)
        
        token = NeuralToken(
            token_type=TokenType.NEURAL_ACTIVITY,
            time_bin=0,
            organoid_id=0,
            population_vector=np.zeros(10),
            stim_event={
                'time_mod': 10.0,
                'electrode_ids': [[[456]]],  # DEEPLY NESTED
                'amplitude': 400.0
            }
        )
        
        encoded = tokenizer.encode_for_model([token])
        electrode_norm = encoded['input_bins'][0, 10 + 1].item()
        assert electrode_norm == pytest.approx(456 / 26000.0), f"Expected {456 / 26000.0}, got {electrode_norm}"

class TestMLPathResolution:
    """Tests for path resolution specific to ML workflows (e.g. /app/data/cache)."""
    
    def test_s3_catalog_with_hardcoded_app_data_cache(self):
        """Test the default /app/data/cache behavior when BRAINDANCE_DATA_DIR is set."""
        import os
        row = pd.Series({
            'base_path': 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/',
            'chip': '23133',
            'experiment': 'exp1/exp1_cartpole_long_4',
            'proj': '24-04-18_butterfly'
        })
        rec = Recording(row) # No base_path override
        
        local_path = rec._spikes_path
        
        assert local_path is not None
        # In Container/ML environment where BRAINDANCE_DATA_DIR is set, verify it's used
        expected_base = os.getenv('BRAINDANCE_DATA_DIR')
        if expected_base:
            assert str(local_path).startswith(expected_base), f"Expected path to start with {expected_base}, got {local_path}"

    def test_metadata_loss_via_identifiers(self):
        """
        Test that from_identifiers with local base_path doesn't create S3 paths.
        """
        proj = '24-04-18_butterfly'
        chip = '23133'
        experiment = 'exp1/exp1_cartpole_long_4'
        local_cache = Path('/app/data/cache')
        
        # It passes the local path as base_path
        rec_broken = Recording.from_identifiers(proj, chip, experiment, base_path=local_cache)
        
        # The S3 path should be None because the base_path is local, not S3
        assert rec_broken._construct_s3_spike_path() is None
        
    def test_metadata_preservation_via_row(self):
        """
        VERIFY FIX: Initializing from full row preserves S3 metadata.
        This ensures that S3 downloads work even when using a local cache directory.
        """
        row = pd.Series({
            'base_path': 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/',
            'chip': '23133',
            'experiment': 'exp1/exp1_cartpole_long_4',
            'proj': '24-04-18_butterfly'
        })
        local_cache = Path('/app/data/cache')
        
        # Initialize directly from row + local cache override (FIXED)
        rec_fixed = Recording(row, base_path=local_cache)
        
        # S3 path remains valid because the row contains the original s3:// URL!
        s3_path = rec_fixed._construct_s3_spike_path()
        assert s3_path is not None
        assert s3_path.startswith('s3://')
        assert str(rec_fixed._spikes_path).startswith('/app/data/cache')

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
