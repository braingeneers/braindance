"""
Test S3 Path Construction for Different Catalog Configurations

This test suite verifies that S3 path construction works correctly for:
- Standard S3 buckets (s3://braingeneers/braindance/)
- Custom S3 buckets (s3://braingeneersdev/asrobbin/braindance_data/)
- Different experiment path formats (exp1, exp1/exp1_cont_95)
- Edge cases (missing fields, trailing slashes, etc.)
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
import os

from braindance.utils.data_manager.utils.data_loading.recording import Recording


class TestS3PathConstruction:
    """Test S3 path construction with different catalog configurations."""
    
    # ==================== Standard S3 Bucket Tests ====================
    
    def test_standard_bucket_spike_path(self):
        """Test spike data S3 path construction for standard braingeneers bucket."""
        row = pd.Series({
            'base_path': 's3://braingeneers/braindance/25-02-25_busybees/',
            'chip': '25123ic',
            'experiment': 'exp1/exp1_cont_95',
            'proj': '25-02-25_busybees'
        })
        rec = Recording(row)
        
        s3_path = rec._construct_s3_spike_path()
        expected = 's3://braingeneers/braindance/25-02-25_busybees/25123ic/exp1/spike_data/exp1_cont_95_spike_data.pkl'
        assert s3_path == expected, f"Expected {expected}, got {s3_path}"
    
    def test_standard_bucket_stim_log_path(self):
        """Test stim log S3 path construction for standard braingeneers bucket."""
        row = pd.Series({
            'base_path': 's3://braingeneers/braindance/25-02-25_busybees/',
            'chip': '25123ic',
            'experiment': 'exp1/exp1_cont_95',
            'proj': '25-02-25_busybees'
        })
        rec = Recording(row)
        
        s3_path = rec._construct_s3_stim_log_path()
        expected = 's3://braingeneers/braindance/25-02-25_busybees/25123ic/exp1/exp1_cont_95_log.csv'
        assert s3_path == expected, f"Expected {expected}, got {s3_path}"
    
    def test_standard_bucket_results_cache_path(self):
        """Test results cache S3 path construction for standard braingeneers bucket."""
        row = pd.Series({
            'base_path': 's3://braingeneers/braindance/25-02-25_busybees/',
            'chip': '25123ic',
            'experiment': 'exp1/exp1_cont_95',
            'proj': '25-02-25_busybees'
        })
        rec = Recording(row)
        
        # Access cache property to trigger construction
        cache = rec.cache
        expected = 's3://braingeneers/braindance/25-02-25_busybees/25123ic/exp1_cont_95/results'
        assert cache.s3_path == expected, f"Expected {expected}, got {cache.s3_path}"
    
    def test_standard_bucket_game_log_path(self):
        """Test game log S3 path construction for standard braingeneers bucket."""
        row = pd.Series({
            'base_path': 's3://braingeneers/braindance/24-04-18_butterfly/',
            'chip': '23133',
            'experiment': 'exp1/exp1_cartpole_long_4',
            'proj': '24-04-18_butterfly'
        })
        rec = Recording(row)
        
        s3_path = rec._construct_s3_game_log_path()
        expected = 's3://braingeneers/braindance/24-04-18_butterfly/23133/exp1/exp1_cartpole_long_4_game_log.csv'
        assert s3_path == expected, f"Expected {expected}, got {s3_path}"
    
    # ==================== Custom S3 Bucket Tests ====================
    
    def test_custom_bucket_spike_path(self):
        """Test spike data S3 path construction for custom braingeneersdev bucket."""
        row = pd.Series({
            'base_path': 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/',
            'chip': '23133',
            'experiment': 'exp1/exp1_cartpole_long_4',
            'proj': '24-04-18_butterfly'
        })
        rec = Recording(row)
        
        s3_path = rec._construct_s3_spike_path()
        expected = 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/23133/exp1/spike_data/exp1_cartpole_long_4_spike_data.pkl'
        assert s3_path == expected, f"Expected {expected}, got {s3_path}"
    
    def test_custom_bucket_stim_log_path(self):
        """Test stim log S3 path construction for custom braingeneersdev bucket."""
        row = pd.Series({
            'base_path': 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/',
            'chip': '23133',
            'experiment': 'exp1/exp1_cartpole_long_7',
            'proj': '24-04-18_butterfly'
        })
        rec = Recording(row)
        
        s3_path = rec._construct_s3_stim_log_path()
        expected = 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/23133/exp1/exp1_cartpole_long_7_log.csv'
        assert s3_path == expected, f"Expected {expected}, got {s3_path}"
    
    def test_custom_bucket_results_cache_path(self):
        """Test results cache S3 path construction for custom braingeneersdev bucket."""
        row = pd.Series({
            'base_path': 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/',
            'chip': '23133',
            'experiment': 'exp1/exp1_cartpole_long_4',
            'proj': '24-04-18_butterfly'
        })
        rec = Recording(row)
        
        # Access cache property to trigger construction
        cache = rec.cache
        expected = 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/23133/exp1_cartpole_long_4/results'
        assert cache.s3_path == expected, f"Expected {expected}, got {cache.s3_path}"
    
    # ==================== Nested Experiment Path Tests ====================
    
    def test_nested_experiment_path_exp_base_extraction(self):
        """Test that exp_base is correctly extracted from nested experiment paths."""
        row = pd.Series({
            'base_path': 's3://braingeneers/braindance/25-02-25_busybees/',
            'chip': '25123ic',
            'experiment': 'exp1/exp1_cont_95',
            'proj': '25-02-25_busybees'
        })
        rec = Recording(row)
        
        # Spike path should use exp_base (exp1) for the folder
        s3_path = rec._construct_s3_spike_path()
        assert '/exp1/spike_data/' in s3_path, f"Expected exp_base 'exp1' in path, got {s3_path}"
    
    def test_nested_experiment_path_exp_name_extraction(self):
        """Test that exp_name is correctly extracted from nested experiment paths."""
        row = pd.Series({
            'base_path': 's3://braingeneers/braindance/25-02-25_busybees/',
            'chip': '25123ic',
            'experiment': 'exp1/exp1_cont_95',
            'proj': '25-02-25_busybees'
        })
        rec = Recording(row)
        
        # Spike path should use exp_name (exp1_cont_95) for the filename
        s3_path = rec._construct_s3_spike_path()
        assert 'exp1_cont_95_spike_data.pkl' in s3_path, f"Expected exp_name in filename, got {s3_path}"
    
    def test_non_nested_experiment_path(self):
        """Test S3 path construction when experiment is not nested (e.g., just 'exp1')."""
        row = pd.Series({
            'base_path': 's3://braingeneers/braindance/25-02-25_busybees/',
            'chip': '25123ic',
            'experiment': 'exp1',
            'proj': '25-02-25_busybees'
        })
        rec = Recording(row)
        
        s3_path = rec._construct_s3_spike_path()
        expected = 's3://braingeneers/braindance/25-02-25_busybees/25123ic/exp1/spike_data/exp1_spike_data.pkl'
        assert s3_path == expected, f"Expected {expected}, got {s3_path}"
    
    # ==================== Edge Cases ====================
    
    def test_missing_base_path(self):
        """Test that None is returned when base_path is missing."""
        row = pd.Series({
            'chip': '25123ic',
            'experiment': 'exp1/exp1_cont_95',
            'proj': '25-02-25_busybees'
        })
        rec = Recording(row)
        
        s3_path = rec._construct_s3_spike_path()
        assert s3_path is None, f"Expected None for missing base_path, got {s3_path}"
    
    def test_missing_chip(self):
        """Test that None is returned when chip is missing."""
        row = pd.Series({
            'base_path': 's3://braingeneers/braindance/25-02-25_busybees/',
            'experiment': 'exp1/exp1_cont_95',
            'proj': '25-02-25_busybees'
        })
        rec = Recording(row)
        
        s3_path = rec._construct_s3_spike_path()
        assert s3_path is None, f"Expected None for missing chip, got {s3_path}"
    
    def test_missing_experiment(self):
        """Test that None is returned when experiment is missing."""
        row = pd.Series({
            'base_path': 's3://braingeneers/braindance/25-02-25_busybees/',
            'chip': '25123ic',
            'proj': '25-02-25_busybees'
        })
        rec = Recording(row)
        
        s3_path = rec._construct_s3_spike_path()
        assert s3_path is None, f"Expected None for missing experiment, got {s3_path}"
    
    def test_base_path_with_trailing_slash(self):
        """Test that trailing slash in base_path is handled correctly."""
        row = pd.Series({
            'base_path': 's3://braingeneers/braindance/25-02-25_busybees/',  # With trailing slash
            'chip': '25123ic',
            'experiment': 'exp1/exp1_cont_95',
            'proj': '25-02-25_busybees'
        })
        rec = Recording(row)
        
        s3_path = rec._construct_s3_spike_path()
        # Should not have double slashes
        assert '//' not in s3_path.replace('s3://', ''), f"Found double slash in path: {s3_path}"
    
    def test_base_path_without_trailing_slash(self):
        """Test that base_path without trailing slash is handled correctly."""
        row = pd.Series({
            'base_path': 's3://braingeneers/braindance/25-02-25_busybees',  # No trailing slash
            'chip': '25123ic',
            'experiment': 'exp1/exp1_cont_95',
            'proj': '25-02-25_busybees'
        })
        rec = Recording(row)
        
        s3_path = rec._construct_s3_spike_path()
        expected = 's3://braingeneers/braindance/25-02-25_busybees/25123ic/exp1/spike_data/exp1_cont_95_spike_data.pkl'
        assert s3_path == expected, f"Expected {expected}, got {s3_path}"
    
    # ==================== Path Consistency Tests ====================
    
    def test_spike_path_uses_spike_data_subfolder(self):
        """Test that spike data paths use the /spike_data/ subfolder."""
        row = pd.Series({
            'base_path': 's3://braingeneers/braindance/25-02-25_busybees/',
            'chip': '25123ic',
            'experiment': 'exp1/exp1_cont_95',
            'proj': '25-02-25_busybees'
        })
        rec = Recording(row)
        
        s3_path = rec._construct_s3_spike_path()
        assert '/spike_data/' in s3_path, f"Expected /spike_data/ in spike path, got {s3_path}"
    
    def test_stim_log_path_in_exp_base_folder(self):
        """Test that stim log paths are in the exp_base folder (not spike_data subfolder)."""
        row = pd.Series({
            'base_path': 's3://braingeneers/braindance/25-02-25_busybees/',
            'chip': '25123ic',
            'experiment': 'exp1/exp1_cont_95',
            'proj': '25-02-25_busybees'
        })
        rec = Recording(row)
        
        s3_path = rec._construct_s3_stim_log_path()
        # Should be in exp1 folder, not spike_data subfolder
        assert '/exp1/' in s3_path, f"Expected /exp1/ folder in stim log path, got {s3_path}"
        assert '/spike_data/' not in s3_path, f"Stim log should not be in spike_data folder, got {s3_path}"
    
    def test_results_path_structure(self):
        """Test that results paths follow the expected structure."""
        row = pd.Series({
            'base_path': 's3://braingeneers/braindance/25-02-25_busybees/',
            'chip': '25123ic',
            'experiment': 'exp1/exp1_cont_95',
            'proj': '25-02-25_busybees'
        })
        rec = Recording(row)
        
        cache = rec.cache
        s3_path = cache.s3_path
        # Results should be at {base_path}/{chip}/{exp_name}/results
        assert s3_path.endswith('/exp1_cont_95/results'), f"Expected results path to end with /exp1_cont_95/results, got {s3_path}"
    
    def test_pattern_log_path(self):
        """Test pattern log S3 path construction."""
        row = pd.Series({
            'base_path': 's3://braingeneers/braindance/25-02-25_busybees/',
            'chip': '25123ic',
            'experiment': 'exp1/exp1_cont_95',
            'proj': '25-02-25_busybees'
        })
        rec = Recording(row)
        
        s3_path = rec._construct_s3_pattern_log_path()
        expected = 's3://braingeneers/braindance/25-02-25_busybees/25123ic/exp1/exp1_cont_95_pattern_log.csv'
        assert s3_path == expected, f"Expected {expected}, got {s3_path}"
    
    def test_reward_log_path(self):
        """Test reward log S3 path construction."""
        row = pd.Series({
            'base_path': 's3://braingeneers/braindance/25-02-25_busybees/',
            'chip': '25123ic',
            'experiment': 'exp1/exp1_cont_95',
            'proj': '25-02-25_busybees'
        })
        rec = Recording(row)
        
        s3_path = rec._construct_s3_reward_log_path()
        expected = 's3://braingeneers/braindance/25-02-25_busybees/25123ic/exp1/exp1_cont_95_reward_log.csv'
        assert s3_path == expected, f"Expected {expected}, got {s3_path}"


class TestLocalPathResolution:
    """Test that S3 URLs in catalogs result in proper local file paths for caching."""
    
    def test_s3_catalog_uses_local_cache_directory(self):
        """Test that S3 base_path in catalog results in local cache directory, not S3 Path object."""
        row = pd.Series({
            'base_path': 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/',
            'chip': '23133',
            'experiment': 'exp1/exp1_cartpole_long_4',
            'proj': '24-04-18_butterfly'
        })
        rec = Recording(row)
        
        # Trigger path resolution
        local_path = rec._spikes_path
        
        # Verify local path is NOT an S3 URL
        assert local_path is not None, "Local spike path should not be None"
        assert not str(local_path).startswith('s3://'), f"Local path should not be S3 URL, got {local_path}"
    
    def test_s3_catalog_includes_project_in_local_path(self):
        """Test that local paths include the project folder when using S3 catalog."""
        row = pd.Series({
            'base_path': 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/',
            'chip': '23133',
            'experiment': 'exp1/exp1_cartpole_long_7',
            'proj': '24-04-18_butterfly'
        })
        rec = Recording(row)
        
        # Trigger path resolution
        local_path = rec._spikes_path
        
        # Verify project folder is included
        assert '24-04-18_butterfly' in str(local_path), f"Expected project '24-04-18_butterfly' in local path, got {local_path}"
    
    def test_s3_catalog_includes_chip_in_local_path(self):
        """Test that local paths include the chip folder when using S3 catalog."""
        row = pd.Series({
            'base_path': 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/',
            'chip': '23133',
            'experiment': 'exp1/exp1_cartpole_long_4',
            'proj': '24-04-18_butterfly'
        })
        rec = Recording(row)
        
        # Trigger path resolution
        local_path = rec._spikes_path
        
        # Verify chip folder is included
        assert '23133' in str(local_path), f"Expected chip '23133' in local path, got {local_path}"
    
    def test_s3_catalog_includes_experiment_structure(self):
        """Test that local paths include proper experiment folder structure."""
        row = pd.Series({
            'base_path': 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/',
            'chip': '23133',
            'experiment': 'exp1/exp1_cartpole_long_4',
            'proj': '24-04-18_butterfly'
        })
        rec = Recording(row)
        
        # Trigger path resolution
        local_path = rec._spikes_path
        
        # Should include experiment structure
        # Could be legacy flat (chip/exp_name.pkl) or standard (chip/exp_name/exp_name.pkl)
        assert 'exp1_cartpole_long_4' in str(local_path), f"Expected experiment in local path, got {local_path}"
    
    def test_standard_s3_bucket_local_path_resolution(self):
        """Test local path resolution for standard braingeneers S3 bucket."""
        row = pd.Series({
            'base_path': 's3://braingeneers/braindance/25-02-25_busybees/',
            'chip': '25123ic',
            'experiment': 'exp1/exp1_cont_95',
            'proj': '25-02-25_busybees'
        })
        rec = Recording(row)
        
        # Trigger path resolution
        local_path = rec._spikes_path
        
        # Verify it's a local path, not S3
        assert not str(local_path).startswith('s3://'), f"Should be local path, got {local_path}"
        # Verify structure is preserved
        assert '25-02-25_busybees' in str(local_path), f"Expected project in path, got {local_path}"
        assert '25123ic' in str(local_path), f"Expected chip in path, got {local_path}"
    
    def test_local_filesystem_catalog_uses_provided_path(self):
        """Test that non-S3 base_path is used directly for local paths."""
        row = pd.Series({
            'base_path': '/custom/local/path/myproject/',
            'chip': '12345',
            'experiment': 'exp1/test_exp',
            'proj': 'myproject'
        })
        rec = Recording(row)
        
        # Trigger path resolution
        local_path = rec._spikes_path
        
        # Should use the provided local path
        assert '/custom/local/path/myproject/' in str(local_path), f"Expected custom path to be used, got {local_path}"
    
    def test_stim_log_local_path_with_s3_catalog(self):
        """Test that stim log local paths are correct when using S3 catalog."""
        row = pd.Series({
            'base_path': 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/',
            'chip': '23133',
            'experiment': 'exp1/exp1_cartpole_long_4',
            'proj': '24-04-18_butterfly'
        })
        rec = Recording(row)
        
        # Trigger path resolution
        local_path = rec._stim_log_path
        
        # Verify it's a local path
        assert not str(local_path).startswith('s3://'), f"Stim log path should be local, got {local_path}"
        # Verify structure
        assert '24-04-18_butterfly' in str(local_path), f"Expected project in path, got {local_path}"
        assert '23133' in str(local_path), f"Expected chip in path, got {local_path}"
    
    def test_results_cache_local_path_with_s3_catalog(self):
        """Test that results cache uses correct local path when catalog has S3 URL."""
        row = pd.Series({
            'base_path': 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/',
            'chip': '23133',
            'experiment': 'exp1/exp1_cartpole_long_4',
            'proj': '24-04-18_butterfly'
        })
        rec = Recording(row)
        
        # Trigger path resolution
        results_path = rec._results_path
        
        # Verify it's a local path
        assert not str(results_path).startswith('s3://'), f"Results path should be local, got {results_path}"
        # Verify structure is preserved
        assert '24-04-18_butterfly' in str(results_path), f"Expected project in results path, got {results_path}"
    
    def test_both_s3_and_local_paths_are_different(self):
        """Test that S3 URL and local path are different for the same recording."""
        row = pd.Series({
            'base_path': 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/',
            'chip': '23133',
            'experiment': 'exp1/exp1_cartpole_long_4',
            'proj': '24-04-18_butterfly'
        })
        rec = Recording(row)
        
        # Get S3 path
        s3_path = rec._construct_s3_spike_path()
        
        # Get local path
        local_path = rec._spikes_path
        
        # They should be different!
        assert s3_path != str(local_path), "S3 path and local path should be different"
        # S3 should start with s3://
        assert s3_path.startswith('s3://'), f"S3 path should start with s3://, got {s3_path}"
        # Local should NOT start with s3://
        assert not str(local_path).startswith('s3://'), f"Local path should not start with s3://, got {local_path}"
    
    def test_project_extracted_from_base_path_when_proj_missing(self):
        """Test that project is extracted from S3 base_path when proj column is missing."""
        row = pd.Series({
            'base_path': 's3://braingeneersdev/asrobbin/braindance_data/24-04-18_butterfly/',
            'chip': '23133',
            'experiment': 'exp1/exp1_cartpole_long_4',
            # NOTE: No 'proj' column!
        })
        rec = Recording(row)
        
        # Trigger path resolution
        local_path = rec._spikes_path
        
        # Should extract project from base_path
        assert '24-04-18_butterfly' in str(local_path), f"Expected project extracted from base_path, got {local_path}"
        # Should include chip
        assert '23133' in str(local_path), f"Expected chip in path, got {local_path}"


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
