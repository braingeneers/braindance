import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, PropertyMock

def test_load_game_log_local(mock_recording, mock_game_log, tmp_path):
    """Verify loading game log from local CSV."""
    # Setup
    log_path = tmp_path / 'game_log.csv'
    mock_game_log.to_csv(log_path, index=False)
    
    with patch.object(type(mock_recording), '_game_log_path', new_callable=PropertyMock) as mock_path:
        mock_path.return_value = log_path
        # Action
        df = mock_recording.game_log
        # Verification
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(mock_game_log)
        assert 'pole_angle' in df.columns

def test_game_log_s3_fallback(mock_recording, tmp_path):
    """Verify S3 download fallback when local log is missing."""
    # Setup
    missing_path = tmp_path / 'missing_game_log.csv'
    # DO NOT touch the file initially; we want it to be missing locally to trigger S3 fallback
    
    # Create a side effect that actually creates the file when download is called
    def mock_download(s3_path, local_path):
        # Simulate successful S3 download by creating a file > 100 bytes
        # (the code rejects files < 100 bytes as empty)
        df = pd.DataFrame({
            'time': list(range(20)),
            'pole_angle': [0.1] * 20,
            'reward': [0, 1] * 10,
            'action': [0, 1, 0, 1] * 5
        })
        df.to_csv(local_path, index=False)
        return True
    
    with patch.object(type(mock_recording), '_game_log_path', new_callable=PropertyMock) as mock_path:
        mock_path.return_value = missing_path
        # Mock the S3 downloader to actually create the file
        with patch.object(mock_recording, '_download_from_s3', side_effect=mock_download) as mock_dl:
            
            # Action
            df = mock_recording.game_log
            
            # Verification
            mock_dl.assert_called()
            assert df is not None
            assert 'reward' in df.columns

def test_load_empty_game_log(mock_recording):
    """Verify handling of empty or missing game log."""
    with patch.object(type(mock_recording), '_game_log_path', new_callable=PropertyMock) as mock_path:
        mock_path.return_value = None
        assert mock_recording.game_log is None
