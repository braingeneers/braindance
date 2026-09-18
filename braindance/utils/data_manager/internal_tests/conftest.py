import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from braindance.utils.data_manager import Recording, RecordingCatalog

try:
    from spikelab import SpikeData
except ImportError:
    # Minimal mock for SpikeData if module not installed
    class SpikeData:
        def __init__(self, train, N, length):
            self.train = train
            self.N = N
            self.length = length

@pytest.fixture
def mock_spike_data():
    """Create a real SpikeData object with synthetic spikes for testing."""
    # 5 neurons, 10 seconds of data
    N = 5
    duration_ms = 10000.0
    
    # Generate some synthetic spikes
    train = []
    for _ in range(N):
        # 10 spontaneous spikes
        spontan = np.random.uniform(0, duration_ms, 10)
        # Add a "burst" at 2000ms
        burst = np.array([1990, 2000, 2010, 2020, 2030])
        # Add another "burst" at 5000ms
        burst2 = np.array([4980, 4990, 5000, 5010, 5020, 5030, 5040, 5050])
        
        neuron_spikes = np.sort(np.concatenate([spontan, burst, burst2]))
        train.append(neuron_spikes)
        
    return SpikeData(train, N=N, length=duration_ms)

@pytest.fixture
def mock_stim_log():
    """Create a mock stimulation log (times in seconds)."""
    return pd.DataFrame({
        'time': [1.0, 3.0, 6.0, 8.0],
        'stim_electrodes': ['[1, 2]', '[3]', 1, '[2, 3]'],
        'time_mod': [1.0, 3.0, 6.0, 8.0]
    })

@pytest.fixture
def mock_game_log():
    """Create a mock game log for RL experiments."""
    return pd.DataFrame({
        'time': np.linspace(0, 10, 100), # seconds
        'pole_angle': np.sin(np.linspace(0, 10, 100)),
        'reward': np.random.choice([0, 1], 100),
        'action': np.random.choice([0, 1], 100)
    })

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for data files."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)

@pytest.fixture
def mock_catalog_df():
    """Sample catalog DataFrame."""
    return pd.DataFrame({
        'proj': ['test_proj'],
        'chip': ['test_chip'],
        'experiment': ['test_exp'],
        'type': ['stimulated'],
        'freq': [0],
        'baseline': [True],
        'data_path': ['test_chip/test_exp_spike_data.pkl'],
        'log_path': ['test_chip/test_exp_stim_log.csv']
    })

@pytest.fixture
def mock_recording(mock_catalog_df, mock_spike_data, mock_stim_log, temp_data_dir):
    """Create a Recording object with mocked data access."""
    import pickle
    
    # Setup temporary directory structure for results
    results_dir = temp_data_dir / "results"
    results_dir.mkdir()
    
    # Create spike data directory and file
    spike_dir = temp_data_dir / "test_chip" / "test_exp" / "spike_data"
    spike_dir.mkdir(parents=True, exist_ok=True)
    spike_file = spike_dir / "test_exp_spike_data.pkl"
    
    # Save the mock spike data to disk
    with open(spike_file, 'wb') as f:
        pickle.dump(mock_spike_data, f)
    
    # Update row with local paths
    row = mock_catalog_df.iloc[0].copy()
    row['results_path'] = str(results_dir)
    row['base_path'] = str(temp_data_dir)
    
    rec = Recording(row)
    # Manually inject stim_log and game_log (spike data will be loaded from file)
    rec._stim_log = mock_stim_log
    rec._game_log = None
    
    return rec
