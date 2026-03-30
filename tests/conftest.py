"""Shared fixtures for the BrainDance test suite."""
import pytest
import numpy as np
import os
import tempfile


@pytest.fixture
def tmp_dir(tmp_path):
    """Provides a temporary directory for test artifacts."""
    return tmp_path


@pytest.fixture
def sample_config_file(tmp_path):
    """Creates a minimal Maxwell .cfg file for testing Config parsing.

    Format: channel/electrode/x/y pairs separated by semicolons,
    all joined in a single whitespace-delimited token.
    """
    config_content = "0/100/1.0/2.0;1/101/3.0/4.0;2/102/5.0/6.0;"
    cfg_path = tmp_path / "test_config.cfg"
    cfg_path.write_text(config_content)
    return str(cfg_path)


@pytest.fixture
def mock_env():
    """Creates a lightweight mock of BaseEnv for phase testing."""

    class _MockEnv:
        def __init__(self):
            self.save_dir = tempfile.mkdtemp()
            self.save_file = os.path.join(self.save_dir, "mock_recording")
            self.step_count = 0
            self.reset_count = 0
            self.close_count = 0
            self.last_action = None
            self.last_tag = None
            self.stim_units = [type("Unit", (), {"power_up": lambda self, v: self})() for _ in range(4)]
            self.stim_electrodes = [0, 1, 2, 3]

        def step(self, action=None, tag=None, buffer_size=None):
            self.step_count += 1
            self.last_action = action
            self.last_tag = tag
            return [], False

        def reset(self):
            self.reset_count += 1
            self.save_file = os.path.join(self.save_dir, f"mock_recording_{self.reset_count}")

        def close(self):
            self.close_count += 1

        def disconnect_all(self):
            pass

        def connect_units(self, units=None, inds=None):
            pass

    return _MockEnv()
