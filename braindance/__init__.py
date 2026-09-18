import os
import pathlib
from .config import (
    get_data_dir, set_data_dir,
    get_catalog_path, set_catalog_path,
    get_auto_extract_spike_info, set_auto_extract_spike_info
)


# Package metadata
__version__ = "0.1.9"

def get_module_path():
    """
    Returns the path to the braindance module.
    
    Returns:
        pathlib.Path: Path to the braindance module
    """
    return pathlib.Path(__file__).parent.resolve()

def get_rt_sort_path():
    """
    Returns the path to the RT-Sort detection models.
    
    Returns:
        pathlib.Path: Path to the RT-Sort detection models directory
    """
    return get_module_path() / "core" / "spikedetector" / "detection_models" / "mea"

def get_data_path():
    """
    Returns the path to the package's data directory.
    
    Returns:
        pathlib.Path: Path to the data directory
    """
    return get_module_path() / "data"

# Create a dictionary of available resources and their paths
PACKAGE_RESOURCES = {
    "rt_sort_detection_models": get_rt_sort_path(),
    "module_path": get_module_path(),
    "data_path": get_data_path()
}

def get_resource_path(resource_name):
    """
    Returns the path to a specific resource in the package.
    
    Args:
        resource_name (str): Name of the resource
        
    Returns:
        pathlib.Path: Path to the resource
        
    Raises:
        ValueError: If the resource is not found
    """
    if resource_name in PACKAGE_RESOURCES:
        return PACKAGE_RESOURCES[resource_name]
    else:
        raise ValueError(f"Resource '{resource_name}' not found. Available resources: {list(PACKAGE_RESOURCES.keys())}")



class Config:
    @property
    def data_dir(self):
        return get_data_dir()

_config = Config()

# Export as module-level attribute
def __getattr__(name):
    if name == 'data_dir':
        return _config.data_dir
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


data_dir = _config.data_dir
# # Also export the setter
# __all__ = ['data_dir', 'set_data_dir']
