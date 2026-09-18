"""
Data Context System for Experiment Framework V3

Provides dynamic attribute access and type-safe data storage for passing
data between experiment phases.
"""
from typing import Dict, Any, Optional, List
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd


class DataContext:
    """
    Dynamic data storage with attribute access.
    
    Example:
        exp.data.neurons = [1, 2, 3]
        print(exp.data.neurons)  # [1, 2, 3]
    """
    
    def __init__(self, overwrite_existing: bool = False):
        self._data: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict] = {}  # Track data types and sources
        self._overwrite_existing = overwrite_existing
        
    def __setattr__(self, name: str, value: Any):
        if name.startswith('_'):
            # Internal attributes
            super().__setattr__(name, value)
        else:
            # User data - check overwrite policy
            if not self._overwrite_existing and name in self._data:
                print(f"⚠️  Warning: Key '{name}' already exists. Skipping assignment (overwrite_existing=False)")
                return
                
            self._data[name] = value
            self._metadata[name] = {
                'type': type(value).__name__,
                'shape': getattr(value, 'shape', None),
                'length': len(value) if hasattr(value, '__len__') else None
            }
    
    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"No data found for key '{name}'")
    
    def __contains__(self, name: str) -> bool:
        return name in self._data
    
    def __repr__(self) -> str:
        items = []
        for key, meta in self._metadata.items():
            type_str = meta['type']
            if meta['shape']:
                type_str += f" shape={meta['shape']}"
            elif meta['length'] is not None:
                type_str += f" len={meta['length']}"
            items.append(f"{key}: {type_str}")
        return f"DataContext({', '.join(items)})"
    
    def keys(self) -> List[str]:
        """Get all data keys."""
        return list(self._data.keys())
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get data with default value."""
        return self._data.get(key, default)
    
    def has_data(self) -> bool:
        """Check if any data is stored."""
        return len(self._data) > 0
    
    def data_summary(self) -> Dict[str, str]:
        """Get a summary of stored data types and shapes."""
        summary = {}
        for key, meta in self._metadata.items():
            type_str = meta['type']
            if meta['shape']:
                type_str += f" shape={meta['shape']}"
            elif meta['length'] is not None:
                type_str += f" len={meta['length']}"
            summary[key] = type_str
        return summary
    
    def update(self, data: Dict[str, Any], overwrite: bool = None):
        """
        Update multiple data values at once.
        
        Args:
            data: Dictionary of key-value pairs to update
            overwrite: Override the instance's overwrite_existing setting for this operation
        """
        # Temporarily override overwrite setting if specified
        original_overwrite = self._overwrite_existing
        if overwrite is not None:
            self._overwrite_existing = overwrite
            
        try:
            for key, value in data.items():
                setattr(self, key, value)
        finally:
            # Restore original setting
            self._overwrite_existing = original_overwrite
    
    def set_overwrite_policy(self, overwrite_existing: bool):
        """Set the overwrite policy for future assignments."""
        self._overwrite_existing = overwrite_existing
    
    def get_overwrite_policy(self) -> bool:
        """Get the current overwrite policy."""
        return self._overwrite_existing
    
    def force_set(self, key: str, value: Any):
        """Force set a value, ignoring overwrite policy."""
        original_overwrite = self._overwrite_existing
        self._overwrite_existing = True
        try:
            setattr(self, key, value)
        finally:
            self._overwrite_existing = original_overwrite
    
    def clear(self):
        """Clear all data."""
        self._data.clear()
        self._metadata.clear()
    
    @staticmethod
    def _is_json_safe(value) -> bool:
        """Return True if *value* can be losslessly represented in JSON."""
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError, OverflowError):
            return False

    def save(self, path: Path, keys: Optional[List[str]] = None, overwrite_files: bool | None = None,
             strict: bool = False):
        """
        Save data context to disk.

        JSON-serializable scalars (strings, numbers, bools, simple lists/dicts)
        are collected into a single ``results.json`` file.  Binary data
        (arrays, DataFrames, pickled objects) are saved as individual files.
        
        Args:
            path: Directory to save data
            keys: Specific keys to save (None = save all)
            overwrite_files: Whether to overwrite existing files (defaults to instance overwrite policy)
            strict: Raise on missing keys or serialization failures. Experiment
                checkpoints use this so a partial save cannot count as success.
        """
        if overwrite_files is None:
            overwrite_files = self._overwrite_existing
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata and update it
        metadata_path = path / 'data_metadata.json'
        existing_metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    existing_metadata = json.load(f)
            except Exception as e:
                print(f"   Warning: Could not load existing metadata: {e}")
                existing_metadata = {}
        
        keys_to_save = keys or self._data.keys()
        keys_to_save = [key for key in keys_to_save if isinstance(key, str)]
        
        for key in keys_to_save:
            if key in self._metadata:
                existing_metadata[key] = self._metadata[key]
        
        with open(metadata_path, 'w') as f:
            json.dump(existing_metadata, f, indent=2)
        
        # ------ Partition keys into JSON-safe vs binary ------
        json_values = {}
        saved_files = {}
        
        for key in keys_to_save:
            if key not in self._data:
                if strict:
                    raise KeyError(f"Cannot save missing data key: {key}")
                continue
                
            value = self._data[key]

            # JSON-safe scalars / simple containers → results.json
            if self._is_json_safe(value):
                json_values[key] = value
                saved_files[key] = 'json'
                continue
            
            # Binary data → individual files
            if isinstance(value, np.ndarray):
                file_path = path / f"{key}.npy"
                file_type = 'numpy'
            elif isinstance(value, pd.DataFrame):
                file_path = path / f"{key}.pkl"
                file_type = 'pandas'
            elif isinstance(value, dict) and any(isinstance(v, np.ndarray) for v in value.values()):
                file_path = path / f"{key}.npz"
                file_type = 'numpy_dict'
            elif type(value).__name__ == 'RTSort':
                value.model = None
                file_path = path / f"{key}.pkl"
                file_type = 'rt_sort'
            else:
                file_path = path / f"{key}.pkl"
                file_type = 'pickle'
            
            if file_path.exists() and not overwrite_files:
                print(f"   ⚠️  Warning: File {file_path.name} already exists. Skipping (overwrite_files=False)")
                continue
            
            try:
                if file_type == 'numpy':
                    np.save(file_path, value)
                elif file_type == 'pandas':
                    value.to_pickle(file_path)
                elif file_type == 'numpy_dict':
                    np.savez(file_path, **value)
                else:
                    with open(file_path, 'wb') as f:
                        pickle.dump(value, f)
                
                saved_files[key] = file_type
                
            except Exception as e:
                if strict:
                    raise RuntimeError(f"Failed to persist data key {key!r}") from e
                print(f"   Error saving {key}: {e}")
                continue

        # ------ Write JSON-safe values into results.json ------
        if json_values:
            results_json_path = path / 'results.json'
            existing_json = {}
            if results_json_path.exists():
                try:
                    with open(results_json_path, 'r') as f:
                        existing_json = json.load(f)
                except Exception:
                    pass
            existing_json.update(json_values)
            with open(results_json_path, 'w') as f:
                json.dump(existing_json, f, indent=2)
        
        # ------ Update data_index.json ------
        index_path = path / 'data_index.json'
        existing_index = {}
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    existing_index = json.load(f)
            except Exception:
                existing_index = {}
        
        existing_saved_files = existing_index.get('saved_files', {})
        existing_saved_files.update(saved_files)
        
        current_time = str(pd.Timestamp.now())
        updated_index = {
            'saved_files': existing_saved_files,
            'last_save_time': current_time,
            'created_time': existing_index.get('created_time', current_time),
            'save_history': existing_index.get('save_history', [])
        }
        
        if saved_files:
            save_record = {
                'timestamp': current_time,
                'files_saved': list(saved_files.keys()),
                'overwrite_policy': overwrite_files
            }
            updated_index['save_history'].append(save_record)
        
        with open(index_path, 'w') as f:
            json.dump(updated_index, f, indent=2)
    
    def load(self, path: Path, keys: Optional[List[str]] = None):
        """
        Load data context from disk.
        
        Args:
            path: Directory containing saved data
            keys: Specific keys to load (None = load all)
        """
        path = Path(path)
        
        # Load metadata
        metadata_path = path / 'data_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self._metadata = json.load(f)

        # Load JSON-safe values from results.json
        results_json_path = path / 'results.json'
        json_data = {}
        if results_json_path.exists():
            try:
                with open(results_json_path, 'r') as f:
                    json_data = json.load(f)
            except Exception:
                pass
        
        # Load data index if available
        index_path = path / 'data_index.json'
        saved_files = {}
        if index_path.exists():
            with open(index_path, 'r') as f:
                index_data = json.load(f)
                saved_files = index_data.get('saved_files', {})
        
        # Determine which keys to load
        if keys is None:
            if saved_files:
                keys = list(saved_files.keys())
            else:
                keys = list(json_data.keys())
                for file in path.iterdir():
                    if file.suffix in ['.pkl', '.npy', '.npz']:
                        keys.append(file.stem)
        
        # Load each data item
        for key in keys:
            try:
                file_type = saved_files.get(key)

                if file_type == 'json':
                    if key in json_data:
                        self._data[key] = json_data[key]
                elif file_type == 'numpy':
                    self._data[key] = np.load(path / f"{key}.npy", allow_pickle=True)
                elif file_type == 'numpy_dict':
                    npz_data = np.load(path / f"{key}.npz", allow_pickle=True)
                    self._data[key] = dict(npz_data)
                elif file_type in ('pandas', 'pickle', 'rt_sort'):
                    with open(path / f"{key}.pkl", 'rb') as f:
                        self._data[key] = pickle.load(f)
                elif key in json_data:
                    self._data[key] = json_data[key]
                else:
                    # Try different file formats by extension
                    npy_path = path / f"{key}.npy"
                    npz_path = path / f"{key}.npz"
                    pkl_path = path / f"{key}.pkl"
                    
                    if npy_path.exists():
                        self._data[key] = np.load(npy_path, allow_pickle=True)
                    elif npz_path.exists():
                        npz_data = np.load(npz_path, allow_pickle=True)
                        self._data[key] = dict(npz_data)
                    elif pkl_path.exists():
                        with open(pkl_path, 'rb') as f:
                            self._data[key] = pickle.load(f)
            except Exception as e:
                print(f"   ⚠️  Warning: Error loading {key}: {e}")
                continue


class DataValidator:
    """
    Validates data types and constraints.
    """
    
    @staticmethod
    def validate_type(value: Any, expected_type: type) -> bool:
        """Check if value matches expected type."""
        return isinstance(value, expected_type)
    
    @staticmethod
    def validate_shape(value: Any, expected_shape: tuple) -> bool:
        """Check if array-like value has expected shape."""
        if not hasattr(value, 'shape'):
            return False
        return value.shape == expected_shape
    
    @staticmethod
    def validate_range(value: Any, min_val: float = None, max_val: float = None) -> bool:
        """Check if numeric value is within range."""
        if hasattr(value, '__iter__'):
            # Array-like
            if min_val is not None and np.any(value < min_val):
                return False
            if max_val is not None and np.any(value > max_val):
                return False
        else:
            # Scalar
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
        return True


class DataDependencyTracker:
    """
    Tracks data dependencies between phases for debugging and visualization.
    """
    
    def __init__(self):
        self.dependencies: Dict[str, List[str]] = {}  # phase -> [required keys]
        self.productions: Dict[str, List[str]] = {}   # phase -> [provided keys]
        self.data_flow: List[tuple] = []  # [(phase, key, 'read'/'write')]
    
    def add_phase(self, phase_name: str, requires: List[str], provides: List[str]):
        """Register a phase's data dependencies."""
        self.dependencies[phase_name] = requires
        self.productions[phase_name] = provides
    
    def record_access(self, phase_name: str, key: str, access_type: str):
        """Record data access for tracking."""
        self.data_flow.append((phase_name, key, access_type))
    
    def get_producers(self, key: str) -> List[str]:
        """Find which phases produce a given data key."""
        producers = []
        for phase, provides in self.productions.items():
            if key in provides:
                producers.append(phase)
        return producers
    
    def get_consumers(self, key: str) -> List[str]:
        """Find which phases consume a given data key."""
        consumers = []
        for phase, requires in self.dependencies.items():
            if key in requires:
                consumers.append(phase)
        return consumers
    
    def visualize_flow(self) -> str:
        """Generate a simple text visualization of data flow."""
        lines = ["Data Flow Visualization", "=" * 50]
        
        # Group by data key
        key_flows: Dict[str, List[tuple]] = {}
        for phase, key, access in self.data_flow:
            if key not in key_flows:
                key_flows[key] = []
            key_flows[key].append((phase, access))
        
        # Display each key's flow
        for key, flows in key_flows.items():
            lines.append(f"\n{key}:")
            for phase, access in flows:
                symbol = "→" if access == 'write' else "←"
                lines.append(f"  {symbol} {phase}")
        
        return "\n".join(lines)
