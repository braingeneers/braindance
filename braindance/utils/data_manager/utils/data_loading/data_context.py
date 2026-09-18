"""
DataContext - Lazy loading and auto-serialization for computed results

This module provides the DataContext class, which handles automatic persistence
and lazy loading of computed results in the BrainDance data manager.

Key Features:
- Lazy attribute access with disk persistence
- Auto-format detection (numpy→.npy, pandas→.csv, objects→.pkl)
- Metadata tracking (type, shape, timestamp)
- Overwrite protection

Usage:
    # Create context for results storage
    ctx = DataContext(path=Path('results/'))

    # Save results
    ctx.firing_rates = np.array([...])
    ctx.save()

    # Lazy load from disk
    rates = ctx.firing_rates  # Auto-loads if not in memory

Architecture:
    The DataContext is used by the Recording class to manage the `rec.results`
    property, enabling "compute once, use forever" workflows where expensive
    computations are cached to disk and automatically reloaded when needed.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from braindance.spike_data import load_spike_pickle


class DataContext:
    """
    Lazy loading + auto-serialization for computed results.

    Provides attribute-style access to data with automatic persistence to disk.
    When a value is set, it's stored in memory. When save() is called, values
    are automatically serialized in the most appropriate format based on type.
    On subsequent access, values are lazy-loaded from disk only when needed.

    Attributes:
        _data: In-memory cache of loaded/set values
        _metadata: Metadata about each stored value
        _path: Directory for persisting data
        _overwrite_existing: Whether to allow overwriting existing values
        _loaded_from_disk: Set of keys that have been attempted to load
    """

    def __init__(self, path: Optional[Path] = None, overwrite_existing: bool = True):
        self._data: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict] = {}
        self._path = Path(path) if path else None
        self._overwrite_existing = overwrite_existing
        self._loaded_from_disk = set()  # Track what's been loaded

    def __setattr__(self, name: str, value: Any):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            if not self._overwrite_existing and name in self._data:
                print(f"⚠️  Warning: Key '{name}' already exists. Skipping (overwrite_existing=False)")
                return
            elif name in self._data:
                print(f"⚠️  Overwriting existing '{name}'")

            self._data[name] = value
            self._metadata[name] = {
                'type': type(value).__name__,
                'shape': getattr(value, 'shape', None),
                'length': len(value) if hasattr(value, '__len__') else None
            }

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(f"No attribute '{name}'")
        if name in self._data:
            return self._data[name]
        # Try lazy load from disk
        if self._path and name not in self._loaded_from_disk:
            if self._try_load_key(name):
                return self._data[name]
        raise AttributeError(f"No data found for key '{name}'")

    def __setitem__(self, key: str, value: Any):
        """Support dict-style assignment: ctx['key'] = value"""
        self.__setattr__(key, value)

    def __getitem__(self, key: str) -> Any:
        """Support dict-style access: value = ctx['key']"""
        return self.__getattr__(key)

    def __contains__(self, name: str) -> bool:
        if name in self._data:
            return True
        # Check if exists on disk
        if self._path:
            return self._exists_on_disk(name)
        return False

    def _exists_on_disk(self, key: str) -> bool:
        """Check if a key exists as a file on disk."""
        if not self._path:
            return False
        for ext in ['.pkl', '.npy', '.npz', '.csv']:
            if (self._path / f"{key}{ext}").exists():
                return True
        return False

    def _try_load_key(self, key: str) -> bool:
        """Attempt to load a single key from disk. Returns True if successful."""
        if not self._path:
            return False

        self._loaded_from_disk.add(key)  # Mark as attempted

        # Try different formats
        pkl_path = self._path / f"{key}.pkl"
        npy_path = self._path / f"{key}.npy"
        npz_path = self._path / f"{key}.npz"
        csv_path = self._path / f"{key}.csv"

        try:
            if pkl_path.exists():
                with open(pkl_path, 'rb') as f:
                    self._data[key] = load_spike_pickle(f)
                return True
            elif npy_path.exists():
                self._data[key] = np.load(npy_path, allow_pickle=True)
                return True
            elif npz_path.exists():
                self._data[key] = dict(np.load(npz_path, allow_pickle=True))
                return True
            elif csv_path.exists():
                self._data[key] = pd.read_csv(csv_path)
                return True
        except Exception as e:
            print(f"⚠️  Warning: Failed to load '{key}': {e}")

        return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get data with default value."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def keys(self) -> List[str]:
        """Get all available keys (in memory + on disk)."""
        all_keys = set(self._data.keys())
        if self._path and self._path.exists():
            for f in self._path.iterdir():
                if f.suffix in ['.pkl', '.npy', '.npz', '.csv']:
                    all_keys.add(f.stem)
        return list(all_keys)

    def save(self, path: Optional[Path] = None, keys: Optional[List[str]] = None):
        """Save data to disk."""
        save_path = Path(path) if path else self._path
        if not save_path:
            raise ValueError("No path specified for saving")

        save_path.mkdir(parents=True, exist_ok=True)
        keys_to_save = keys or list(self._data.keys())

        for key in keys_to_save:
            if key not in self._data:
                continue
            value = self._data[key]
            self._save_value(save_path, key, value)

    def _save_value(self, path: Path, key: str, value: Any):
        """
        Save a single value to disk with auto-format detection.

        Format rules:
        - np.ndarray → .npy
        - pd.DataFrame → .csv (for readability) or .pkl (if complex index)
        - dict with arrays → .npz
        - SpikeData or other objects → .pkl
        """
        path.mkdir(parents=True, exist_ok=True)

        try:
            if isinstance(value, np.ndarray):
                # Save numpy arrays as .npy
                np.save(path / f"{key}.npy", value)
            elif isinstance(value, pd.DataFrame):
                # Save DataFrames as CSV if simple index, otherwise pickle
                if isinstance(value.index, pd.RangeIndex) or value.index.name is None:
                    value.to_csv(path / f"{key}.csv", index=False)
                else:
                    # Complex index - use pickle to preserve it
                    value.to_pickle(path / f"{key}.pkl")
            elif isinstance(value, dict):
                # Check if dict contains only arrays (use npz), otherwise pickle
                if all(isinstance(v, np.ndarray) for v in value.values()):
                    np.savez(path / f"{key}.npz", **value)
                else:
                    with open(path / f"{key}.pkl", 'wb') as f:
                        pickle.dump(value, f)
            else:
                # Everything else (SpikeData, custom objects, etc.) -> pickle
                with open(path / f"{key}.pkl", 'wb') as f:
                    pickle.dump(value, f)

            # Save metadata
            self._save_index(path, key, value)
        except Exception as e:
            print(f"⚠️  Failed to save '{key}': {e}")

    def _save_index(self, path: Path, key: str, value: Any, note: str = None):
        """
        Save provenance metadata for a single key.

        Args:
            path: Directory path
            key: Data key name
            value: The value being saved
            note: Optional note about the data
        """
        metadata = {
            'saved_at': datetime.now().isoformat(),
            'type': type(value).__name__,
            'shape': getattr(value, 'shape', None),
            'note': note
        }

        try:
            with open(path / f"{key}_meta.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            # Don't fail the whole save if metadata fails
            print(f"⚠️  Warning: Failed to save metadata for '{key}': {e}")

    def load(self, path: Optional[Path] = None, keys: Optional[List[str]] = None):
        """Load data from disk."""
        load_path = Path(path) if path else self._path
        if not load_path:
            raise ValueError("No path specified for loading")

        self._path = load_path

        if keys:
            for key in keys:
                self._try_load_key(key)
        # If no keys specified, lazy loading will handle it
