"""
RecordingCatalog - Collection interface for loading recordings from CSV

This module provides the RecordingCatalog class for loading and filtering
neural recordings based on experimental metadata stored in a CSV catalog.

Key Features:
- Load catalog from CSV with RecordingCatalog.from_csv()
- Pandas-style filtering with Django-style lookups (gt, gte, lt, lte, in, contains)
- List-like iteration and indexing
- Lazy loading of Recording objects
- Batch property access

Architecture:
    load_catalog(path) → RecordingCatalog → catalog.filter() → catalog[i] → Recording

Usage:
    # Load catalog from CSV
    catalog = load_catalog('catalog.csv')
    
    # Filter recordings
    stim_recs = catalog.filter(freq__gt=0, baseline=False)
    
    # Iterate and process
    for rec in stim_recs:
        spikes = rec.spikes
        rec.clear_cache()
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator, Callable
import numpy as np
import pandas as pd

# Import from same package
from braindance.utils.data_manager.utils.data_loading.recording import Recording
from braindance.utils.data_manager.utils.data_loading.data_context import DataContext


def _experiment_sort_key(exp_name):
    """Sort key for natural experiment ordering.

    Groups by experiment day first (exp1 < exp2 < ... < exp10 < RL2 < RL_pharma),
    then orders within each group:
        0 - baseline:  no suffix (e.g. 'exp1/exp1')
        1 - causal:    contains 'causal' (e.g. 'exp1/exp1_causal')
        2 - game base: game name without trailing number (e.g. 'exp1/exp1_cartpole_long')
        3 - numbered:  game with trailing number (e.g. 'exp1/exp1_cartpole_long_1', '_2', ...)
    """
    import re
    name = str(exp_name)

    # Extract group prefix: "exp1/exp1" from "exp1/exp1_cartpole_long_3"
    # Split on "/" to get folder and recording parts
    parts = name.split("/")
    folder = parts[0] if len(parts) > 1 else ""
    recording = parts[-1]

    # Natural sort key for the folder (e.g. exp1 < exp2 < exp10)
    folder_key = tuple(
        int(c) if c.isdigit() else c.lower()
        for c in re.split(r"(\d+)", folder)
    )

    # Determine the base name (folder prefix repeated in recording name)
    # e.g. for "exp1/exp1_cartpole_long_3", base is "exp1"
    base = folder.split("/")[-1] if folder else recording

    # Get the suffix after the base name
    rec_lower = recording.lower()
    base_lower = base.lower()
    suffix = rec_lower[len(base_lower):] if rec_lower.startswith(base_lower) else rec_lower

    # Classify within-group ordering
    if not suffix or suffix == "":
        # Baseline: just "exp1/exp1"
        return (folder_key, 0, "")
    elif "causal" in suffix:
        # Causal: "exp1/exp1_causal" or "exp1/exp1_causal_1"
        causal_match = re.search(r"_(\d+)$", suffix)
        causal_num = int(causal_match.group(1)) if causal_match else 0
        return (folder_key, 1, causal_num)
    else:
        # Game recordings: check for trailing number
        match = re.search(r"_(\d+)$", suffix)
        if match:
            num = int(match.group(1))
            prefix = suffix[: match.start()]
            return (folder_key, 3, prefix, num)
        else:
            # Unnumbered game: "exp1/exp1_cartpole_long"
            return (folder_key, 2, suffix)


class BatchResults:
    """
    Provides batch access to results across multiple recordings.
    
    Usage:
        catalog.results.connectivity  # Returns list of connectivity matrices
    """
    
    def __init__(self, catalog: 'RecordingCatalog'):
        self._catalog = catalog
    
    def __getattr__(self, name: str) -> List[Any]:
        """Access a result across all recordings."""
        results = []
        for rec in self._catalog:
            val = rec.results.get(name)
            results.append(val)
        return results
    
    def __contains__(self, name: str) -> bool:
        """Check if any recording has this result."""
        for rec in self._catalog:
            if name in rec.results:
                return True
        return False
    
    def available(self) -> Dict[str, int]:
        """Get count of how many recordings have each result."""
        counts: Dict[str, int] = {}
        for rec in self._catalog:
            for key in rec.results.keys():
                counts[key] = counts.get(key, 0) + 1
        return counts


class RecordingCatalog:
    """
    A collection of recordings with list-like and pandas-like access.
    
    Supports:
        - List-style indexing: catalog[0], catalog[-1], catalog[1:5]
        - Pandas-style filtering: catalog[catalog.freq > 0]
        - Django-style filtering: catalog.filter(freq__gt=0)
        - Iteration: for rec in catalog
        - Batch property access: catalog.spikes (returns list)
    
    Usage:
        catalog = load_catalog("path/to/catalog.csv")
        catalog = catalog.filter(freq__gt=0, baseline=False)
        
        for rec in catalog:
            print(rec.spikes.N)
    """
    
    def __init__(self, df: pd.DataFrame, base_path: Optional[Path] = None,
                 legacy_flat: Optional[bool] = None, sort_by_recording: bool = True):
        """
        Initialize from a DataFrame.

        Args:
            df: Catalog DataFrame with recording metadata
            base_path: Optional base path for resolving data file paths
            legacy_flat: Directory structure mode for all recordings:
                        - None (default): Auto-detect per recording
                        - True: Force legacy flat {base}/{proj}/{chip}/{exp_name}_spike_data.pkl
                        - False: Force standard {base}/{proj}/{chip}/{exp_name}/{exp_name}_spike_data.pkl
            sort_by_recording: If True (default), sort by natural experiment order
                              and add a 'recording_num' column.
        """
        if sort_by_recording and 'experiment' in df.columns and len(df) > 0:
            sort_keys = df['experiment'].apply(_experiment_sort_key)
            df = df.iloc[sort_keys.argsort()].reset_index(drop=True)
            df['recording_num'] = range(len(df))
        self._df = df.reset_index(drop=True)
        self._base_path = Path(base_path) if base_path else None
        self._legacy_flat = legacy_flat
        self._recordings: Dict[int, Recording] = {}  # Cache
    
    def __repr__(self) -> str:
        return f"RecordingCatalog({len(self)} recordings)"
    
    def __len__(self) -> int:
        return len(self._df)
    
    def __iter__(self) -> Iterator[Recording]:
        for i in range(len(self)):
            yield self[i]
    
    def __getitem__(self, idx) -> Union['RecordingCatalog', Recording]:
        """
        Flexible indexing:
            - int: Return single Recording
            - slice: Return new RecordingCatalog
            - boolean array/Series: Pandas-style filtering
        """
        # Boolean mask (pandas-style filtering)
        if isinstance(idx, (pd.Series, np.ndarray)):
            if hasattr(idx, 'dtype') and idx.dtype == bool:
                return RecordingCatalog(self._df[idx], base_path=self._base_path,
                                       legacy_flat=self._legacy_flat,
                                       sort_by_recording=False)

        # List of booleans
        if isinstance(idx, list) and all(isinstance(x, bool) for x in idx):
            return RecordingCatalog(self._df[idx], base_path=self._base_path,
                                   legacy_flat=self._legacy_flat,
                                   sort_by_recording=False)

        # Slice
        if isinstance(idx, slice):
            return RecordingCatalog(self._df.iloc[idx], base_path=self._base_path,
                                   legacy_flat=self._legacy_flat,
                                   sort_by_recording=False)
        
        # Integer index
        if isinstance(idx, int):
            # Handle negative indexing
            if idx < 0:
                idx = len(self) + idx
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range for catalog of size {len(self)}")
            
            # Return cached or create new Recording
            if idx not in self._recordings:
                self._recordings[idx] = Recording(self._df.iloc[idx], base_path=self._base_path)
            return self._recordings[idx]
        
        raise TypeError(f"Invalid index type: {type(idx)}")
    
    def __getattr__(self, name: str) -> Any:
        """
        Attribute access:
            - DataFrame columns return pd.Series (for filtering)
            - 'results' returns BatchResults
            - Data properties return lists
        """
        if name.startswith('_'):
            raise AttributeError(f"No attribute '{name}'")
        
        # Check if it's a DataFrame column
        if name in self._df.columns:
            return self._df[name]
        
        # Check if it's a data property (batch access)
        if name in Recording.DATA_PROPERTIES:
            return [getattr(rec, name) for rec in self]
        
        raise AttributeError(f"RecordingCatalog has no attribute '{name}'")
    
    # ==================== Results ====================
    
    @property
    def results(self) -> BatchResults:
        """Batch access to results across all recordings."""
        return BatchResults(self)
    
    # ==================== Filtering ====================
    
    def filter(self, sort_by_recording=True, **kwargs) -> 'RecordingCatalog':
        """
        Chainable filtering with Django-style lookups.
        
        Supports:
            - Exact match: filter(chip="25178ic")
            - Greater than: filter(freq__gt=0)
            - Greater or equal: filter(freq__gte=10)
            - Less than: filter(freq__lt=100)
            - Less or equal: filter(freq__lte=50)
            - In list: filter(freq__in=[1, 5, 10])
            - Contains (string): filter(chip__contains="251")
            - Is null: filter(drug__isnull=True)
            - Is not null: filter(drug__isnull=False)
        
        Args:
            sort_by_recording: If True (default), sort results by recording ID
                              extracted from experiment name for correct temporal order.
                              Set to False to preserve catalog order.
            **kwargs: Filter criteria
        
        Example:
            catalog.filter(freq__gt=0, baseline=False, chip__contains="251")
            catalog.filter(freq__gt=0, sort_by_recording=False)  # Don't sort
        """
        mask = pd.Series(True, index=self._df.index)
        
        for key, value in kwargs.items():
            mask &= self._apply_filter(key, value)
        
        filtered_df = self._df[mask]

        return RecordingCatalog(filtered_df, base_path=self._base_path,
                               legacy_flat=self._legacy_flat,
                               sort_by_recording=sort_by_recording)
    
    def _apply_filter(self, key: str, value: Any) -> pd.Series:
        """Apply a single filter and return boolean mask."""
        # Parse key for lookup type
        if '__' in key:
            parts = key.rsplit('__', 1)
            column, lookup = parts[0], parts[1]
        else:
            column, lookup = key, 'exact'
        
        if column not in self._df.columns:
            raise ValueError(f"Unknown column: '{column}'")
        
        col = self._df[column]
        
        # Apply lookup
        if lookup == 'exact':
            return col == value
        elif lookup == 'gt':
            return col > value
        elif lookup == 'gte':
            return col >= value
        elif lookup == 'lt':
            return col < value
        elif lookup == 'lte':
            return col <= value
        elif lookup == 'in':
            return col.isin(value)
        elif lookup == 'contains':
            return col.str.contains(value, na=False)
        elif lookup == 'startswith':
            return col.str.startswith(value, na=False)
        elif lookup == 'endswith':
            return col.str.endswith(value, na=False)
        elif lookup == 'isnull':
            return col.isnull() if value else col.notnull()
        else:
            raise ValueError(f"Unknown lookup type: '{lookup}'")
    
    def _sort_by_recording_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort DataFrame by experiment name in natural recording order.

        Order: baseline (no trailing number) -> causal -> numbered experiments.
        Also adds a 'recording_num' column (0, 1, 2, ...) to the DataFrame.

        Examples:
            'exp1/exp1_cartpole_long'     -> recording 0 (baseline, no trailing number)
            'exp1/exp1_cartpole_causal'   -> recording 1 (causal)
            'exp1/exp1_cartpole_long_1'   -> recording 2
            'exp1/exp1_cartpole_long_2'   -> recording 3
            ...
            'exp1/exp1_cartpole_long_19'  -> recording 20
        """
        sort_keys = df['experiment'].apply(_experiment_sort_key)
        sorted_df = df.iloc[sort_keys.argsort()].reset_index(drop=True)
        sorted_df['recording_num'] = range(len(sorted_df))
        return sorted_df
    
    # ==================== DataFrame Access ====================
    
    @property
    def df(self) -> pd.DataFrame:
        """Access underlying DataFrame (escape hatch for complex operations)."""
        return self._df
    
    @property
    def columns(self) -> List[str]:
        """List of available metadata columns."""
        return list(self._df.columns)
    
    # ==================== Aggregation ====================
    
    def collect(self, attr: str) -> List[Any]:
        """
        Collect an attribute from all recordings.
        
        Args:
            attr: Attribute path (e.g., 'spikes', 'results.connectivity')
        
        Returns:
            List of values
        """
        results = []
        for rec in self:
            # Handle nested attributes like 'results.connectivity'
            value = rec
            for part in attr.split('.'):
                value = getattr(value, part, None)
                if value is None:
                    break
            results.append(value)
        return results
    
    def apply(
        self,
        func: Callable[[Recording], Any],
        parallel: bool = False,
        max_workers: Optional[int] = None,
        on_error: str = 'raise'
    ) -> List[Any]:
        """
        Apply a function to each recording.

        Args:
            func: Function that takes a Recording and returns a result
            parallel: Whether to run in parallel
            max_workers: Number of parallel workers (default: min(32, len(catalog)))
            on_error: 'raise', 'warn', or 'ignore'

        Returns:
            List of results in same order as catalog
        """
        if not parallel:
            results = []
            for rec in self:
                try:
                    results.append(func(rec))
                except Exception as e:
                    if on_error == 'raise':
                        raise
                    elif on_error == 'warn':
                        print(f"⚠️  Error: {rec.identifier}: {e}")
                        results.append(None)
                    else:
                        results.append(None)
            return results
        else:
            return self._apply_parallel(func, max_workers, on_error)
    
    def _apply_parallel(
        self,
        func: Callable,
        max_workers: Optional[int],
        on_error: str
    ) -> List[Any]:
        """
        Parallel execution using ThreadPoolExecutor.

        Args:
            func: Function to apply to each recording
            max_workers: Number of workers (default: min(32, len(catalog)))
            on_error: 'raise', 'warn', or 'ignore'

        Returns:
            List of results in same order as catalog
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        if max_workers is None:
            max_workers = min(32, len(self))

        results = [None] * len(self)  # Preserve order
        print_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(func, rec): i
                for i, rec in enumerate(self)
            }

            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    if on_error == 'raise':
                        raise
                    elif on_error == 'warn':
                        with print_lock:
                            rec = self[idx]
                            print(f"⚠️  Error: {rec.identifier}: {e}")
                    # If 'ignore', result stays None

        return results
    
    # ==================== Grouping ====================
    
    def group_by(self, column: str) -> Dict[Any, 'RecordingCatalog']:
        """
        Group recordings by a column value.
        
        Args:
            column: Column name to group by
        
        Returns:
            Dict mapping values to RecordingCatalog objects
        """
        groups = {}
        for value in self._df[column].unique():
            mask = self._df[column] == value
            groups[value] = RecordingCatalog(self._df[mask], base_path=self._base_path,
                                            legacy_flat=self._legacy_flat,
                                            sort_by_recording=False)
        return groups
    
    # ==================== Summary ====================
    
    def summary(self) -> pd.Series:
        """Get summary statistics about the catalog."""
        summary_data = {
            'total_recordings': len(self),
            'unique_chips': self._df['chip'].nunique() if 'chip' in self._df else 0,
            'unique_experiments': self._df['experiment'].nunique() if 'experiment' in self._df else 0,
        }
        
        # Add column value counts for categorical columns
        for col in ['baseline', 'type', 'drug']:
            if col in self._df.columns:
                summary_data[f'{col}_counts'] = self._df[col].value_counts().to_dict()
        
        return pd.Series(summary_data)
    
    def describe(self):
        """Print a description of the catalog."""
        print(f"RecordingCatalog: {len(self)} recordings")
        print(f"Columns: {', '.join(self.columns)}")
        print(f"\nSummary:")
        print(self.summary().to_string())
    
    # ==================== Persistence ====================
    
    def save_all_results(self, keys: Optional[List[str]] = None):
        """Save results for all recordings."""
        for rec in self:
            rec.save_results(keys=keys)
    
    # ==================== Construction ====================
    
    @classmethod
    def from_csv(cls, path: Union[str, Path], base_path: Optional[Path] = None) -> 'RecordingCatalog':
        """Load catalog from CSV file."""
        path = Path(path)
        if not path.exists():
            raise ValueError(
                f"Catalog file not found: {path}\n"
                "To generate a catalog, run:\n"
                "  python -m braindance.utils.data_manager setup --catalog /path/to/catalog.csv\n"
                "Or generate one with:\n"
                "  python -m braindance.utils.data_manager.catalogging generate"
            )

        df = pd.read_csv(path)
        return cls(df, base_path=base_path)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, base_path: Optional[Path] = None) -> 'RecordingCatalog':
        """Create catalog from existing DataFrame."""
        return cls(df, base_path=base_path)


# ==================== Module-Level Functions ====================


def load_catalog(path: Optional[Union[str, Path]] = None) -> RecordingCatalog:
    """
    Load a recording catalog from CSV.
    
    If no path is provided, uses the catalog path from DataContext configuration.
    
    Args:
        path: Optional path to catalog CSV file
    
    Returns:
        RecordingCatalog
    
    Example:
        >>> catalog = load_catalog()  # Uses configured catalog
        >>> catalog = load_catalog('my_catalog.csv')  # Custom catalog
    """
    if path is None:
        # Get from config
        try:
            from braindance.config import get_catalog_path
            path = get_catalog_path()
            if path is None or not path.exists():
                raise ValueError(
                    "No catalog path provided and none configured.\n"
                    "Either pass a path or run 'python -m braindance.utils.data_manager setup'"
                )
        except Exception as e:
            raise ValueError(
                f"Failed to load catalog path from config: {e}\n"
                "Either pass a path or run 'python -m braindance.utils.data_manager setup'"
            )
    
    return RecordingCatalog.from_csv(path)
