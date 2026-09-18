"""
Recording - Single recording wrapper with lazy data loading

This module provides the Recording class, which represents a single neural
recording with lazy-loaded data and automatic results caching.

Key Features:
- Lazy loading of spike data, stim logs, game logs, pattern logs, reward logs, and mapping files
- Path resolution from proj/chip/experiment identifiers
- Results caching via DataContext
- S3-transparent loading with automatic caching
- Analysis methods (calculate_latencies, group_stimulations_by_electrode)

Architecture:
    load_recording(proj, chip, exp)  ← Simple entry point
           ↓
    Recording.from_identifiers()     ← Create Recording from identifiers
           ↓
    Recording._resolve_paths()       ← Build file paths
           ↓
    rec.spikes (property access)     ← Lazy load when accessed
           ↓
    Recording._load_spikes()         ← Load + format conversion
           ↓
    DataContext (results caching)    ← Cache computed metrics

Usage:
    # Load a single recording
    rec = load_recording('proj', 'chip', 'experiment')

    # Access data (lazy loaded)
    spikes = rec.spikes
    stim_log = rec.stim_log

    # Compute and cache results
    rec.results.firing_rates = spikes.rates(unit="Hz")
    rec.save_results()

    # Results are persisted and reloaded automatically
    rates = rec.results.firing_rates  # Loads from disk
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import ast

# Import from same package
from braindance.utils.data_manager.utils.data_loading.data_context import DataContext
from braindance.utils.data_manager.utils.data_loading.results_cache import (
    ResultsCache,
    get_auto_upload_enabled,
)

from braindance.spike_data import SpikeData, as_spike_data, load_spike_pickle

# Import ephys_manager CacheManager for loading autocuration zips.
# Mirrors the vendored proj/predictor/nrp/scripts/utils/data_manager/utils/recording.py
# block so source=="ephys_manager" rows can be loaded directly via this Recording
# class (otherwise their NaN base_path forces the data_manager spike_data.pkl
# path which doesn't exist for Anthos imports).
try:
    import sys
    import zipfile

    # __file__ = braindance/utils/data_manager/utils/data_loading/recording.py
    # ↑6 = repo root, then proj/predictor/nrp/scripts/utils/ephys_manager.
    _em_candidates = [
        Path(__file__).resolve().parents[5]
        / "proj"
        / "predictor"
        / "nrp"
        / "scripts"
        / "utils"
        / "ephys_manager",
    ]
    # Container layout: /app/src/proj/predictor/nrp/scripts/utils/ephys_manager.
    if os.path.isdir("/app/src/proj/predictor/nrp/scripts/utils/ephys_manager"):
        _em_candidates.append(
            Path("/app/src/proj/predictor/nrp/scripts/utils/ephys_manager")
        )
    _em_loaded = False
    for _p in _em_candidates:
        if _p.is_dir():
            sys.path.insert(0, str(_p))
            try:
                from utils.data_loading.cache import CacheManager  # noqa: E402

                _em_loaded = True
                break
            except ImportError:
                pass
    if not _em_loaded:
        CacheManager = None
except Exception:
    CacheManager = None


class Recording:
    """
    A single recording with lazy data loading.

    Wraps a catalog row (pd.Series) and provides attribute access to both
    metadata (from the row) and data (lazy loaded from files).

    Attributes:
        Metadata (from catalog row): chip, freq, baseline, experiment, etc.
        Data (lazy loaded): spikes, stim_log, game_log, pattern_log, reward_log,
                            mapping, raw_data, wf (extracted waveforms)
        Results: User-computed derived data (connectivity, cleaned_spikes, etc.)
    """

    # Columns that are metadata (from catalog row)
    METADATA_COLUMNS = {
        "proj",
        "chip",
        "experiment",
        "exp",
        "freq",
        "baseline",
        "drug",
        "type",
        "start_time",
        "duration",
        "n_neurons",
        "data_path",
        "log_path",
        "results_path",
        "base_path",
    }

    # Properties that trigger data loading
    DATA_PROPERTIES = {
        "spikes",
        "stim_log",
        "mapping",
        "spike_locations",
        "raw_data",
        "game_log",
        "pattern_log",
        "reward_log",
        # `wf` is the odd one out: it is a DERIVED product read from
        # `self.cache`, not a raw file resolved through `_resolve_paths()`. It
        # sits here anyway so `rec.wf` feels like `rec.spikes` and so
        # `clear_cache()` drops it. See `_load_wf`.
        "wf",
    }

    def __init__(
        self,
        row: pd.Series,
        base_path: Optional[Path] = None,
        legacy_flat: Optional[bool] = None,
        auto_upload: Optional[bool] = None,
    ):
        """
        Initialize a Recording from a catalog row.

        Args:
            row: A pandas Series from the catalog DataFrame
            base_path: Optional base path override for data files
            legacy_flat: Directory structure mode:
                        - None (default): Auto-detect by checking which structure exists
                        - True: Force legacy flat structure {base}/{proj}/{chip}/{exp_name}_spike_data.pkl
                        - False: Force standard structure {base}/{proj}/{chip}/{exp_name}/{exp_name}_spike_data.pkl
            auto_upload: Automatically upload derived data (binned_fr, etc.) to S3.
                        If None, defaults to BRAINDANCE_AUTO_UPLOAD environment variable.
        """
        self._row = row
        # Don't convert S3 URLs to Path objects
        if base_path:
            if isinstance(base_path, str) and base_path.startswith("s3://"):
                self._base_path = base_path  # Keep as string
            else:
                self._base_path = Path(base_path)
        else:
            self._base_path = None
        self._legacy_flat = legacy_flat  # None = auto-detect
        self._auto_upload = (
            auto_upload if auto_upload is not None else get_auto_upload_enabled()
        )
        self._detected_format: Optional[str] = None  # Track which format was detected
        self._data = DataContext(overwrite_existing=True)
        self._results: Optional[DataContext] = None
        self._results_cache: Optional[ResultsCache] = None
        self._paths_resolved = False
        self._resolved_paths: Dict[str, Path] = {}
        self._s3_loader: Optional["S3Loader"] = None  # Lazy init

    def __repr__(self) -> str:
        name = self._row.get("experiment", "unknown")
        chip = self._row.get("chip", "unknown")
        return f"Recording('{chip}/{name}')"

    def __getattr__(self, name: str) -> Any:
        """
        Attribute access priority:
        1. Internal attributes (_*)
        2. Data properties (spikes, stim_log, etc.) - lazy loaded
        3. Catalog row columns (metadata)
        """
        if name.startswith("_"):
            raise AttributeError(f"No attribute '{name}'")

        # Check if it's a data property
        if name in self.DATA_PROPERTIES:
            return self._load_data_property(name)

        # Check if it's in the catalog row
        if name in self._row.index:
            return self._row[name]

        raise AttributeError(f"Recording has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        """Dict-like access to metadata."""
        if key in self._row.index:
            return self._row[key]
        raise KeyError(f"No column '{key}' in recording metadata")

    def __contains__(self, key: str) -> bool:
        """Check if key exists in metadata or results."""
        return key in self._row.index or key in self.results

    # ==================== S3 Loader ====================

    def _get_s3_loader(self):
        """
        Get or create S3Loader instance.

        Returns:
            S3Loader instance with cache configured
        """
        if self._s3_loader is None:
            from braindance.utils.data_manager.utils.data_loading.s3_loader import (
                S3Loader,
            )
            from braindance import get_data_dir

            cache_dir = get_data_dir() / ".s3_cache"
            self._s3_loader = S3Loader(cache_dir=cache_dir)
        return self._s3_loader

    def _parse_experiment_components(self) -> Tuple[str, str]:
        """
        Parse experiment into exp_base and exp_name components.

        Handles multiple formats:
        1. Catalog has 'exp_base' column: Use it directly
        2. Experiment has '/': Split on '/' (e.g., 'rl_only/rl_only_cartpole_long_1')
        3. Fallback: Use experiment as both exp_base and exp_name

        Returns:
            Tuple of (exp_base, exp_name)
        """
        experiment = self._row.get("experiment", "")

        # First priority: Use exp_base column if available (for legacy data)
        if "exp_base" in self._row.index and pd.notna(self._row.get("exp_base")):
            exp_base = str(self._row["exp_base"])
            # exp_name is the last component or the full experiment
            exp_name = (
                str(experiment).split("/")[-1]
                if "/" in str(experiment)
                else str(experiment)
            )
            return exp_base, exp_name

        # Second priority: Parse experiment path (e.g., 'exp1/exp1_cont_95')
        if "/" in str(experiment):
            exp_base = str(experiment).split("/")[0]  # 'exp1'
            exp_name = str(experiment).split("/")[-1]  # 'exp1_cont_95'
            return exp_base, exp_name

        # Fallback: Use experiment as both
        return str(experiment), str(experiment)

    def _construct_s3_spike_path(self) -> Optional[str]:
        """
        Construct S3 path for spike data using catalog's base_path.

        S3 structure:
            {base_path}{chip}/{exp_base}/spike_data/{exp_name}_spike_data.pkl

        Where:
            - base_path: From catalog (e.g., 's3://braingeneers/braindance/proj/' or 's3://braingeneersdev/asrobbin/braindance_data/proj/')
            - experiment column: Like 'exp1/exp1_cont_95' or just 'exp1_cont_95' (if exp_base column exists)
            - exp_base: 'exp1' (from exp_base column or parsed from experiment)
            - exp_name: 'exp1_cont_95'
        """
        base_path = self._row.get("base_path", "")
        if not base_path or not str(base_path).startswith("s3://"):
            return None

        chip = self._row.get("chip", "")
        experiment = self._row.get("experiment", "")

        if not chip or not experiment:
            return None

        # Parse experiment components using unified method
        exp_base, exp_name = self._parse_experiment_components()

        # Ensure base_path has trailing slash
        if not base_path.endswith("/"):
            base_path = base_path + "/"

        # Construct S3 path: {base_path}{chip}/{exp_base}/spike_data/{exp_name}_spike_data.pkl
        s3_path = f"{base_path}{chip}/{exp_base}/spike_data/{exp_name}_spike_data.pkl"
        print(f"  [DEBUG] _construct_s3_spike_path():")
        print(f"    base_path={base_path}")
        print(f"    chip={chip}, experiment={experiment}")
        print(f"    exp_base={exp_base}, exp_name={exp_name}")
        print(f"    → s3_path={s3_path}")
        return s3_path

    def _construct_s3_stim_log_path(self) -> Optional[str]:
        """
        Construct S3 path for stim log using catalog's base_path.

        S3 structure:
            {base_path}{chip}/{exp_base}/{exp_name}_log.csv
        """
        base_path = self._row.get("base_path", "")
        if not base_path or not str(base_path).startswith("s3://"):
            return None

        chip = self._row.get("chip", "")
        experiment = self._row.get("experiment", "")

        if not chip or not experiment:
            return None

        # Parse experiment components using unified method
        exp_base, exp_name = self._parse_experiment_components()

        # Ensure base_path has trailing slash
        if not base_path.endswith("/"):
            base_path = base_path + "/"

        s3_path = f"{base_path}{chip}/{exp_base}/{exp_name}_log.csv"
        return s3_path

    def _construct_s3_game_log_path(self) -> Optional[str]:
        """
        Construct S3 path for game log using catalog's base_path.

        S3 structure:
            {base_path}{chip}/{exp_base}/{exp_name}_game_log.csv
        """
        base_path = self._row.get("base_path", "")
        if not base_path or not str(base_path).startswith("s3://"):
            return None

        chip = self._row.get("chip", "")
        experiment = self._row.get("experiment", "")

        if not chip or not experiment:
            return None

        # Parse experiment components using unified method
        exp_base, exp_name = self._parse_experiment_components()

        # Ensure base_path has trailing slash
        if not base_path.endswith("/"):
            base_path = base_path + "/"

        s3_path = f"{base_path}{chip}/{exp_base}/{exp_name}_game_log.csv"
        return s3_path

    def _construct_s3_pattern_log_path(self) -> Optional[str]:
        """
        Construct S3 path for pattern log using catalog's base_path.

        S3 structure:
            {base_path}{chip}/{exp_base}/{exp_name}_pattern_log.csv
        """
        base_path = self._row.get("base_path", "")
        if not base_path or not str(base_path).startswith("s3://"):
            return None

        chip = self._row.get("chip", "")
        experiment = self._row.get("experiment", "")

        if not chip or not experiment:
            return None

        # Parse experiment components using unified method
        exp_base, exp_name = self._parse_experiment_components()

        # Ensure base_path has trailing slash
        if not base_path.endswith("/"):
            base_path = base_path + "/"

        s3_path = f"{base_path}{chip}/{exp_base}/{exp_name}_pattern_log.csv"
        return s3_path

    def _construct_s3_reward_log_path(self) -> Optional[str]:
        """
        Construct S3 path for reward log using catalog's base_path.

        S3 structure:
            {base_path}{chip}/{exp_base}/{exp_name}_reward_log.csv
        """
        base_path = self._row.get("base_path", "")
        if not base_path or not str(base_path).startswith("s3://"):
            return None

        chip = self._row.get("chip", "")
        experiment = self._row.get("experiment", "")

        if not chip or not experiment:
            return None

        # Parse experiment components using unified method
        exp_base, exp_name = self._parse_experiment_components()

        # Ensure base_path has trailing slash
        if not base_path.endswith("/"):
            base_path = base_path + "/"

        s3_path = f"{base_path}{chip}/{exp_base}/{exp_name}_reward_log.csv"
        return s3_path

    def _construct_s3_raw_data_path(self) -> Optional[str]:
        """
        Construct S3 path for raw data (.raw.h5) using catalog fields.

        Priority:
        1. full_path (if present)
        2. base_path + chip + exp_base + exp_name
        """
        full_path = self._row.get("full_path", "")
        if isinstance(full_path, str) and full_path.startswith("s3://"):
            full_path = full_path.strip()
            if full_path.endswith(".raw.h5"):
                return full_path
            if full_path.endswith("/"):
                _, exp_name = self._parse_experiment_components()
                if exp_name:
                    return f"{full_path}{exp_name}.raw.h5"
            # Assume it's the file stem without extension
            return f"{full_path}.raw.h5"

        base_path = self._row.get("base_path", "")
        if not base_path or not str(base_path).startswith("s3://"):
            return None

        chip = self._row.get("chip", "")
        exp_base, exp_name = self._parse_experiment_components()
        if not chip or not exp_name:
            return None

        if not base_path.endswith("/"):
            base_path = base_path + "/"

        if self._legacy_flat:
            return f"{base_path}{chip}/{exp_name}.raw.h5"

        return f"{base_path}{chip}/{exp_base}/{exp_name}.raw.h5"

    def _construct_s3_experiment_dir(self) -> Optional[str]:
        """
        Construct S3 experiment directory for listing raw data files.
        """
        full_path = self._row.get("full_path", "")
        if isinstance(full_path, str) and full_path.startswith("s3://"):
            full_path = full_path.strip()
            if full_path.endswith(".raw.h5"):
                return f"{full_path.rsplit('/', 1)[0]}/"
            if full_path.endswith("/"):
                return full_path
            return f"{full_path.rsplit('/', 1)[0]}/"

        base_path = self._row.get("base_path", "")
        if not base_path or not str(base_path).startswith("s3://"):
            return None

        chip = self._row.get("chip", "")
        exp_base, _ = self._parse_experiment_components()
        if not chip or not exp_base:
            return None

        if not base_path.endswith("/"):
            base_path = base_path + "/"

        return f"{base_path}{chip}/{exp_base}/"

    def _download_from_s3(self, s3_path: str, local_path: Path) -> bool:
        """
        Download a file from S3 to local storage.

        Args:
            s3_path: S3 path (e.g., 's3://bucket/key.pkl')
            local_path: Local path to save to

        Returns:
            True if successful, False otherwise
        """
        try:
            loader = self._get_s3_loader()
            return loader.download_file(s3_path, local_path)
        except Exception as e:
            print(f"  [S3] ✗ Failed to download {s3_path}: {e}")
            return False

    # ==================== Path Resolution ====================

    def _resolve_paths(self):
        """
        Resolve all data paths from catalog row and base_path.

        Strategy:
        1. Get base_path from: self._base_path → row['base_path'] → get_data_dir()
        2. Extract proj/chip/experiment from row
        3. Auto-detect directory structure (if legacy_flat not specified)
        4. Build standard paths following BusyBeeLoader conventions
        5. Override with explicit paths from catalog if present

        Supports two directory structures:
        - Legacy flat: {base}/{proj}/{chip}/{exp_name}_spike_data.pkl
        - Standard:    {base}/{proj}/{chip}/{exp_name}/{exp_name}_spike_data.pkl

        Auto-detection checks which structure exists locally, preferring standard format.
        """
        if self._paths_resolved:
            return

        # Get base path
        base_path = self._base_path

        # Convert S3 paths to local data directory for caching
        if (
            base_path is not None
            and isinstance(base_path, (str, Path))
            and str(base_path).startswith("s3://")
        ):
            from braindance import get_data_dir

            base_path = get_data_dir()

        if base_path is None:
            if "base_path" in self._row.index and pd.notna(self._row["base_path"]):
                raw_base_path = str(self._row["base_path"])
                # Don't convert S3 paths to Path objects - they should only be used by _construct_s3_*_path() methods
                if isinstance(raw_base_path, str) and raw_base_path.startswith("s3://"):
                    # S3 path detected - use default local data dir for local caching
                    from braindance import get_data_dir

                    base_path = get_data_dir()
                else:
                    # Local filesystem path
                    base_path = Path(raw_base_path)
            else:
                # Import here to avoid circular dependency
                from braindance import get_data_dir

                base_path = get_data_dir()

        # Extract metadata
        proj = self._row.get("proj", "")
        chip = self._row.get("chip", "")
        experiment = self._row.get("experiment", "")

        # Bug Fix: If proj is missing, try to extract it from S3 base_path
        if (
            not proj
            and "base_path" in self._row.index
            and pd.notna(self._row["base_path"])
        ):
            raw_base_path = str(self._row["base_path"])
            if isinstance(raw_base_path, str) and raw_base_path.startswith("s3://"):
                # Extract last component: s3://bucket/path/to/project/ -> project
                path_parts = raw_base_path.rstrip("/").split("/")
                if len(path_parts) > 3:
                    proj = path_parts[-1]

        # Get experiment name and base
        # experiment can be like "exp1/exp1_cont_95" or just "exp1_cont_95"
        if isinstance(experiment, str) and experiment:
            exp_name = Path(experiment).name
            # First priority: Use exp_base column if available (for legacy data)
            if "exp_base" in self._row.index and pd.notna(self._row.get("exp_base")):
                exp_base = str(self._row["exp_base"])
            # Second priority: Parse from experiment path
            elif "/" in experiment:
                exp_base = experiment.split("/")[0]
            else:
                exp_base = exp_name
        else:
            exp_name = ""
            exp_base = ""

        # Preserve a full experiment/recording path when that layout exists.
        nested_dir = base_path / proj / chip / experiment if exp_name else base_path
        use_nested = self._legacy_flat is None and bool(exp_name) and '/' in experiment and nested_dir.is_dir()
        # Auto-detect directory structure if not specified
        use_legacy_flat = self._legacy_flat
        if use_nested:
            use_legacy_flat = False
        elif use_legacy_flat is None and exp_name:
            use_legacy_flat = self._detect_directory_structure(
                base_path, proj, chip, exp_name
            )

        # Build directory structure based on mode
        if use_legacy_flat:
            # Legacy flat: files directly in chip folder
            # {base}/{proj}/{chip}/{exp_name}_spike_data.pkl
            rec_dir = base_path / proj / chip
            results_dir = rec_dir / "results" / exp_name
            self._detected_format = "legacy_flat"
        else:
            # Standard: files in exp_name subfolder
            # {base}/{proj}/{chip}/{exp_name}/{exp_name}_spike_data.pkl
            rec_dir = nested_dir if use_nested else (base_path / proj / chip / exp_name if exp_name else base_path)
            results_dir = rec_dir / "results"
            self._detected_format = "standard"

        # Standard file paths (following BusyBeeLoader conventions)
        self._resolved_paths["spikes"] = (
            rec_dir / f"{exp_name}_spike_data.pkl" if exp_name else None
        )
        self._resolved_paths["stim_log"] = (
            rec_dir / f"{exp_name}_log.csv" if exp_name else None
        )
        self._resolved_paths["game_log"] = (
            rec_dir / f"{exp_name}_game_log.csv" if exp_name else None
        )
        self._resolved_paths["pattern_log"] = (
            rec_dir / f"{exp_name}_pattern_log.csv" if exp_name else None
        )
        self._resolved_paths["reward_log"] = (
            rec_dir / f"{exp_name}_reward_log.csv" if exp_name else None
        )
        # Mapping is shared at experiment base level (exp1), not continuation level (exp1_cont_95)
        self._resolved_paths["mapping"] = (
            (base_path / proj / chip / "mapping.csv")
            if use_legacy_flat
            else (nested_dir.parent / "mapping.csv" if use_nested
                  else base_path / proj / chip / exp_base / "mapping.csv")
        )
        self._resolved_paths["spike_locations"] = (
            (base_path / "spike_info" / proj / chip / "spike_info.json")
            if exp_name
            else None
        )
        self._resolved_paths["raw_data"] = (
            rec_dir / f"{exp_name}.raw.h5" if exp_name else None
        )
        self._resolved_paths["results"] = results_dir

        # Override with explicit paths from catalog if present
        for column, key in (("data_path", "spikes"), ("log_path", "stim_log"),
                            ("results_path", "results"), ("raw_data_path", "raw_data"),
                            ("mapping_path", "mapping")):
            value = self._row.get(column)
            if pd.notna(value) and str(value).strip():
                path = Path(value).expanduser()
                self._resolved_paths[key] = path if path.is_absolute() else base_path / path

        self._paths_resolved = True

    def _detect_directory_structure(
        self, base_path: Path, proj: str, chip: str, exp_name: str
    ) -> bool:
        """
        Auto-detect whether data uses legacy flat or standard directory structure.

        Args:
            base_path: Base data directory
            proj: Project name
            chip: Chip identifier
            exp_name: Experiment name

        Returns:
            True if legacy flat structure detected, False for standard structure.
            Defaults to legacy_flat=True if neither exists (for S3 download compatibility).
        """
        # Check standard structure first (preferred for new data)
        standard_path = (
            base_path / proj / chip / exp_name / f"{exp_name}_spike_data.pkl"
        )
        if standard_path.exists():
            return False  # Use standard structure

        # Check legacy flat structure
        legacy_path = base_path / proj / chip / f"{exp_name}_spike_data.pkl"
        if legacy_path.exists():
            return True  # Use legacy flat

        # Neither exists locally - default to legacy_flat for S3 compatibility
        # (Most existing S3 data is in legacy flat format)
        return True

    @property
    def _spikes_path(self) -> Optional[Path]:
        """Path to spike data file."""
        self._resolve_paths()
        return self._resolved_paths.get("spikes")

    @property
    def _stim_log_path(self) -> Optional[Path]:
        """Path to stimulus log file."""
        self._resolve_paths()
        return self._resolved_paths.get("stim_log")

    @property
    def _mapping_path(self) -> Optional[Path]:
        """Path to mapping file."""
        self._resolve_paths()
        return self._resolved_paths.get("mapping")

    @property
    def _raw_data_path(self) -> Optional[Path]:
        """Path to raw data file (.raw.h5)."""
        self._resolve_paths()
        return self._resolved_paths.get("raw_data")

    @property
    def _game_log_path(self) -> Optional[Path]:
        """Path to game log file."""
        self._resolve_paths()
        return self._resolved_paths.get("game_log")

    @property
    def _pattern_log_path(self) -> Optional[Path]:
        """Path to pattern log file."""
        self._resolve_paths()
        return self._resolved_paths.get("pattern_log")

    @property
    def _reward_log_path(self) -> Optional[Path]:
        """Path to reward log file."""
        self._resolve_paths()
        return self._resolved_paths.get("reward_log")

    @property
    def _spike_locations_path(self) -> Optional[Path]:
        """Path to spike locations file (spike_info.json or spike_info.pkl)."""
        self._resolve_paths()
        return self._resolved_paths.get("spike_locations")

    @property
    def _results_path(self) -> Path:
        """Path to results directory."""
        self._resolve_paths()
        return self._resolved_paths.get("results", Path(".") / "results")

    @property
    def base_output_dir(self) -> Path:
        """
        Base output directory (without project/chip structure).

        Returns the configured output directory without any automatic subdirectories.
        Useful for cross-project files or custom organization.

        Returns:
            Path: Base output directory path
        Examples:
            >>> rec.base_output_dir
            PosixPath('/Volumes/hunter_ssd/busy_bee/plots')

            >>> rec.base_output_dir / 'summary_report.txt'
            PosixPath('/Volumes/hunter_ssd/busy_bee/plots/summary_report.txt')
        """
        from braindance.config import get_output_dir

        return get_output_dir()

    @property
    def catalog_path(self) -> Path:
        """
        Path to the catalog CSV file.

        Returns the path to the all_catalog.csv file in the project root.
        Useful for loading the full catalog to select recordings.

        Returns:
            Path: Catalog file path

        Examples:
            >>> rec.catalog_path
            PosixPath('/path/to/BrainDance-dev/proj/predictor/nrp/all_catalog.csv')

            >>> import pandas as pd
            >>> catalog_df = pd.read_csv(rec.catalog_path)
        """
        from pathlib import Path

        # Try to find catalog in project root
        # Assume we're in braindance/utils/data_manager/utils/data_loading/recording.py
        # Go up to project root: ../../../../proj/predictor/nrp/all_catalog.csv
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent.parent
        catalog_path = project_root / "proj" / "predictor" / "nrp" / "all_catalog.csv"

        if catalog_path.exists():
            return catalog_path

        # Fallback: check if we're already in the project root
        alt_path = Path.cwd() / "proj" / "predictor" / "nrp" / "all_catalog.csv"
        if alt_path.exists():
            return alt_path

        # Return the expected path even if it doesn't exist (will raise clear error when used)
        return catalog_path

    @property
    def output_dir(self) -> Path:
        """
        Output directory for this recording's plots and figures.

        Returns a Path object with the structure:
            {output_dir}/{proj}/{chip}/

        The directory is automatically created when accessed.

        Returns:
            Path: Output directory path (can be combined with / operator)

        Examples:
            >>> rec.output_dir / 'plot.png'
            PosixPath('/Volumes/hunter_ssd/busy_bee/plots/25-02-25_busybees/25123ic/plot.png')

            >>> rec.output_dir / 'analysis' / 'firing_rates.png'
            PosixPath('/Volumes/hunter_ssd/busy_bee/plots/25-02-25_busybees/25123ic/analysis/firing_rates.png')
        """
        path = self.base_output_dir / self.proj / self.chip
        path.mkdir(parents=True, exist_ok=True)

        return path

    # ==================== Data Loading ====================

    def _load_data_property(self, name: str) -> Any:
        """Load a data property if not already cached."""
        if name in self._data._data:
            return self._data._data[name]

        loader_method = getattr(self, f"_load_{name}", None)
        if loader_method is None:
            raise AttributeError(f"No loader for '{name}'")

        value = loader_method()
        if value is not None:
            self._data._data[name] = value
        return value

    def _load_pickle_safe(self, file_path: Path) -> Any:
        """
        Load a trusted pickle with legacy SpikeData format conversion.
        """
        try:
            with open(file_path, "rb") as f:
                return as_spike_data(load_spike_pickle(f))
        except Exception as e:
            print(f"⚠️  Failed to load {file_path.name}: {e}")
            return None

    def _load_spikes_from_autocuration(self) -> Any:
        """Load spike data from ephys_manager autocuration zip.

        Mirrors the vendored Recording's loader. Uses `_original_uuid`,
        `_original_experiment`, and `file_extension` from the catalog row
        (these are populated for every ephys_manager row) so `base_path`
        being NaN does not block loading.
        """
        if CacheManager is None:
            raise ImportError(
                "ephys_manager CacheManager not available — cannot load "
                "spike data for source=ephys_manager rows from this "
                "braindance install"
            )

        uuid = self._row.get("_original_uuid", "")
        experiment = self._row.get("_original_experiment", "")
        file_extension = self._row.get("file_extension", "_acqm.zip")
        if pd.isna(file_extension) or not file_extension:
            file_extension = "_acqm.zip"
        # Exact S3 key when the catalog has one (required for v2-layout sorts,
        # whose path contains content hashes). NaN for data_manager rows.
        acqm_key = self._row.get("acqm_key", None)

        if not uuid or not experiment or pd.isna(uuid) or pd.isna(experiment):
            print(
                "✗ ERROR: Missing _original_uuid or _original_experiment "
                "for ephys_manager recording"
            )
            return None

        try:
            from braindance import get_data_dir

            cache_mgr = CacheManager(data_dir=get_data_dir())
            local_path = cache_mgr.get_or_download(
                uuid, experiment, file_extension, acqm_key
            )

            with zipfile.ZipFile(local_path, "r") as f_zip:
                qm = f_zip.open("qm.npz")
                data = np.load(qm, allow_pickle=True)
                spike_times = data["train"].item()
                fs = data["fs"]
                # index-align train and neuron_attributes off the SAME iteration,
                # keyed by the actual cluster_id -- do not assume cluster_id is a
                # dense 0..N-1 range (autocuration can drop clusters).
                cluster_ids = list(spike_times.keys())
                train = [spike_times[cid] / fs * 1000 for cid in cluster_ids]
                # neuron_data only covers clusters that survived autocuration as
                # single units (e.g. MUA clusters are dropped) -- SpikeData requires
                # neuron_attributes to be DENSE (one entry per train index), so units
                # without a footprint get an empty dict rather than being omitted.
                neuron_data = data["neuron_data"].item() if "neuron_data" in data.files else {}
                # 🚨 `neuron_data` is keyed 0..N-1 DENSE POSITIONAL, while `train`
                # is keyed by the ACTUAL cluster_id, which is sparse. Looking up
                # neuron_data[cluster_id] therefore silently returns a DIFFERENT
                # neuron's entry whenever cluster_id happens to fall inside
                # 0..N-1, and {} otherwise. Measured on three files: 8/34, 35/65
                # and 87/149 lookups "hit", and 7, 35 and 83 of those hits
                # belonged to another cluster. The rest were read as "this unit
                # has no footprint", which is how it hid.
                #
                # Every entry carries its own `cluster_id`, and those match the
                # sorted train keys exactly -- neuron_data covers the SAME
                # clusters, nothing is dropped. So map by that field and never by
                # the dict key.
                by_cid = {}
                for entry in neuron_data.values():
                    if isinstance(entry, dict) and "cluster_id" in entry:
                        by_cid[entry["cluster_id"]] = entry
                if by_cid:
                    neuron_attributes = {
                        i: by_cid.get(cid, {}) for i, cid in enumerate(cluster_ids)
                    }
                else:
                    # No cluster_id field to key on. Fall back to POSITION, which
                    # is the layout neuron_data actually has -- not to the sparse
                    # cluster id, which is what was wrong here.
                    nd_by_pos = [neuron_data[k] for k in sorted(neuron_data)]
                    neuron_attributes = {
                        i: (nd_by_pos[i] if i < len(nd_by_pos) else {})
                        for i in range(len(cluster_ids))
                    }
                metadata = {"fs": float(fs)}
                if "redundant_pairs" in data.files:
                    metadata["redundant_pairs"] = data["redundant_pairs"]
                if "config" in data.files:
                    metadata["array_config"] = data["config"].item()
                return SpikeData(
                    train=train, neuron_attributes=neuron_attributes, metadata=metadata
                )
        except Exception as e:
            print(f"✗ Failed to load spikes from autocuration: {e}")
            return None

    def _load_spikes(self) -> Any:
        """
        Load spike data from .pkl file (local or S3).
        - Try to load pickle
        - Handle multiple formats (SpikeData object, dict with 'train', list of arrays)
        - Convert to SpikeData if needed
        - Return None on failure
        """
        # ephys_manager rows: load from Anthos autocuration zip rather than
        # the data_manager spike_data.pkl path (which requires base_path and
        # doesn't exist for these recordings).
        if self._row.get("source", "") == "ephys_manager":
            return self._load_spikes_from_autocuration()

        self._resolve_paths()
        spike_path = self._spikes_path

        if spike_path is None:
            # Get base path for error message
            base_path = self._base_path
            if base_path is None:
                if "base_path" in self._row.index and pd.notna(self._row["base_path"]):
                    base_path = Path(str(self._row["base_path"]))
                else:
                    from braindance import get_data_dir

                    base_path = get_data_dir()

            print(f"✗ ERROR: Could not resolve spike data path")
            print(f"  Identifier: {self.identifier}")
            print(f"  BASE_PATH: {base_path}")
            print(f"  Hint: Check if BASE_PATH is correct. Expected structure:")
            print(
                f"        {base_path}/<proj>/<chip>/<experiment>/<experiment>_spike_data.pkl"
            )
            return None

        # Check if S3 path
        if isinstance(spike_path, (str, Path)) and str(spike_path).startswith("s3://"):
            loader = self._get_s3_loader()
            data = loader.load_pickle(str(spike_path))
            return as_spike_data(data)

        # Local path
        if not spike_path.exists():
            # Try to download from S3
            s3_path = self._construct_s3_spike_path()
            if s3_path:
                print(f"✗ Spike data file not found locally: {spike_path}")
                print(f"  [S3] Attempting to download from S3...")
                if self._download_from_s3(s3_path, spike_path):
                    # Successfully downloaded, continue to load
                    pass
                else:
                    print(f"  [S3] ✗ Could not download from: {s3_path}")
                    return None
            else:
                # Get base path for error message
                base_path = self._base_path
                if base_path is None:
                    if "base_path" in self._row.index and pd.notna(
                        self._row["base_path"]
                    ):
                        base_path = Path(str(self._row["base_path"]))
                    else:
                        from braindance import get_data_dir

                        base_path = get_data_dir()

                print(f"✗ ERROR: Spike data file not found")
                print(f"  Identifier: {self.identifier}")
                print(f"  Expected path: {spike_path}")
                print(f"  BASE_PATH: {base_path}")
                print(f"  Hint: Check if BASE_PATH is correct. Use:")
                print(
                    f"        rec = load_recording(proj, chip, experiment, base_path='{base_path}')"
                )
                return None

        # Final check after potential download
        if not spike_path.exists():
            return None

        # Use safe loading pattern from BusyBeeLoader
        return self._load_pickle_safe(spike_path)

    def _adjust_stim_times(
        self,
        stim_log: pd.DataFrame,
        spike_data: "SpikeData",
        adjust_window_ms: float = 100.0,
    ) -> pd.DataFrame:
        """
        Adjust stimulation times using artifact detection.

        Reuses BusyBeeLoader.adjust_stim_times (vectorized implementation):
        - Vectorized nearest-neighbor search using np.searchsorted
        - Only adjust if artifact within window
        - Add 'time_mod' column (keeps original 'time')

        Args:
            stim_log: DataFrame with 'time' column (in seconds)
            spike_data: SpikeData with metadata['artifact_times'] (in ms)
            adjust_window_ms: Window for considering artifact match

        Returns:
            DataFrame with added 'time_mod' column
        """
        if (
            not hasattr(spike_data, "metadata")
            or "artifact_times" not in spike_data.metadata
        ):
            # No artifacts available - return original without modification
            # DO NOT copy 'time' to 'time_mod' - that would defeat artifact correction
            return stim_log

        # Convert artifacts to seconds
        artifacts = spike_data.metadata["artifact_times"] / 1000
        # Coerce stim time column to float — some CSVs (e.g. p001237
        # 24-04-18_butterfly cartpole_long_*) save 'time' as object/string,
        # which propagates here and breaks np.searchsorted with a
        # `'<' not supported between float and str` TypeError. Drop rows
        # that can't be parsed (rather than silently producing NaN times).
        stim_times = pd.to_numeric(stim_log["time"], errors="coerce").values
        valid_mask = ~np.isnan(stim_times)
        if not valid_mask.all():
            n_dropped = int((~valid_mask).sum())
            print(
                f"  [stim_log] dropped {n_dropped}/{len(stim_times)} rows with "
                f"non-numeric 'time' values"
            )
            stim_log = stim_log.loc[valid_mask].reset_index(drop=True)
            stim_times = stim_times[valid_mask]
        adjust_window_s = adjust_window_ms / 1000

        # Vectorized nearest-neighbor search using searchsorted
        insert_indices = np.searchsorted(artifacts, stim_times)

        # Handle edge cases where insert_indices might be at boundaries
        left_indices = np.clip(insert_indices - 1, 0, len(artifacts) - 1)
        right_indices = np.clip(insert_indices, 0, len(artifacts) - 1)

        # Calculate distances to left and right artifacts
        left_distances = np.abs(stim_times - artifacts[left_indices])
        right_distances = np.abs(stim_times - artifacts[right_indices])

        # Choose the closest artifact for each stim time
        use_left = left_distances <= right_distances
        closest_artifacts = np.where(
            use_left, artifacts[left_indices], artifacts[right_indices]
        )
        closest_distances = np.where(use_left, left_distances, right_distances)

        # Only adjust if within the window, otherwise keep original time
        stim_times_mod = np.where(
            closest_distances < adjust_window_s, closest_artifacts, stim_times
        )

        # Create modified log
        stim_log_mod = stim_log.copy()
        stim_log_mod["time_mod"] = stim_times_mod
        return stim_log_mod

    def _load_stim_log(self) -> Optional[pd.DataFrame]:
        """
        Load stimulus log with artifact-based time adjustment.

        Implementation:
        1. Load CSV from self._resolved_paths['stim_log']
        2. Check if file exists and is not empty (<1000 bytes)
        3. Apply artifact-based time adjustment if artifacts available
        4. Add 'time_mod' column with corrected times
        """
        self._resolve_paths()
        stim_path = self._stim_log_path

        if stim_path is None:
            return None

        # Check if path exists (for local files)
        if not isinstance(stim_path, (str, Path)) or (
            not str(stim_path).startswith("s3://") and not stim_path.exists()
        ):
            # Try to download from S3
            s3_path = self._construct_s3_stim_log_path()
            if s3_path:
                if self._download_from_s3(s3_path, stim_path):
                    # Successfully downloaded, continue to load
                    pass
                else:
                    return None
            else:
                return None

        # Check file size (empty logs are <1000 bytes)
        if not isinstance(stim_path, (str, Path)) or not str(stim_path).startswith(
            "s3://"
        ):
            if stim_path.stat().st_size < 1000:
                return None

        # Load CSV (handle S3 or local)
        try:
            if isinstance(stim_path, (str, Path)) and str(stim_path).startswith(
                "s3://"
            ):
                loader = self._get_s3_loader()
                stim_log = loader.load_csv(str(stim_path))
                if stim_log is None:
                    return None
            else:
                stim_log = pd.read_csv(stim_path)
        except Exception as e:
            print(f"⚠️  Failed to load stim log: {e}")
            return None

        if stim_log.empty:
            return None

        # Check if artifact correction already done in the file we just loaded
        if "time_mod" in stim_log.columns:
            return stim_log

        # Apply artifact-based time adjustment if spikes available
        spike_data = self.spikes  # Trigger lazy load
        if spike_data is not None:
            stim_log = self._adjust_stim_times(stim_log, spike_data)

            # Persist only an actual correction. Without artifact metadata the
            # original CSV must remain intact (including its verified checksum).
            if "time_mod" in stim_log.columns and not str(stim_path).startswith("s3://"):
                try:
                    stim_log.to_csv(stim_path, index=False)
                    print(
                        f"  [CACHE] Saved corrected stim log with time_mod to {stim_path.name}"
                    )

                    # Upload to S3 if auto_upload enabled
                    if self._auto_upload:
                        s3_path = self._construct_s3_stim_log_path()
                        if s3_path:
                            # Use the new upload_file
                            loader = self._get_s3_loader()
                            loader.upload_file(stim_path, s3_path)
                except Exception as e:
                    print(f"  ⚠️  Failed to save/upload corrected stim log: {e}")

        return stim_log

    def _load_game_log(self) -> Optional[pd.DataFrame]:
        """
        Load game log from CSV (for RL/CartPole experiments).

        Game log contains: time, pole_angle, reward, action, spike_count_l,
        spike_count_r, state (or similar columns depending on experiment).

        Returns:
            DataFrame or None if file not found/empty
        """
        self._resolve_paths()
        game_log_path = self._game_log_path

        if game_log_path is None:
            return None

        # Check if path exists (for local files)
        if not isinstance(game_log_path, (str, Path)) or (
            not str(game_log_path).startswith("s3://") and not game_log_path.exists()
        ):
            # Try to download from S3
            s3_path = self._construct_s3_game_log_path()
            if s3_path:
                if self._download_from_s3(s3_path, game_log_path):
                    # Successfully downloaded, continue to load
                    pass
                else:
                    return None
            else:
                return None

        # Check file size (only reject truly empty files)
        # Verify file exists after potential S3 download
        if not isinstance(game_log_path, (str, Path)) or not str(
            game_log_path
        ).startswith("s3://"):
            if not game_log_path.exists():
                return None
            if game_log_path.stat().st_size < 100:
                return None

        # Load CSV (handle S3 or local)
        try:
            if isinstance(game_log_path, (str, Path)) and str(game_log_path).startswith(
                "s3://"
            ):
                loader = self._get_s3_loader()
                game_log = loader.load_csv(str(game_log_path))
                if game_log is None:
                    return None
            else:
                game_log = pd.read_csv(game_log_path)
        except Exception as e:
            print(f"⚠️  Failed to load game log: {e}")
            return None

        if game_log.empty:
            return None

        # Automatically parse array columns if they are stringified
        array_columns = ["state", "spike_count_l", "spike_count_r"]
        for col in array_columns:
            if col in game_log.columns:
                try:
                    # Check if the column is stringified (starts with [ or looks like a list)
                    sample = game_log[col].dropna().iloc[0]
                    if isinstance(sample, str) and (
                        sample.startswith("[") or " " in sample
                    ):

                        def safe_parse(x):
                            if isinstance(x, str):
                                try:
                                    # Handle numpy-style space-separated strings "[ 0.1 0.2 ]"
                                    # ast.literal_eval needs commas, so we clean it up if needed
                                    cleaned = x.strip()
                                    if (
                                        cleaned.startswith("[")
                                        and "," not in cleaned
                                        and " " in cleaned
                                    ):
                                        # Convert "[ 0.1 0.2 ]" to "[0.1, 0.2]"
                                        cleaned = (
                                            "[" + ",".join(cleaned[1:-1].split()) + "]"
                                        )
                                    return np.array(ast.literal_eval(cleaned))
                                except:
                                    return x
                            return x

                        game_log[col] = game_log[col].apply(safe_parse)
                except Exception:
                    # Silently skip if parsing fails or column is empty
                    pass

        return game_log

    def _load_pattern_log(self) -> Optional[pd.DataFrame]:
        """
        Load pattern log from CSV (for RL/CartPole experiments).

        Pattern log contains: time, pattern, reward, probs, vals
        (stimulation patterns and RL policy information).

        Returns:
            DataFrame or None if file not found/empty
        """
        self._resolve_paths()
        pattern_log_path = self._pattern_log_path

        if pattern_log_path is None:
            return None

        # Check if path exists (for local files)
        if not isinstance(pattern_log_path, (str, Path)) or (
            not str(pattern_log_path).startswith("s3://")
            and not pattern_log_path.exists()
        ):
            # Try to download from S3
            s3_path = self._construct_s3_pattern_log_path()
            if s3_path:
                if self._download_from_s3(s3_path, pattern_log_path):
                    # Successfully downloaded, continue to load
                    pass
                else:
                    return None
            else:
                return None

        # Check file size (only reject truly empty files)
        if not isinstance(pattern_log_path, (str, Path)) or not str(
            pattern_log_path
        ).startswith("s3://"):
            if pattern_log_path.stat().st_size < 100:
                return None

        # Load CSV (handle S3 or local)
        try:
            if isinstance(pattern_log_path, (str, Path)) and str(
                pattern_log_path
            ).startswith("s3://"):
                loader = self._get_s3_loader()
                pattern_log = loader.load_csv(str(pattern_log_path))
                if pattern_log is None:
                    return None
            else:
                pattern_log = pd.read_csv(pattern_log_path)
        except Exception as e:
            print(f"⚠️  Failed to load pattern log: {e}")
            return None

        if pattern_log.empty:
            return None

        return pattern_log

    def _load_reward_log(self) -> Optional[pd.DataFrame]:
        """
        Load reward log from CSV (for RL/CartPole experiments).

        Reward log contains: time, episode, reward, episode_steps
        (episode-level reward information).

        Unlike stim_log, no artifact-based time adjustment is applied.

        Returns:
            DataFrame or None if file not found/empty
        """
        self._resolve_paths()
        reward_log_path = self._reward_log_path

        if reward_log_path is None:
            return None

        # Check if path exists (for local files)
        if not isinstance(reward_log_path, (str, Path)) or (
            not str(reward_log_path).startswith("s3://")
            and not reward_log_path.exists()
        ):
            # Try to download from S3
            s3_path = self._construct_s3_reward_log_path()
            if s3_path:
                print(f"  [S3] Attempting to download reward_log from: {s3_path}")
                if self._download_from_s3(s3_path, reward_log_path):
                    # Successfully downloaded, continue to load
                    pass
                else:
                    return None
            else:
                return None

        # Check file size (reward logs are small - just episode summaries)
        # Use lower threshold than game_log since reward logs have fewer rows
        if not isinstance(reward_log_path, (str, Path)) or not str(
            reward_log_path
        ).startswith("s3://"):
            file_size = reward_log_path.stat().st_size
            if file_size < 100:  # Only reject truly empty files
                print(
                    f"  [DEBUG] reward_log file too small ({file_size} bytes), skipping"
                )
                return None

        # Load CSV (handle S3 or local)
        try:
            if isinstance(reward_log_path, (str, Path)) and str(
                reward_log_path
            ).startswith("s3://"):
                loader = self._get_s3_loader()
                reward_log = loader.load_csv(str(reward_log_path))
                if reward_log is None:
                    return None
            else:
                reward_log = pd.read_csv(reward_log_path)
        except Exception as e:
            print(f"⚠️  Failed to load reward log: {e}")
            return None

        if reward_log.empty:
            return None

        return reward_log

    def _load_mapping(self) -> Any:
        """
        Load electrode mapping from CSV or pickle and return as Mapping object.

        Returns:
            Mapping object from braindance.analysis.mapping, or None if not available
        """
        self._resolve_paths()
        mapping_path = self._mapping_path

        if mapping_path is None:
            return None

        # Check if path exists (for local files)
        if not isinstance(mapping_path, (str, Path)) or (
            not str(mapping_path).startswith("s3://") and not mapping_path.exists()
        ):
            # Try to download from S3 if catalog has base_path
            s3_base_path = self._row.get("base_path", "")
            if isinstance(s3_base_path, str) and s3_base_path.startswith("s3://"):
                chip = self._row.get("chip", "")
                experiment = self._row.get("experiment", "")
                if chip and experiment:
                    # Parse experiment components using unified method
                    exp_base, _ = self._parse_experiment_components()

                    # Ensure base_path has trailing slash
                    if not s3_base_path.endswith("/"):
                        s3_base_path = s3_base_path + "/"

                    # Try S3 path: {s3_base_path}{chip}/{exp_base}/{exp_base}_mapping.csv
                    s3_mapping_path = (
                        f"{s3_base_path}{chip}/{exp_base}/{exp_base}_mapping.csv"
                    )
                    print(
                        f"  [S3] Attempting to download mapping from: {s3_mapping_path}"
                    )
                    if self._download_from_s3(s3_mapping_path, mapping_path):
                        # Successfully downloaded, continue to load
                        pass
                    else:
                        return None
                else:
                    return None
            else:
                return None

        # Load based on file extension
        mapping_df = None
        if isinstance(mapping_path, Path) and mapping_path.suffix == ".csv":
            try:
                mapping_df = pd.read_csv(mapping_path)
            except Exception as e:
                print(f"⚠️  Failed to load mapping from {mapping_path.name}: {e}")
                return None
        else:
            # Assume pickle for other extensions
            mapping_df = self._load_pickle_safe(mapping_path)

        if mapping_df is None:
            return None

        # Import and create Mapping object
        try:
            from braindance.analysis.mapping import Mapping

            return Mapping.from_df(mapping_df)
        except ImportError:
            print("⚠️  Could not import Mapping class from braindance.analysis.mapping")
            # Fallback to returning DataFrame
            return mapping_df

    def _load_spike_locations(self) -> Any:
        """
        Load spike locations (neuron spatial coordinates) from JSON or pickle.

        Returns:
            List of (x, y) numpy arrays for each neuron, or None if not available
        """
        self._resolve_paths()
        spike_loc_path = self._spike_locations_path

        if spike_loc_path is None:
            return None

        # Check if files need to be downloaded from S3
        json_path = (
            spike_loc_path.with_suffix(".json")
            if isinstance(spike_loc_path, Path) and spike_loc_path.suffix != ".json"
            else spike_loc_path
        )
        pkl_path = (
            spike_loc_path.with_suffix(".pkl")
            if isinstance(spike_loc_path, Path) and spike_loc_path.suffix != ".pkl"
            else spike_loc_path
        )

        # Try S3 download if neither local file exists
        if (isinstance(json_path, Path) and not json_path.exists()) and (
            isinstance(pkl_path, Path) and not pkl_path.exists()
        ):
            # Try to download from S3 if catalog has base_path
            s3_base_path = self._row.get("base_path", "")
            if isinstance(s3_base_path, str) and s3_base_path.startswith("s3://"):
                chip = self._row.get("chip", "")
                experiment = self._row.get("experiment", "")
                # Ensure base_path has trailing slash
                if not s3_base_path.endswith("/"):
                    s3_base_path = s3_base_path + "/"
                if chip:
                    # Try chip-level first: {s3_base}/{chip}/spike_info.{json,pkl}
                    for suffix in [".json", ".pkl"]:
                        s3_spike_info_path = f"{s3_base_path}{chip}/spike_info{suffix}"
                        local_path = spike_loc_path.with_suffix(suffix)
                        print(
                            f"  [S3] Attempting to download spike_info from: {s3_spike_info_path}"
                        )
                        if self._download_from_s3(s3_spike_info_path, local_path):
                            # Successfully downloaded
                            break

                    # Clean up empty files from failed chip-level downloads
                    if (
                        isinstance(json_path, Path)
                        and json_path.exists()
                        and json_path.stat().st_size == 0
                    ):
                        json_path.unlink()
                    if (
                        isinstance(pkl_path, Path)
                        and pkl_path.exists()
                        and pkl_path.stat().st_size == 0
                    ):
                        pkl_path.unlink()

                    # If chip-level didn't work and we have experiment, try experiment-level
                    # Check file size to detect failed downloads that created empty files
                    json_exists_valid = (
                        isinstance(json_path, Path)
                        and json_path.exists()
                        and json_path.stat().st_size > 0
                    )
                    pkl_exists_valid = (
                        isinstance(pkl_path, Path)
                        and pkl_path.exists()
                        and pkl_path.stat().st_size > 0
                    )
                    if not json_exists_valid and not pkl_exists_valid and experiment:
                        # Parse experiment components using unified method
                        exp_base, _ = self._parse_experiment_components()

                        # Ensure base_path has trailing slash
                        if not s3_base_path.endswith("/"):
                            s3_base_path = s3_base_path + "/"

                        for suffix in [".json", ".pkl"]:
                            s3_spike_info_path = (
                                f"{s3_base_path}{chip}/{exp_base}/spike_info{suffix}"
                            )
                            local_path = spike_loc_path.with_suffix(suffix)
                            print(
                                f"  [S3] Attempting to download spike_info from: {s3_spike_info_path}"
                            )
                            if self._download_from_s3(s3_spike_info_path, local_path):
                                # Successfully downloaded
                                break

                        # Clean up empty files from failed downloads
                        if (
                            isinstance(json_path, Path)
                            and json_path.exists()
                            and json_path.stat().st_size == 0
                        ):
                            json_path.unlink()
                        if (
                            isinstance(pkl_path, Path)
                            and pkl_path.exists()
                            and pkl_path.stat().st_size == 0
                        ):
                            pkl_path.unlink()

        # If spike_info still doesn't exist or is empty, try extracting from RT-Sort (if enabled)
        json_exists_valid = (
            isinstance(json_path, Path)
            and json_path.exists()
            and json_path.stat().st_size > 0
        )
        pkl_exists_valid = (
            isinstance(pkl_path, Path)
            and pkl_path.exists()
            and pkl_path.stat().st_size > 0
        )

        if not json_exists_valid and not pkl_exists_valid:
            from braindance.config import get_auto_extract_spike_info

            if get_auto_extract_spike_info():
                # Clean up empty files first
                if (
                    isinstance(json_path, Path)
                    and json_path.exists()
                    and json_path.stat().st_size == 0
                ):
                    json_path.unlink()
                if (
                    isinstance(pkl_path, Path)
                    and pkl_path.exists()
                    and pkl_path.stat().st_size == 0
                ):
                    pkl_path.unlink()
                self._extract_spike_info_from_rt_sort(json_path)

        # Try JSON first (preferred format)
        if isinstance(json_path, Path) and json_path.exists():
            try:
                import json
                import numpy as np

                with open(json_path, "r") as f:
                    spike_info = json.load(f)

                # Convert lists back to numpy arrays
                spike_locs = [np.array(loc) for loc in spike_info.get("spike_locs", [])]

                if len(spike_locs) > 0:
                    return spike_locs
            except Exception as e:
                print(f"⚠️  Failed to load spike locations from JSON: {e}")

        # Try pickle as fallback
        pkl_path = (
            spike_loc_path.with_suffix(".pkl")
            if isinstance(spike_loc_path, Path) and spike_loc_path.suffix != ".pkl"
            else spike_loc_path
        )
        if isinstance(pkl_path, Path) and pkl_path.exists():
            try:
                import pickle
                import numpy as np

                with open(pkl_path, "rb") as f:
                    spike_info = pickle.load(f)

                # Handle both list and numpy array formats
                if spike_info.get("data_type") == "list":
                    spike_locs = [
                        np.array(loc) for loc in spike_info.get("spike_locs", [])
                    ]
                else:
                    spike_locs = spike_info.get("spike_locs", [])

                if len(spike_locs) > 0:
                    return spike_locs
            except Exception as e:
                print(f"⚠️  Failed to load spike locations from pickle: {e}")

        return None

    def _extract_spike_info_from_rt_sort(self, json_path: Path) -> bool:
        """
        Extract spike_info from RT-Sort pickle and save as JSON.

        Args:
            json_path: Path where spike_info.json should be saved

        Returns:
            bool: True if extraction succeeded, False otherwise
        """
        try:
            import json
            import pickle
            import numpy as np
            import smart_open

            # Get S3 paths
            s3_base_path = self._row.get("base_path", "")
            if (
                not isinstance(s3_base_path, str)
                or not s3_base_path
                or not s3_base_path.startswith("s3://")
            ):
                return False

            chip = self._row.get("chip", "")
            experiment = self._row.get("experiment", "")

            if not chip or not experiment:
                return False

            # Parse experiment components using unified method
            exp_base, _ = self._parse_experiment_components()

            # Ensure base_path has trailing slash
            if not s3_base_path.endswith("/"):
                s3_base_path = s3_base_path + "/"

            # RT-Sort pickle path
            rt_sort_patterns = [
                f"{s3_base_path}{chip}/{exp_base}/rt_sort/{exp_base}_rt_sort.pickle",
            ]

            rt_sorter = None
            source_path = None
            last_error = None

            for rt_sort_path in rt_sort_patterns:
                try:
                    # RT-Sort files may contain PyTorch tensors saved on CUDA
                    # Use torch.load with map_location='cpu' to handle this on CPU-only machines
                    import torch
                    import io

                    with smart_open.open(rt_sort_path, "rb", transport_params={
                        "client": self._get_s3_loader()._s3_client,
                    }) as f:
                        # Read into buffer first since smart_open may not support seek
                        buffer = io.BytesIO(f.read())
                        rt_sorter = torch.load(
                            buffer, map_location="cpu", weights_only=False
                        )
                    source_path = rt_sort_path
                    break
                except Exception as e:
                    last_error = str(e)
                    continue

            if rt_sorter is None:
                print(f"  [RT-Sort] ✗ Could not find RT-Sort pickle on S3")
                if last_error:
                    print(f"  [RT-Sort] Last error: {last_error}")
                return False

            # Extract spike locations and channels
            spike_locs = rt_sorter.seq_locs  # List of np arrays of shape (2,)
            spike_channels = rt_sorter.seq_comp_elecs  # List of electrode IDs

            # Convert to pure Python types (avoid numpy in JSON)
            def convert_to_python(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, list):
                    return [convert_to_python(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_to_python(item) for item in obj)
                else:
                    return obj

            spike_locs_clean = convert_to_python(spike_locs)
            spike_channels_clean = convert_to_python(spike_channels)

            # Package spike_info
            spike_info = {
                "spike_locs": spike_locs_clean,
                "spike_channels": spike_channels_clean,
                "n_neurons": len(spike_locs),
                "proj": self._row.get("proj", ""),
                "chip": chip,
                "experiment": exp_base,
                "source_path": source_path,
                "format_version": "2.0",
                "data_type": "json",
            }

            # Save locally
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(spike_info, f, indent=2)

            file_size = json_path.stat().st_size / 1024 / 1024
            print(
                f"  ✓ Extracted spike_info from RT-Sort: {len(spike_locs)} neurons ({file_size:.2f} MB)"
            )

            # Upload to S3 for future use
            s3_spike_info_path = f"{s3_base_path}{chip}/{exp_base}/spike_info.json"
            try:
                with smart_open.open(s3_spike_info_path, "wb") as f_out:
                    with open(json_path, "rb") as f_in:
                        f_out.write(f_in.read())
                print(f"  ✓ Uploaded spike_info to S3 for future use")
            except Exception as e:
                print(f"  ⚠️  Could not upload to S3: {e}")

            return True

        except Exception as e:
            print(f"  [RT-Sort] ✗ Failed to extract spike_info: {e}")
            return False

    def _load_raw_data(
        self,
        channels: Optional[List[int]] = None,
        start: int = 0,
        length: int = -1,
        spikes: bool = False,
        dtype: np.dtype = np.float32,
        suffix: Optional[str] = None,
        verbose: bool = False,
        sync_if_missing: bool = False,
    ) -> Any:
        """
        Load raw data (lazy, returns loader or memmap).

        Args:
            channels: Channel indices to load (optional)
            start: Starting frame offset
            length: Number of frames to load (-1 for all)
            spikes: Whether to load spikes table
            dtype: Data type to load
            suffix: Optional suffix override (e.g., '.raw.h5')
            verbose: Enable verbose logging in loader
            sync_if_missing: Download from S3 if local path is missing
        """
        from braindance.analysis import data_loader

        self._resolve_paths()

        raw_path = None
        if "full_path" in self._row.index and pd.notna(self._row.get("full_path")):
            raw_path = str(self._row["full_path"]).strip()
            if raw_path.endswith("/"):
                _, exp_name = self._parse_experiment_components()
                if exp_name:
                    raw_path = f"{raw_path}{exp_name}"

        if not raw_path:
            raw_path = self._raw_data_path

        if raw_path is None:
            print("⚠️  Raw data path not found in catalog or resolved paths.")
            return None

        # If local path is missing but S3 exists, either use S3 directly or sync
        s3_path = None
        if not str(raw_path).startswith("s3://"):
            raw_path_obj = Path(raw_path)
            if not raw_path_obj.exists():
                s3_path = self._construct_s3_raw_data_path()
                if s3_path and not sync_if_missing:
                    raw_path = s3_path
                elif s3_path and sync_if_missing:
                    if self._download_from_s3(s3_path, raw_path_obj):
                        raw_path = raw_path_obj
                    else:
                        print(f"  [S3] ✗ Could not download raw data from: {s3_path}")
                        return None
        elif sync_if_missing:
            s3_path = str(raw_path)
            local_target = self._raw_data_path
            if local_target is None:
                print("⚠️  Raw data local path not resolved for sync.")
                return None
            if not local_target.exists():
                if self._download_from_s3(s3_path, local_target):
                    raw_path = local_target
                else:
                    print(f"  [S3] ✗ Could not download raw data from: {s3_path}")
                    return None

        return data_loader.load_data_maxwell(
            raw_path,
            channels=channels,
            start=start,
            length=length,
            spikes=spikes,
            dtype=dtype,
            suffix=suffix,
            verbose=verbose,
        )

    def sync_raw_data(
        self, overwrite: bool = False, local_path: Optional[Union[str, Path]] = None
    ) -> Optional[Path]:
        """
        Download raw data for this recording to local storage.

        Args:
            overwrite: If True, re-download even if local file exists
            local_path: Optional explicit local path to save the raw file

        Returns:
            Path to local raw data file or None on failure
        """
        self._resolve_paths()

        target_path = Path(local_path) if local_path else self._raw_data_path
        if target_path is None:
            print("⚠️  Raw data local path not resolved.")
            return None

        if target_path.exists() and not overwrite:
            return target_path

        s3_path = self._construct_s3_raw_data_path()
        if not s3_path:
            print("⚠️  Raw data S3 path not found in catalog.")
            return None

        if self._download_from_s3(s3_path, target_path):
            return target_path
        return None

    def sync_experiment_raw_data(
        self, overwrite: bool = False, legacy_flat: Optional[bool] = None
    ) -> List[Path]:
        """
        Download all raw data files for this experiment to local storage.

        Args:
            overwrite: If True, re-download existing local files
            legacy_flat: Override legacy/standard local structure

        Returns:
            List of local file paths that exist after sync
        """
        from braindance import get_data_dir
        from braindance.utils.data_manager.utils.catalogging.generator import (
            DataPathManager,
        )

        exp_dir = self._construct_s3_experiment_dir()
        if not exp_dir:
            print("⚠️  Experiment S3 path not found in catalog.")
            return []

        manager = DataPathManager()
        raw_files = manager.get_raw_data_files(exp_dir)
        if not raw_files:
            print(f"⚠️  No raw data files found at: {exp_dir}")
            return []

        proj = self._row.get("proj", "")
        if not proj:
            base_path = str(self._row.get("base_path", "")).rstrip("/")
            if base_path.startswith("s3://"):
                proj = base_path.split("/")[-1]

        chip = self._row.get("chip", "")
        use_legacy = legacy_flat if legacy_flat is not None else self._legacy_flat
        if use_legacy is None:
            use_legacy = True

        local_root = get_data_dir()
        local_paths: List[Path] = []

        for s3_file in raw_files:
            filename = str(s3_file).rstrip("/").split("/")[-1]
            if not filename.endswith(".raw.h5"):
                continue
            exp_name = filename[: -len(".raw.h5")]

            if use_legacy:
                target_path = local_root / proj / chip / filename
            else:
                target_path = local_root / proj / chip / exp_name / filename

            if target_path.exists() and not overwrite:
                local_paths.append(target_path)
                continue

            if self._download_from_s3(s3_file, target_path):
                local_paths.append(target_path)

        return local_paths

    # ==================== Results ====================

    @property
    def results(self) -> DataContext:
        """
        Access results (derived/computed data).

        Results are stored separately from primary data and persist across sessions.

        Usage:
            rec.results.connectivity = compute_connectivity(rec.spikes)
            rec.results.save()

            # Later
            conn = rec.results.connectivity  # Loads from disk
        """
        if self._results is None:
            results_path = self._results_path
            self._results = DataContext(path=results_path, overwrite_existing=True)
        return self._results

    @property
    def cache(self) -> ResultsCache:
        """
        Access S3-backed results cache for derived data.

        The cache provides get_or_compute semantics with automatic
        S3 sync for container deployments.

        Usage:
            # Get binned firing rates (cached)
            binned = rec.cache.get_or_compute(
                'binned_fr',
                params={'bin_ms': 20},
                compute_fn=lambda: bin_spike_data_vectorized(rec.spikes.train, 20)
            )
        """
        if self._results_cache is None:
            self._resolve_paths()

            # Get base_path from catalog for S3 results path
            base_path = self._row.get("base_path", "")
            if pd.isna(base_path):
                base_path = ""
            chip = self._row.get("chip", "")
            experiment = self._row.get("experiment", "")
            if pd.isna(experiment):
                experiment = ""
            exp_name = Path(experiment).name if experiment else ""

            # 🚨 `base_path` IS NOT A "CAN THIS RESOLVE" TEST. The catalogs carry two mutually
            # exclusive loading paths: data_manager rows have base_path/exp and a null `uuids`;
            # ephys_manager rows have `uuids` and a NULL base_path. Falling straight to
            # s3_path=None on a null base_path therefore disabled the S3 results cache for
            # EVERY ephys recording -- silently, in both directions: `exists_s3` always
            # returned False, `auto_download` could never fetch, and `upload_to_s3`
            # short-circuited on `if not self.s3_path`. Measured 2026-08-04 on the pan-cohort
            # FT cohort: 53/53 data_manager rows resolved, 0/247 ephys rows did, and all 14
            # sampled "missing" recordings already had their cache on S3 -- so the loader was
            # re-deriving them from raw acqm spikes every run, for nothing.
            #
            # ⚠️ Layout is per-experiment SUBDIRECTORY:
            #     s3://braingeneers/ephys/<uuid>/derived/results/<experiment>/binned_fr_10ms.npz
            # NOT an `<experiment>_` filename prefix (the convention the nrp ephys_manager copy
            # of ResultsCache builds via its `experiment_name` kwarg). Verified against the
            # bucket: 14/14 in the subdirectory form, 0/14 in the prefix form.
            uuid = self._row.get("_original_uuid", None)
            if uuid is None or pd.isna(uuid) is True:
                uuid = self._row.get("uuids", None)
            uuid = "" if uuid is None or pd.isna(uuid) is True else str(uuid).strip()

            orig_exp = self._row.get("_original_experiment", None)
            orig_exp = (exp_name if orig_exp is None or pd.isna(orig_exp) is True
                        else str(orig_exp).strip())

            # Construct S3 results path using catalog's base_path
            if base_path and chip and exp_name:
                s3_path = f"{base_path.rstrip('/')}/{chip}/{exp_name}/results"
            elif uuid and orig_exp:
                s3_path = f"s3://braingeneers/ephys/{uuid}/derived/results/{orig_exp}"
            else:
                s3_path = None

            self._results_cache = ResultsCache(
                local_path=self._results_path,
                s3_path=s3_path,
                auto_upload=self._auto_upload,
                auto_download=True,
            )
        return self._results_cache

    def get_binned_fr(
        self,
        bin_ms: float = 20.0,
        time_range: Optional[Tuple[float, float]] = None,
        force_recompute: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Get binned firing rates with automatic caching.

        Uses the ResultsCache to store/retrieve binned firing rates.
        Cached results are keyed by bin_ms parameter (e.g., binned_fr_20ms.npz).

        🚨 THE CACHE OBJECTS ON S3 ARE `binned_fr_<N>ms.npz.zst`, NOT `.npz`.
        `binned_fr` is in ResultsCache's zstd WIRE **and** LOCAL tiers, so both the
        S3 object and the on-disk file are zstd-compressed. This method handles that
        transparently — you do not need to decompress anything, and there is no
        `bin_ms` for which a plain `.npz` sibling exists.

        ⛔ DO NOT conclude "the binned FR was never extracted" from an
        `aws s3 ls .../binned_fr_10ms.npz` that comes back empty. That filename is
        not what is in the bucket. Test coverage with
        `rec.cache.exists_s3('binned_fr', {'bin_ms': N})` (or just call this method),
        NEVER by listing an S3 prefix for a guessed filename. The model was trained
        on these caches, so for any recording in the training corpus they exist.

        The only way this legitimately fails is a missing `zstandard` package, which
        now raises a named ImportError telling you to `pip install zstandard` —
        it no longer degrades into a silent cache miss that re-derives from spikes.

        Args:
            bin_ms: Bin size in milliseconds (default: 20)
            time_range: Optional (start_ms, end_ms) time range
            force_recompute: Bypass cache and recompute

        Returns:
            Dict with:
            - 'rates': (n_bins, n_neurons) array of spike counts per bin
            - 'time_axis_ms': Array of bin center times in milliseconds

        Example:
            >>> rec = load_recording('proj', 'chip', 'exp')
            >>> binned = rec.get_binned_fr(bin_ms=20)
            >>> rates = binned['rates']  # (n_bins, n_neurons)
            >>> times = binned['time_axis_ms']  # (n_bins,)
            >>>
            >>> # On subsequent calls, loads from cache:
            >>> binned2 = rec.get_binned_fr(bin_ms=20)  # Instant!
        """
        from braindance.utils.data_manager.utils.analysis.vectorized_binning import (
            bin_spike_data_vectorized,
        )

        # Build params dict (only include non-default time_range)
        params = {"bin_ms": int(bin_ms)}

        def compute_fn():
            """Compute binned firing rates."""
            spike_data = self.spikes
            if spike_data is None:
                raise ValueError(f"No spike data for {self.identifier}")

            # Get spike trains from SpikeData
            if hasattr(spike_data, "train"):
                spike_trains = spike_data.train
            elif isinstance(spike_data, dict) and "train" in spike_data:
                spike_trains = spike_data["train"]
            else:
                raise ValueError(f"Unknown spike data format for {self.identifier}")

            # Run binning
            rates, time_axis = bin_spike_data_vectorized(
                spike_trains, bin_size_ms=bin_ms, time_range=time_range, verbose=True
            )

            return {"rates": rates, "time_axis_ms": time_axis}

        return self.cache.get_or_compute(
            "binned_fr",
            params=params,
            compute_fn=compute_fn,
            force_recompute=force_recompute,
        )

    # ==================== Waveforms ====================

    def _load_wf(self) -> Optional["Waveforms"]:
        """Loader behind `rec.wf`. Reads the extracted-waveform npz at the
        default parameters; see `get_waveforms()` for anything else.

        Misses are memoised too. `_load_data_property` only caches non-None, so
        without this the 32 rows with no npz would re-attempt an S3 lookup on
        every `rec.wf` — and `if rec.wf is not None: ... rec.wf` already touches
        it twice. The flag lives in `_data` so `clear_cache()` drops it.
        """
        if self._data._data.get("_wf_missing"):
            return None
        wf = self.get_waveforms()
        if wf is None:
            self._data._data["_wf_missing"] = True
        return wf

    def has_waveforms(self, **kwargs) -> bool:
        """True if this recording has an extracted-waveform npz, local or on S3.

        This is the ONLY correct way to measure waveform coverage. Listing the
        S3 results prefix is not equivalent and under-counts badly, because rows
        resolve to prefixes you did not think to list.

        Takes the same keyword arguments as `get_waveforms()`.
        """
        from braindance.utils.data_manager.utils.data_loading.waveforms import (
            WAVEFORM_NAME,
            waveform_params,
        )

        params = waveform_params(**kwargs)
        return bool(
            self.cache.exists_local(WAVEFORM_NAME, params)
            or self.cache.exists_s3(WAVEFORM_NAME, params)
        )

    def get_waveforms(
        self,
        ms_before: float = 1.0,
        ms_after: float = 2.0,
        n_footprint: int = 64,
        artifact_guard_ms: float = 10.0,
        download: bool = True,
    ) -> Optional["Waveforms"]:
        """Per-unit mean spike waveforms, cut from the raw MaxWell `.raw.h5`.

        `rec.wf` is this at the default parameters, memoised. Use this method
        directly only to read a non-default extraction.

        Args:
            ms_before / ms_after: window around the spike, ms
            n_footprint: K, channels kept per unit (nearest by DISTANCE)
            artifact_guard_ms: stimulation-artifact guard half-width. Part of
                the cache key -- a guarded and an unguarded npz are different
                products, so changing it selects a different file.
            download: fetch from S3 if not already local

        Returns:
            A `Waveforms`, or **None** if this recording has no npz. Handle the
            None: coverage is not uniform. In the `wf_v1` run 2,820 of 2,852
            rows had one, but that is training 99.9% against holdout 97.1% --
            most misses are holdout drug recordings that were never sorted.

        Never computes on a miss. Computing means reading a multi-GB raw h5,
        which is a fan-out job (`waveform_extractor/`), not something an
        attribute access should trigger.

        Example:
            >>> wf = rec.wf
            >>> wf.waveforms.shape          # (n_units, K, n_samples)
            >>> wf.summary(live_only=True)  # per-unit table
            >>> wf.t_ms                     # time axis, t=0 at the spike
        """
        from braindance.utils.data_manager.utils.data_loading.waveforms import (
            WAVEFORM_NAME,
            Waveforms,
            waveform_params,
        )

        params = waveform_params(
            ms_before=ms_before,
            ms_after=ms_after,
            n_footprint=n_footprint,
            artifact_guard_ms=artifact_guard_ms,
        )

        if not self.cache.exists_local(WAVEFORM_NAME, params):
            if not download:
                return None
            if not self.cache.download_from_s3(WAVEFORM_NAME, params):
                return None

        data = self.cache.load_local(WAVEFORM_NAME, params)
        if data is None:
            return None
        return Waveforms(data, params=params, identifier=self.identifier)

    def get_binned_isi(
        self,
        bin_ms: float = 20.0,
        time_range: Optional[Tuple[float, float]] = None,
        force_recompute: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Get binned Inter-Spike Intervals (ISI) with automatic caching.

        Uses the ResultsCache to store/retrieve binned ISI.
        Cached results are keyed by bin_ms parameter (e.g., binned_isi_20ms.npz).

        ISI values are normalized by bin_ms by default (sentinel value for 0/1 spikes = 1.0).

        Args:
            bin_ms: Bin size in milliseconds (default: 20)
            time_range: Optional (start_ms, end_ms) time range
            force_recompute: Bypass cache and recompute

        Returns:
            Dict with:
            - 'isi': (n_bins, n_neurons) array of normalized mean ISI per bin
            - 'time_axis_ms': Array of bin center times in milliseconds
        """
        from braindance.utils.data_manager.utils.analysis.vectorized_binning import (
            compute_binned_isi_vectorized,
        )

        params = {"bin_ms": int(bin_ms)}

        def compute_fn():
            """Compute binned ISI."""
            spike_data = self.spikes
            if spike_data is None:
                raise ValueError(f"No spike data for {self.identifier}")

            # Get spike trains
            if hasattr(spike_data, "train"):
                spike_trains = spike_data.train
            elif isinstance(spike_data, dict) and "train" in spike_data:
                spike_trains = spike_data["train"]
            else:
                raise ValueError(f"Unknown spike data format for {self.identifier}")

            # Run ISI computation
            isi, time_axis = compute_binned_isi_vectorized(
                spike_trains,
                bin_size_ms=bin_ms,
                time_range=time_range,
                normalize=True,
                verbose=True,
            )

            return {"isi": isi, "time_axis_ms": time_axis}

        return self.cache.get_or_compute(
            "binned_isi",
            params=params,
            compute_fn=compute_fn,
            force_recompute=force_recompute,
        )

    def get_binned_fr_and_isi(
        self,
        bin_ms: float = 20.0,
        time_range: Optional[Tuple[float, float]] = None,
        force_recompute: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Get both binned firing rates and ISI in a single efficient call.

        This is the most efficient method for model training as it shares
        expensive sorting and digitizing operations.

        Args:
            bin_ms: Bin size in milliseconds (default: 20)
            time_range: Optional time range
            force_recompute: Bypass cache

        Returns:
            Dict with 'rates', 'isi', and 'time_axis_ms'.
        """
        from braindance.utils.data_manager.utils.analysis.vectorized_binning import (
            compute_binned_fr_isi_vectorized,
        )

        params = {"bin_ms": int(bin_ms)}

        def compute_fn():
            spike_data = self.spikes
            if spike_data is None:
                raise ValueError(f"No spike data for {self.identifier}")

            if hasattr(spike_data, "train"):
                spike_trains = spike_data.train
            else:
                spike_trains = spike_data["train"]

            # Compute both efficiently
            rates, isi, time_axis = compute_binned_fr_isi_vectorized(
                spike_trains,
                bin_size_ms=bin_ms,
                time_range=time_range,
                normalize_isi=True,
                verbose=True,
            )

            return {"rates": rates, "isi": isi, "time_axis_ms": time_axis}

        return self.cache.get_or_compute(
            "binned_fr_isi",
            params=params,
            compute_fn=compute_fn,
            force_recompute=force_recompute,
        )

    def detect_bursts(
        self,
        bin_size: float = 1.0,
        smoothing_window: int = 50,
        burst_detection_params: Optional[Dict] = None,
        burst_edge_params: Optional[Dict] = None,
        backbone_threshold: float = 0.9,
        compute_all_metrics: bool = True,
        use_cache: bool = True,
        check_s3: bool = False,
        force_recompute: bool = False,
    ):
        """
        Detect bursts with automatic caching.

        Uses the ResultsCache to store/retrieve burst detection results.
        Cached results are keyed by detection parameters.

        Parameters
        ----------
        bin_size : float, default=1.0
            Size of time bins in ms for population activity calculation
        smoothing_window : int, default=50
            Size of the smoothing window for population activity
        burst_detection_params : dict, optional
            Parameters for burst detection (baseline_percentile, peak_threshold_factor, etc.)
        burst_edge_params : dict, optional
            Parameters for burst edge detection (edge_threshold_factor, min/max_burst_width)
        backbone_threshold : float, default=0.9
            Threshold for classifying a neuron as rigid
        compute_all_metrics : bool, default=True
            Whether to compute all metrics or just the basic ones
        use_cache : bool, default=True
            Whether to use cached results if available
        check_s3 : bool, default=False
            Whether to check S3 for cached results (may be slower than recompute)
        force_recompute : bool, default=False
            Force recomputation bypassing cache

        Returns
        -------
        BurstResults
            Object containing burst detection results with clean property access

        Example
        -------
        >>> rec = load_recording('proj', 'chip', 'exp')
        >>> results = rec.detect_bursts()  # First run: computes and caches
        >>> print(f"Found {results.n_bursts} bursts")
        >>>
        >>> # On subsequent calls, loads from cache (instant!):
        >>> results2 = rec.detect_bursts()  # Loads from cache
        """
        from braindance.utils.data_manager.utils.analysis.burst_detector import (
            BurstDetector,
        )

        detector = BurstDetector(
            spike_data=self.spikes,
            bin_size=bin_size,
            smoothing_window=smoothing_window,
            burst_detection_params=burst_detection_params,
            burst_edge_params=burst_edge_params,
            backbone_threshold=backbone_threshold,
            compute_all_metrics=compute_all_metrics,
        )

        return detector.detect_bursts(
            cache=self.cache if use_cache else None,
            use_cache=use_cache,
            check_s3=check_s3,
            force_recompute=force_recompute,
        )

    @property
    def pl(self) -> "PlotAccessor":
        """
        Access plotting functions for this recording.

        Provides scanpy-style plotting interface:
            rec.pl.raster_with_pop(time_window=(0, 60))
            rec.pl.sttc_matrix()
            rec.pl.firing_rate_hist()

        You can also set a custom styler:
            from braindance.utils.data_manager import Styler
            rec.pl.styler = Styler(journal="draft")
        """
        from braindance.utils.data_manager.utils.plotting import PlotAccessor

        if not hasattr(self, "_plot_accessor") or self._plot_accessor is None:
            self._plot_accessor = PlotAccessor(self)
        return self._plot_accessor

    # ==================== Metadata Access ====================

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get all metadata as a dictionary."""
        return self._row.to_dict()

    @property
    def name(self) -> str:
        """Recording name (experiment name or identifier)."""
        return self._row.get("experiment", self._row.get("basename", "unknown"))

    @property
    def identifier(self) -> str:
        """Unique identifier string: proj/chip/experiment."""
        proj = str(self._row.get("proj", ""))
        chip = str(self._row.get("chip", ""))
        exp = str(self._row.get("experiment", ""))
        return f"{proj}/{chip}/{exp}".strip("/")

    # ==================== Saving ====================

    def save_results(self, keys: Optional[List[str]] = None):
        """Save results to disk."""
        if self._results is not None:
            self._results.save(keys=keys)

    def save(self):
        """Save all modified data (primarily results)."""
        self.save_results()

    # ==================== Utilities ====================

    def info(self) -> Dict[str, Any]:
        """Get summary information about this recording."""
        info = {
            "identifier": self.identifier,
            "metadata": {
                k: self._row.get(k)
                for k in self.METADATA_COLUMNS
                if k in self._row.index
            },
            "data_loaded": list(self._data._data.keys()),
            "results_available": self.results.keys() if self._results else [],
        }
        return info

    def clear_cache(self):
        """Clear cached data (forces reload on next access)."""
        self._data = DataContext(overwrite_existing=True)

    # ==================== Analysis Methods ====================

    def calculate_latencies(
        self,
        min_response_ratio: float = 1.5,
        max_p_value: float = 0.0001,
        baseline_window: Tuple[float, float] = (-100, 0),
        response_window: Tuple[float, float] = (0, 100),
        use_time_mod: bool = True,
        save_results: bool = True,
        result_key: str = "evoked_latencies",
        verbose: bool = False,
        use_cache: bool = True,
        force_recompute: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect stimulus-evoked neural responses with statistical validation.

        Convenience method that loads spike data and stim log, runs latency
        analysis, and optionally saves results to rec.results.

        If results already exist in rec.results for the given result_key,
        they are returned immediately unless force_recompute is True.

        Args:
            min_response_ratio: Minimum response/baseline firing rate ratio
            max_p_value: Maximum p-value for Mann-Whitney U test
            baseline_window: (start, end) times in ms for baseline period
            response_window: (start, end) times in ms for response period
            use_time_mod: Use artifact-corrected times if available
            save_results: If True, saves results to rec.results.{result_key}
            result_key: Name for saved results (default: 'evoked_latencies')
            verbose: Print progress information
            use_cache: If True, try to load from rec.results first
            force_recompute: If True, ignore cache and recompute
            **kwargs: Additional arguments for UltraOptimizedLatencyHelper

        Returns:
            Dict mapping "electrode_{id}_neuron_{idx}" to latency data with:
            - peak_latency (time of max response in response_window)
            - baseline_rate, response_rate, response_ratio, p_value
            - mean_psth, time_bins (for visualization)

        Example:
            >>> rec = load_recording('proj', 'chip', 'exp')
            >>> evoked = rec.calculate_latencies(min_response_ratio=1.5)
            >>>
            >>> # Second call loads from cache automatically
            >>> evoked = rec.calculate_latencies(min_response_ratio=1.5)  # Instant!
        """
        # Check cache first
        if use_cache and not force_recompute:
            if result_key in self.results:
                if verbose:
                    print(f"  ⚡ Loading cached results for '{result_key}'...")
                return getattr(self.results, result_key)

        # Import analysis function
        from braindance.utils.data_manager.utils.analysis.latency_helper import (
            calculate_latencies,
        )

        # Load data (triggers lazy loading)
        spike_data = self.spikes
        stim_log = self.stim_log

        # Validate data
        if spike_data is None:
            raise ValueError(f"No spike data for {self.identifier}")
        if stim_log is None or len(stim_log) == 0:
            raise ValueError(f"No stim log for {self.identifier}")

        if verbose:
            print(f"  Computing latencies for {self.identifier}...")

        # Run analysis
        evoked_pairs = calculate_latencies(
            spike_data,
            stim_log,
            min_response_ratio=min_response_ratio,
            max_p_value=max_p_value,
            baseline_window=baseline_window,
            response_window=response_window,
            use_time_mod=use_time_mod,
            verbose=verbose,
            **kwargs,
        )

        # Save results
        if save_results:
            setattr(self.results, result_key, evoked_pairs)
            self.save_results()

        return evoked_pairs

    def group_stimulations_by_electrode(
        self, use_time_mod: bool = True
    ) -> Dict[int, np.ndarray]:
        """
        Group stimulation times by electrode ID.

        Utility method for custom analysis or plotting.

        Args:
            use_time_mod: Use artifact-corrected times if available

        Returns:
            Dict mapping electrode_id -> array of stim times (in ms)

        Example:
            >>> electrode_groups = rec.group_stimulations_by_electrode()
            >>> for electrode_id, stim_times in electrode_groups.items():
            >>>     print(f"Electrode {electrode_id}: {len(stim_times)} stims")
        """
        from braindance.utils.data_manager.utils.analysis.latency_helper import (
            group_stimulations_by_electrode,
        )

        stim_log = self.stim_log
        if stim_log is None:
            raise ValueError(f"No stim log for {self.identifier}")

        return group_stimulations_by_electrode(stim_log, use_time_mod=use_time_mod)

    # ==================== Class Methods ====================

    @classmethod
    def load(
        cls, path: Union[str, Path], catalog_row: Optional[pd.Series] = None
    ) -> "Recording":
        """
        Load a Recording from a directory path.

        Args:
            path: Path to recording directory
            catalog_row: Optional pre-loaded catalog row

        Returns:
            Recording object
        """
        path = Path(path)

        if catalog_row is None:
            # Try to load metadata from the directory
            catalog_row = cls._load_metadata_from_path(path)

        rec = cls(catalog_row, base_path=path)
        return rec

    @classmethod
    def from_identifiers(
        cls,
        proj: str,
        chip: str,
        experiment: str,
        base_path: Optional[Path] = None,
        legacy_flat: Optional[bool] = None,
        auto_upload: Optional[bool] = None,
    ) -> "Recording":
        """
        Load a single recording by project/chip/experiment identifiers.

        Args:
            proj: Project name
            chip: Chip ID
            experiment: Experiment name
            base_path: Base data directory
            legacy_flat: Directory structure mode (None=auto-detect, True=legacy, False=standard)
            auto_upload: Automatically upload derived data to S3

        Returns:
            Recording object
        """
        if base_path is not None:
            if isinstance(base_path, str) and base_path.startswith("s3://"):
                pass  # Keep as string
            else:
                base_path = Path(base_path)

        # Try to load metadata from catalog if not provided
        row = None
        try:
            from braindance.utils.data_manager.utils.data_loading.catalog import (
                load_catalog,
            )

            catalog = load_catalog()
            if catalog is not None:
                # Search for matching recording
                matches = catalog.df[
                    (catalog.df["proj"] == proj)
                    & (catalog.df["chip"] == chip)
                    & (catalog.df["experiment"] == experiment)
                ]
                if len(matches) > 0:
                    row = matches.iloc[0]
                    # Update base_path if found in catalog and not provided
                    if base_path is None and "base_path" in row and row["base_path"]:
                        base_path = row["base_path"]
        except Exception:
            # Silent fail - fall back to detached mode
            pass

        if row is None:
            row = pd.Series(
                {
                    "proj": proj,
                    "chip": chip,
                    "experiment": experiment,
                    "base_path": str(base_path) if base_path else None,
                }
            )

        return cls(
            row, base_path=base_path, legacy_flat=legacy_flat, auto_upload=auto_upload
        )


def load_recording(
    proj: str,
    chip: str,
    experiment: str,
    base_path: Optional[Union[str, Path]] = None,
    legacy_flat: Optional[bool] = None,
    auto_upload: Optional[bool] = None,
) -> Recording:
    """
    Load a single recording by project/chip/experiment identifiers.

    Convenience function that wraps Recording.from_identifiers().

    Args:
        proj: Project name
        chip: Chip ID
        experiment: Experiment name
        base_path: Base data directory
        legacy_flat: Directory structure mode:
                    - None (default): Auto-detect by checking which structure exists
                    - True: Force legacy flat {base}/{proj}/{chip}/{exp_name}_spike_data.pkl
                    - False: Force standard {base}/{proj}/{chip}/{exp_name}/{exp_name}_spike_data.pkl
        auto_upload: Automatically upload derived data to S3

    Returns:
        Recording object

    Example:
        >>> # Auto-detect structure (recommended)
        >>> rec = load_recording('25-02-25_busybees', '25123ic', 'exp1_cont_95',
        ...                      base_path='/data')
        >>> binned = rec.get_binned_fr(bin_ms=20)
    """
    return Recording.from_identifiers(
        proj,
        chip,
        experiment,
        base_path,
        legacy_flat=legacy_flat,
        auto_upload=auto_upload,
    )
