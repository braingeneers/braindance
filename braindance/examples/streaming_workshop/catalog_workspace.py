"""Local workshop experiment index alongside the configured recording catalog."""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from braindance.config import get_catalog_path, get_data_dir

from .discovery import discover_experiments


class CatalogWorkspace:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.index_path = self.output_dir / 'catalog_experiments.json'
        self.lock = threading.RLock()

    def control(self, command):
        action = command.get('action', 'refresh')
        if action not in ('refresh', 'add'):
            raise ValueError('Unknown catalog action')
        with self.lock:
            saved = json.loads(self.index_path.read_text()) if self.index_path.exists() else []
            if action == 'add':
                if not str(command.get('path') or '').strip():
                    raise ValueError('Enter an experiment directory or recording path')
                path = Path(command['path']).expanduser().resolve()
                if not path.exists():
                    raise ValueError(f'Path does not exist: {path}')
                if not path.is_dir() and path.suffix.lower() not in ('.json', '.h5', '.hdf5', '.npz'):
                    raise ValueError('Choose an experiment directory, experiment JSON, HDF5 or NPZ recording')
                if str(path) not in saved:
                    saved.append(str(path))
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    temporary = self.index_path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(saved, indent=2))
                    temporary.replace(self.index_path)
            entries, warnings = [], []
            catalog_path = get_catalog_path()
            if catalog_path and Path(catalog_path).is_file():
                try:
                    from braindance.utils.data_manager import load_catalog, Recording
                    catalog_root = Path(catalog_path).parent
                    base_path = catalog_root if (catalog_root / '.complete.json').is_file() else get_data_dir()
                    catalog = load_catalog(base_path=base_path)
                    frame = catalog.df
                    # CSV inference often makes chip IDs numeric; path components are strings.
                    for key in ('proj', 'chip', 'experiment'):
                        if key in frame:
                            frame[key] = frame[key].fillna('').astype(str)
                    metadata = json.loads(frame.to_json(orient='records'))
                    for index, row in enumerate(metadata):
                        rec = Recording(frame.iloc[index], base_path=base_path)
                        try:
                            explicit = str(row.get('full_path') or '')
                            path = Path(explicit).expanduser() if explicit and '://' not in explicit else None
                            if path is not None and not path.is_absolute():
                                path = base_path / path
                            if path is None or not path.exists():
                                path = rec._raw_data_path
                            available = bool(path and path.exists())
                            entries.append(dict(name=str(row.get('experiment') or row.get('exp') or 'Recording'),
                                                project=str(row.get('proj') or ''), chip=str(row.get('chip') or ''),
                                                kind=str(row.get('type') or ''), source='Configured catalog',
                                                path=str(path.resolve()) if available else explicit,
                                                available=available, metadata=row,
                                                reason='' if available else 'Recording is not available locally. Add its local path to analyze it.'))
                        except Exception as exc:
                            warnings.append(f"Could not resolve {row.get('experiment', index)}: {exc}")
                        finally:
                            rec.clear_cache()
                except Exception as exc:
                    warnings.append(f'Could not load configured catalog: {exc}')
            else:
                warnings.append('No configured catalog CSV found. Set its path in Settings; saved workshop experiments are listed below.')

            discovered = {}
            for root, source in ((self.output_dir, 'Workshop run'),
                                 (get_data_dir() / 'tutorials', 'Tutorial data')):
                for experiment in discover_experiments(root):
                    discovered[str(experiment)] = source
            for location in saved:
                path = Path(location)
                if path.name in ('experiment.json', 'experiment_log.json'):
                    path = path.parent
                experiments = discover_experiments(path) if path.is_dir() else []
                for experiment in experiments or [path]:
                    discovered[str(experiment)] = 'Added locally'
            indexed = {entry['path'] for entry in entries if entry['available']}
            for location in sorted(discovered, reverse=True):
                path = Path(location)
                if location in indexed:
                    continue
                entries.append(dict(name=path.name, project='', chip='', kind='Experiment' if path.is_dir() else 'Recording',
                                    path=location, source=discovered[location],
                                    available=path.exists(), metadata={},
                                    reason='' if path.exists() else 'This saved path is no longer available.'))
            return dict(entries=entries, warnings=warnings, catalog_path=str(catalog_path or ''),
                        index_path=str(self.index_path), refreshed_at=datetime.now(timezone.utc).isoformat())
