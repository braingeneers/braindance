"""Find experiments beneath organizational folders without merging their recordings."""
import os
from pathlib import Path


def is_experiment(path):
    return any((path / name).is_file() for name in ('experiment.json', 'experiment_log.json'))


def discover_experiments(root):
    root = Path(root)
    found = []
    for directory, children, _ in os.walk(root, followlinks=False):
        path = Path(directory)
        children[:] = sorted(name for name in children if not name.startswith('.'))
        if is_experiment(path):
            found.append(path.resolve())
            children[:] = []
    return found


def recording_files(root):
    """Yield recordings belonging to this root, excluding nested experiments."""
    for directory, children, filenames in os.walk(root, followlinks=False):
        path = Path(directory)
        children[:] = sorted(name for name in children
                             if not name.startswith('.') and not is_experiment(path / name))
        for name in sorted(filenames):
            file = path / name
            if file.suffix.lower() in ('.h5', '.hdf5', '.npz') and not file.is_symlink():
                yield file
