import os
from pathlib import Path
import json
import tempfile


def get_global_settings():
    """Describe saved values and effective values, including environment overrides."""
    path = Path.home() / '.braindance' / 'config.json'
    saved = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(saved, dict):
        raise ValueError('BrainDance config.json must contain a JSON object')
    getters = {'data_dir': get_data_dir, 'catalog_path': get_catalog_path,
               'output_dir': get_output_dir,
               'auto_extract_spike_info': get_auto_extract_spike_info}
    return {'config_path': str(path), 'settings': {
        key: {'saved': saved.get(key),
              'effective': str(getter()) if key != 'auto_extract_spike_info' else getter(),
              'environment': 'BRAINDANCE_' + key.upper() if os.getenv('BRAINDANCE_' + key.upper()) else None}
        for key, getter in getters.items()}}


def save_global_settings(updates):
    """Validate and atomically persist global settings; null restores a default."""
    allowed = {'data_dir', 'catalog_path', 'output_dir', 'auto_extract_spike_info'}
    if not isinstance(updates, dict) or not set(updates) <= allowed:
        raise ValueError('Unsupported global setting')
    normalized = {}
    for key, value in updates.items():
        if value is None:
            normalized[key] = None
        elif key == 'auto_extract_spike_info':
            if not isinstance(value, bool):
                raise ValueError('Automatic spike-info extraction must be true or false')
            normalized[key] = value
        else:
            if not isinstance(value, str) or not value.strip() or '\x00' in value:
                raise ValueError(f'{key} must be a nonempty path')
            path = Path(value.strip()).expanduser()
            if not path.is_absolute():
                raise ValueError(f'{key} must be an absolute path (or start with ~)')
            if path.exists() and (path.is_dir() if key == 'catalog_path' else not path.is_dir()):
                raise ValueError(f'{key} must be a {"file" if key == "catalog_path" else "directory"} path')
            normalized[key] = str(path.resolve())
    config_path = Path.home() / '.braindance' / 'config.json'
    saved = json.loads(config_path.read_text()) if config_path.exists() else {}
    if not isinstance(saved, dict):
        raise ValueError('BrainDance config.json must contain a JSON object')
    for key, value in normalized.items():
        if value is None:
            saved.pop(key, None)
        else:
            saved[key] = value
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=config_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(saved, stream, indent=2)
            stream.write('\n')
        temporary.replace(config_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return get_global_settings()

def get_data_dir():
    """Get data directory from config, env var, or default"""
    
    # 1. Check environment variable first
    env_path = os.getenv('BRAINDANCE_DATA_DIR')
    if env_path:
        return Path(env_path)
    
    # 2. Check config file
    config_path = Path.home() / '.braindance' / 'config.json'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'data_dir' in config:
                    return Path(config['data_dir'])
        except (json.JSONDecodeError, KeyError):
            pass
    
    # 3. Default fallback
    print("Data directory not found, using default: ", Path.home() / 'braindance_data')
    print("To set the data directory, run using: python -m braindance.config --set_data_dir '/path/to/data'")
    return Path.home() / 'braindance_data'

def set_data_dir(path):
    """Set the data directory in config file"""
    config_dir = Path.home() / '.braindance'
    config_dir.mkdir(exist_ok=True)

    config_path = config_dir / 'config.json'
    config = {}
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            pass

    config['data_dir'] = str(Path(path).resolve())

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print("Data directory set to: ", config['data_dir'])

def get_catalog_path():
    """Get default catalog path from config, env var, or default"""

    # 1. Check environment variable first
    env_path = os.getenv('BRAINDANCE_CATALOG_PATH')
    if env_path:
        return Path(env_path)

    # 2. Check config file
    config_path = Path.home() / '.braindance' / 'config.json'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'catalog_path' in config:
                    return Path(config['catalog_path'])
        except (json.JSONDecodeError, KeyError):
            pass

    # 3. Default fallback - catalog.csv in data directory
    return get_data_dir() / 'catalog.csv'

def set_catalog_path(path):
    """Set the default catalog path in config file"""
    config_dir = Path.home() / '.braindance'
    config_dir.mkdir(exist_ok=True)

    config_path = config_dir / 'config.json'
    config = {}
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            pass

    config['catalog_path'] = str(Path(path).resolve())

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def get_auto_extract_spike_info():
    """
    Check if automatic spike_info extraction from RT-Sort is enabled.

    Returns:
        bool: True if auto-extraction is enabled, False otherwise (default)
    """
    # 1. Check environment variable first
    env_val = os.getenv('BRAINDANCE_AUTO_EXTRACT_SPIKE_INFO')
    if env_val:
        return env_val.lower() in ('true', '1', 'yes')

    # 2. Check config file
    config_path = Path.home() / '.braindance' / 'config.json'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'auto_extract_spike_info' in config:
                    return bool(config['auto_extract_spike_info'])
        except (json.JSONDecodeError, KeyError):
            pass

    # 3. Default to False (most users work locally)
    return False

def set_auto_extract_spike_info(enabled: bool):
    """
    Enable/disable automatic spike_info extraction from RT-Sort.

    Args:
        enabled: True to enable auto-extraction, False to disable
    """
    config_dir = Path.home() / '.braindance'
    config_dir.mkdir(exist_ok=True)

    config_path = config_dir / 'config.json'
    config = {}
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            pass

    config['auto_extract_spike_info'] = bool(enabled)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def get_output_dir():
    """Get output directory for plots/figures from config, env var, or default"""

    # 1. Check environment variable first
    env_path = os.getenv('BRAINDANCE_OUTPUT_DIR')
    if env_path:
        return Path(env_path)

    # 2. Check config file
    config_path = Path.home() / '.braindance' / 'config.json'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'output_dir' in config:
                    return Path(config['output_dir'])
        except (json.JSONDecodeError, KeyError):
            pass

    # 3. Default fallback - outputs subdirectory in data dir
    return get_data_dir() / 'outputs'

def set_output_dir(path):
    """Set the output directory in config file"""
    config_dir = Path.home() / '.braindance'
    config_dir.mkdir(exist_ok=True)

    config_path = config_dir / 'config.json'
    config = {}
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            pass

    config['output_dir'] = str(Path(path).resolve())

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_data_dir", type=str, default=None)
    parser.add_argument("--get_data_dir", action="store_true")
    args = parser.parse_args(argv)
    if args.set_data_dir is not None:
        set_data_dir(args.set_data_dir)
    elif args.get_data_dir:
        print("Current data directory: ", get_data_dir())
    else:
        print("Current data directory: ", get_data_dir())
        print("To set the data directory, run using: python -m braindance.config --set_data_dir '/path/to/data'")


if __name__ == "__main__":
    main()
