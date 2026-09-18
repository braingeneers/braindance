"""Global configuration and lightweight installation status for the local UI."""
import importlib.util

from braindance.config import get_global_settings, save_global_settings
from .analysis_sorting import sorting_capabilities


def settings_payload(updates=None):
    result = get_global_settings() if updates is None else save_global_settings(updates)
    features = []
    for key, label, modules in [
        ('cartpole', 'Environment · CartPole', ['numpy', 'gymnasium']),
        ('foodland', 'Environment · FoodLand', ['numpy', 'gym', 'pygame']),
        ('ant', 'Environment · Ant / Walker2D', ['numpy', 'gymnasium', 'mujoco']),
        ('rl', 'Reinforcement learning / Atari', ['stable_baselines3', 'ale_py', 'gymnasium', 'pygame']),
        ('analysis', 'Data analysis', ['scipy', 'matplotlib', 'seaborn', 'scienceplots', 'numba',
                                     'tables', 'smart_open', 'xarray', 'boto3', 'spikelab']),
        ('gui', 'Legacy desktop GUI', ['PyQt5', 'matplotlib', 'scipy', 'tqdm']),
        ('maxwell', 'Maxwell acquisition SDK', ['maxlab']),
        ('open-ephys', 'Open Ephys tools', ['open_ephys']),
    ]:
        missing = [name for name in modules if importlib.util.find_spec(name) is None]
        features.append({'id': key, 'label': label, 'available': not missing,
                         'detail': 'Missing: ' + ', '.join(missing) if missing else 'Dependencies detected'})
    sorting = sorting_capabilities()
    features.insert(0, {'id': 'rtsort', 'label': 'RT-Sort', 'available': sorting['available'],
                        'detail': sorting['reason'] or 'Dependencies and detection model detected'})
    result['features'] = features
    return result
