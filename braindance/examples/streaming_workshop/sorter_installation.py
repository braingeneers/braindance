"""Fixed, opt-in pip installation plans for the workshop's Python environment."""
from pathlib import Path
import platform
import subprocess
import sys


# Match BrainDance's supported numerical stack and RT-Sort extra. Do not accept
# package names, pip flags, URLs or executable paths from browser requests.
COMMON = ('numpy>=1.26,<2', 'pandas>=2.1,<3', 'spikeinterface>=0.104.9,<0.105',
          'scipy>=1.11', 'scikit-learn>=1.3,<1.8', 'h5py>=3.10', 'pynwb>=2.6')
PLANS = {
    'spikeinterface': dict(label='SpikeInterface Python sorters',
        description='Installs Simple, SpyKING CIRCUS 2 and Tridesclous 2 dependencies and NWB support. Other sorters may need separate software.',
        packages=COMMON + ('numba>=0.59',
            'numba<0.63; sys_platform == "darwin" and platform_machine == "x86_64"',
            'hdbscan>=0.8.33', 'torch>=2.6,<3', 'networkx', 'threadpoolctl>=3.2', 'tqdm>=4.66')),
    'rt-sort': dict(label='RT-Sort dependencies',
        description='Installs PyTorch and RT-Sort dependencies. Uses the detection model bundled with BrainDance. CUDA drivers require separate setup.',
        packages=COMMON + ('torch>=2.6,<3', 'matplotlib>=3.8', 'diptest>=0.8',
                           'nvidia-ml-py>=12', 'threadpoolctl>=3.2', 'tqdm>=4.66')),
}


def installation_plans():
    return [dict(id=key, label=value['label'], description=value['description']) for key, value in PLANS.items()]


def validate_installation(plan_id):
    if plan_id not in PLANS:
        raise ValueError('Unknown sorter installation plan')
    if sys.platform == 'darwin' and platform.machine().lower() in ('x86_64', 'amd64'):
        raise ValueError(
            f'Cannot install sorter dependencies into {sys.executable}: this Python '
            'runs as Intel (x86_64) on macOS. Both installation plans require '
            'PyTorch >=2.6, but Intel macOS wheels stop at 2.2.2. On Apple Silicon, '
            'create a separate Python 3.11 environment using native ARM64 Miniforge '
            'or Miniconda, install BrainDance there, and launch the workshop with '
            'that environment. Verify that python -c "import platform; '
            'print(platform.machine())" prints arm64. On an Intel Mac, use a '
            'supported Linux or Windows machine for these sorters. '
            'pip was not started; no packages were changed.')


def install_sorter(plan_id, log_path):
    validate_installation(plan_id)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
               '--no-input', *PLANS[plan_id]['packages']]
    with log_path.open('w', encoding='utf-8') as log:
        log.write(f'Installing into {sys.executable}\n')
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    if result.returncode:
        raise RuntimeError(f'pip exited with code {result.returncode}. See the installation log below.')
