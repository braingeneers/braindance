from pathlib import Path

from setuptools import setup, find_packages

# Function to read the list of requirements from requirements.txt
def read_requirements():
    return [line for line in Path(__file__).with_name('requirements.txt').read_text().splitlines()
            if line and not line.startswith('#')]

analysis_requirements = [
    'scipy>=1.11', 'matplotlib>=3.8', 'seaborn>=0.13', 'scienceplots>=2.1',
    'numba>=0.59',
    # Newer Numba/llvmlite releases no longer ship Intel macOS wheels.
    # This also applies to Intel Python running under Rosetta on Apple Silicon.
    'numba<0.63; sys_platform == "darwin" and platform_machine == "x86_64"',
    'tables>=3.9', 'smart_open>=6', 'xarray>=2023.1',
    'boto3>=1.28', 'spikelab>=0.1.2,<0.2',
]
foodland_requirements = ['gym==0.26.2', 'pygame>=2.5']
ant_requirements = [
    'gymnasium[mujoco]>=1.1,<2',
    # MuJoCo 3.10 is the last release with Intel macOS wheels.
    'mujoco<3.11; sys_platform == "darwin" and platform_machine == "x86_64"',
]
gui_requirements = ['PyQt5>=5.15', 'matplotlib>=3.8', 'scipy>=1.11', 'tqdm>=4.66']
default_requirements = (read_requirements() + analysis_requirements
                        + foodland_requirements + ant_requirements)

setup(
    name='braindance',
    version='0.1.9',
    description='Neural stimulation, simulation, and streaming experiments',
    long_description=Path(__file__).with_name('README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    python_requires='>=3.11,<3.13',
    packages=(find_packages(include=['braindance', 'braindance.*'])
              + ['braindance.utils']
              + ['braindance.utils.' + name for name in find_packages(
                  where='braindance/utils',
                  exclude=['data_manager.internal_tests', 'data_manager.internal_tests.*',
                           'data_manager.internal_usage', 'data_manager.internal_usage.*'])]),
    install_requires=default_requirements,
    extras_require={
        'rtsort': [
            'torch>=2.6,<3', 'scipy>=1.11', 'matplotlib>=3.8',
            'scikit-learn>=1.3', 'diptest>=0.8', 'nvidia-ml-py>=12',
            'spikeinterface>=0.104.9,<0.105', 'threadpoolctl>=3.2', 'tqdm>=4.66',
        ],
        'analysis': analysis_requirements,
        'gui': gui_requirements,
        # CPU-friendly full install; hardware, GPU sorting and RL remain opt-in.
        'full': gui_requirements,
        'all': gui_requirements,
        'rl': ['stable-baselines3>=2.0,<3', 'ale-py>=0.8', 'gymnasium[classic-control]>=1.1,<2'],
        'foodland': foodland_requirements,
        'ant': ant_requirements,
        'kilosort': ['spikeinterface>=0.104.9,<0.105', 'natsort>=8',
                     'scipy>=1.11', 'matplotlib>=3.8', 'tqdm>=4.66'],
        'open-ephys': ['open-ephys-python-tools'],
        'dev': ['build>=1.2', 'twine>=5', 'pytest>=8', 'pytest-timeout>=2', 'pygments>=2'],
    },
    include_package_data=False,
    package_data={
        'braindance': ['tutorial_data.json'],
        'braindance.core.maxwell': ['stim_buffers.npy'],
        'braindance.core.spikedetector': [
            'detection_models/mea/init_dict.json',
            'detection_models/mea/state_dict.pt',
            'detection_models/neuropixels/init_dict.json',
            'detection_models/neuropixels/state_dict.pt',
        ],
        'braindance.examples.streaming_workshop': ['*.html', '*.js', '*.css'],
        'braindance.gui': ['resources/*.png'],
    },
    entry_points={
        'console_scripts': [
            'bdquery = braindance.core.maxwell.query_electrodes:main',
            'braindance = braindance.examples.streaming_workshop.main:cli',
            'braindance-gui = braindance.gui.experiment_launcher:main',
            'bdreplay = braindance.cli.replay:main',
        ],
    },
)
