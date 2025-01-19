from setuptools import setup, find_packages

# Function to read the list of requirements from requirements.txt
def read_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()

setup(
    name='braindance',
    version='0.1.7',
    packages=find_packages(),
    install_requires=read_requirements(),
    extras_require={
        "rt-sort": ["diptest",
                    "natsort",
                    "pynvml",
                    "scikit-learn",
                    "spikeinterface[full]"]
    },
    include_package_data=True,
    package_data={
        '': ['*.npy', '*.json'],
        'braindance': ['core/spikedetector/detection_models/mea/*',
                       'core/spikedetector/detection_models/neuropixels/*'],
    },
    entry_points={
        'console_scripts': [
            'bdquery = braindance.core.maxwell.query_electrodes:main'
        ],
    },
)
