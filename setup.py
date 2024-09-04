from setuptools import setup, find_packages

# Function to read the list of requirements from requirements.txt
def read_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()

setup(
    name='brainloop',
    version='0.0.1',
    packages=find_packages(),
    install_requires=read_requirements(),
    include_package_data=True,
    package_data={
        '': ['*.npy', '*.json'],
    },
    entry_points={
        'console_scripts': [
            'bdquery = brainloop.core.maxwell.query_electrodes:main'
        ],
    },
)
