---
sidebar_position: 1
---

# Quick Start Guide

Get up and running with BrainDance in just a few minutes!

## Installation

0. Create a virtual environment
   ```
   conda create -n brain python=3.11
   ```

1. Clone the BrainDance repository:
   ```
   pip install git+https://github.com/braingeneers/braindance.git
   ```

## Your First Experiment

Here's a simple experiment to get you started:

```python
from braindance.core.maxwell_env import MaxwellEnv
from braindance.core.params import maxwell_params
from braindance.core.phases import PhaseManager, RecordPhase, NeuralSweepPhase

# Set up the environment
params = maxwell_params
params['save_dir'] = './my_first_experiment'
params['name'] = 'quick_start'
params['max_time_sec'] = 60*5  # 5 minutes
params['stim_electrodes'] = [20421, 1925] # electrodes in your current config
params['config'] = 'config.cfg' # this is the path to your config file for Maxwell
env = MaxwellEnv(**params)

# Create phases
record_phase = RecordPhase(env, duration=60)
sweep_phase = NeuralSweepPhase(env, neuron_list=[0, 1], amp_bounds=300, stim_freq=1, tag="QuickSweep", replicates=3)

# Build and run the experiment
phase_manager = PhaseManager(env, verbose=True)
phase_manager.add_phase_group([record_phase, sweep_phase, record_phase])

print(phase_manger.summary())

phase_manager.run()
```

This experiment will:
1. Record baseline activity for 1 minute
2. Perform a neural sweep on two electrodes
3. Record post-stimulation activity for 1 minute

## Next Steps

- Learn about [Core Concepts](core-concepts) in BrainDance
- ~~Explore more complex Experiment Designs~~
- ~~Dive into Data Analysis techniques~~