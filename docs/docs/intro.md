---
title: BrainDance
description: Neural stimulation framework for organoids and cultures with micro electrode arrays
slug: /
sidebar_position: 0
---

# BrainDance

BrainDance is a powerful framework for neural stimulation experiments with live tissue, such as organoids, using electrode arrays. It provides a flexible and intuitive interface for designing, executing, and analyzing complex stimulation experiments.

## Key Features

- **Modular Design**: Easily create and combine different experimental phases
- **Real-time Control**: Precise control over stimulation parameters and timing
- **Hardware Integration**: Seamless integration with MaxWell CMOS HD-MEAs

## Getting Started

Are you ready to start your neural stimulation experiments? Check out our [Quick Start Guide](quick-start) to set up BrainDance and run your first experiment.

## Example Experiment

Here's a glimpse of what a BrainDance experiment looks like:

```python
from braindance.core.maxwell_env import MaxwellEnv
from braindance.core.params import maxwell_params
from braindance.core.phases import PhaseManager, NeuralSweepPhase, RecordPhase, FrequencyStimPhase

# Set up environment
params = maxwell_params
params['save_dir'] = './causal_tetanus'
params['name'] = 'test'
params['max_time_sec'] = 60*60*3  # 3 hours
env = MaxwellEnv(**params)

# Define experimental phases
record_phase = RecordPhase(env, duration=60*18)
causal_phase = NeuralSweepPhase(env, neuron_list, amp_bounds=400, stim_freq=1, tag="Causal", replicates=30)
tetanus_phase = FrequencyStimPhase(env, tetanus_command, 10, duration=60*1, tag="Tetanus")

# Build and run the experiment
phase_manager = PhaseManager(env, verbose=True)
phase_manager.add_phase_group([record_phase, causal_phase, tetanus_phase])
phase_manager.run()
```

## Learn More

Explore our documentation to learn about:

- [Core Concepts](/docs/core-concepts)
- ~~Experiment Design~~
- ~~Data Analysis~~
- ~~API Reference~~

## Get Involved

BrainDance is an open-source project. We welcome contributions and feedback from the scientific community. Check out our [GitHub repository](https://github.com/your-repo-link) to get involved!