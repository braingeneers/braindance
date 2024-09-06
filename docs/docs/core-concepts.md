---
sidebar_position: 2
---

# Core Concepts

Understanding the core concepts of BrainDance will help you design and execute more complex experiments.

## Maxwell Environment

The `MaxwellEnv` class is the central component that interfaces with the micro electrode array hardware. It handles:

- Initialization of the hardware
- Data acquisition
- Stimulation control
- Data saving

Example usage:

```python
from brainloop.core.maxwell_env import MaxwellEnv
from brainloop.core.params import maxwell_params

params = maxwell_params
params['save_dir'] = './experiment_data'
params['name'] = 'my_experiment'
env = MaxwellEnv(**params)
```

## Phases

Phases are the building blocks of experiments in BrainDance. Each phase represents a specific experimental action or protocol. Common phase types include:

- `RecordPhase`: Simple recording of neural activity
- `NeuralSweepPhase`: Systematic stimulation of specified electrodes
- `FrequencyStimPhase`: Stimulation at a specified frequency

Example of creating a phase:

```python
from brainloop.core.phases import RecordPhase

record_phase = RecordPhase(env, duration=60*5)  # 5-minute recording phase
```

## Phase Manager

The `PhaseManager` class orchestrates the execution of multiple phases in an experiment. It allows you to:

- Add phases to the experiment
- Specify the order of phase execution
- Run the entire experiment

Example usage:

```python
from brainloop.core.phases import PhaseManager

phase_manager = PhaseManager(env, verbose=True)
phase_manager.add_phase_group([record_phase, stim_phase, record_phase])
phase_manager.run()
```

By combining these core concepts, you can create complex, multi-stage experiments with precise control over stimulation and recording parameters.