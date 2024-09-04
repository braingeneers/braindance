from brainloop.core.maxwell_env import MaxwellEnv
from brainloop.core.params import maxwell_params
from brainloop.core.phases import FrequencyStimPhase, PhaseManager
from brainloop.core.trainer import generate_stimulations
import numpy as np

params = maxwell_params
params['name'] = 'multi_freq_stim' # Name of the experiment
params['stim_electrodes'] = [15006, 18317, 22714, 22497,23600, 20744, 24926, 24510, 17673,
                             18791, 24510, 21223]
params['save_dir'] = 'test' # Path to the data directory, will be created if it doesn't exist
params['config'] = None # Path to the config file
params['verbose'] = True
params['dummy'] = 'sine'


env = MaxwellEnv(**params)
neuron_list = np.arange(len(params['stim_electrodes']))

# We aim to use the following design:
# First, stimulate the pattern:
#   0,1,2, then 4, then 2,3,4, then 5, then 4, then 3, then 2, then 1, then 0
#   At 2 Hz
# Second, stimulate the pattern:
#   0, then 1, then 2
#   At 5 Hz

# Lets stim 
electrode_inds = [[0,1,2],4,[2,3,4],5,4,3,2,1,0]

stim_commands = generate_stimulations(electrode_inds, amp=400, phase_width=200)
print(stim_commands)

freq_stim_phase = FrequencyStimPhase(env, stim_command=stim_commands,stim_freq=2,
                               verbose=True)

freq_stim_single_phase = FrequencyStimPhase(env, stim_command=stim_commands[0],stim_freq=5,
                               verbose=True)

phase_manager = PhaseManager(env)
phase_manager.add_phase(freq_stim_phase)
phase_manager.add_phase(freq_stim_single_phase)

phase_manager.summary()

phase_manager.run()

