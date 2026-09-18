from braindance.core.maxwell_env import MaxwellEnv
from braindance.core.params import maxwell_params
from braindance.core.phases import NeuralSweepPhase, PhaseManager

import numpy as np

params = maxwell_params
params['name'] = 'amplitude_sweep' # Name of the experiment
params['stim_electrodes'] = [15006, 18317, 22714, 22497,23600, 20744, 24926, 24510, 17673,
                             18791, 24510, 21223]
params['save_dir'] = 'c20185' # Path to the data directory, will be created if it doesn't exist
params['config'] = 'config.cfg' # Path to the config file


env = MaxwellEnv(**params)
neuron_list = np.arange(len(params['stim_electrodes']))


amp_sweep = NeuralSweepPhase(env, neuron_list=neuron_list, replicates=50,
                        amp_bounds=(150, 150, 1), stim_freq=2,
                        order='ran', verbose=True)

# amp_sweep = AmplitudeSweep(env, neuron_list=[0,1,2], replicates=1,
#                         amp_bounds=(50, 200, 1), stim_freq=4,
#                         type='random', verbose=True)

phase_manager = PhaseManager(env)
phase_manager.add_phase(amp_sweep)

phase_manager.summary()

phase_manager.run()

