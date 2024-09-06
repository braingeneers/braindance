from brainloop.core.maxwell_env import MaxwellEnv
from brainloop.core.params import maxwell_params
from brainloop.core.phases import NeuralSweepPhase, PhaseManager

import numpy as np

params = maxwell_params
params['name'] = 'amplitude_sweep' # Name of the experiment
params['stim_electrodes'] = [3689, 5891, 5019, 7007, 9405, 9614, 7171]
params['save_dir'] = 'c20185' # Path to the data directory, will be created if it doesn't exist
params['config'] = 'config.cfg' # Path to the config file


env = MaxwellEnv(**params)
neuron_list = np.arange(len(params['stim_electrodes']))


amp_sweep = NeuralSweepPhase(env, neuron_list=neuron_list, replicates=50,
                        amp_bounds=(80, 180, 10), stim_freq=2,
                        order='random', verbose=True)


phase_manager = PhaseManager(env)
phase_manager.add_phase(amp_sweep)

phase_manager.summary()

phase_manager.run()

