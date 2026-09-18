from braindance.core.maxwell_env import MaxwellEnv
from braindance.core.params import maxwell_params
from braindance.core.phases import CartPolePhase, PhaseManager, NeuralSweepPhase

import numpy as np

params = maxwell_params
params['save_dir'] = '/media/mxwbio/poop/data_ash/cartpole_replicate' # Path to the data directory, will be created if it doesn't exist
params['name'] = '09-27' # Name of the experiment


# Last 2 electrodes here are for the ones on the motor neurons, should be just used for tetanus
params['stim_electrodes'] =  [5234, 7229, 7429, 12258, 11606, 4134, 5453] # Define the 4 electrodes to stimulate here
params['max_time_sec'] = 60*30 # 30 mins

params['config'] = 'config-phase.cfg'#'config.cfg' # Path to the config file



env = MaxwellEnv(**params)
neuron_list = np.arange(len(params['stim_electrodes']))
sensory_neurons = [0,1] # Indexes of the stim channels

motor_neurons = [325,869] # In channels to read
training_neurons= [2,3,4,5,6] # Indexes of the stim channels


cartpole = CartPolePhase(env, sensory_neurons=sensory_neurons, motor_neurons=motor_neurons,
                          training_neurons=training_neurons, verbose=True,
                          read_period_ms = 200)

phase_manager = PhaseManager(env)
phase_manager.add_phase(cartpole)


phase_manager.summary()

phase_manager.run()