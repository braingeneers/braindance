import numpy as np

from brainloop.core.maxwell_env import MaxwellEnv
from brainloop.core.params import maxwell_params

params = maxwell_params
params['name'] = 'causal_freq' # Name of the experiment
params['stim_electrodes'] = [10254,14130, 2, 4] # Define the 4 electrodes to stimulate here
params['max_time_sec'] = 60*60*2 # 2 hours
params['save_dir'] = 'data' # Path to the data directory, will be created if it doesn't exist
params['config'] = 'config.cfg' # Path to the config file


# ================== Custom experiment ==================
# This is NOT THE RECOMMENDED way to run experiments, but it is possible
# to directly interact with the environment and use exact timing.
# It is generally better to use the PhaseManager and Phase classes to
# define experiments. ( brainloop.core.phases )


# Here we will run a simple experiment with 3 phases:
# 1. 1 minute of tetanic stimulation
# 2. 1 minute of silence
# 3. 2 minutes of stimulation
# 4. 1 minute of silence

# We will define the experiment in terms of phases


# Lets define the stim_cycle in seconds
tetanus_Hz = 50
causal_Hz = 1

stim_cycle = [
    ('tetanus', 60),
    ('silent', 60),
    ('causal', 120),
    ('silent', 60)
]

silent_cycle = [
    ('silent', 2),
]

full_exp = 3*silent_cycle + 5*stim_cycle + 1*silent_cycle + 6*stim_cycle + 1*silent_cycle + 5*stim_cycle + 3*silent_cycle

delay_ms = 5

# We will do this at tetanus_Hz
# 'stim' is the action, [0] is the neuron, 150 is the amplitude, 100 is the phase duration of the square pulse
tetanus_action = [('stim',[0],150,100), ('delay', delay_ms), 
                  ('stim',[1],150,100), ('delay', delay_ms),
                  ('stim',[2],150,100), ('delay', delay_ms),
                  ('stim',[3],150,100)]




# Set up the variables
env = MaxwellEnv(**params)
done = False

fs = 20000
neurons = len(params['stim_electrodes'])

# Get the starting phase, and the time at which the next phase starts
phase, phase_dt = full_exp.pop(0) 
phase_start_time = env.time_elapsed()

# For bookkeeping
causal_count = 0


print('Starting experiment at {:.3f}'.format(env.time_elapsed()))
print('Beginning with phase',phase,'for',phase_dt,'seconds')

while not done:
    # ~~~~~~~~~~~ Phase changing Logic ~~~~~~~~~~~
    if phase_dt <= env.time_elapsed() - phase_start_time:
        # We are in the next phase
        if len(full_exp) != 0:
            phase, phase_dt = full_exp.pop(0)
        else:
            done = True
            print('Experiment complete')
            break
        phase_start_time = env.time_elapsed()
        print("Starting phase",phase,"at\t",'{:.3f}'.format(env.time_elapsed()), "for", phase_dt, "seconds")

        if phase == 'causal':
            # We reset the causal count
            causal_count = 0

    # ~~~~~~~~~~~ Silent Phase Logic ~~~~~~~~~~~
    if phase == 'silent':
        # We do nothing
        env.step()

    # ~~~~~~~~~~~ Causal Phase Logic ~~~~~~~~~~~
    elif phase == 'causal':
        # We stimulate the electrodes one-by-one at 1Hz 30 times
        if env.stim_dt >= 1/causal_Hz:
            action = ([causal_count//30],150,100) # Neuron, amplitude, phase duration
            print('Causal on {} at {:.3f}'.format(causal_count//30,env.time_elapsed()))
            env.step(action=action, tag='causal')
            causal_count += 1
        else:
            env.step()
    
    # ~~~~~~~~~~~ Tetanus Phase Logic ~~~~~~~~~~~
    elif phase == 'tetanus':
        # We stimulate the electrodes in order at tetanus_Hz
        if env.stim_dt >= 1/tetanus_Hz:
            print('Tetanus at {:.3f}'.format(env.time_elapsed()))
            env.step(action=tetanus_action, tag='tetanus')
            
        else:
            env.step()
          

env.close()