import numpy as np

from brainloop.core.maxwell_env import MaxwellEnv
from brainloop.core.params import maxwell_params
import time
import sys


params = maxwell_params
params['name'] = 'rec2_post-thc-stim'
params['save_dir'] = 'mo_experiment'
params['stim_electrodes'] =  [16786, 17453, 18337, 19220, 21641] # Define the 4 electrodes to stimulate here
params['max_time_sec'] = 60*5 # 5 mins
params['config'] = 'config_stim.cfg'

# Custom experiment, complicated hand-built stimulation pattern
# yucky line of code :shrug:
stim_pattern = [['stim', [0], 150, 4], ['delay', 20], ['stim', [0], 150, 4], ['delay', 20], ['stim', [0], 150, 4], ['delay', 20], ['stim', [0], 150, 4], ['stim', [0], 150, 4], ['stim', [0], 150, 4], ['stim', [0], 150, 4], ['stim', [0], 150, 4], ['stim', [1], 150, 4], ['stim', [1], 150, 4], ['stim', [1], 150, 4], ['stim', [1], 150, 4], ['delay', 10], ['stim', [1], 150, 4], ['stim', [1], 150, 4], ['stim', [1], 150, 4], ['stim', [1], 150, 4], ['delay', 10], ['stim', [1], 150, 4], ['stim', [1], 150, 4], ['stim', [1], 150, 4], ['stim', [2], 150, 10], ['stim', [2], 150, 10], ['stim', [2], 150, 10], ['stim', [2], 150, 10], ['stim', [2], 150, 10], ['delay', 20], ['stim', [2], 150, 10], ['stim', [2], 150, 10], ['stim', [2], 150, 10], ['stim', [2], 150, 10], ['stim', [2], 150, 10], ['delay', 20], ['stim', [2], 150, 10], ['stim', [2], 150, 10], ['stim', [2], 150, 10], ['stim', [2], 150, 10], ['stim', [2], 150, 10], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['delay', 35], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['stim', [3], 150, 8], ['stim', [4], 150, 2], ['stim', [4], 150, 2], ['stim', [4], 150, 2], ['stim', [4], 150, 2], ['stim', [4], 150, 2], ['stim', [4], 150, 2], ['stim', [4], 150, 2], ['stim', [4], 150, 2], ['stim', [4], 150, 2], ['stim', [4], 150, 2], ['stim', [4], 150, 2], ['stim', [4], 150, 2], ['stim', [4], 150, 2], ['stim', [4], 150, 2], ['stim', [4], 150, 2], ['stim', [4], 150, 2], ['stim', [4], 150, 2], ['stim', [4], 150, 2]]


if __name__ == '__main__':
    env = MaxwellEnv(**params)
    fs = 20000
    stim_Hz = 1

    done = False

    try:
        while not done:
            # Bind the loop to 1ms step
            # if env.dt >= 1/fs:
            if env.stim_dt > 1/stim_Hz and stim_pattern != None:
                print("Stimulating at\t",env.time_elapsed())
                obs, done = env.step(action = stim_pattern)
            else:
                obs,done = env.step()
            

    finally:
        print()
        print('Experiment complete')
        env._cleanup()
            