import numpy as np

from braindance.core.maxwell_env import MaxwellEnv
from braindance.core.params import maxwell_params
import time
import sys


params = maxwell_params
params['name'] = 'test'
params['stim_electrodes'] = [10254,14130]
params['max_time_sec'] = 60
params['config'] = None

params['multiprocess'] = False
params['render'] = False
# params['dummy'] = 'sine'
params['dummy'] = '/media/danser-lab/hippo/cartpole/23-11-22_cartpole/c20215-run1/exp2/exp2_cartpole_48.raw.h5'
params['observation_type'] = 'raw' # spikes

if __name__ == '__main__':
    env = MaxwellEnv(**params)

    done = False

    neuron = 0

    q = 0
    fs = 20000
    t = time.perf_counter()
    times = []

    while not done:
        # Bind the loop to 1ms step
        if env.dt >= 1/fs:
            if env.stim_dt > 1:
                print("Stimulating at\t",env.time_elapsed())
                obs, done = env.step(action = ([neuron],150,100))
                neuron ^= 1
                print(obs)
            else:
                obs,done = env.step()
            