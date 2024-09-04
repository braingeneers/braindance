import numpy as np

from brainloop.core.maxwell_env import MaxwellEnv
from brainloop.core.params import maxwell_params
import time
import sys


params = maxwell_params
params['name'] = 'test'
params['stim_electrodes'] = [10254,14130]
params['max_time_sec'] = 60
params['config'] = None

params['multiprocess'] = False
params['render'] = False
params['dummy'] = 'sine'
params['observation_type'] = 'raw' # can be raw or spikes (spikes can be read through different methods)

# ================== Custom experiment ==================
# This is NOT THE RECOMMENDED way to run experiments, but it is possible
# to directly interact with the environment and use exact timing.
# It is generally better to use the PhaseManager and Phase classes to
# define experiments. ( brainloop.core.phases )

# Here we will run a simple experiment which reads consistently, and stimulates
# when time since the last stim is > 1 second
# It flops between stimulation electrode 0 and 1 (called neuron here)
# amplitude is 150 uAmps, phase duration is 100 ms

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
            