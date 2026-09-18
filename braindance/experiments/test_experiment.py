import numpy as np

from braindance.core.maxwell_env import MaxwellEnv
from braindance.core.params import maxwell_params
import time
import sys

params = maxwell_params
params['name'] = 'test'
params['stim_electrodes'] = [10254,14130]
params['max_time_sec'] = 60


params['multiprocess'] = False
params['render'] = False
if __name__ == '__main__':
    env = MaxwellEnv(**params)

    done = False

    neuron = 0

    q = 0
    fs = 20000
    t = time.perf_counter()
    times = []

    try:
        while not done:
            # Bind the loop to 1ms step
            dt = env.dt
            if dt >= 1/fs:
            # if time.perf_counter() - t >= 1/(fs+800):
                # q += 1
                # print (time.perf_counter() - t)
                # times.append(time.perf_counter() - t)
                times.append(dt)
                # t = time.perf_counter()
                if env.stim_dt > 1:
                    print("Stimulating at\t",env.time_elapsed())
                    obs, done = env.step(action = ([neuron],150,100))
                    # print(obs)
                #     neuron += 1
                #     neuron %= 2
                #     print("with neuron",neuron)
                # else:
                obs,done = env.step()
                # print(q)
    except KeyboardInterrupt:
        env.worker.terminate()
        env.plot_worker.terminate()
        sys.exit(1)
