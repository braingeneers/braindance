import numpy as np

from braindance.core.maxwell_env import MaxwellEnv
from braindance.core.params import maxwell_params
import time
import sys
from braindance.core.spikedetector.model import ModelSpikeSorter
from braindance.core.spikedetector.spikedetector import SpikeDetector
import matplotlib.pyplot as plt

params = maxwell_params
params['name'] = 'test'
params['stim_electrodes'] = [10254,14130]
params['max_time_sec'] = 60

params['config'] = None


params['multiprocess'] = False
params['render'] = False
params['observation_type'] = 'raw'
params['dummy'] = 'sine'# put a file here

n_channels=4


ind = 0

fig, ax = plt.subplots(1,1)

if __name__ == '__main__':
    env = MaxwellEnv(**params)

    done = False

    neuron = 0

    q = 0
    fs = 20000
    buffer_size = 20*30 # 30ms
    t = time.perf_counter()
    times = []
    time_bef = 0

    read_ch = 0

    plot_buffer = np.zeros((n_channels, fs*4))
    ind = 0
    try:
        while not done:
            # Bind the loop to 1ms step
            dt = env.dt
            if dt >=  0:#1/fs:
                if env.stim_dt > .5:
                    print("Stimulating at\t",env.time_elapsed())
                    time_bef = time.perf_counter()
                    obs, done = env.step(action = ([neuron],150,100), buffer_size=buffer_size)
                    times.append(time.perf_counter() - time_bef)

                    # Buffer_size, n_channels
                    obs = np.array(obs)
                    print(obs.shape)
                    # Stack data

                    plot_buffer[:,ind:ind+buffer_size] = obs[:,:n_channels].T
                    ind += 1

                    

                    # Clear
                    ax.clear()
                    # ax.plot(probs[read_ch,0,:])

                    ax.plot(plot_buffer[read_ch,:])
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    fig.show()
                    continue


                obs, done = env.step(buffer_size=buffer_size)
                obs = np.array(obs)
                plot_buffer[:,ind:ind+buffer_size] = obs[:,:n_channels].T
                ind += buffer_size
                if ind >= fs*4 - buffer_size:
                    ind = 0
                

    except Exception as e:
        print(e)
    finally:
        ms_per_step = np.array(times) * 1000
        print("Average time per buff step: {:0.2f}ms".format(np.mean(ms_per_step)))
        if env.worker:
            env.worker.terminate()
        if env.plot_worker:
            env.plot_worker.terminate()
        sys.exit(1)
