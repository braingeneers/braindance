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
params['dummy'] = '/media/danser-lab/hippo/cartpole/24-01-07_data/20217/exp1/exp1.raw.h5'

n_channels=4
model_path = '../core/spikedetector/model_8ch_torch.pt'
import torch
map_location=torch.device('cpu')
detector = SpikeDetector(model_path, n_channels=n_channels, device=map_location)

# model = torch.jit.load('../core/spikedetector/model_256ch.pt')
# data_slice = torch.zeros(256,1, 200)
# Data should be type half
# data_slice = torch.zeros(256,1, 200).half().cuda()
# Suppress torch warning
# torch.Tensor._use_fallback_warn = False

ind = 0

fig, ax = plt.subplots(1,1)

if __name__ == '__main__':
    env = MaxwellEnv(**params)

    done = False

    neuron = 0

    q = 0
    fs = 20000
    t = time.perf_counter()
    times = []
    time_bef = 0

    read_ch = 0

    buffer = np.zeros((n_channels, fs*4))
    ind = 0
    try:
        while not done:
            # Bind the loop to 1ms step
            dt = env.dt
            if dt >=  0:#1/fs:
            # if time.perf_counter() - t >= 1/(fs+800):
                # q += 1
                # print (time.perf_counter() - t)
                # times.append(time.perf_counter() - t)
                # times.append(dt)
                # t = time.perf_counter()
                if env.stim_dt > .5:
                    print("Stimulating at\t",env.time_elapsed())
                    obs, done = env.step(action = ([neuron],150,100))
                    # Stack data
                    

                    # Feed through model
                    # This is ch, 0, time (120frames -> 6ms)

                    obs, done = env.step()
                    arr = np.array(obs)[:n_channels]
                    probs = detector.detect(arr)

                    buffer[:,ind] = arr
                    ind += 1
                    
                    # data_slice[:,0,ind] = torch.from_numpy(np.array(obs)[:256]).half()
                    time_bef = time.perf_counter()
                    probs = detector.detect(np.array(obs)[:n_channels])
                    # output = model(data_slice)
                    # probs = output.cpu().detach().numpy()
                    times.append(time.perf_counter() - time_bef)

                    arr = detector.data_slice[read_ch,0,:].cpu().detach().numpy()

                    # Clear
                    ax.clear()
                    # ax.plot(probs[read_ch,0,:])

                    ax.plot(buffer[read_ch,:])
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    fig.show()
                    continue


                obs, done = env.step()
                arr = np.array(obs)[:n_channels]
                probs = detector.detect(arr)
                buffer[:,ind] = arr
                ind += 1
                if ind == fs*4:
                    ind = 0
                

    except Exception as e:
        print(e)
    finally:
        ms_per_step = np.array(times) * 1000
        print("Average time per ML step: {:0.2f}ms".format(np.mean(ms_per_step)))
        if env.worker:
            env.worker.terminate()
        if env.plot_worker:
            env.plot_worker.terminate()
        sys.exit(1)
