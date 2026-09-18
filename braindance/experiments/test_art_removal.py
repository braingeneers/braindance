import numpy as np

from braindance.core.maxwell_env import MaxwellEnv
from braindance.core.params import maxwell_params
import time
import sys
import matplotlib.pyplot as plt

params = maxwell_params
params['name'] = 'test'
params['stim_electrodes'] = [10254,14130]
params['max_time_sec'] = 60

params['config'] = None


params['multiprocess'] = False
params['render'] = False
params['observation_type'] = 'raw'

n_channels=4
from braindance.core.artifact_removal import ArtifactRemoval


# model = torch.jit.load('../core/spikedetector/model_256ch.pt')
# data_slice = torch.zeros(256,1, 200)
# Data should be type half
# data_slice = torch.zeros(256,1, 200).half().cuda()
# Suppress torch warning
# torch.Tensor._use_fallback_warn = False

ind = 0



if __name__ == '__main__':
    env = MaxwellEnv(**params)

    done = False

    neuron = 0

    N = 10
    nc_start = N
    spike_thresh = -15
    ar = ArtifactRemoval(N=N, nc_start=nc_start, min_val=-200, max_val=200,
                        spike_thresh = spike_thresh)


    step = 0
    fs = 20000

    data_len = fs//2
    data_raw = np.zeros((data_len, 1))
    data_clean = np.zeros((data_len, 1))

    
    fig, ax = plt.subplots(1,1)
    ax.set_ylim(-50,50)
    line_raw, = ax.plot(data_raw)
    line_clean, = ax.plot(data_clean, alpha = .5)
    scat = ax.scatter([], [], c = 'r', marker = 'x', s = 20)

    # Plot green dashed line at spike_thresh
    ax.axhline(spike_thresh, c = 'g', ls = '--')


    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()


    t = time.perf_counter()
    times = []
    time_bef = 0

    read_ch = 0
    cur_spikes = []

    print('Starting')

    try:
        while not done:
            # Bind the loop to 1ms step
            dt = env.dt
            if dt >= 1/fs:
            # if time.perf_counter() - t >= 1/(fs+800):
                # q += 1
                # print (time.perf_counter() - t)
                # times.append(time.perf_counter() - t)
                # times.append(dt)
                # t = time.perf_counter()
                if env.stim_dt > .1:
                    print("Stimulating at\t",env.time_elapsed())
                    obs, done = env.step(action = ([neuron],150,100))
                    # Stack data
                    

                    # Set data in graph
                    scat.remove()
                    line_raw.set_ydata(data_raw)
                    line_clean.set_ydata(data_clean)
                    scat = ax.scatter(cur_spikes, data_clean[cur_spikes], c = 'r',
                                        marker = 'x', s = 20)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    # ax.clear()
                    # ax.plot(data_raw)
                    # ax.plot(data_clean, alpha = .5)
                    # fig.canvas.draw()
                    # fig.canvas.flush_events()
                    # fig.show()
                else:
                    obs, done = env.step()

                # Process
                obs = np.array(obs)
                obs[0] = obs[0] + np.random.normal(0, 1)
                if np.random.rand() < .001:
                    obs[0] = obs[0] + -np.random.randint(5, 30)

                if np.random.rand() < .005:
                    obs[0] = obs[0] + -np.random.randint(400, 600)
                clean_data_step, art_step, spike = ar.fit_step(obs[0])
                
                # Store
                data_raw[step] = obs[0]
                data_clean[step] = clean_data_step



                if spike:
                    print('Spike detected')
                    cur_spikes.append(step)

                step += 1
                if step % data_len == 0:
                    step = 0
                    cur_spikes = []
                    data_raw = np.zeros((data_len))
                    data_clean = np.zeros((data_len))

                # This is a frame, now we remove artifacts
                

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
