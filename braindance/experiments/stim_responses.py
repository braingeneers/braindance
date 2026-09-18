import numpy as np

from braindance.core.maxwell_env import MaxwellEnv
from braindance.core.params import maxwell_params
import time
import sys
import matplotlib.pyplot as plt

params = maxwell_params
params['name'] = 'test'

params['max_time_sec'] = 60*5

params['config'] = 'config_st.cfg'


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
    
    params['stim_electrodes'] = [16079, 14549, 17414, 15430, 11255]
    read_chs = [772, 514]
    
    
    env = MaxwellEnv(**params)

    done = False

    neuron = 0

    N = 20
    nc_start = N
    spike_thresh = [-3.5,-20]
    ar = ArtifactRemoval(N=N, nc_start=nc_start, min_val=-25, max_val=25,
                        spike_thresh = spike_thresh)
    
    ar2 = ArtifactRemoval(N=N, nc_start=nc_start, min_val=-25, max_val=25,
                        spike_thresh = spike_thresh)

    
    
    step = 0
    fs = 20000

    data_len = fs*4
    data_raw = np.zeros((data_len, 2))
    data_clean = np.zeros((data_len, 2))

    
    fig, ax = plt.subplots(1,1)
    ax.set_ylim(-50,50)
    line_raw1, = ax.plot(data_raw[:,0])
    line_clean1, = ax.plot(data_clean[:,0], alpha = .8)
    line_raw2, = ax.plot(data_raw[:,1])
    line_clean2, = ax.plot(data_clean[:,1], alpha = .8)
    scat = ax.scatter([], [], c = 'r', marker = 'x', s = 20)
    scat2 = ax.scatter([], [], c = 'g', marker = 'x', s = 20)

    # Plot green dashed line at spike_thresh
    ax.axhline(spike_thresh[0], c = 'g', ls = '--')
    ax.axhline(spike_thresh[1], c = 'g', ls = '--')


    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()


    t_start = time.perf_counter()
    times = []
    time_bef = 0

    read_ch = 0
    cur_spikes1 = []
    cur_spikes2 = []
    
    stim_counts = 0
    amp = [0,400]*1000
    
    elec_switch_count = 0
    elec_switch_p = 10 # 10 seconds
    
    cur_neuron = 0
    
    plot_time = time.perf_counter()
    print('Starting')

    try:
        while not done:
            
            # Switch stim electrode
            if time.perf_counter() - t_start > elec_switch_count*elec_switch_p:
                cur_neuron += 1
                elec_switch_count += 1
                if cur_neuron >= len(params['stim_electrodes']):
                    cur_neuron = 0
                    
            if True: #dt >= 1/fs:
                if env.stim_dt > .5:
                    print("Stimulating at\t",env.time_elapsed())
                    # obs, done = env.step(action = ([0],400,100))
                    obs, done = env.step(action = ([cur_neuron],400,100))
                    
                    stim_counts += 1
                    # Stack data
                else:
                    obs, done = env.step()
                    
                # Process
                clean_data_step, art_step, spike = ar.fit_step(obs[read_chs[0]])
                clean_data_step2, art_step2, spike2 = ar2.fit_step(obs[read_chs[1]])
                
                # Store
                data_raw[step,0] = obs[read_chs[0]]
                data_raw[step,1] = obs[read_chs[1]]
                data_clean[step,0] = clean_data_step
                data_clean[step,1] = clean_data_step2

                if spike:
                    # print('Spike detected 1')
                    cur_spikes1.append(step)
                
                if spike2:
                    cur_spikes2.append(step)
                    
                    
                if time.perf_counter() - plot_time > .5:
                    plot_time = time.perf_counter()
                    print(ar.state, clean_data_step2)
                
                    # Set data in graph
                    scat.remove()
                    scat2.remove()
                    # line_raw1.set_ydata(data_raw[:,0])
                    line_clean1.set_ydata(data_clean[:,0])
                    # line_raw2.set_ydata(data_raw[:,1])
                    line_clean2.set_ydata(data_clean[:,1])
                    scat = ax.scatter(cur_spikes1, data_clean[cur_spikes1,0], c = 'r',
                                        marker = 'x', s = 20)
                    scat2 = ax.scatter(cur_spikes2, data_clean[cur_spikes2,1], c = 'g',
                                        marker = 'x', s = 20)
                    ax.set_title(f"Elec: {params['stim_electrodes'][cur_neuron]} Spikes1: {len(cur_spikes1)}  Spikes2: {len(cur_spikes2)}")
                    fig.canvas.draw()
                    fig.canvas.flush_events()


                step += 1
                if step % data_len == 0:
                    step = 0
                    cur_spikes1 = []
                    cur_spikes2 = []
                    data_raw = np.zeros((data_len,2))
                    data_clean = np.zeros((data_len,2))

                # This is a frame, now we remove artifacts
                

    except Exception as e:
        print(e)
        # print(traceback.format_exc())
    finally:
        print("Finished")
        env.close()
        if env.worker:
            env.worker.terminate()
        if env.plot_worker:
            env.plot_worker.terminate()
        sys.exit(1)
