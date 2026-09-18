import numpy as np

from braindance.core.maxwell_env import MaxwellEnv
from braindance.core.params import maxwell_params
import time
import sys

try:
    import maxlab
    import maxlab.system
    import maxlab.chip
    import maxlab.util
    import maxlab.saving
except ImportError:
    print("No maxlab found, instead using dummy maxlab module!")
    
    import braindance.core.dummy_maxlab as maxlab


params = maxwell_params
params['name'] = 'test'
params['stim_electrodes'] = [10254,14130]
params['max_time_sec'] = 60
params['config'] = None

# params['dummy'] = 'sine'
# params['dummy'] = '/media/danser-lab/hippo/cartpole/23-11-22_cartpole/c20215-run1/exp2/exp2_cartpole_48.raw.h5'
params['observation_type'] = 'raw' # spikes


def my_waveform(amplitude_mV=150, freq=1000):
    '''
    Adds a square wave to the sequence with a set amplitude and frequency.
    
    Parameters
    ----------
    amplitude_mV : float
        The amplitude of the square wave, in mV.
    freq : float
        The frequency of the square wave, in Hz.
        
    Returns
    -------
    maxlab.Sequence
        A sequence containing the square wave.
    '''
    # System sampling rate (20 kHz)
    sampling_rate = 20000  # Hz
    
    # Calculate samples per period based on frequency
    samples_per_period = round(sampling_rate / freq)
    
    # Calculate half period (samples spent in high and low states)
    half_period = samples_per_period // 2
    
    # Ensure we have at least 1 sample per half period
    if half_period < 1:
        half_period = 0
    
    # Convert amplitude from mV to LSBs (least significant bits)
    amplitude_lsbs = round(amplitude_mV / 2.9)  # scaling factor given by Maxwell
    
    # Create sequence
    seq = maxlab.Sequence()
    
    # Generate square wave
    # We'll create enough periods to cover 10000 samples (similar to original code)
    total_samples = 20000
    num_periods = (total_samples + samples_per_period - 1) // samples_per_period  # Ceiling division
    
    for i in range(num_periods):
        # Up (high state)
        seq.append(maxlab.chip.DAC(0, 512 - amplitude_lsbs))
        if half_period > 0:
            seq.append(maxlab.system.DelaySamples(half_period))
        
        # Down (low state)
        seq.append(maxlab.chip.DAC(0, 512 + amplitude_lsbs))
        if half_period > 0:
            seq.append(maxlab.system.DelaySamples(half_period))
    
    return seq



if __name__ == '__main__':
    env = MaxwellEnv(**params)
    
    stim_mv = 150
    my_seq = my_waveform(stim_mv)

    my_action = ("manual",my_seq,[0,1])

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
                obs, done = env.step(action = my_action)
                
                # print(obs)
            else:
                obs,done = env.step()
            