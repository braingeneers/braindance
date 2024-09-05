from brainloop.core.maxwell_env import MaxwellEnv
from brainloop.core.params import maxwell_params
from brainloop.core.phases import PhaseManager, NeuralSweepPhase, RecordPhase, FrequencyStimPhase

from brainloop.core.trainer import generate_tetanus_pattern

import numpy as np

params = maxwell_params
params['save_dir'] = './causal_tetanus' # Path to the data directory, will be created if it doesn't exist
params['name'] = 'test' # Name of the experiment

params['max_time_sec'] = 60*60*1 # 3 hours

params['config'] = None#'config.cfg'# Path to the config file
params['observation_type'] = 'raw'

params['stim_electrodes'] = [400,800,2200,2400]







env = MaxwellEnv(**params)
# ----- Let us define our phases -----

# ~~~ Record Phase ~~~
# Start with recording
record_phase1 = RecordPhase(env, duration = 60*1)
record_phase2 = RecordPhase(env, duration = 60*1)
record_phase_small = RecordPhase(env, duration = 60*1)
record_phase_small4min = RecordPhase(env, duration = 60*2)

# This sweeps every stim electrode n_replicates times at a given frequency/amplitude
neuron_list = np.arange(len(params['stim_electrodes'])) 
causal_freq = 2 # Hz
n_replicates = 30

# ~~~ Causal Phase ~~~
causal_phase = NeuralSweepPhase(env, neuron_list, amp_bounds=400,stim_freq=causal_freq,
                                 tag="Causal", replicates=n_replicates, order='nra', verbose=True)

# ~~~ Tetanus Phase ~~~
# Neurons are INDEXES of the stim_electrodes
tetanus_command = generate_tetanus_pattern(neurons = [0,1,2,3], stim_count=4, delay_ms=5,
                                        amp_mv = 400, pulse_width=100, random=False)
tetanus_freq = 10 # Hz
tetanus_phase = FrequencyStimPhase(env, tetanus_command, tetanus_freq, duration=60*1, tag="Tetanus")



# Analysis phases which can be run before, but will not be ran here...
# heatmap_phase = HeatmapPhase("Heatmap1", verbose=True,  make_gif=True, make_plots=True)
# footprint_phase = FootprintPhase(verbose=True, rms_mult=3, wind=200, load_whole_recording=False)



# ----- Let us build our experiment ----
phase_manager = PhaseManager(env, verbose=True)

# We want the following:
#18 mins
exp = [record_phase1]
no_tetanus_chunk = [causal_phase, record_phase_small4min] 
tetanus_chunk = [causal_phase, tetanus_phase, record_phase_small]

exp.extend(no_tetanus_chunk*2)
exp.extend(tetanus_chunk*4)
exp.extend(no_tetanus_chunk*2)

#18 mins
exp.append(record_phase1)
#42 mins
exp.append(record_phase2)
#18 mins
exp.append(record_phase1)


phase_manager.add_phase_group(exp)

print(phase_manager.summary())
phase_manager.run()


# analysis_dao = phase_manager.analysis_dao