from braindance.core.maxwell_env import MaxwellEnv
from braindance.core.params import maxwell_params
from braindance.core.phases import PhaseManager, NeuralSweepPhase, RecordPhase
from braindance.core.phases_analysis import HeatmapPhase, FootprintPhase
from braindance.analysis.data_loader import AnalysisDAO

import numpy as np

params = maxwell_params
params['save_dir'] = './neural_config' # Path to the data directory, will be created if it doesn't exist
params['name'] = 'test' # Name of the experiment

params['max_time_sec'] = 60*5 # 30 sec

params['config'] = None#'config.cfg'# Path to the config file
params['observation_type'] = 'raw'

params['stim_electrodes'] = []



env = MaxwellEnv(**params)

# Start with recording
record_phase = RecordPhase(env, 60*3)
heatmap_phase = HeatmapPhase("Heatmap1", verbose=True,  make_gif=False, make_plots=False)
footprint_phase = FootprintPhase(verbose=True, rms_mult=1, wind=100, load_whole_recording=False, 
                                num_channel_thresh=120, remove_bad=True, remove_redundant=True,
                                similarity_thresh=.65)

# Build experiment
phase_manager = PhaseManager(env, verbose=True)

phase_manager.add_phase(record_phase)
phase_manager.add_phase(heatmap_phase)
phase_manager.add_phase(footprint_phase)

phase_manager.summary()
phase_manager.run()

# Get the analysis object
analysis = phase_manager.analysis_dao
print(analysis.selected_electrodes)


# ALL OF THIS BELOW IS TO PLOT FOOTPRINTS

footprint_chs = analysis.selected_footprint_chans
footprint_waves = analysis.selected_footprint_waves
mapping = analysis.mapping
selected_channels = analysis.selected_channels



from braindance.analysis.plot_helper import plot_footprints

print('Plotting footprints')
plot_footprints(footprint_chs, footprint_waves, mapping, selected_channels)
print('Done plotting footprints')

print(analysis.selected_electrodes)
analysis.save_params()
