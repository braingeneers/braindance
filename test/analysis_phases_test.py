from tabnanny import verbose
# from braindance.core.maxwell_env import MaxwellEnv
from braindance.core.params import maxwell_params
from braindance.core.phases import CartPolePhase, PhaseManager, NeuralSweepPhase, RecordPhase
from braindance.core.phases_analysis import HeatmapPhase, FootprintPhase
from braindance.analysis.data_loader import AnalysisDAO
import braindance

import os
import numpy as np
import datetime



heatmap_phase = HeatmapPhase("Heatmap1", verbose=True,  make_gif=False, make_plots=False)
footprint_phase = FootprintPhase(verbose=True, rms_mult=1, wind=100, load_whole_recording=False, 
                                num_channel_thresh=120, remove_bad=True, remove_redundant=True,
                                similarity_thresh=.65)

data_path = "/home/danser-lab/research/bgr/BrainDance/braindance/experiments/neural_config/test"
# data_path = "/home/danser-lab/research/bgr/BrainDance/test_data/test_footprint.raw.h5"

analysis = AnalysisDAO.from_file_path(data_path)

heatmap_phase.run(analysis)
# mapping = braindance.analysis.data_loader.load_mapping_maxwell(data_path + '.raw.h5')


print(analysis.selected_channels)

# Same as above but with commas
# selected_channels = [3,4,31,46,90,99,106,159,179,211,246,261,286,374,385,428,431,443,
            # 499,500,639,653,674,692,705,734,739,753,818,855,872,876,893]
footprint_phase.run(analysis)

footprint_chs = analysis.selected_footprint_chans
footprint_waves = analysis.selected_footprint_waves
mapping = analysis.mapping
selected_channels = analysis.selected_channels

from braindance.analysis.plot_helper import plot_footprints

print('Plotting footprints')
plot_footprints(footprint_chs, footprint_waves, mapping, selected_channels)
print('Done plotting footprints')

print(analysis.selected_channels)

# Save to json

analysis.update_json(data_path + '.json')






print("Done")

