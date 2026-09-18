from braindance.analysis import data_loader
from braindance.core.phases_analysis import HeatmapPhase, FootprintPhase



file_path = '/media/danser-lab/hippo1/cartpole/24-01-07_data/20217/exp1/exp1'

# Get footprint

analysis_obj = data_loader.AnalysisDAO(file_path=file_path)




heatmap_phase = HeatmapPhase("Heatmap1", verbose=True,  make_gif=False, make_plots=False, save_plots=False)

footprint_phase = FootprintPhase(verbose=True, rms_mult=1, wind=100, load_whole_recording=False, 
                                num_channel_thresh=120, remove_bad=True, remove_redundant=True,
                                similarity_thresh=.65)

analysis_obj = heatmap_phase.run(analysis_obj)
analysis_obj = footprint_phase.run(analysis_obj)


footprint_chs = analysis_obj.selected_footprint_chans
footprint_waves = analysis_obj.selected_footprint_waves
mapping = analysis_obj.mapping
selected_channels = analysis_obj.selected_channels


from braindance.analysis.plot_helper import plot_footprints
import matplotlib.pyplot as plt

print('Plotting footprints')
fig, ax = plot_footprints(footprint_chs, footprint_waves, mapping, selected_channels)
# plt.savefig(f'{analysis_obj.file_path}_footprints.png')
plt.show()
print('Done plotting footprints')

print(analysis_obj.selected_electrodes)
# analysis_obj.save_params()