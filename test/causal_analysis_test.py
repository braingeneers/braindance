from braindance.core.phases_analysis import HeatmapPhase, FootprintPhase, CausalAnalysis
from braindance.analysis.data_loader import AnalysisDAO

import matplotlib.pyplot as plt
import numpy as np

# f = '/media/danser-lab/hippo1/ephys/2023-10-26-e-tj-causal2/original/data/causal_freq_17618_5Hz_231026'
f = '/media/danser-lab/hippo1/ephys/2023-11-21-e-causal_analysis/original/data/test_3'
analysis = AnalysisDAO.from_file_path(f)
causal_analysis = CausalAnalysis(wind_ms = 150, verbose=True, file_path = f, tag='Causal', channels_of_interest = [0,200,231,245])

causal_analysis.run(analysis)

from braindance.analysis import plot_helper as ph
# Plot heatmap of the reactivity matrix
ph.plot_reactivity(analysis)
plt.show()

# Now plot the reactions on index 2 of reactivity channels to
# each of the stim electrodes
plt.figure()
for i in range(len(analysis.stim_electrodes)):
    plt.plot(analysis.reactivity[i,3,:], label = analysis.stim_electrodes[i])
plt.legend()
plt.show()

