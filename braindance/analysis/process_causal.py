from braindance.analysis import data_loader
from braindance.analysis import analysis_helper as helper
import braingeneers.utils.s3wrangler as wr
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from braingeneers.analysis import SpikeData
import argparse

#import path to analysis_helper
import sys
from braindance.core.phases_analysis import CausalAnalysis
# argparse path to data
parser = argparse.ArgumentParser(description='Process Causal Analysis')
parser.add_argument('--path', type=str, default='./',
                    help='path to .raw.h5 data file')
parser.add_argument('--save', type=bool, default=True,
                    help='save clean data and reactive spikes times')
parser.add_argument('--tag', type=str, default='causal',
                    help='tag for saving files')


args = parser.parse_args()
data_path = args.path
save = args.save
tag = args.tag



# Get filepath and recording name
rec_name = data_path.split('/')[-1].split('.')[0]

data_dir = data_path.split(rec_name)[0]
print("Recording name:", rec_name)

# if False:
#     chip = data_path.split('/')[-3]
#     exp = data_path.split('/')[-2]

#     print("Chip:", chip, "Exp:", exp)

#     chip_dir = data_path.split(chip)[0]
#     exp_dir = data_path.split(exp)[0]

#     print("Chip dir:", chip_dir)
#     helper.set_path(chip_dir)

# # Get paramaters
#     obj_params = helper.get_json(chip, exp)

#     print("~"*20)
#     print("Parameters:")
#     print(obj_params)

# # Get channels of interest
#     elecs = obj_params['selected_electrodes']


analysis_obj = data_loader.AnalysisDAO().from_file_path(data_path)
channels = analysis_obj.get_channels()
electrodes = analysis_obj.get_electrodes(channels)
# channels = [channels[0]]
stim_log = data_loader.adjust_stim_times2(data_path, stim_offset_ms = 10, 
                                                        tag=tag)
stim_log['stim_electrodes'] = stim_log['stim_electrodes'].apply(lambda x: x if isinstance(x, list) and len(x) > 0 else None)

analysis_obj.set_stim_log(stim_log)
stim_electrodes = np.unique(stim_log['stim_electrodes'].apply(lambda x: tuple(x) if isinstance(x, list) else x))


print("-"*20)
print("\nRunning Causal Analysis on recording name:", rec_name)

print("Stim electrodes:", stim_electrodes)

raise ValueError("Check the code below")
causal_analysis = CausalAnalysis(file_path = data_path,clean_data_path = True,tag="causal", channels_of_interest=channels,
                                 wind_ms=150)


if not os.path.exists(f"{data_dir}{rec_name}_analysis"):
    os.mkdir(f"{data_dir}{rec_name}_analysis")
for ind, stim_electrode in enumerate(stim_electrodes):
    analysis_obj = causal_analysis.run(analysis_obj, stim_electrode_ind=ind)
    # Save clean data and reactive_spikes_times
    if save:
        np.save(f"{data_dir}{rec_name}_analysis/{rec_name}_clean_{stim_electrode}.npy", analysis_obj.clean_data)
        np.save(f"{data_dir}{rec_name}_analysis/{rec_name}_reactive_spikes_times_{stim_electrode}.npy", analysis_obj.reactivity_times)
