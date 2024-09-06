import numpy as np
import braindance
import argparse
import json

if __name__ == "__main__":
    # get path to braindance module
    bd_path = braindance.__path__[0]

    # Argparse the stim electrodes
    parser = argparse.ArgumentParser()
    parser.add_argument("--stim_electrodes", "-s", nargs="+", type=int)
    parser.add_argument('--json',"-j", type=str, default=None)
    args = parser.parse_args()
    stim_electrodes = args.stim_electrodes

    if args.json is not None:
        print('Loading from JSON:')
        json_file = args.json
        json_params = json.load(open(json_file, 'r'))
        
        # Check and see if the stim electrodes are in the json file
        # or if the selected electrodes are in the json file
        stim_electrodes = json_params.get('stim_electrodes', None)
        if stim_electrodes is None or stim_electrodes == []:
            print("No stim electrodes in json file")
            stim_electrodes = json_params.get('selected_electrodes', None)
            if stim_electrodes is None:
                print("No selected electrodes in json file")
                raise Exception("No electrodes in json file to parse")
            
        # see if there is a mapping file
        mapping_file = json_params.get('mapping_file_path', None)
        if mapping_file is not None:
            mapping_file = mapping_file
            mapping = np.load(mapping_file)
            print(mapping)
    # stim_electrodes = [19098, 15349, 1258, 15281]

    a = np.load(bd_path + "/core/maxwell/stim_buffers.npy")
    elec_arr = np.arange(26400)
    b = elec_arr.reshape(26400 // 220, 220)
    c = [a[np.where(b == i)] for i in stim_electrodes]

    for ci, si in zip(c, stim_electrodes):
        print(si, ci)