try:
    import maxlab
except:
    # print("Could not import maxlab")
    # print("Please make sure you are running this on a maxwell computer")
    import braindance.core.dummy_maxlab as maxlab
import argparse
import json

    

def query_electrode_list(e):
    array = maxlab.chip.Array('online')
    array.reset()
    array.select_electrodes(e)
    array.select_stimulation_electrodes(e)
    array.route()
    for ei in e:
        w = array.connect_electrode_to_stimulation(ei)
        
        stimmy = array.query_stimulation_at_electrode( ei )
        print(f"elec {ei}: {stimmy} : {w}")



def query_config(config, electrodes=None, remove_duplicates=False):
    array = maxlab.chip.Array('online')
    array.reset()
    array.load_config(config)
    
    ids = {str(i):0 for i in range(32)}
    if remove_duplicates:
        electrodes_out = []
    for e in electrodes:
        try:
            status = array.connect_electrode_to_stimulation(e)
            id = array.query_stimulation_at_electrode( e )
            array.disconnect_electrode_from_stimulation(e)
            if id not in ids.keys():
                print(f"ID: {id} not valid")
                continue
        except:
            print(f"elec {e}: \tFAILED")
            continue
        ids[id] += 1
        print(f"elec {e}: \t{status}\t{id} \t({ids[id]})", end="")
        
        # If we are removing duplicates, we need to keep track of the electrodes
        if remove_duplicates:
            if ids[id] == 1:
                electrodes_out.append(e)
                print (" *", end="")
            else:
                print(" X", end="")
        
        print()

    if remove_duplicates:
        print("Electrodes without duplicates", electrodes_out)
        return electrodes_out
    else:
        return electrodes
        


def main():

    # Argparse the stim electrodes
    parser = argparse.ArgumentParser()
    parser.add_argument("--stim_electrodes", "-s", nargs="+", type=int)
    parser.add_argument('--json',"-j", type=str, default=None)
    parser.add_argument('--remove_duplicates',"-r", action='store_true')
    args = parser.parse_args()
    stim_electrodes = args.stim_electrodes
    remove_duplicates = args.remove_duplicates

    if args.json is not None:
        json_file = args.json
        json_params = json.load(open(json_file, 'r'))
        
        # Check and see if the stim electrodes are in the json file
        # or if the selected electrodes are in the json file
        
        config = json_params.get('config', None)
        print("Loading config", config)
        # print("Loading json", json_params)
        
        stim_electrodes = json_params.get('stim_electrodes', None)
        if stim_electrodes is None:
            print("No stim_electrodes in json file")
            stim_electrodes = json_params.get('selected_electrodes', None)
            if stim_electrodes is None:
                print("No selected electrodes in json file")
                raise Exception("No electrodes in json file to parse")
    else:
        config = None

    print("Querying electrodes")
    print("Stim electrodes are,", stim_electrodes)
    if config is not None:
        print('Querying')
        electrodes = query_config(config, stim_electrodes, remove_duplicates)
        if remove_duplicates:
            print("Verifying no duplicates")
            _ = query_config(config, electrodes)
            # Now we save back to the json file
            if args.json is not None:
                json_params['stim_electrodes'] = electrodes
                with open(json_file, 'w') as f:
                    json.dump(json_params, f)
    else:
        query_electrode_list(stim_electrodes)

    print("Done")

if __name__ == "__main__":
    main()
