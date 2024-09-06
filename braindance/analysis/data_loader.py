import smart_open
import pandas as pd
import ast
import os
import numpy as np
import h5py


def load_info_maxwell(filepath):
    """ Loads metadata from a maxwell file
    
    Parameters
    ----------
    filepath : str
        path to maxwell file
        
    Returns
    -------
    metadata : dict
        metadata dictionary
    """
    from datetime import datetime

    # if no .raw.h5 extension, add it
    if not filepath.endswith('.raw.h5'):
        filepath += '.raw.h5'
        
    info = {}
    # open file
    with smart_open.open(filepath, 'rb') as file:
        with h5py.File(file, 'r', libver='latest', rdcc_nbytes=2 ** 25) as h5file:
            # know that there are 1028 channels which all record and make 'num_frames'
            # lsb = np.float32(h5file['settings']['lsb'][0]*1000) #1000 for uv to mv  # voltage scaling factor is not currently implemented properly in maxwell reader
            
            lsb = np.float32(h5file['/data_store/data0000/settings/lsb'][0])
            gain = np.float32(h5file['/data_store/data0000/settings/gain'][0])
            hpf = np.float32(h5file['/data_store/data0000/settings/hpf'][0])
            table = 'sig' if 'sig' in h5file.keys() else '/data_store/data0000/groups/routed/raw'
            info['shape'] = h5file[table].shape
            start_time = h5file['data_store/data0000/start_time'][0]
            info['start_time'] = datetime.fromtimestamp(start_time / 1e3).strftime('%Y%m%d-%H%M%S')
            info['lsb'] = lsb
            info['gain'] = gain
            info['hpf'] = hpf


    return info

def load_mapping_maxwell(filepath, channels=None):
    """ Loads mapping from a maxwell file
    
    Parameters
    ----------
    filepath : str
        path to maxwell file
    channels : list of int
        channels of interest
        
    Returns
    -------
    mapping : dict
        mapping dictionary
    """
    # if no .raw.h5 extension, add it
    if not filepath.endswith('.raw.h5'):
        filepath += '.raw.h5'

    # open file
    with smart_open.open(filepath, 'rb') as f:
        with h5py.File(f, 'r') as h5:
            # version is 20160704 - ish?, old format
            if 'mapping' in h5:
                mapping = np.array(h5['mapping']) #ch, elec, x, y
                mapping = pd.DataFrame(mapping)
                # Set orig_channel to be the same as channel
                mapping['orig_channel'] = mapping['channel']
                # set channel to be the 
                mapping['channel'] = np.arange(mapping.shape[0])
            # version is 20190530 - ish?
            else:
                mapping = np.array(h5['data_store/data0000/settings/mapping'])
                mapping = pd.DataFrame(mapping)
                # Set orig_channel to be the same as channel
                mapping['orig_channel'] = mapping['channel']
                # set channel to be the 
                mapping['channel'] = np.arange(mapping.shape[0])
    if channels is not None:
        return mapping[mapping['channel'].isin(channels)]
    else:
        return mapping

           
def load_data_maxwell(filepath, channels=None, start=0, length=-1, spikes=False, dtype=np.float32,
                      suffix = None, verbose=False):
    """
    Loads specified amount of data from a maxwell file
    :param filepath: 
        Path to filename.raw.h5 file
    :param channels: list of int
        Channels of interest
    :param start: int
        Starting frame (offset) of the datapoints to use
    :param length: int
        Length of datapoints to take
    :param spikes: bool
        Whether to only thresholded spikes or raw data
    :param dtype: np.dtype
        Data type to load
    :param suffix: str
        Suffix to add to filepath
    :param verbose: bool
        Whether to print errors

    :return:
    dataset: nparray
        Dataset of datapoints.
    """
    # if no .raw.h5 extension, add it
    if suffix is not None:
        filepath += suffix
    elif not filepath.endswith('.raw.h5'):
        filepath += '.raw.h5'

    if channels is not None:
        # Ensure unique channels
        assert len(channels) == len(np.unique(channels)), f"Channels must be unique, but have length {len(channels)} and unique {len(np.unique(channels))} unique channels"
    
    frame_end = start + length 

    # Defaults if not in file
    lsb = 6.294*10**-6
    gain = 512
    sig_offset = 512

    # open file
    with smart_open.open(filepath, 'rb') as file:
        with h5py.File(file, 'r', libver='latest', rdcc_nbytes=2 ** 25) as h5file:
            # know that there are 1028 channels which all record and make 'num_frames'
            # lsb = np.float32(h5file['settings']['lsb'][0]*1000) #1000 for uv to mv  # voltage scaling factor is not currently implemented properly in maxwell reader
            try:
                if np.float32(h5file['/data_store/data0000/settings/lsb'][0]) == 0:
                    raise ValueError('lsb is 0, cannot trust scaling values, using defaults')

                lsb = np.float32(h5file['/data_store/data0000/settings/lsb'][0])
                gain = np.float32(h5file['/data_store/data0000/settings/gain'][0])
                hpf = np.float32(h5file['/data_store/data0000/settings/hpf'][0])
            except Exception as e:
                if verbose:
                    print(e)

            # print(h5file['/data_store/data0000/groups/routed'].keys())
            table = 'sig' if 'sig' in h5file.keys() else '/data_store/data0000/groups/routed/raw'
            if spikes:
                spikes = h5file['/data_store/data0000/spikes']
                mapping = load_mapping_maxwell(filepath)

                start_frame = h5file['/data_store/data0000/groups/routed/frame_nos'][0]
                # Convert spikes to a DataFrame
                columns_to_load = ['frameno', 'channel', 'amplitude']  # adjust as needed
    

                if 's3' in filepath:
                    spikes_data = np.array(h5file['/data_store/data0000/spikes'])
                    spikes_df = pd.DataFrame(spikes_data, columns=columns_to_load)
                else:
                    spikes_df = pd.read_hdf(filepath, '/data_store/data0000/spikes', columns=columns_to_load)
                
                # Change frameno to frame
                spikes_df.rename(columns={'frameno': 'frame'}, inplace=True)

                # Filter out spikes that don't have a corresponding channel in the mapping DataFrame
                filtered_spikes_df = spikes_df[spikes_df['channel'].isin(mapping['orig_channel'])].copy()

                # Convert channel to the new channel number
                # Use .loc to avoid SettingWithCopyWarning
                channel_map = mapping.set_index('orig_channel')['channel']
                

                filtered_spikes_df.loc[:, 'channel'] = filtered_spikes_df['channel'].map(channel_map)

                # Adjust the 'frame' column
                filtered_spikes_df.loc[:, 'frame'] = filtered_spikes_df['frame'] - start_frame

                # Remove all rows with negative times
                filtered_spikes_df = filtered_spikes_df[filtered_spikes_df['frame'] >= 0]

                
                return filtered_spikes_df
                # return np.array(h5file['/data_store/data0000/spikes'])
            
            dataset = h5file[table]
            
            if channels is not None:
                sorted_channels = np.sort(channels)
                undo_sort_channels = np.argsort(np.argsort(channels))

                dataset = dataset[sorted_channels, start:frame_end]
            else:
                dataset = dataset[:, start:frame_end]
    
    if channels is not None:
        # Unsort data
        dataset = dataset[undo_sort_channels, :]
    
    
    if dtype is np.float32:
        return (np.array(dataset, dtype=np.float32) - sig_offset)* lsb * gain * 1000 # convert to mV
    elif dtype is np.int16:
        return (np.array(dataset, dtype=np.int16))
    else:
        print('Hmm you shouldnt be here, probably use float32 or int')
        return (np.array(dataset, dtype=dtype) - sig_offset)* lsb * gain * 1000 # convert to mV


def convert_uint16_maxwell(data):
    """
    Converts uint16 data to float32
    :param data: nparray
        Data to convert
    :return:
    data: nparray
        Converted data
    """
    # Defaults if not in file
    lsb = 6.294*10**-6
    gain = 512
    sig_offset = 512
    return (np.array(data, dtype=np.float32) - sig_offset)* lsb * gain * 1000 # convert to mV


def load_windows_maxwell(filepath, starts, window_sz=2000, channels=None):
    """
    Loads a fixed window size from the list of starts
    :param filepath: 
        Path to filename.raw.h5 file
    :param starts: list of int
        List of start frames
    :param window_sz: int
        Window size in frames
    :param channels: list of int

    :return:
    dataset: nparray
        Dataset of datapoints.
    """
    if channels is None:
        data_shape = load_info_maxwell(filepath)['shape']
        channels = np.arange(data_shape[0])
        
    data_chunks = np.zeros((len(starts), len(channels), window_sz))

    for i, start in enumerate(starts):
        try:
            data_chunks[i] = load_data_maxwell(filepath, channels=channels,
                                    start=start, length=window_sz)
        except Exception as e:
            print('Error loading window', start)
            print(e)
            data_chunks[i] = np.zeros((len(channels), window_sz))
    return data_chunks


def load_stim_log(filepath, adjust=False, suffix='_log.csv'):

    if not filepath.endswith(suffix):
        if filepath.endswith('.csv'):
            print("Warning: filepath does not end with", suffix)
        else:
            filepath += suffix


    def string_to_list(s):
        return ast.literal_eval(s)

    with smart_open.open(filepath, 'rb') as file:
        stim_log = pd.read_csv(file, sep=',', header=0, converters={'stim_electrodes': string_to_list})

    if adjust:
        # Adjust stim times
        pass

    stim_log['stim_electrodes'] = stim_log['stim_electrodes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    if 'stim_pattern' in stim_log.columns:
        stim_log['stim_pattern'] = stim_log['stim_pattern'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return stim_log


def apply_literal_eval_stim_log(stim_log):
    """
    Applies ast.literal_eval to the stim electrodes and stim pattern columns of the stim log
    needed because the columns are read as strings"""
    stim_log['stim_electrodes'] = stim_log['stim_electrodes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    if 'stim_pattern' in stim_log.columns:
        stim_log['stim_pattern'] = stim_log['stim_pattern'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return stim_log

def get_stim_electrodes(stim_log):
    """
    Get the unique stim electrodes from the stim log
    """
    stim_electrodes = []
    for i, stim in stim_log.iterrows():
        stim_electrodes.extend(stim['stim_electrodes'])
    return np.unique(stim_electrodes)



def load_windows(metadata, exp, window_centers, window_sz, dtype=np.float16,
                channels=None):
    '''Loads a window of data from an experiment
    window is in frames
    Parameters
    ----------
    metadata : dict
        metadata dictionary
    exp : str
        experiment name
    window_centers : list
        list of window centers in frames
    window_sz : int
        window size in frames
    dtype : np.dtype
        data type to load
    
    Returns
    -------
    data : np.array (n_windows, n_channels, window_sz)

    '''
    data = []
    dataset_length = metadata['ephys_experiments'][exp]['blocks'][0]['num_frames']
    if channels is None: 
        num_channels = metadata['ephys_experiments'][exp]['num_channels']
    else:
        num_channels = len(channels)


    for i,center in enumerate(window_centers):
        # window is (start, end)
        window = (center - window_sz//2, center + window_sz//2)

        # Check if window is out of bounds
        if window[0] < 0 or window[1] > dataset_length:
            print("Window out of bounds, inserting zeros for window",window)
            data_temp = np.zeros((num_channels,window_sz),dtype=dtype)
        else:
            data_temp = load_window(metadata, exp, window, dtype=dtype, channels=channels)
        
        # Check if window is the right size
        if data_temp.shape[1] != window_sz:
            print("Data shape mismatch, inserting zeros for window",window)
            data_temp = np.zeros((data_temp.shape[0],window_sz),dtype=dtype)
        
        data.append(data_temp)
    return np.stack(data, axis=0)
