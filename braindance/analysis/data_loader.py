import numpy as np
import h5py
from braindance.io import open_file
import posixpath
import pathlib
from numpy import ndarray
from dataclasses import dataclass, field
from copy import deepcopy
from itertools import product
import pandas as pd
import pickle
import os
import warnings


@dataclass
class AnalysisDAO:
    def __init__(self, spikes=None, file_path=None, mapping=None):
        """
        Initializes the AnalysisDAO.
        
        Args:
            spikes (pd.DataFrame): A pandas dataframe containing at least:
                - time: The spike time in frames
                - channel: The channel number
                - amp: The amplitude number
            file_path (str, optional): The path to the raw data file. Defaults to None.
            mapping (pd.DataFrame, optional): A pandas dataframe containing at least:
                - channel: The channel number
                - electrode: The electrode number
                - x: The x coordinate
                - y: The y coordinate
                Defaults to None.
        """
        warnings.warn(
            "AnalysisDAO is deprecated and retained for legacy selection workflows. "
            "Use Recording for recording data; see docs/data-api-migration-review.md "
            "before migrating selection or experiment state.",
            FutureWarning,
            stacklevel=2,
        )
        self.file_path = file_path
        if file_path is not None:
            self.name = file_path.split('/')[-1].split('.')[0]
        else:
            self.name = None

        if spikes is not None:
            self.set_spikes(spikes)
        else:
            if file_path is not None:
                self.set_spikes(load_data_maxwell(file_path, spikes=True))

        if mapping is not None:
            self.set_mapping(mapping)
        else:
            if file_path is not None:
                self.set_mapping(load_mapping_maxwell(file_path))


        self.py_file = None


        self.selected_electrodes = None
        self.selected_channels = None

        self.selected_footprint_chans = None
        self.selected_footprint_waves = None
        self.selected_footprint_pk_to_pks = None

        self.stim_electrodes = None
        self.stim_log = None
        self.channels_of_interest = None
        self.reactivity = None 

    
    @classmethod
    def from_file_path(cls, file_path, get_spikes=True):
        """
        Initializes the AnalysisDAO from a file path.
        
        Args:
            file_path (str): The path to the raw data file.
            
        Returns:
            An AnalysisDAO object.
        """
        if get_spikes:
            spikes = load_data_maxwell(file_path, spikes=True)
        else:
            spikes = None
        mapping = load_mapping_maxwell(file_path)
        return cls(spikes, file_path, mapping)



    def __repr__(self):
        return f"AnalysisDAO(num_spikes={len(self.spikes)}, file_path={self.file_path}, n_channels={len(self.channels)})"

    def set_spikes(self, spikes):
        if spikes is None:
            self.spikes = None
            self.num_spikes = 0
            return
        self.spikes = spikes
        self.num_spikes = len(spikes)

    def set_footprints(self, footprint_chans, footprint_waves, footprint_pk_to_pks):
        self.selected_footprint_chans = footprint_chans
        self.selected_footprint_waves = footprint_waves
        self.selected_footprint_pk_to_pks = footprint_pk_to_pks

    def select_electrodes(self, electrodes):
        """
        Selects electrodes.
        
        Args:
            electrodes (list): A list of electrodes.
        """
        self.selected_electrodes = electrodes
        self.selected_channels = [self.mapping[self.mapping['electrode'] == electrode]['channel'].values[0]
                        for electrode in electrodes]

    def select_channels(self, channels):
        """
        Selects channels.
        
        Args:
            channels (list): A list of channels.
        """
        self.selected_channels = channels
        self.selected_electrodes = [self.mapping[self.mapping['channel'] == channel]['electrode'].values[0] 
                        for channel in channels]

    def set_file_path(self, file_path, get_mapping=True):
        self.file_path = file_path
        if get_mapping:
            self.set_mapping(load_mapping_maxwell(file_path))

    def set_mapping(self, mapping):
        if mapping is None:
            print("No mapping provided")
            self.mapping = None
            self.channels = None
            self.electrodes = None
            return
        self.mapping = mapping
        self.mapping_file = None
        self.channels = mapping['channel'].values
        # Ensure channels are integers
        self.channels = [int(ch) for ch in self.channels]
        
        self.electrodes = mapping['electrode'].values
        self.electrodes = [int(elec) for elec in self.electrodes]


    def set_stim_electrodes(self, stim_electrodes):
        self.stim_electrodes = stim_electrodes

    def set_stim_log(self, stim_log):
        self.stim_log = stim_log

    def set_channels_of_interest(self, channels_of_interest):
        self.channels_of_interest = channels_of_interest

    def set_reactivity(self, reactivity):
        self.reactivity = reactivity

    def set_reactivity_times(self, reactivity_times):
        self.reactivity_times = reactivity_times

    def set_clean_data(self, clean_data):
        self.clean_data = clean_data

    def get_electrodes(self, channels=None):
        """
        Get electrodes from the AnalysisDAO.
        
        Args:
            channels (list, optional): A list of channels. Defaults to None.
            
        Returns:
            A list of electrodes.
        """
        if channels is None:
            channels = self.channels
        return [self.mapping[self.mapping['channel'] == channel]['electrode'].values[0] 
                        for channel in channels]

    def get_channels(self, electrodes=None):
        """
        Get channels from the AnalysisDAO.
        
        Args:
            electrodes (list, optional): A list of electrodes. Defaults to None.
            
        Returns:
            A list of channels.
        """
        if electrodes is None:
            electrodes = self.electrodes
        return [self.mapping[self.mapping['electrode'] == electrode]['channel'].values[0]
                        for electrode in electrodes]

    def get_orig_channels(self, channels=None, electrodes=None):
        """
        Get original channels from the AnalysisDAO, converting either channels or electrodes to original channels.
        
        Args:
            channels (list, optional): A list of channels. Defaults to None.
            electrodes (list, optional): A list of electrodes. Defaults to None.
            
        Returns:
            A list of original channels with original order preserved.
        """
        if channels:
            return [self.mapping[self.mapping['channel'] == ch]['orig_channel'].values[0]
                        for ch in channels]
        elif electrodes:
            return [self.mapping[self.mapping['electrode'] == elec]['orig_channel'].values[0]
                        for elec in electrodes]
        else:
            return self.mapping['orig_channel'].values
        
    def get_nearest_channels(self, channels=None, electrodes=None, n=1):
        """
        Get the nearest channels to the given channels or electrodes.
        
        Args:
            channels (list, optional): A list of channels. Defaults to None.
            electrodes (list, optional): A list of electrodes. Defaults to None.
            n (int, optional): The number of nearest channels to get. Defaults to 1.
            
        Returns:
            A list of lists of n nearest channels or electrodes for each provided channel or electrode.
            If you provide a list of channels, you will get a list of lists of n nearest channels.
            If you provide a list of electrodes, you will get a list of lists of n nearest electrodes.
        """
        if channels and electrodes:
            raise ValueError("Cannot provide both channels and electrodes")
        elif channels:
            pass
        elif electrodes:
            channels = self.get_channels(electrodes)
        else:
            raise ValueError("Must provide either channels or electrodes")
        nearest_channels = []
        for channel in channels:
            # Get the x and y coordinates of the channel
            x, y = self.mapping[self.mapping['channel'] == channel][['x', 'y']].values[0]
            # Make temporary dataframe with distances
            temp_df = self.mapping.copy()
            temp_df['distance'] = np.sqrt((temp_df['x'] - x)**2 + (temp_df['y'] - y)**2)
            # Sort by distance
            temp_df.sort_values(by='distance', inplace=True)
            # Get the n nearest channels
            nearest_channels.append(temp_df['channel'].values[1:n+1])

        # Convert to electrodes if electrodes were provided
        if electrodes:
            nearest_channels = [self.get_electrodes(chans) for chans in nearest_channels]
        return nearest_channels

    def get_positions(self, channels=None, electrodes=None):
        """
        Get the x and y coordinates of the given channels or electrodes.
        
        Args:
            channels (list, optional): A list of channels. Defaults to None.
            electrodes (list, optional): A list of electrodes. Defaults to None.
            
        Returns:
            A list of tuples of x and y coordinates.
        """
        if channels is not None and electrodes is not None:
            raise ValueError("Cannot provide both channels and electrodes")
        elif channels is not None:
            pass
        elif electrodes is not None:
            channels = self.get_channels(electrodes)
        else:
            raise ValueError("Must provide either channels or electrodes")
        return self.mapping[self.mapping['channel'].isin(channels)][['x', 'y']].values        
        

    def get_spikes(self, channels=None, amplitudes=None, frame_bounds=None):
        """
        Get spikes from the AnalysisDAO.
        
        Args:
            channels (list, optional): A list of channels. Defaults to None.
            amplitudes (list, optional): A list of amplitudes. Defaults to None.
            frame (tuple, length 2, optional): A tuple of start and end frames. Defaults to None.
            
        Returns:
            A filtered AnalysisDAO object.
        """
        if np.issubdtype(type(channels), int):
            channels = [channels]
        if np.issubdtype(type(amplitudes), int):
            amplitudes = [amplitudes]
        if channels is None:
            channels = self.channels
        if frame_bounds is None:
            frame_bounds = (0, np.inf)
        if amplitudes is None:
            return self.spikes[(self.spikes['channel'].isin(channels)) & (self.spikes['frame'] >= frame_bounds[0]) & (self.spikes['frame'] <= frame_bounds[1])]
        else:
            return self.spikes[(self.spikes['channel'].isin(channels)) & (self.spikes['frame'] >= frame_bounds[0]) & (self.spikes['frame'] <= frame_bounds[1]) & (self.spikes['amplitude'].isin(amplitudes))]


    def nbytes(self):
        """
        Get the number of bytes of the AnalysisDAO. This includes the dataframes for spikes and mapping.
        
        Returns:
            The number of bytes of the AnalysisDAO.
        """
        return self.spikes.memory_usage(deep=True).sum() + self.mapping.memory_usage(deep=True).sum()

    def save_params(self, file_path=None):
        """
        Saves the AnalysisDAO to a python file which has:
        selected_channels: The selected channels.
        selected_electrodes: The selected electrodes.
        selected_footprint_chans: The selected footprint channels.
        selected_footprint_waves: The selected footprint waves.

        saves mapping to a csv file
        
        Args:
            file_path (str): The path to the file.
        """
        # if no .py extension, add it
        if file_path is None:
            file_path = self.file_path
        file_base_name = file_path.split('.')[0] # remove extension if it exists
        
        py_file = file_base_name + '_params.py'
        csv_file = file_base_name + '_mapping.csv'

        with open(py_file, 'w') as f:
            f.write(f"selected_channels = {self.selected_channels}\n")
            f.write(f"selected_electrodes = {self.selected_electrodes}\n")
            f.write(f"selected_footprint_chans = {self.selected_footprint_chans}\n")
            selected_footprint_elecs = [self.get_electrodes(chans) for chans in self.selected_footprint_chans]
            f.write(f"selected_footprint_elecs = {selected_footprint_elecs}\n")

        self.mapping.to_csv(csv_file)
        self.mapping_file = csv_file
        self.py_file = py_file

    def update_json(self, json_file_path, overwrite=False):
        """
        Updates the json file with the parameters in the AnalysisDAO.

        
        Args:
            json_file_path (str): The path to the json file.
            overwrite (bool, optional): Whether to overwrite the file. Defaults to False.
        """
        import json
        if not overwrite:
            with open(json_file_path, 'r') as f:
                json_params = json.load(f)
        else:
            json_params = {}

        # json_params['selected_channels'] = self.selected_channels
        json_params['selected_electrodes'] = self.selected_electrodes
        json_params['stim_electrodes'] = self.selected_electrodes
        # json_params['selected_footprint_chans'] = self.selected_footprint_chans
        # json_params['selected_footprint_elecs'] = [self.get_electrodes(chans) for chans in self.selected_footprint_chans]

        # Check if we have a mapping file
        if self.mapping_file is not None:
            json_params['mapping_file_path'] = self.mapping_file

        if self.py_file is not None:
            json_params['py_file_path'] = self.py_file


        with open(json_file_path, 'w') as f:
            # Dump to json file with newline after each value
            json.dump(json_params, f, cls=NumpyEncoder, indent=4, sort_keys=True)
            

    def save(self, file_path):
        """
        Saves the AnalysisDAO to a file.
        
        Args:
            file_path (str): The path to the file.
        """
        base_dir = os.path.dirname(file_path)
        
        pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)


    @classmethod
    def load(cls, file_path):
        """
        Loads the AnalysisDAO from a file.
        
        Args:
            file_path (str): The path to the file.
            
        Returns:
            An AnalysisDAO object.
        """
        warnings.warn(
            "AnalysisDAO is deprecated; loading legacy artifacts remains supported. "
            "See docs/data-api-migration-review.md before migrating.",
            FutureWarning,
            stacklevel=2,
        )
        with open(file_path, 'rb') as f:
            return pickle.load(f)

import json

class NumpyEncoder(json.JSONEncoder): 
    def default(self, obj):
        # Convert numpy types to Python types for JSON serialization
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, 
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
                                np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)): # Handle numpy arrays
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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
    if not str(filepath).endswith('.raw.h5'):
        filepath = str(filepath) + '.raw.h5'
        
    info = {}
    # open file
    with open_file(filepath, 'rb') as file:
        with h5py.File(file, 'r', libver='latest', rdcc_nbytes=2 ** 25) as h5file:
            # know that there are 1028 channels which all record and make 'num_frames'
            # lsb = np.float32(h5file['settings']['lsb'][0]*1000) #1000 for uv to mv  # voltage scaling factor is not currently implemented properly in maxwell reader
            
            lsb = np.float32(h5file['/data_store/data0000/settings/lsb'][0])
            gain = np.float32(h5file['/data_store/data0000/settings/gain'][0])
            hpf = np.float32(h5file['/data_store/data0000/settings/hpf'][0])
            has_sig = 'sig' in h5file.keys()
            table = 'sig' if has_sig else '/data_store/data0000/groups/routed/raw'
            info['shape'] = h5file[table].shape
            start_time = h5file['data_store/data0000/start_time'][0]
            info['start_time'] = datetime.fromtimestamp(start_time / 1e3).strftime('%Y%m%d-%H%M%S')
            if not has_sig:
                info['start_frame'] = h5file['/data_store/data0000/groups/routed/frame_nos'][0]

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
    if not str(filepath).endswith('.raw.h5'):
        filepath = str(filepath) + '.raw.h5'

    import pandas as pd
    # open file
    with open_file(filepath, 'rb') as f:
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
    Loads specified amount of data from one block
    :param filepath: 
        Path to filename.raw.h5 file
    :param channels: list of int
        Channels of interest
    :param start: int
        Starting frame (offset) of the datapoints to use
    :param length: int
        Length of datapoints to take
    :param spikes: bool
        Whether to load spikes or not
    :param dtype: np.dtype
        Data type to load

    :return:
    dataset: nparray
        Dataset of datapoints.
    """
    # if no .raw.h5 extension, add it
    if suffix is not None:
        filepath = str(filepath) + suffix
    elif not str(filepath).endswith('.raw.h5'):
        filepath = str(filepath) + '.raw.h5'

    if channels is not None:
        # Ensure unique channels
        assert len(channels) == len(np.unique(channels)), f"Channels must be unique, but have length {len(channels)} and unique {len(np.unique(channels))} unique channels"
    
    frame_end = start + length 

    # Defaults if not in file
    lsb = 6.294*10**-6
    gain = 512
    sig_offset = 512

    # open file
    with open_file(filepath, 'rb') as file:
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
                import pandas as pd
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

                filtered_spikes_df['channel'] = filtered_spikes_df['channel'].astype('int64')
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
        return (np.array(dataset, dtype=np.float32) - sig_offset)* lsb * gain * 1000
    elif dtype is np.int16:
        return (np.array(dataset, dtype=np.int16))
    else:
        print('Hmm you shouldnt be here, probably use float32 or int')
        return (np.array(dataset, dtype=dtype) - sig_offset)* lsb * gain * 1000


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
    return (np.array(data, dtype=np.float32) - sig_offset)* lsb * gain * 1000

def load_windows_maxwell(filepath, starts, window_sz=2000, channels=None, dtype=np.float32):
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
        
    # Convert starts to integer numpy array to ensure integer indexing
    starts = np.array(starts, dtype=np.int64)
    
    data_chunks = np.zeros((len(starts), len(channels), window_sz))

    for i, start in enumerate(starts):
        try:
            data_chunks[i] = load_data_maxwell(filepath, channels=channels,
                                start=start, length=window_sz, dtype=dtype)
        except Exception as e:
            print('Error loading window', start)
            print(e)
            data_chunks[i] = np.zeros((len(channels), window_sz))
    return data_chunks


def load_stim_log(filepath, adjust=False, suffix='_log.csv'):
    import pandas as pd
    import ast

    if not str(filepath).endswith(suffix):
        if str(filepath).endswith('.csv'):
            print("Warning: filepath does not end with", suffix)
        else:
            filepath = str(filepath) + suffix

    def string_to_list(s):
        return ast.literal_eval(s)

    with open_file(filepath, 'rb') as file:
        stim_log = pd.read_csv(file, sep=',', header=0, converters={'stim_electrodes': string_to_list})

    if adjust:
        # Adjust stim times
        pass

    stim_log['stim_electrodes'] = stim_log['stim_electrodes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    if 'stim_pattern' in stim_log.columns:
        stim_log['stim_pattern'] = stim_log['stim_pattern'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return stim_log

def apply_literal_eval_stim_log(stim_log):
    import ast
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

def set_stim_patterns(stim_log):
    """
    Get the unique stim patterns from the stim log
    """
    # TODO This will make anything with the same stim electrodes, regardless 
    # of the time delays, be the same pattern
    stim_log['stim_pattern'] = stim_log['stim_electrodes'].apply(lambda x: tuple(x) if isinstance(x, list) else x)


    return stim_log#, stim_counts




def adjust_stim_times(data, stim_df, window_size_ms = 200, fs=20000, ch_choice = 0
                      , stim_offset_ms = 10, plot_check= False):
    '''
    Adjusts the stim times to account for jitter.
    This looks within a window surrounding each stim time and finds the peak in the data.
    The stim time is then set to the time of the peak.
    
    :param data: data to adjust stim times
    :param stim_df: dataframe of stim times
    :param window_size_ms: window size to look for peak in ms
    :param fs: sampling frequency
    :param ch_choice: channel to look for peak in
    :param stim_offset: offset to add to stim times
    :return: stim_df with adjusted stim times
    '''
    fs_ms = fs/1000
    stim_df2 = stim_df.copy()
    for i, stim in stim_df.iterrows():
        stim_time_frame = int(stim_df['time'][i] * fs) + stim_offset_ms*fs_ms
        #print('Old time', stim_df['time'][i])
        start = int(max(0, stim_time_frame - window_size_ms*fs_ms/2))
        # End should be half window after stim or end of data
        end = int(min(data.shape[1], stim_time_frame + window_size_ms*fs_ms/2))
        #print('Start: ', start, 'end: ', end)
        if end - start < 0:
            print('Error: start' , start, 'end', end)
            print('Stim time: ', stim_time_frame, 'has no data')
            print("Returning dataframe (you probably didn't load the full dataset)")
            stim_df2.fillna(np.inf, inplace=True)
            return stim_df2

        peak = np.argmin(data[ch_choice,start:end]) + start
        stim_df2.loc[i, 'time_mod'] = peak / fs
        #print('New time: ', stim_df2.loc[i, 'time'])

        # plot to check
        if plot_check:
            import matplotlib.pyplot as plt
            plt.plot(data[ch_choice,start:end].T)
            plt.scatter(peak-start, data[ch_choice,peak])
            stim_time = int(fs_ms*window_size_ms/2)
            #plt.scatter(stim_time, data[ch_choice,stim_time])
    stim_df2.fillna(np.inf, inplace=True)

        
    return stim_df2


def adjust_stim_times2(file_path, stim_log = None, stim_log_path = None, window_size_ms = 200, fs=20000, ch_choice = None,
                      stim_offset_ms = 10, plot_check= False, tag=None, save_adjustments = True,
                      force_adjustment = False, verbose=False):
    '''
    Adjusts the stim times to account for jitter.
    This looks within a window surrounding each stim time and finds the peak in the data.
    The stim time is then set to the time of the peak.
    
    :param data: data to adjust stim times
    :param stim_df: dataframe of stim times
    :param window_size_ms: window size to look for peak in ms
    :param fs: sampling frequency
    :param ch_choice: channel to look for peak in
    :param stim_offset: offset to add to stim times
    
    :return: stim_df with adjusted stim times
    '''
    if stim_log_path is None:
        stim_log_path = file_path

    suffix = '_log.csv'

    # get mapping
    mapping = load_mapping_maxwell(file_path)
    
    def electrode_to_channel(electrode):
        return mapping[mapping['electrode'] == electrode]['channel'].values[0]
    
    if stim_log is None:
        stim_log = load_stim_log(stim_log_path, suffix=suffix)
    else:
        # Verify that stim_electrodes is not a string
        if isinstance(stim_log['stim_electrodes'][0], str):
            stim_log = apply_literal_eval_stim_log(stim_log)
    

    # Check if time_mod already exists
    if 'time_mod' in stim_log.columns and not force_adjustment:
        # Return if already adjusted
        if verbose:
            print('Stim times already adjusted')
        return stim_log

    info = load_info_maxwell(file_path)
    end_frame = info['shape'][1]

    if tag is not None:
        stim_log = stim_log[stim_log['tag'].str.lower() == tag.lower()]  # so that "Causal" or "causal" in log file works

    fs_ms = fs/1000
    stim_df = stim_log.copy()
    for i, stim in stim_log.iterrows():
        cur_stim_electrodes = stim['stim_electrodes'][0]
        if type(cur_stim_electrodes) == list:
            cur_stim_electrodes = cur_stim_electrodes[0]
        ch_choice = electrode_to_channel(cur_stim_electrodes)
        
        stim_time_frame = int(stim_log['time'][i] * fs) + stim_offset_ms*fs_ms
        #print('Old time', stim_df['time'][i])
        start = int(max(0, stim_time_frame - window_size_ms*fs_ms/2))
        # End should be half window after stim or end of data
        end = int(min(end_frame, stim_time_frame + window_size_ms*fs_ms/2))
        #print('Start: ', start, 'end: ', end)
        if end - start < 0:
            print('Error: start' , start, 'end', end)
            print('Stim time: ', stim_time_frame, 'has no data')
            print("Returning dataframe (you probably didn't load the full dataset)")
            stim_df.fillna(np.inf, inplace=True)
            return stim_df

        data_i = load_data_maxwell(file_path, start=start, length=end-start)
        # print(data_i.shape)

        peak = np.argmin(data_i[ch_choice,:]) + start
        stim_df.loc[i, 'time_mod'] = peak / fs
        #print('New time: ', stim_df.loc[i, 'time'])

        # plot to check
        if plot_check:
            import matplotlib.pyplot as plt
            plt.plot(data[ch_choice,start:end].T)
            plt.scatter(peak-start, data[ch_choice,peak])
            stim_time = int(fs_ms*window_size_ms/2)
            #plt.scatter(stim_time, data[ch_choice,stim_time])
    stim_df.fillna(np.inf, inplace=True)

    # Set stim patterns
    stim_df = set_stim_patterns(stim_df)

    if save_adjustments:
        if verbose:
            print('Saving stim log to ', str(stim_log_path) + suffix)
        stim_df.to_csv(str(stim_log_path) + suffix, index=False)
    return stim_df



def adjust_stim_times3(file_path, stim_log = None, stim_log_path = None, window_size_ms = 200, fs=20000, ch_choice = None,
                      stim_offset_ms = 10, plot_check= False, tag=None, save_adjustments = True,
                      force_adjustment = False, verbose=False):
    '''
    Adjusts the stim times to account for jitter.
    This looks within a window surrounding each stim time and finds the peak in the data.
    The stim time is then set to the time of the peak.
    
    :param data: data to adjust stim times
    :param stim_df: dataframe of stim times
    :param window_size_ms: window size to look for peak in ms
    :param fs: sampling frequency
    :param ch_choice: channel to look for peak in
    :param stim_offset: offset to add to stim times
    
    :return: stim_df with adjusted stim times
    '''
    if stim_log_path is None:
        stim_log_path = file_path

    suffix = '_log.csv'

    # get mapping
    mapping = load_mapping_maxwell(file_path)
    
    def electrode_to_channel(electrode):
        return mapping[mapping['electrode'] == electrode]['channel'].values[0]
    
    if stim_log is None:
        stim_log = load_stim_log(stim_log_path, suffix=suffix)
    else:
        # Verify that stim_electrodes is not a string
        if isinstance(stim_log['stim_electrodes'][0], str):
            stim_log = apply_literal_eval_stim_log(stim_log)
    

    # Check if time_mod already exists
    if 'time_mod' in stim_log.columns and not force_adjustment:
        # Return if already adjusted
        if verbose:
            print('Stim times already adjusted')
        return stim_log

    info = load_info_maxwell(file_path)
    end_frame = info['shape'][1]

    if tag is not None:
        stim_log = stim_log[stim_log['tag'].str.lower() == tag.lower()]  # so that "Causal" or "causal" in log file works

    fs_ms = fs/1000
    stim_df = stim_log.copy()
    for i, stim in stim_log.iterrows():
        cur_stim_electrodes = stim['stim_electrodes'][0]
        if type(cur_stim_electrodes) == list:
            cur_stim_electrodes = cur_stim_electrodes[0]
        ch_choice = electrode_to_channel(cur_stim_electrodes)
        
        stim_time_frame = int(stim_log['time'][i] * fs) + stim_offset_ms*fs_ms
        #print('Old time', stim_df['time'][i])
        start = int(max(0, stim_time_frame - window_size_ms*fs_ms/2))
        # End should be half window after stim or end of data
        end = int(min(end_frame, stim_time_frame + window_size_ms*fs_ms/2))
        #print('Start: ', start, 'end: ', end)
        if end - start < 0:
            print('Error: start' , start, 'end', end)
            print('Stim time: ', stim_time_frame, 'has no data')
            print("Returning dataframe (you probably didn't load the full dataset)")
            stim_df.fillna(np.inf, inplace=True)
            return stim_df

        data_i = load_data_maxwell(file_path, start=start, length=end-start)
        # print(data_i.shape)

        peak = np.argmin(data_i[ch_choice,:]) + start
        stim_df.loc[i, 'time_mod'] = peak / fs
        #print('New time: ', stim_df.loc[i, 'time'])

        # plot to check
        if plot_check:
            import matplotlib.pyplot as plt
            plt.plot(data[ch_choice,start:end].T)
            plt.scatter(peak-start, data[ch_choice,peak])
            stim_time = int(fs_ms*window_size_ms/2)
            #plt.scatter(stim_time, data[ch_choice,stim_time])
    stim_df.fillna(np.inf, inplace=True)

    # Set stim patterns
    stim_df = set_stim_patterns(stim_df)

    if save_adjustments:
        if verbose:
            print('Saving stim log to ', str(stim_log_path) + suffix)
        stim_df.to_csv(str(stim_log_path) + suffix, index=False)
    return stim_df


if __name__ == '__main__':
    print('Doing nothing')
