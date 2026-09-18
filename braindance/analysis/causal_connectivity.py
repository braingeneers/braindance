"""
This file contains causal connectivity analysis functions which are used to analyze
stimulus responses. We assume that you want to "wrap" stimulus responses around, creating 
windows of data. The main goal is to characterize the response distribution which
happens on reacting electrodes/neurons.

We yield 3 connectivity matrices along with the cleaned data and event times.
1. First order connectivity matrix
    - This matrix shows the connectivity between the stimulus and the short-latency
        reacting electrodes. (usually < 10ms)
2. Multi-order connectivity matrix
    - This matrix shows the connectivity between the stimulus and the long-latency
        reacting electrodes. (usually 10-200ms)
3. Total connectivity matrix
    - This matrix shows the connectivity between the stimulus and all reacting electrodes.

"""

# Deprecated: evaluate SpikeLab or design a replacement for stimulation-timed
# response analysis before introducing new consumers. See docs/data-api-migration-review.md.
import warnings
from functools import wraps


def _deprecated(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        warnings.warn(
            "braindance.analysis.causal_connectivity is deprecated. "
            "Its replacement is pending evaluation of SpikeLab and stimulation-timed analysis.",
            FutureWarning,
            stacklevel=2,
        )
        return function(*args, **kwargs)
    return wrapped


import numpy as np

from braindance.analysis import data_loader, mapping
from braindance.core.artifact_removal import ArtifactRemoval, cubic_fit5

try:
    from braindance.core.spikesorter.rt_sort import RTSort, detect_sequences
except:
    print("RTSort not available")
    RTSort = None
    detect_sequences = None


import argparse
import os
import pathlib
import multiprocessing

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from scipy.signal import find_peaks
from numba import njit

params = {
    'window_ms' : 200,
    "ms_before_stim": None,
    'verbose' : True
}

@njit
def _clean_stim_response(data):
    """
    Processes response of one stimulus
    Removes artifacts, thresholds, and returns the processed data
    
    Parameters:
    data : np.ndarray (time_window)
        Data to process

    Returns:
    np.ndarray (time_window)
        Processed data
    list
        Spike times
    """
    # import IPython
    # IPython.embed()
    # Remove artifacts
    data_i,_ = cubic_fit5(data, 60, 60)
    # if np.min(data_i) < -200 or np.max(data_i) > 200:
    #     data_i[:] = 0
    return data_i

from numba import prange
@njit(parallel=True)
def _clean_stim_responses(data):
    """
    Calls _clean_stim_response repeatedly through multiprocessing
    """
    
    # pool = multiprocessing.Pool()
    # x = np.array(pool.map(_clean_stim_response, [d for d in data]))
    # x = np.array([_clean_stim_response(d) for d in data])
    # return x
    arr = np.zeros(data.shape)
    for i in prange(data.shape[0]):
        arr[i] = _clean_stim_response(data[i])
    return arr

    # return _clean_stim_response(data)

from numba import prange
@njit(parallel=True)
def _clean_stim_responses_all(data):
    """
    Calls _clean_stim_response repeatedly through multiprocessing.
    Does this for axis except for the last one.
    """
    
    shape = data.shape
    
    # flatten
    data = data.reshape(-1, shape[-1])
    arr = np.zeros(data.shape)
    for i in prange(data.shape[0]):
        arr[i] = _clean_stim_response(data[i])
    return arr.reshape(shape)
    # return data.reshape(shape)


@_deprecated
def clean_stim_response(data):
    """Deprecated compatibility entrypoint for the legacy cleaning kernel."""
    return _clean_stim_response(data)


@_deprecated
def clean_stim_responses(data):
    """Deprecated compatibility entrypoint for the legacy cleaning kernel."""
    return _clean_stim_responses(data)


@_deprecated
def clean_stim_responses_all(data):
    """Deprecated compatibility entrypoint for the legacy cleaning kernel."""
    return _clean_stim_responses_all(data)


@_deprecated
def causal_reactions(data_path, react_channels, wind_ms=200,
                        stim_log = None, save_dir=None, tag='causal',
                        save_clean_data = False, stim_patterns = None,
                        remove_start_frames = 60,
                        ms_before_stim=None,
                        verbose = False):
    """
    For a file, stim_log, stim indices, and react indices:
    1. Clean data
    # 2. Threshold data
    # 3. Calculate separate matrices
    # 4. Save matrices

    Parameters:
    data_path : str
        Path to the data file
    react_inds : list
        List of indices of the reacting channels. These should be the same ones as when
        indexing the loaded data.
    stim_log : pd.DataFrame
        Stimulus log. If not provided, it will be loaded from the data_path.
    save_dir : str
        Path to save the results. If not provided, it will be saved in the same directory
        as the data_path.
    tag : str
        Tag to use for reading the stimulus
    save_clean_data : bool
        Whether to save the clean data
    stim_patterns : list
        List of stimulus patterns to use. If not provided, it will be inferred from the stim_log.
    ms_before_stim : None or int
        If not None, include frames_before_stim frames before stimulus in clean_data, so the data window would be [frames before stim] concatenated with [frames after stim]
        Needed for RT-Sort's noise estimator
    """
    import psutil
    
    # Create save directory if it doesn't exist
    if save_dir is not None:
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    info = {}

    if ms_before_stim is not None:
        frames_before_stim = int(ms_before_stim*20)
        wind_ms += ms_before_stim

    wind_frames = int(wind_ms*20)
    N = 60 # For artifact removal
    fs = 20000

    window_sz = remove_start_frames + wind_frames + N

    # Get stim log
    if stim_log is None:
        stim_log = data_loader.load_stim_log(data_path)
        if verbose:
            print(f"Stimulus log has {len(stim_log)} entries, adjusting times...")
        # Adjust and use tag
        stim_log = data_loader.adjust_stim_times2(data_path,stim_log = stim_log, stim_offset_ms = 10, 
                                                            tag=tag, verbose=verbose, force_adjustment=False)

    if stim_patterns is None:
        # Get unique patterns in the order they first appear in the DataFrame
        stim_patterns = stim_log['stim_pattern'].unique().tolist()
        # Calculate counts separately while preserving order
        pattern_counts = {pattern: len(stim_log[stim_log['stim_pattern']==pattern])
                         for pattern in stim_patterns}
    else:
        pattern_counts = {pattern: len(stim_log[stim_log['stim_pattern']==pattern])
                         for pattern in stim_patterns}

    if verbose:
        print(f"Stim patterns: \n\t{stim_patterns}")

    stim_times_dict = {s: (stim_log[stim_log['stim_pattern']==s]['time_mod'].to_numpy()*fs).astype(int)
                        for s in stim_patterns}
    
    # If a stim time occurs within ms_before_stim of the start of the recording, then skip it 
    if ms_before_stim is not None:  
        for s, times in stim_times_dict.items():
            times = times - frames_before_stim  # This is so that (times + frames_before_stim) = the actual stim time
            times = times[times >= 0]  
            stim_times_dict[s] = times

    split_stim_patterns = False
    if save_clean_data:        
        # Data shape: (stim_reps, stim electrodes, react electrodes, time window)
        
        # clean_data_size = clean_data.nbytes
        # Calculate clean data size without making the array
        clean_data_size = max(pattern_counts.values())*len(stim_patterns)*len(react_channels)*wind_frames*8 # 8 bytes per float64
        virtual_memory = psutil.virtual_memory()
        if verbose:
            print(f"Clean data predicted size: {clean_data_size/1e9} GB/ {virtual_memory.available/1e9} GB")
        # Overestimate the memory size a bit
        if clean_data_size*1.3 > virtual_memory.available:
            if verbose:
                print("Data size (after 1.3x overestimation) is too large, splitting the data")
            split_stim_patterns = True
            clean_data_shape = (max(pattern_counts.values()), 1, len(react_channels), wind_frames)
            clean_data = np.zeros(clean_data_shape)  # 1 is so clean_data is same shape regardless of whether splitting
        else:
            if verbose:
                print("Data size is small enough to load all at once")
            split_stim_patterns = False
            clean_data = np.zeros((max(pattern_counts.values()), len(stim_patterns), len(react_channels), wind_frames)) 
        if verbose:
            if split_stim_patterns:
                print(f"For one stim pattern, clean data size is {clean_data.nbytes/1e9} GB")
            else:
                print(f"Clean data size is {clean_data.nbytes/1e9} GB")
            print("="*50)
        clean_data_paths = []

    if verbose:
        print(f"Running causal analysis on {data_path}")
        print('='*20)
        
    # Iterate over all stimulus electrodes
    
    for stim_pattern_ind, stim_pattern in tqdm(enumerate(stim_patterns), leave=False, total=len(stim_patterns), desc="Processing:"):
        if save_clean_data:
            if split_stim_patterns:  # Skip saving if already saved (useful if script is interrupted before finishing and needs to be reran)
                # remove parentheses, change commas to hyphens, and remove spaces
                stim_pattern_str = str(stim_pattern).replace('(','').replace(')','').replace(',','-').replace(' ','')
                # Save the data
                cur_save_path = f"{save_dir}/clean_data_{stim_pattern_str}.npy"
                if os.path.isfile(cur_save_path):
                    clean_data_paths.append(cur_save_path)
                    continue
                # Reset clean_data (frees `windows` memory)
                del clean_data
                clean_data = np.zeros(clean_data_shape)
            else:
                cur_save_path = f"{save_dir}/clean_data.npy"
                if os.path.isfile(cur_save_path):
                    clean_data_paths = [cur_save_path]
                    continue
        
        if verbose:
            tqdm.write(f"Processing stim electrode {stim_pattern} ({stim_pattern_ind+1}/{len(stim_patterns)})")

        windows = data_loader.load_windows_maxwell(data_path, stim_times_dict[stim_pattern],
                                                    window_sz=window_sz, channels = react_channels)
         
        current_pattern_count = len(stim_times_dict[stim_pattern]) # pattern_counts[stim_pattern]
        # clean_data[:current_pattern_count,stim_electrode_ind,:,:] = _clean_stim_responses_all(windows)[...,remove_start_frames:-N]
        if ms_before_stim is None:
            windows = _clean_stim_responses_all(windows)[...,remove_start_frames:-N]
        else:            
            after_stim_size = (windows.shape[-1] - frames_before_stim) - remove_start_frames - N
            windows[..., frames_before_stim:frames_before_stim+after_stim_size] = _clean_stim_responses_all(np.ascontiguousarray(windows[..., frames_before_stim:]))[..., remove_start_frames:-N]
            windows = windows[..., :frames_before_stim+after_stim_size] 
        
        if save_clean_data:
            if split_stim_patterns:
                clean_data[:current_pattern_count,0,:,:] = windows
                np.save(cur_save_path, clean_data)
                tqdm.write(f"Saved the data to {cur_save_path}")
                clean_data_paths.append(cur_save_path)
            else:
                clean_data[:current_pattern_count,stim_pattern_ind,:,:] = windows

        del windows

    if save_clean_data:
        if not split_stim_patterns and not os.path.isfile(cur_save_path):
            if verbose:
                print("Saving the data...")
            # Save the data
            np.save(cur_save_path, clean_data)
            clean_data_paths = [cur_save_path]
    else:
        clean_data = None

    if verbose:
        print("Finished cleaning the data")
        # Try it all at once


        # Iterate over all reacting electrodes
        # for react_ind, react_channel in enumerate(react_inds):
        #     if verbose:
        #         print(f"\tProcessing react electrode {react_channel} ({react_ind+1}/{len(react_inds)})")

        #     # Window shape: (stim_reps, react electrodes, time_window)
        #     data = windows[:,react_ind,:]

        #     # Clean data
        #     clean_data_reps = _clean_stim_responses(data)

        #     if save_clean_data:
        #         clean_data[:,stim_electrode_ind,react_ind,:] = clean_data_reps[:,remove_start_frames:-N]

    info['split_stim_patterns'] = split_stim_patterns
    info['stim_patterns'] = stim_patterns
    info['react_inds'] = react_channels
    info['wind_ms'] = wind_ms
    info['remove_start_frames'] = remove_start_frames
    info['N'] = N
    info['pattern_counts'] = pattern_counts
    info['stim_times_dict'] = stim_times_dict
    info['stim_log'] = stim_log
    info['data_path'] = data_path
    if save_clean_data:
        info['clean_data_paths'] = clean_data_paths

    if save_dir is not None:
        np.save(f"{save_dir}/info.npy", info)

    return clean_data, info


@_deprecated
def get_spikes(data, sorter="thresh", sorter_params={'thresh':-9}, save_dir=None,
               spikes_offset_ms=0):
    """
    Takes data of shape (reps, ..., time_window), and returns a sparse listing of 
    spike times, where it is a np object of shape (...) with each element being
    a list of spike times.

    Parameters:
    data : np.ndarray (reps, ..., time_window)
        Data to threshold
    sorter : str
        The sorter to use
        -----------------------------------------------------------------------------------------------
        > 'thresh' : Thresholds the data based on the given threshold
        >>>> sorter_params : dict
        >>>>     thresh : int, float
        >>>>         Threshold to use for each data point. If a single value is given, it is used for all data points.
        -----------------------------------------------------------------------------------------------
        > 'std_thresh' : Thresholds the data based on the standard deviation of the data
        >>>> sorter_params : dict
        >>>>     std : int, float, np.ndarray
        >>>>         Thresholds to use for each data point scaled by the standard deviation.
                     If a single value is given, it is used for all data points.
        -----------------------------------------------------------------------------------------------
        > 'RT-Sort' : Sorts the data based on the given RT-Sort object
        >>>> sorter_params : dict
        >>>>     TODO: Add documentation. data should be list of str/Path to paths
    save_dir : None or path to directory (str or Path) where spike_data and amps will be saved 
    spike_offset : int, Saved/returned spike times = time detected in data + spike_offset

    Returns:
    np.ndarray (reps, ..., time_window)
        Spike times and amplitudes
    """    
    if sorter == "std_thresh":

        original_shape = data.shape
        stds = data.std(axis=(0,-1)) # std should be averaged over reps

        # Make stds of shape original_shape[:-1]
        stds = np.broadcast_to(stds, original_shape[:-1])

        # Flatten
        data = data.reshape(-1, data.shape[-1])
        stds = stds.flatten()
        
        
        spikes, amps = threshold_data(data, -stds*sorter_params['std'])
        spikes = spikes.reshape(original_shape[:-1])
        amps = amps.reshape(original_shape[:-1])
        # return spikes, amps 
    
    elif sorter == "thresh":
        original_shape = data.shape
        # Flatten
        data = data.reshape(-1, data.shape[-1])
        
        
        spikes, amps = threshold_data(data, sorter_params['thresh'])
        spikes = spikes.reshape(original_shape[:-1])
        amps = amps.reshape(original_shape[:-1])
        # return spikes, amps 
    elif sorter == "RT-Sort":     
        # TODO: If no spikes are detected in any repetition for a stimulus pattern, the np.array may have length 0 in a dimension, which may cause problems. Fix by initializing array as np.empty and setting array[i] = np.array(data)
                   
        rt_sort = sorter_params['rt_sort']  # type: RTSort
        include_spikes_ms_before_stim = sorter_params['include_spikes_ms_before_stim']
        
        # spikes = []  # Is filled with shape (num_stim_patterns, num_stim_reps, num_seqs), but will be transposed to (num_stim_reps, num_stim_patterns, num_seqs)
        for path in tqdm(data):  
            tqdm.write(f"Processing {path}")
            if "clean_data_" in path:  # Multiple clean data paths
                spikes_path = "spikes_".join(path.rsplit("clean_data_", 1))  # Change last occurrence of clean_data_ (the file name and not folder names) to spikes_ 
            else:  # Single "clean_data.npy"
                spikes_path = path.replace("clean_data", "spikes")
            spikes = []  # Is filled with shape (num_stim_patterns, num_stim_reps, num_seqs), but will be transposed to (num_stim_reps, num_stim_patterns, num_seqs)
            if os.path.isfile(spikes_path):  # Data already saved
                continue
            clean_data = np.load(path, mmap_mode="r")  # (num_stim_reps, num_stim_patterns, num_react_elecs, num_frames)
                        
            if len(clean_data.shape) != 4:
                raise NotImplementedError("Saved cleaned data must have shape (num_stim_reps, num_stim_patterns, num_react_elecs, num_frames)")
            
            for pattern_idx in range(clean_data.shape[1]):
                pattern_windows = clean_data[:, pattern_idx]   # (num_stim_reps, num_react_elecs, num_frames)
                reps_spikes = []  # Spikes in repetition windows of a stim pattern, will have shape (num_stim_reps, num_seqs)
                for window in pattern_windows:  # (num_react_elecs, num_frames)                    
                    all_detections = rt_sort.sort_offline(window)
                    for i, detections in enumerate(all_detections):
                        # 1000 pre frames, 50 pre stim frames -- stim -- 4000 after stim frames
                        detections -= rt_sort.total_num_pre_median_frames / rt_sort.samp_freq + include_spikes_ms_before_stim  # Make spikes relative to (stim+remove_start_frames)
                        # Handle spikes before stim
                        selected_detections = detections[
                            (detections >= -include_spikes_ms_before_stim) &   # In the input chunk when rt_sort.total_num_pre_median_frames is reached, spikes in that input chunk can be sorted. Don't include these 
                            (detections < -rt_sort.full_end_buffer/rt_sort.samp_freq)  # Don't include ending full_end_buffer milliseconds because this will result in using data that occurred after stim 
                        ] 
                        # Handle spikes after stim
                        selected_detections = np.concatenate((
                            selected_detections,
                            (detections[detections >= rt_sort.full_front_buffer/rt_sort.samp_freq]) + spikes_offset_ms  # Spikes detected before this are the result of concatenating data before stim, so don't include them
                        ))
                        all_detections[i] = selected_detections
                    reps_spikes.append(all_detections)
                spikes.append(reps_spikes)
                # spikes.append(np.array(reps_spikes, dtype=object))
            np.save(spikes_path, np.array(spikes, dtype=object).transpose((1, 0, 2)))
                
        # spikes = np.array(spikes, dtype=object).transpose((1, 0, 2))
        amps = None
        save_dir = None  # So spike data is not saved later on because it is already saved
        # return spikes, None
       
    if save_dir is not None:
        np.save(f"{save_dir}/spikes.npy", spikes)
        if amps is not None:  # RT-Sort does not save amps for now
            np.save(f"{save_dir}/spike_amps.npy", amps)

    return spikes, amps


# @njit(parallel=True)
@_deprecated
def threshold_data(data, thresholds, fs_ms=20):
    """
    Takes data of shape (n, time_window), and thresholds it based on the given thresholds.

    Parameters:
    data : np.ndarray (n, time_window)
        Data to threshold
    thresholds : int, float, np.ndarray
        Thresholds to use for each data point. If a single value is given, it is used for all data points.
        If an array is given, it should have the same shape as the first dimension of data.

    Returns:
    np.ndarray (n, time_window)
        Thresholded data
    """
    if isinstance(thresholds, int) or isinstance(thresholds, float):
        thresholds = np.ones(data.shape[0])*thresholds
    elif isinstance(thresholds, np.ndarray):
        if thresholds.shape[0] != data.shape[0]:
            raise ValueError(f"std should have the same shape as the first dimension of data. {thresholds.shape} != {data.shape[0]}")
    else:
        raise ValueError(f"std should be an int, float, or np.ndarray. {type(thresholds)}")
    
    spikes = np.zeros(data.shape[:-1], dtype=object)
    amps = np.zeros(data.shape[:-1], dtype=object)
    for i in prange(data.shape[0]):
        # scipy find_peaks
        # data[i] = data[i] > std[i]*data[i].std()
        #peaks, _ = find_peaks(-data_i[:-self.art_rem_N], height=cur_std*5, distance=3*fs_ms)
        peaks, _ = find_peaks(-data[i], height=-thresholds[i], distance=fs_ms)
        spikes[i] = peaks
        amps[i] = data[i,peaks]
    return spikes, amps


@_deprecated
def causal_connectivity_metrics(spike_data, first_order_ms = 10, multi_order_ms = (10,200),
                                burst_mad_thresh = 3, save_dir=None,
                                count_bursts=True):
    """
    
    Parameters:
    spike_data : np.ndarray (reps, stim_electrodes, react_electrodes, ) type=obj
        Array of lists of spike times on the react electrode from the 
        stimulations on the stim electrode.
    first_order_ms : int
        Time window to consider as first order

    Returns:
    np.ndarray (stim_electrodes, react_electrodes)
        First order connectivity matrix, percentage evoked FIRST spikes within first_order_ms

    np.ndarray (stim_electrodes, react_electrodes)
        Multi order connectivity matrix, mean number evoked spikes within multi_order_ms:
        (multi_order_ms[0] and multi_order_ms[1])
    """
    first_order_frame = int(first_order_ms*20)
    multi_order_frame = (int(multi_order_ms[0]*20), int(multi_order_ms[1]*20))
    first_order_connectivity = np.zeros((spike_data.shape[1], spike_data.shape[2]))
    multi_order_connectivity = np.zeros((spike_data.shape[1], spike_data.shape[2]))

    # Find bursts
    # stim neurons x stim reps
    stim_response_counts = np.zeros((spike_data.shape[1], spike_data.shape[0])) 
    for stim_ind in range(spike_data.shape[1]):
        for rep in range(spike_data.shape[0]):
            stim_response_counts[stim_ind, rep] = np.sum([len(spike_data[rep,stim_ind,react_ind]) for react_ind in range(spike_data.shape[2])])

    # Calculate statistics
    stim_response_med = np.median(stim_response_counts)
    stim_response_mad = np.median(np.abs(stim_response_counts - stim_response_med))
    # Mean Absolute Deviation
    threshold = stim_response_med + burst_mad_thresh * stim_response_mad

    # Calculate the percentage of burst responses
    burst_percent = (stim_response_counts > threshold).sum(axis=1) / spike_data.shape[0]
    # Keep track of burst indexes (stim ind, stim rep)
    # In order to remove bursts from multi_order_connectivity
    burst_inds = np.where(stim_response_counts > threshold)
    # Should be a tuple of (stim_inds, rep_inds)
    # Convert to list of tuples
    burst_inds = list(zip(*burst_inds))

    for stim_ind in range(spike_data.shape[1]):
        
        for react_ind in range(spike_data.shape[2]):
            for rep in range(spike_data.shape[0]):
                cur_spike_data = spike_data[rep,stim_ind,react_ind]
                # if stim_ind != react_ind:
                # count += len(cur_spike_data)
                if len(cur_spike_data) == 0:
                    continue
                # Ensure first_order can only have one spike per rep
                first_order_connectivity[stim_ind, react_ind] += np.sum(cur_spike_data < first_order_frame) > 0
                
                # Don't count bursts
                if not count_bursts and (stim_ind, rep) in burst_inds:
                    continue
                multi_order_connectivity[stim_ind, react_ind] += np.sum((cur_spike_data > multi_order_frame[0]) & 
                                                                        (cur_spike_data < multi_order_frame[1]))

    first_order_connectivity /= spike_data.shape[0] # Normalize by number of reps, probability of first spike
    # multi order needs to take into account the removed bursts
    if not count_bursts:
        # multi order needs to take into account the removed bursts
        multi_order_connectivity /= (spike_data.shape[0] - burst_percent*spike_data.shape[0])[:, None]
    else:
        multi_order_connectivity /= spike_data.shape[0]

    if save_dir is not None:
        print(f"Saving the connectivity matrices to {save_dir}")
        np.save(f"{save_dir}/first_order_connectivity.npy", first_order_connectivity)
        np.save(f"{save_dir}/multi_order_connectivity.npy", multi_order_connectivity)
        np.save(f"{save_dir}/burst_percent.npy", burst_percent)
    
    return first_order_connectivity, multi_order_connectivity, burst_percent



# Test with:
# python causal_connectivity.py /media/danser-lab/hippo/cartpole/24-03-25_plasticity/p001237/RL_pharma/RL_pharma_causal.raw.h5
# python causal_connectivity.py /media/danser-lab/hippo/cartpole/24-03-25_plasticity/p001237/RL_pharma/RL_pharma_causal.raw.h5 -r json --stim json

# python causal_connectivity.py /data/MEAprojects/primary_mouse/e_stim_seq/231122/22217/rec/causal_freq_22217_5Hz_231122.raw.h5
sorter="std_thresh"
sorter_params={'std': 3.5}
@_deprecated
def main(data_path=None,
         save_dir='none', tag='causal', react_inds='all', stim='infer',
         remove_start_frames=40, count_bursts=False):
    """
    Params:
        data_path
            If None, use command line args for parameters
            Else, use arguments to this function
    """
    
    import time

    if data_path is None:
        # Create an argument parser
        parser = argparse.ArgumentParser(description='Causal connectivity analysis')
        parser.add_argument('data_path', type=str, help='Path to the .raw.h5 data')
        parser.add_argument('--save_dir', type=str, default='none', help='Path to save the results')
        parser.add_argument('--tag','-t', type=str, default='causal',
                        help='tag for saving files')
        parser.add_argument('--react','-r', type=str, default='all',
                        help='Reacting electrodes. Either "all" or path to json file with "stim_electrodes" key, \
                        or list of electrodes.')
        parser.add_argument('--stim', type=str, default='infer',
                        help='Stimulus electrodes. Either "infer" or an explicit JSON file path.')
        
        # Parse the arguments
        args = parser.parse_args()
        data_path = args.data_path
        save_dir = args.save_dir
        react_inds = args.react
        tag = args.tag
        stim = args.stim
        
        # Additional arguments (not included in cmd line arguments)
        remove_start_frames=60
        count_bursts=True

    # Check if the data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file '{data_path}' does not exist.")
    elif params['verbose']:
        print(f"Data file\n\t'{data_path}'\n")

    if data_path.endswith('.raw.h5'):
        # Get the recording name
        data_path = data_path.split('.raw.h5')[0]
        if params['verbose']:
            print(f"Recording name:\n\t'{data_path}'\n")

    # If the save directory is not specified, save in the same directory as the data
    # except add a 'causal' folder
    if save_dir == 'none':
        save_dir = os.path.join(os.path.dirname(data_path), 'causal') 

    if params['verbose']:
        print(f"Save directory:\n\t'{save_dir}'\n")

    # Pathlib to create directories
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    
    if react_inds == 'all':
        info = data_loader.load_info_maxwell(data_path)
        react_channels = np.arange(info['shape'][0])
        print(f"Using all electrodes: {len(react_channels)}")

    elif 'json' in react_inds:
        import json
        if react_inds == 'json':
            raise ValueError("Implicit project JSON lookup was removed; supply an explicit JSON file path.")
        else:
            json_params = json.load(open(react_inds, 'r'))

        # mapping = data_loader.load_mapping_maxwell(data_path)
        mapper = mapping.Mapping(data_path)
        react_elecs = json_params['stim_electrodes']

        react_channels = mapper.get_channels(electrodes = react_elecs)

        print(f"Using {len(react_channels)} electrodes from ['stim_electrodes'] in json file:\n\t{react_channels}")
        # Convert from elects to channels

    else:
        try:
            react_channels = [int(e) for e in react_inds.split(',')]
        except:
            raise ValueError(f"Could not parse reacting electrodes: {react_inds}")
        print(f"Using {len(react_channels)} electrodes from command line:\n\t{react_channels}")

    if stim == 'infer':
        stim_patterns = None
        print("Stimulus pattern will be inferred from the log file.")
    elif 'json' in stim:
        import json
        if stim == 'json':
            raise ValueError("Implicit project JSON lookup was removed; supply an explicit JSON file path.")
        else:
            json_params = json.load(open(stim, 'r'))

        # mapping = data_loader.load_mapping_maxwell(data_path)
        mapper = mapping.Mapping(data_path)
        stim_elecs = json_params['stim_electrodes']

        stim_patterns = [(p,) for p in stim_elecs]

        print(f"Using {len(stim_patterns)} electrodes from ['stim_electrodes'] in json file:\n\t{stim_patterns}")

    window_ms = params['window_ms']
    ms_before_stim = None
    # Load RT-Sort if using it
    if sorter == "RT-Sort":
        if isinstance(sorter_params['rt_sort'], RTSort):
            rt_sort = sorter_params['rt_sort']
        else:
            print("Loading RT-Sort...")
            
            sorter_params['rt_sort'] = rt_sort = RTSort.load_from_file(sorter_params['rt_sort'], sorter_params['detection_model'])
        ms_before_stim = round(rt_sort.total_num_pre_median_frames / rt_sort.samp_freq) + sorter_params['include_spikes_ms_before_stim']
        spikes_offset_ms = remove_start_frames / rt_sort.samp_freq
        window_ms += rt_sort.full_end_buffer / rt_sort.samp_freq  # Need to add end buffers, so RT-Sort can detect spikes at end of window
        window_ms -= spikes_offset_ms  # Subtracting spikes_offset_ms to account for windows being extracted after remove_start_frames after stim
        # This window_ms adjustment is so that the latest spike after stim that can be sorted (after stim, NOT after remove_start_frames after stim) is window_ms
    else:
        spikes_offset_ms = 0
    start_time = time.perf_counter()

    print('\n====================================================')
    print("Running causal connectivity analysis...\n\n")
    clean_data, info = causal_reactions(data_path, react_channels, window_ms,  
                        save_dir=save_dir, tag=tag, save_clean_data=True, stim_patterns=stim_patterns,
                        remove_start_frames=remove_start_frames, ms_before_stim=ms_before_stim,
                        verbose=params['verbose'])
    # info = np.load(f"{save_dir}/info.npy", allow_pickle=True).item()  # TODO: Remove this (was used for testing)
    if sorter == "RT-Sort":
        clean_data = info['clean_data_paths']

    end_time = time.perf_counter()
    print(f"Finished in causal connectivity in  {end_time-start_time} seconds")
    print('====================================================\n')

    # Happy face
    print(' '*10 + 'O  O')
    print(' '*10 + '  ~  ')
    print(' '*10 + "\___/")

    # Threshold the data
    start_time = time.perf_counter()
    print('\n====================================================')
    if sorter != "RT-Sort":
        print("Thresholding the data...")
    else:
        print("Spike sorting the data...")
    # spikes, amps = get_spikes(clean_data, sorter="thresh", sorter_params={'thresh': -22})
    # spikes, amps = get_spikes(clean_data, sorter="std_thresh", sorter_params={'std': 3.5})
    
    spikes, amps = get_spikes(clean_data, sorter=sorter, sorter_params=sorter_params, save_dir=save_dir,
                              spikes_offset_ms=spikes_offset_ms)
    # spikes = np.load(f"{save_dir}/spikes.npy", allow_pickle=True)  # TODO: Remove this (was used for testing)
    
    end_time = time.perf_counter()
    if sorter != "RT-Sort":
        print(f"Finished thresholding in {end_time-start_time} seconds")
    else:
        print(f"Finished sorting in {end_time-start_time} seconds")
        return  # Skip causal_connectivity_metrics because output of get_spikes is not implemented with it yet

    # Calculate connectivity metrics
    start_time = time.perf_counter()
    print('\n====================================================')
    print("Calculating connectivity metrics...")
    first_order_connectivity, multi_order_connectivity, burst_percent = causal_connectivity_metrics(spikes, save_dir=save_dir, count_bursts=count_bursts)
    end_time = time.perf_counter()
    print(f"Finished calculating connectivity metrics in  {end_time-start_time} seconds")

    # Test plot
    # import matplotlib.pyplot as plt

    # stim_ind = 6
    # react_ind = 1
    # plt.plot(clean_data[:,stim_ind,react_ind,:].T)
    # for rep in range(spikes.shape[0]):
    #     plt.scatter(spikes[rep,stim_ind,react_ind], amps[rep,stim_ind,react_ind], color='red', marker='x')
    # plt.show()


@_deprecated
def main_rt_sort(recording_path, 
                 recording_window_ms=(2*60*1000, 6*60*1000),
                 include_spikes_ms_before_stim=0,
                 tag="Causal", phases=None,
                 verbose=True):
    """
    Run main() with RT-Sort
    
    Params:
        recoridng
            Paths to Maxwell recording
        delete_inter
            If True, delete intermediate data needed to run RT-Sort
    """
    from pathlib import Path
    from braindance.core.spikedetector.model2 import ModelSpikeSorter

    recording_path = Path(recording_path)
    parent_path = recording_path.parent
    rt_sort_inter_path = parent_path / (recording_path.name.split(".")[0] + "_rt_sort")
    causal_save_dir = rt_sort_inter_path / "causal"
    causal_save_dir.mkdir(exist_ok=True, parents=True)
        
    model = ModelSpikeSorter.load_mea()
    
    if (rt_sort_inter_path / "rt_sort.pickle").exists():
        rt_sort = RTSort.load_from_file(rt_sort_inter_path / "rt_sort.pickle", model)
    else:
        rt_sort = detect_sequences(recording_path, rt_sort_inter_path, model, 
                                   recording_window_ms, verbose=verbose, debug=True)
        
    # Save seq_locs
    if not hasattr(rt_sort, "seq_locs"):  # For backwards compatibility 
        root_elecs = rt_sort.get_seq_root_elecs()
        from spikeinterface.extractors import MaxwellRecordingExtractor
        elec_locs = MaxwellRecordingExtractor(recording_path).get_channel_locations()
        seq_locs = np.array([elec_locs[root] for root in root_elecs])
        assert len(seq_locs) == rt_sort.num_seqs
        np.save(rt_sort_inter_path / "seq_locs.npy", seq_locs)
    else:
        np.save(rt_sort_inter_path / "seq_locs.npy", rt_sort.seq_locs)
    
    global sorter, sorter_params
    sorter = "RT-Sort"
    sorter_params = {"rt_sort": rt_sort, 
                     "detection_model": model,
                     "include_spikes_ms_before_stim": include_spikes_ms_before_stim,
                     "verbose": verbose}
            
    main(data_path=str(recording_path), save_dir=str(causal_save_dir),
         remove_start_frames=40, count_bursts=False, tag=tag)
    
    if phases is not None:
        from braindance.experiments.causal_freq_sort_intrinsic import sort_intrinsic
        sort_intrinsic(recording_path, rt_sort_inter_path / "intrinsic", rt_sort, phases=phases)
    
    # main() manually 
    # info = np.load(f"{causal_save_dir}/info.npy", allow_pickle=True).item()
    # clean_data = info['clean_data_paths']
    # spikes, amps = get_spikes(clean_data, sorter=sorter, sorter_params=sorter_params, save_dir=causal_save_dir)
    # first_order_connectivity, multi_order_connectivity, burst_percent = causal_connectivity_metrics(spikes, save_dir=causal_save_dir)
    
    return rt_sort_inter_path / "seq_locs.npy", rt_sort_inter_path / "rt_sort.pickle", causal_save_dir, rt_sort_inter_path / "intrinsic"
    
@_deprecated
def remove_duplicate_chans_elecs(rec_path):
    import h5py
    rec = h5py.File(rec_path)

    mapping = rec['recordings']['rec0000']['well000']['settings']['mapping']
    repeat_chan = {}
    repeat_elec = {}
    for i, row in enumerate(mapping):
        chan = row['channel']
        if chan in repeat_chan:
            repeat_chan[chan].append(i)
        else:
            repeat_chan[chan] = [i]

        elec = row['electrode']
        if elec in repeat_elec:
            repeat_elec[elec].append(i)
        else:
            repeat_elec[elec] = [i]

    # print(f"Repeated channels:")
    reset_chan_nums = False
    for chan, ind in repeat_chan.items():
        if len(ind) == 1:
            continue
        reset_chan_nums = True

        # print(f"Channel: {chan}")
        # for idx in ind:
        #     print(idx, mapping[idx])
        # print()

    # print(f"\nRepeated electrodes:")
    remove_ind = []
    repeated_elecs = []
    for elec, ind in repeat_elec.items():
        if len(ind) == 1:
            continue

        # print(f"Electrode: {elec}")
        for i, idx in enumerate(ind):
            # print(idx, mapping[idx])
            if i > 0:  # Keep first occurrence of electrode
                remove_ind.append(idx)
                repeated_elecs.append(elec)

    # print(f"Removing {remove_ind}")
    mapping = rec['data_store']['data0000']['settings']['mapping']
    edit = False
    if len(remove_ind) > 0:
        # Remove repeated electrodes
        print(f"Removing repeated electrodes: {repeated_elecs}")
        mapping = np.delete(mapping, remove_ind, axis=0)
        edit = True
    if reset_chan_nums:
        print(f"Resetting channel numbers")
        mapping['channel'] = np.arange(len(mapping['channel']))
        edit = True
    if edit:
        rec.close()
        rec = h5py.File(rec_path, "r+")
        del rec['data_store']['data0000']['settings']['mapping']
        rec['data_store']['data0000']['settings']['mapping'] = mapping

    rec.close()

        
@_deprecated
def main_kosik(recording_paths, 
               recording_window_ms=(2*60*1000, 6*60*1000),
               ind_24h=[], include_spikes_ms_before_stim=0,
               tag='Causal'):
    """
    For Kosik Lab, UCSB, setup:
        1. Download recording and log file from server onto GPU computer
        2. Process data on GPU computer
        3. Upload processed data to server
        4. Delete data from GPU computer
        
    If there are too many stimulation patterns and not enough disk space,
    a solution is to split up the log file into multiple different ones and
    handle, upload, and delete a fraction of all .npy files at a time (they are independent of each other)
    
    Params:
        ind_24h
            Which indices in recording_paths have a 24h recording (recording that happened next day that should be processed with same RT-Sort object)
    """
    from pathlib import Path
    import subprocess
    import shutil
    import traceback
    def download(server_path, local_path=None):
        if local_path is None:
            local_path = Path(server_path).parent
        
        # mkdir
        subprocess.run(['mkdir', '-p', local_path])

        # download
        rsync_cmd = ["rsync", "-av", "--chmod=u+rw,g+rw,o+r", "--ignore-existing", "--progress", 
                    f"maxlim@kosik.cnsi.ucsb.edu:{server_path}", local_path]
        subprocess.run(rsync_cmd)
        
    def upload(local_path, server_path=None):
        if server_path is None:
            server_path = Path(local_path).parent
        # server_path = str(server_path).replace("/data/MEAprojects/human_slice/e_stim_seq", "/data/MEAprojects/human_slice/e_stim_seq/temp")  # If not have write permissions
            
        # mkdir
        subprocess.run(['ssh', 'maxlim@kosik.cnsi.ucsb.edu', 'mkdir', '-p', server_path])
        
        # upload
        rsync_cmd = ["rsync", "-av", "--chmod=u+rw,g+rw,o+r", "--ignore-existing", "--progress", 
                    local_path, f"maxlim@kosik.cnsi.ucsb.edu:{server_path}"]
        subprocess.run(rsync_cmd)
                                             
    for i, rec_path in enumerate(recording_paths):
        rec_path = Path(rec_path)
        print("="*100)
        print(f"STARTING ON RECORDING")
        print(rec_path)
        print("="*100)
        remove = not rec_path.parent.exists()
        try:            
            download(rec_path)
            log_file = rec_path.name.split(".")[0] + "_log.csv"
            log_path = rec_path.parent / log_file
            download(log_path)
            remove_duplicate_chans_elecs(rec_path)
                        
            upload_paths = main_rt_sort(rec_path, recording_window_ms=recording_window_ms,
                                        include_spikes_ms_before_stim=include_spikes_ms_before_stim, tag=tag,
                                        phases=[["pre", 3, 2, 6],
                                                ["seq_stim", 14, 3, 6],
                                                ["post1", 3, 2, 6],
                                                ["intrinsic", 7, 0, 6],
                                                ["post2", 3, 2, 6]],)
            for path in upload_paths:
                upload(path)

            if i in ind_24h:
                rec_path = Path(str(rec_path).replace("seq_stim", "seq_stim_24h"))
                download(rec_path)
                log_file = rec_path.name.split(".")[0] + "_log.csv"
                log_path = rec_path.parent / log_file
                download(log_path)
                remove_duplicate_chans_elecs(rec_path)
                
                inter_path = rec_path.parent / (rec_path.name.split(".")[0] + "_rt_sort")
                inter_path.mkdir(exist_ok=True, parents=True)
                rt_sort_path = upload_paths[1]
                shutil.copy2(rt_sort_path, inter_path)
            
                upload_paths = main_rt_sort(rec_path, recording_window_ms=recording_window_ms, 
                                            include_spikes_ms_before_stim=include_spikes_ms_before_stim, tag=tag,
                                            phases=[["post3", 3, 2, 6]])
                for path in upload_paths:
                    upload(path)
        except Exception as e:
            print("---ERROR---")
            traceback.print_exc()

        if remove:
            shutil.rmtree(rec_path.parent)
        
              
if __name__ == '__main__':  
    pass