import os
from pathlib import Path

import numpy as np
from spikeinterface.extractors import BaseRecording
from tqdm import tqdm

from braindance.core.spikesorter.rt_sort import save_traces, load_recording

class Sort:
    """
    Base class for ___Sort objects
    TODO: Implement this in ThreshSort and RTSort
    """
    
    def __init__(self, num_units, samp_freq):
        self.num_units = num_units
        self.samp_freq = samp_freq
        self.latest_frame = 0
        
    def reset(self):
        self.latest_frame = 0
                
    def sort_offline(self, recording, buffer_size,
                     inter_path=None, recording_window_ms=None,
                     reset=True,
                     verbose=False):
        """
        TODO: Give option to pass in model_traces_path? So when detecting sequences, the model does not need to be rerun on the recording when detecting sequences
        
        Params:
            recording
                Can be a numpy array of shape (n_channels, n_samples), path to scaled_traces.npy, or a recording object
            inter_path
                If :recording: is a recording object, where scaled_traces.npy will be saved
                    If None, scaled_traces.npy will be saved to the current directory and deleted after sorting
            recording_window_ms
                If recording is a recording object, use recording_window_ms to only sort a part of the recording
                    Should be in format of (start_ms, end_ms)
            reset
                Whether to call self.reset()
                    In most cases, self.reset() should be called. 
        """
        if reset:
            self.reset()

        remove_traces = False
        if isinstance(recording, np.ndarray):
            scaled_traces = recording
        else:
            if (isinstance(recording, str) or isinstance(recording, Path)) and str(recording).endswith(".npy"):
                scaled_traces = np.load(recording, mmap_mode="r")
            else:       
                recording = load_recording(recording)         
                if inter_path is None:
                    inter_path = Path.cwd()
                    remove_traces = True

                if recording_window_ms is None:
                    recording_window_ms = (0, recording.get_total_duration() * 1000)
                recording = save_traces(recording, inter_path, *recording_window_ms, verbose=verbose)
                scaled_traces = np.load(recording, mmap_mode="r")

        # all_start_frames = range(0, scaled_traces.shape[1]+1-self.buffer_size, self.buffer_size) 
        all_start_frames = range(0, scaled_traces.shape[1], buffer_size) 
        
        if verbose:
            print("Sorting recording")
            all_start_frames = tqdm(all_start_frames)
            
        all_detections = [[] for _ in range(self.num_units)]
        for start_frame in all_start_frames:
            # .T because self.sort() expects data to be in obs format (which has channels last)
            detections = self.running_sort(scaled_traces[:, start_frame:start_frame+buffer_size].T)
            for seq_idx, spike_time in detections:
                all_detections[int(seq_idx)].append(spike_time)
        
        # # Check if there is data remaining at end of recording that was not included in all_start_frames for-loop
        # Now handled by all_start_frames and internally running_sort
        # if start_frame + self.buffer_size < scaled_traces.shape[1]:
        #     detections = self.running_sort(scaled_traces[:, start_frame+self.buffer_size:].T)            
        #     for seq_idx, spike_time in detections:
        #         all_detections[seq_idx].append(spike_time)
                
        # Can't use list comprehension because then if no sequences detect spikes, the array will have shape (num_seqs, 0)
        array_detections = np.empty(self.num_units, dtype=object) 
        for seq_idx, detections in enumerate(all_detections):
            array_detections[seq_idx] = np.array(detections)

        if remove_traces:
            os.remove(recording)

        return array_detections
        
    def running_sort(self, obs) -> list:
        raise NotImplementedError
    
    def sort_chunk(self, obs) -> list:
        raise NotImplementedError
    
    def to_spikedata(self):
        # Return SpikeData object created from this Sort object
        raise NotImplementedError