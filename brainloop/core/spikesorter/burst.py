import os 
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from spikedata import SpikeData
from tqdm import tqdm

from braindance.core.spikesorter.rt_sort import save_traces
from braindance.core.spikesorter.sort import Sort


class BurstDetector(Sort):
    """
    Detect when a burst of spikes occurs. 
        A wrapper of a Sort class
    """

    def __init__(self, sorter,
                 burst_window_ms=200, burst_rms_thresh=3, min_ms_between_bursts=800):
        """
        Params
            sorter:
                Used for spike sorting data
        """
        
        super().__init__(num_units=1, samp_freq=sorter.samp_freq)

        self.sorter = sorter
        
        self.burst_window_frames = burst_window_ms * self.samp_freq
        self.burst_rms_thresh = burst_rms_thresh
        self.min_burst_spikes = None  # Need to call self.sort_offline(calibrate=True) to determine this
        
        self.min_ms_between_bursts = min_ms_between_bursts
        self.last_burst = -np.inf

        self.min_num_obs = None  # Assuming :obs: is of constant duration for running_sort, the minimum number of obs to start detecting bursts
        self.spike_counts = []  # Number of spikes passed in successive iterations of running_sort
        self.num_window_spikes = 0  # Number of spikes in sliding burst_window_ms window (faster than doing sum(self.spike_counts) every iteration)
                
    def reset(self):
        super().reset()
        self.sorter.reset()
        self.last_burst = -np.inf
        self.min_num_obs = None
        self.spike_counts = []
        self.num_window_spikes = 0
        
    def sort_offline(self, recording, buffer_size, 
                     inter_path=None, recording_window_ms=None, 
                     reset=True, 
                     verbose=False,
                     calibrate=False):     
        """
        Params
            calibrate
                If True, determine self.min_burst_spikes and return burst times
        
        """
           
        if calibrate:
            if not reset:
                print("'calibrate' was set to True, so forcing 'reset' to be True as well")
                reset = True
            self.calibrate = True
            self.window_spike_counts = []
        
        burst_times = [super().sort_offline(recording, buffer_size, inter_path, recording_window_ms, reset, verbose)]
        
        if not calibrate:
            return burst_times
        
        # Determine self.min_burst_spikes
        window_spike_counts = np.array(self.window_spike_counts)
        del self.window_spike_counts
        self.calibrate = False

        latest_mses = window_spike_counts[:, 0]
        window_spike_counts = window_spike_counts[:, 1]
        rms = np.sqrt(np.mean(np.square(window_spike_counts)))
        self.min_burst_spikes = self.burst_rms_thresh * rms
        
        # Get burst times (burst_times from sort_offline is an empty array in this casse)
        burst_times = []
        last_burst = -np.inf
        for ms, count in zip(latest_mses, window_spike_counts):
            if ms - last_burst < self.min_ms_between_bursts:
                continue
            
            if count >= self.min_burst_spikes:
                last_burst = ms
                burst_times.append((0, ms))

        # UNFINISHED: Return duration of burst                
        # burst_times = []
        # i = 0
        # while i < len(window_spike_counts):
        #     num_spikes = sum(window_spike_counts[i:i+self.min_num_obs])
        #     if num_spikes >= self.min_burst_spikes:
        #         time = latest_mses[i]
        #         duration = 0
        #         # Find burst duration
        #         while sum(window_spike_counts[i:i+self.min_num_obs]) > self.min_burst_spikes:
        #             duration += buffer_size / self.samp_freq
        #             i += 1
        #         burst_times.append((0, time, duration))
                
        #         # Wait enough time for next burst
        #         while latest_mses[i] - time < self.min_ms_between_bursts:
        #             i += 1
        #     else:
        #         i += 1
        return [burst_times]  # Extra [] to follow Sort class standard

    def running_sort(self, obs):
        """
        Time of burst is determined by number of milliseconds passed in recording before burst could have been detected
            I.e. time that has passed in the recording when the burst is detected
        
        TODO: This current implemention assumes duration of :obs: passed to running_sort in each iteration remains constant. Allow for variable size :obs:
        """
        
        if self.min_num_obs is None:
            self.min_num_obs = round(self.burst_window_frames / len(obs))
                
        num_spikes = len(self.sorter.running_sort(obs))
        self.num_window_spikes += num_spikes
        self.spike_counts.append(num_spikes)

        num_window_spikes = self.num_window_spikes
        
        if len(self.spike_counts) >= self.min_num_obs:
            self.num_window_spikes -= self.spike_counts.pop(0)  # Update internal counter now before possible return statement
        
        latest_ms = self.sorter.latest_frame / self.samp_freq
        
        if self.calibrate:
            self.window_spike_counts.append((latest_ms, num_window_spikes))
            return []
        
        if latest_ms - self.last_burst < self.min_ms_between_bursts:
            return [] 
        
        if num_window_spikes >= self.min_burst_spikes:
            self.last_burst = latest_ms
            return [[0, latest_ms]]  # Keep format of (seq_idx, spike_time)
        
        return []
    
    
class SmoothBurstDetector:
    """
    Detect when a burst of spikes occurs using a half Gaussian smoothing (adapted from code by Jerry Zhang, UCSB)
        A wrapper of a Sort class
    """

    def __init__(self, sorter,
                 rms_peak_threshold=3, rms_trough_threshold=1, 
                 future_bins=25, ibi_threshold_ms=800,
                 burst_duration_thresh_f=0.1):
        """
        Params
            sorter:
                Used for spike sorting data
        
            Finding bursts in pre-recording to calibrate burst detector:
                rms_peak_threshold
                rms_trough_threshold
                    Population rate must be lower than this value at some point between two bursts to avoid finding multiple peaks in one burst
                burst_duration_thresh_f
                    Burst 'starts' when pop rate crosses 10% of peak, used to find average burst duration 
                    (Decimal, not percent)

            Detecting bursts 
                future_ms:
                    Time (frames) used to predict whether to have a burst peak or not in the following frames of data
                ibi_threshold_ms_ms:
                    Minimum separation between two consecutive predicted bursts (ms) to avoid counting the same burst multiple times
        """
        self.sorter = sorter
        self.samp_freq = sorter.samp_freq
        self.buffer_size = sorter.buffer_size
        
        self.bin_size_ms = round(self.buffer_size / self.samp_freq)  # Bin size used to calculate population firing rate 
        self.rms_peak_threshold = rms_peak_threshold
        self.rms_trough_threshold = rms_trough_threshold
        
        self.burst_duration_thresh_f = burst_duration_thresh_f
        self.remaining_burst_time_ms = None

        self.future_bins = future_bins     
        
        # Create half gaussian (reversed so it can be used in np.dot (convolve operation reverses the kernel))
        x = np.arange(-1, -(future_bins*2)-1, -1)
        # x = np.arange(-future_bins * 2, 0)  # Not reversed version
        norm = 1 / future_bins / np.sqrt(2 * np.pi)
        self.half_gaussian = np.exp(-(x / future_bins) ** 2 / 2) * norm

        self.pop_rate_threshold = None  # For detecting bursts in real time, need to call self.sort_offline(calibrate=True) to set this

        self.ibi_threshold_ms = ibi_threshold_ms 
        
        self.reset()  # Initializes some variables
                
    def reset(self):
        self.sorter.reset()
        self.last_burst = -np.inf
        self.pop_rate_cache = []
        
    def offline_predict(self, recording,
                        inter_path=None, recording_window_ms=None, 
                        reset=True, 
                        verbose=False,
                        calibrate=False,
                        plot=False):     
        """
        Params
            recording
                If a numpy array with 1D shape and dtype object (or a list), recording should be detected spike trains
                    Useful if self.sorter already spike sorted recording
            calibrate
                If True, determine self.pop_rate_threshold and then return times of detected bursts
            plot
                Whether to plot smoothed population rate and where bursts occur
        """
           
        if calibrate:
            if not reset:
                print("'calibrate' was set to True, so forcing 'reset' to be True as well")
                reset = True
            self.calibrate = True
        
        # Get popluation firing rate
        if isinstance(recording, np.ndarray) or isinstance(recording, list):  # and len(recording.shape) == 1 and recording.dtype == "object":
            all_spike_trains = recording
        else:
            all_spike_trains = self.sorter.sort_offline(recording, inter_path=inter_path, recording_window_ms=recording_window_ms, verbose=verbose, reset=reset, buffer_size=self.buffer_size)
        
        spike_data = SpikeData(all_spike_trains)
        pop_rate = np.sum(spike_data.raster(self.bin_size_ms), axis=0)
        pop_rate = SmoothBurstDetector.smooth_half_gaussian(pop_rate, self.half_gaussian.size)
        
        # Use multiple of RMS as the threshold for detecting bursts
        rms = np.sqrt(np.mean(pop_rate**2))
        threshold = self.rms_peak_threshold*rms
        
        # Find_peaks returns list[array()], so zeroth element is the result
        raw_peaks = np.array(find_peaks(pop_rate, height=(threshold, None), distance=self.ibi_threshold_ms/self.bin_size_ms)[0])

        # Make sure there must be troughs between two peaks to avoid multiple peaks within one burst
        peaks = [raw_peaks[0]]
        for i in range(raw_peaks.size - 1):
            for j in range(raw_peaks[i], raw_peaks[i+1]):
                if pop_rate[j] < self.rms_trough_threshold:
                    peaks.append(raw_peaks[i+1])
                    break
        peaks = np.array(peaks)
        burst_times = peaks * self.bin_size_ms
        
        if calibrate:
            # Determine self.pop_rate_threshold
            self.pop_rate_threshold = np.mean(pop_rate[peaks-self.future_bins])
            
            # Determine average burst duration to find self.remaining_burst_time_ms
            #   When running online, need to know how much of the burst is remaining:
            #   time_remaining = end_idx - (peak_idx - future_bins)
            burst_start_end_ind = []
            all_time_remaining_burst_ms = []
            for peak_idx in peaks:
                peak_pop_rate = pop_rate[peak_idx]
                # Find when burst starts
                for start_idx in range(peak_idx-1, -1, -1):
                    if pop_rate[start_idx] <= self.burst_duration_thresh_f * peak_pop_rate:
                        break
                # Find when burst ends
                for end_idx in range(peak_idx+1, len(pop_rate)):
                    if pop_rate[end_idx] <= self.burst_duration_thresh_f * peak_pop_rate:
                        break
                burst_start_end_ind.extend((start_idx, end_idx))
                all_time_remaining_burst_ms.append(
                    (end_idx - (peak_idx - self.future_bins)) * self.bin_size_ms
                )
            self.remaining_burst_time_ms = np.mean(all_time_remaining_burst_ms)
            
        if plot:
            x_values = np.arange(0, pop_rate.shape[0]/1000, 0.001) * self.bin_size_ms
            plt.plot(x_values, pop_rate)
            plt.scatter(x_values[peaks], pop_rate[peaks], marker="x", color="orange")  # Mark burst pekas
            plt.axhline(y=threshold, color="r", linestyle="--")  # Mark threshold
            
            plt.xlabel('time(s)', fontsize=14)
            plt.ylabel('Population rate (kHz)', fontsize=14)
            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)
            
            if calibrate:
                plt.scatter(x_values[burst_start_end_ind], pop_rate[burst_start_end_ind], marker="o", color='orange',
                            zorder=10)
            
            
            # plt.xlim(10, 20)  # Zoom into a burst (manually set lim)
            plt.show()
            
        return burst_times

    def running_predict(self, obs):
        """
        Return a tuple of
            list: Result of self.sorter.running_sort(obs)
            boolean: Whether there will be a burst in self.future_bins
        """
        burst = False

        all_detections = self.sorter.running_sort(obs)
        self.pop_rate_cache.append(len(all_detections))
        
        # # Testing (obs = a frame of population rate before smoothing)
        # all_detections = []
        # self.pop_rate_cache.append(obs)
        # self.sorter.latest_frame += 100
        
        if len(self.pop_rate_cache) < self.half_gaussian.size:
            return all_detections, burst
        
        # Smooth pop_rate
        pop_rate = np.dot(self.pop_rate_cache, self.half_gaussian)
                
        # Update pop_rate_cache, so it is always one less than self.half_gaussian.size at the start of this method
        if len(self.pop_rate_cache) == self.half_gaussian.size:
            self.pop_rate_cache.pop(0)  
        
        # If the half smoothed real time population rate exceeds the PR_threshold,
        # and the separation between current moment and last predicted peak is more than IBI_threshold,
        # predict to have the burst peak in self.future_bins amount of time
        if pop_rate >= self.pop_rate_threshold and not self.is_bursting():
            self.last_burst = self.sorter.latest_frame/self.samp_freq  # self.last_burst + self.future_bins * self.bin_size_ms == predicted peak of burst
            burst = True
    
        return all_detections, burst
    
    def is_bursting(self):
        """
        Determine if currently in a burst
        """
        return self.sorter.latest_frame/self.samp_freq <= self.last_burst + self.remaining_burst_time_ms
    
    @staticmethod
    def smooth_half_gaussian(signal, half_gaussian_size):
        """
        Half-gaussian smooth a 1d array :signal: using a sliding window
        
        Params:
            signal:
                1d array
            half_gaussian_size:
                Duration of the half-smooth sliding window (frames)
        """
        
        sigma = half_gaussian_size / 2
        x = np.arange(-half_gaussian_size, 0)
        norm = 1 / sigma / np.sqrt(2 * np.pi)
        weight = np.exp(-(x / sigma) ** 2 / 2) * norm

        # # For posterity: Jerry's original code return a 0 at index 0 and does not include the final frame 
        # convolved = np.convolve(spikes, weight)[:spikes.size-1]
        # return np.concatenate([[0], convolved])
        
        return np.convolve(signal, weight)[:signal.size]
