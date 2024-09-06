from random import randint
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from collections.abc import Iterable
import json

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from braindance.core.spikedetector import plot 
from braindance.core.spikesorter.kilosort2 import run_kilosort2
from braindance.core.spikesorter.rt_sort import save_traces


class Recording:
    """
    Represents a raw ephys recording

    Samples are indexed with [sample_time, sample_time+sample_size)
    """

    def __init__(self, rec_path, sample_size, start, 
                 # gain_to_uv,
                 mmap_mode="r",
                 n_before=60, n_after=60):
        """
        :param rec_path: str
        :param sample_size: int
            The size of the samples the dataloader should be returning.
            This is NOT the size of the dataset being loaded in by load_data.
        :param start: int
            Start frame of larger dataset
        :param spike_times: np.array
            Each element is a spike time (in samples)
        # :param gain_to_uv: float
        #     Multiply traces by this to convert to uV
        :param mmap_mode
            For np.load
        :param n_before: int
            Number of frames before spike location that contain remnants of spike
        :param n_after: int
            Number of frames after spike location that contain remnants of spike
        """
        self.traces = np.load(rec_path / "scaled_traces.npy", mmap_mode=mmap_mode)
        spike_times = np.load(rec_path / "sorted.npz", allow_pickle=True)["spike_times"]

        self.n_channels = self.traces.shape[0]

        self.sample_size = sample_size
        self.sample_times = self.get_sample_times(start, self.traces.shape[1], spike_times, n_before, n_after)
        self.n_samples = len(self.sample_times)

    def get_sample_times(self, start, total_duration, spike_times, n_before, n_after):
        """
        Get the timepoints in a recording where there are no spikes, and there is enough
        space to extract a sample. A timepoint t will be returned as if it will be used to
        extract a sample corresponding to the interval [t, t+sample_size)

        A timepoint t from the recording will be returned if at least one of the following
        conditions is met:
            1) There is no spike within the interval [t-(n_before + self.sample_size), t+n_after]
            2) t+sample_size > total_duration

        :param start: int
            Frame to start getting samples from recording
            I.e. if start=5, all samples before sample 5 will be disregarded
        :param total_duration: int
            Number of total samples in the recording
        :param spike_times: np.array
            Each element is a spike time (in samples)
        :param n_before: int
            Number of frames before spike location that contain remnants of spike
        :param n_after: int
            Number of frames after spike location that contain remnants of spike

        :return: non_spike_times: np.array
            Each element is a timepoint (in samples) that can be used to extract a sample
            from the raw recording
        """

        # Get all sample times
        sample_times = np.arange(start, total_duration - self.sample_size+1)

        # Get times that meet condition 1        
        n_before += self.sample_size - 1                
        failed_times = np.unique(spike_times + np.arange(-n_before, n_after+1)[:, None])
        sample_times = np.setdiff1d(sample_times, failed_times)
        if sample_times.size == 0:
            # This ValueError uses the parameter names of RecordingCrossVal
            raise ValueError(f"There are no windows of size {self.sample_size} frames where there are no spikes {n_before - (self.sample_size-1)} frames before and {n_after} frames after.\nReduce the value of sample_size_ms, recording_spike_before_ms, and/or recording_spike_after_ms.")
        return sample_times

    def get_sample(self, channel=None):
        """
        Get a random sample from the recording

        :param: channel: None or int
            Channel of recording for sample
            If None, a random channel will be returned

        :return: sample: np.array
            With shape (n_channels, n_samples)
        """

        sample_time_idx = torch.randint(self.n_samples, (1,))
        sample_time = self.sample_times[sample_time_idx]
        if channel is None:
            channel = torch.randint(self.n_channels, (1,))

        return self[channel, sample_time:sample_time+self.sample_size] 

    def __getitem__(self, idx):
        return self.traces[idx]

    def __len__(self):
        return self.n_samples * self.n_channels        


class Waveform:
    """Class to represent a single waveform and its properties"""
    def __init__(self, waveform, peak_idx, alpha, curated):
        self.waveform = waveform
        self.peak_idx = peak_idx
        self.len = waveform.size
        self.alpha = alpha
        self.curated = curated
        
    def unravel(self):
        return self.waveform, self.peak_idx, self.len, self.alpha, self.curated   
    
    
class Unit:
    def __init__(self, waveforms):
        self.wfs = waveforms
        
    def plot_stack(self, axis=None):
        if axis is None:
            show = False
            fig, axis = plt.subplots(1)
        else:
            show = True
        
        amp_max = -1
        wf_max = None
        for wf in self.wfs[::-1]:  # Lowest to highest amp
            wf, peak_idx, size, alpha, curated = wf.unravel()
           
            color = "black" if curated else "red"
            axis.plot(wf, alpha=0.4, color=color)
            
            amp = np.max(np.abs(wf))
            if amp > amp_max:
                amp_max = amp
                wf_max = wf
            
        print(amp_max)
        plot.set_ticks((axis,), wf_max)
        
        if show:
            plt.show() 
        

class WaveformDataset(Dataset):
    """
    Dataset that represents template waveforms that will be pasted into noise for generating data samples
    
    Data should already be scaled to uV
    """

    def __init__(self, rec_path, thresh_amp, thresh_std,
                 use_positive_peak=False,
                 x_highest=None, ms_before_after=None):
        """
        :param rec_path:
            See MultiRecordingDataset
        :param thresh_amp: float
            For each unit, only template waveforms with amplitudes >= amp_thresh will be included
            (Each unit has waveforms defined across all channels. These are treated as the individual waveforms of a unit)
        :param thresh_std: float
            For each unit, only template waveforms with mean std divided by amplitude <= thresh_std will be included
        :param mmap_mode:
            For np.load
            
        :param use_positive_peak:
            If True, use positive peak if positive peak is twice as large as negative peak (used for training MEA model)
            If False, never use positive peak (used for training neuropixels model)
            
        :param x_highest:
            If not None, then only use the x_highest waveform of a unit (0 = highest amp)
            
        :param ms_before_after:
            If None, use full template stored in rec_path/"sorted.npz"
            If a tuple, use rec_path/"spikesort_matlab4/waveforms/templates/templates_average.npy" templates and extract ms_before_after of template to use
                Tuple needs to be (ms_before_and_after_to_extract_around_peak, peak_idx_npz, peak_idx_temps_avg) 
                where peak_idx_npz is the index of peak in sorted.npz, peak_idx_temps_avg is the index of peak in templates_average,
                    For the original 6 MEA recordings, sorted.npz templates contain 61 frames (20 before peak, 1 for peak, 41 after peak).
                    templates_average.npy templates contain 201 frames (100 before peak, 1 for peak, 100 after peak)
                    So peak_idx_npz=20 and peak_idx_temps_avg=100
        """
        sorted = np.load(rec_path / "sorted.npz", allow_pickle=True)
        
        if ms_before_after is not None: 
            all_templates = np.load(rec_path / "spikesort_matlab4/waveforms/templates/templates_average.npy", mmap_mode="r").transpose(0, 2, 1)  # (num_units, num_chans, num_frames)
            ms_before_after, peak_idx_npz, peak_idx_temps_avg = ms_before_after
            samp_freq = sorted['fs'] / 1000  # kHz
            n_before_after = round(samp_freq * ms_before_after)
            peak_offset = n_before_after - peak_idx_npz
        
        self.num_units = len(sorted["units"])
        # self.locations = npz["locations"]
        # self.sampling = npz["fs"]

        self.units = []
        self.waveforms = []
        for i, unit in enumerate(sorted["units"]):
            if ms_before_after is None:
                templates = unit["template"].T  # (channels, length)
            else:
                templates = all_templates[unit['unit_id'], :, peak_idx_temps_avg-n_before_after:peak_idx_temps_avg+n_before_after]
            
            # region Old version 
            # Use values given in sorted.npz
            # peak_ind = unit["peak_ind"]
            # amplitudes = unit["amplitudes"]
            # channels = (unit["amplitudes"] >= thresh_amp) * (unit["std_norms"] <= thresh_std)  # type: np.ndarray
            # if sum(channels) == 0:
            #     continue

            # region Only use positive peak if positive peak amplitude is twice as large as negative peak amplitude
            # waveforms = templates[channels, :]
            # peak_pos = np.max(waveforms, axis=1)
            # peak_neg = np.abs(np.min(waveforms, axis=1))
            # peak_ind_pos = np.argmax(waveforms, axis=1)
            # peak_ind_neg = np.argmin(waveforms, axis=1)
            # peaks = [
            #     (peak_ind_pos[i], peak_pos[i]) if peak_pos[i] > peak_neg[i] * 2 else (peak_ind_neg[i], peak_neg[i])
            #     for i in range(len(peak_pos))
            # ]

            # # Visualize waveforms defined by positive peak
            # for i in range(len(peak_pos)):
            #     if peak_pos[i]  > peak_neg[i] * 2:
            #         print(peak_pos[i], peak_neg[i])
            #         _, ax = plt.subplots(1)
            #         plot.plot_waveform(waveforms[i], peak_ind[i], 121, 0, 80, ax)
            #         plt.show()
            # endregion
            
            # Use negative peak for peak_ind and amplitudes
            # 230703 - spikesort_matlab4 extracted waveforms sometimes using positive peak, so template may not be centered around negative peak 
            # endregion
            
            if x_highest is None:
                amplitudes = unit["amplitudes"]
                std_norms = unit["std_norms"]
                channels = (amplitudes >= thresh_amp) * (std_norms <= thresh_std)
                if sum(channels) == 0:
                    continue
                
                if use_positive_peak:
                    peak_ind = unit['peak_ind']
                    # peak_pos = np.max(templates, axis=1)
                    # peak_neg = np.abs(np.min(templates, axis=1))
                    # peak_ind_pos = np.argmax(templates, axis=1)
                    # peak_ind_neg = np.argmin(templates, axis=1)
                    # amplitudes = []
                    # peak_ind = []
                    # for c in range(peak_pos.size):
                    #     if peak_pos[c] > peak_neg[c] * 2:
                    #         amplitudes.append(peak_pos[c])
                    #         peak_ind.append(peak_ind_pos[c])
                    #     else:
                    #         amplitudes.append(peak_neg[c])
                    #         peak_ind.append(peak_ind_neg[c])
                else:
                    peak_ind = np.argmin(templates, axis=1) 
                unit_wfs = []
                for c, is_curated in enumerate(channels):
                    temp = templates[c, :]
                    peak_idx = peak_ind[c]
                    if ms_before_after is not None:
                        peak_idx += peak_offset
                    wf = Waveform(temp, peak_idx, amplitudes[c], is_curated)
                    unit_wfs.append(wf)
                    if is_curated:
                        self.waveforms.append(wf)
                self.units.append(Unit(unit_wfs))
            else:
                if use_positive_peak or ms_before_after is not None:
                    raise NotImplementedError("use_positive_peak==True or ms_before_after is not None is not implemented when x_highest is not None")
                amplitudes = unit['amplitudes']
                chan = np.argsort(-amplitudes)[x_highest]
                temp = templates[chan, :]
                peak_idx = np.argmin(temp)
                wf = Waveform(temp, peak_idx, amplitudes[chan], curated=True)
                self.waveforms.append(wf)
                self.units.append(Unit([wf]))
                

    def __getitem__(self, idx):
        """
        Get a waveform

        :param idx: int
            Index of the unit

        :return: template_max: np.array
            With shape (n_samples,)
        :return: peak_idx: int
            Index of the peak of template_max
        :return: wf_len: int
            Number of samples in waveform (the length)
        :return: alpha: float
            A unique value for the waveform
        :return: beta: float
            A unique value for the waveform
        """

        return self.waveforms[idx]

    def __len__(self):
        return len(self.waveforms)

    def plot_waveforms(self, num_rows=4, num_cols=4, max_plots=None):
        # Plot all waveforms in dataset
        # Each box is a separate unit
        # multiple plots since all units may not fit on one plot
        # :param max_plots: stop plotting after max_plots plots. if list, plot only those plot indices

        units_per_plot = num_rows * num_cols
        num_plots = int(np.ceil(len(self.units) / units_per_plot))
        for p in range(num_plots):
            if p == max_plots:
                break
            
            if isinstance(max_plots, Iterable) and p not in max_plots:
                continue
            
            # if p not in (0, num_plots//2, num_plots-2):
            #     continue
            
            fig, axs = plt.subplots(num_rows, num_cols, tight_layout=True, figsize=(15, 15))
            axs = np.atleast_2d(axs)
            
            for i, unit in enumerate(self.units[p*units_per_plot:p*units_per_plot + units_per_plot]):
                num_curated = 0
                row = i // num_rows
                col = i % num_rows
                for wf in unit.wfs:
                    wf, peak_idx, wf_len, alpha, curated = wf.unravel()
                    if curated:
                        color = "black"
                        alpha = 0.4
                        zorder = 1
                        
                        num_curated += 1
                    else:
                        color = "red"
                        alpha = 0.05
                        zorder = 0
                        
                        peak_idx = wf_len // 2
                        
                        
                    wf_x = np.arange(wf_len) - peak_idx
                    # wf_x = np.arange(wf_len)
                    axs[row, col].plot(wf_x, wf, alpha=alpha, color=color, zorder=zorder)
                    
                axs[row, col].axvline(wf_len // 2, color="black", linestyle="dashed", alpha=0.5, label=f"{num_curated}/{len(unit.wfs)}")
                axs[row, col].legend()
                 #axs[row, col].set_xlim(0, wf_len-1)

            # for r in range(n_rows):
            #     for c in range(n_cols):
            #         plot.unscaled_ticks_to_uv(axs[r, c])

            plt.show()


class MultiRecordingDataset(Dataset):
    """
    Dataset that represents 1 or more  recordings
    A wrapper of Recording and WaveformDataset
    
    traces.npy should not be in uV (instead in MEA's aribtrary units)
    sorted.npz templates should be in uV
    """

    def __init__(self, rec_paths,
                 # MultiRecordingDataset
                 samples_per_waveform=2, front_buffer=40, end_buffer=40,
                 num_wfs_probs=[0.5, 0.3, 0.12, 0.06, 0.02], isi_wf_min=4, isi_wf_max=None,
                 # WaveformDataset
                 thresh_amp=18.88275, thresh_std=0.6, x_highest=None, use_positive_peaks=False, ms_before_after=None,
                 # Recording
                 sample_size=200, start=0, ms_before=3, ms_after=3, # gain_to_uv=1,  
                 # All
                 device="cuda", dtype=torch.float32,
                 mmap_mode="r"):
        """
        Params for MultiRecordingDataset
        :param samples_per_waveform: int
            Number of recording samples per waveform.
        :param front_buffer: int
            Waveforms' indices are in [front_buffer, sample_size-end_buffer)
        :param end_buffer: int
            Waveforms' indices are in [front_buffer, sample_size-end_buffer)
        :param num_wfs_probs: list
            For each sample with a waveform, the ith element in num_wfs_probs gives the decimal probability
            of the sample containing i waveforms.
            Ex) The 0th element gives the probability of there being 0 waveform per sample with a waveform.
                The 1st element gives the probability of there being 1 waveforms per sample with a waveform.
            
            If None, use probs found in rec_path/num_wfs_probs.npy  
            
        :param isi_wf_min: int
            When multiple waveforms are in a sample, the distance between their frames must be > isi_wf_min
        :param isi_wf_max: int
            When multiple waveforms are in a sample, the distance between their frames must be <= isi_wf_max
            If None, there is no max value

        Params for Recording and WaveformDataset
        :param data_path: 
            List where each item is a path to dir containing:
                1. traces.npy: recording traces in shape of (num_chans, num_samples).  # Scaled to uV
                2. sorted.npz: from spikesort_matlab4.py with SAVE_DL_DATA equal to True

        Params for Recording
        :param sample_size: int
            This value should be passed in every time; this determines the size of the samples the dataloader
            should be returning. This is NOT the size of the dataset being loaded in by load_data.
        :param start:
        :param ms_before:
        :param ms_after:

        Params for WaveformDataset
        :param thresh_amp: float
            For each unit, only template waveforms with amplitudes >= amp_thresh will be included
            (Each unit has waveforms defined across all channels. These are treated as the individual waveforms of a unit)
        :param thresh_std: float

        Params for MultiRecordingDataset
        :param device: str
            Device to load PyTorch tensors on
        :param mmap_mode
            mmap_mode for np.load
        """
        # if np.abs(1 - sum(num_wfs_probs)) > 1e-15:  # For floating point rounding error
        #     raise ValueError("'sum(num_wfs_probs)' must equal 1")

        # Cache for __len__ and __getitem__
        self.sample_size = sample_size
        self.samples_per_waveform = samples_per_waveform
        self.torch_dtype = dtype
        # self.np_dtype = str(dtype).split(".")[1]
        self.device = device
        
        # Range of random spawning locations for waveforms
        self.loc_range = (front_buffer, sample_size - end_buffer) 
        # self.num_wfs_probs = num_wfs_probs
        self.isi_wf_min = isi_wf_min
        self.isi_wf_max = isi_wf_max

        assert isi_wf_max is None or isi_wf_min < isi_wf_max, "isi_wf_min must be less than isi_wf_max"

        self.recs = []
        self.wf_datasets = []
        if num_wfs_probs is None:
            sum_num_wfs_probs = np.zeros(sample_size, float)  # max wf count that could be for a recording is probably <= sample size 
        for rec_folder in rec_paths:
            rec_folder = Path(rec_folder)
            
            sorted = np.load(rec_folder / "sorted.npz", allow_pickle=True)
            fs = round(sorted["fs"] / 1000)  # Hz to kHz
            
            rec = Recording(rec_path=rec_folder, 
                            sample_size=sample_size,
                            start=start,
                            mmap_mode=mmap_mode,
                            n_before=ms_before * fs,
                            n_after=ms_after * fs)
                            # gain_to_uv=gain_to_uv)
            self.recs.append(rec)
            
            # print("Instantiating WaveformDataset ...")
            wf_ds = WaveformDataset(
                rec_path=rec_folder,
                thresh_amp=thresh_amp,
                thresh_std=thresh_std,
                x_highest=x_highest,
                use_positive_peak=use_positive_peaks,
                ms_before_after=ms_before_after
            )
            self.wf_datasets.append(wf_ds)
        
            if num_wfs_probs is None:
                rec_num_wfs_probs = np.load(rec_folder / "num_wfs_probs.npy")
                sum_num_wfs_probs[:len(rec_num_wfs_probs)] += rec_num_wfs_probs
        self.samp_freq = fs
        if num_wfs_probs is None:
            num_wfs_probs = sum_num_wfs_probs[sum_num_wfs_probs > 0] / len(rec_paths)
        self.num_wfs_probs = num_wfs_probs
        
        self.n_recs = len(self.recs)
        self.n_wf_datasets = len(self.wf_datasets)

        # Cache for __len__
        self._len = self.samples_per_waveform * sum([len(wf) for wf in self.wf_datasets])
        
        # Cache for __getitem__
        wfs = []  # Idx to Waveform in one of the WaveformDatasets
        idx = 0
        for wf_ds in self.wf_datasets:
            for w in range(len(wf_ds)):
                wf = wf_ds[w]
                wf.amp = wf.alpha
                wf.alpha = idx
                wfs.append(wf)
                idx += 1
        self.wfs = wfs
        
        # Set length equal to number of waveforms in the WaveformDatasets
        # Cache for __len__
        # self._len = idx
             

    def __len__(self):
        # samples_per_waveform * num_waveforms_across_all_wf_datasets
        return self._len

    def __getitem__(self, idx):
        """
        Get a sample: (noise from a constituent recording) + (waveform based on idx)

        :param idx: int
            Index of sample

        :return: trace: np.arary
            With shape (n_samples,), the trace with noise and possibly a spike
        :return: num_wfs: int
            Number of waveforms in sample
        :return: wf_trace_loc: int
            Index of waveform in trace
            If num_wfs==0, wf_idx is None
        """
        num_wfs = np.random.choice(len(self.num_wfs_probs), p=self.num_wfs_probs)

        rec_i = randint(0, self.n_recs-1)
        trace = self.recs[rec_i].get_sample().copy()
        # trace = noise.astype(self.np_dtype)

        wf_trace_locs = [-1] * (len(self.num_wfs_probs) - 1)
        wf_alphas = [-1] * (len(self.num_wfs_probs) - 1)

        if num_wfs > 0:
            # # Get which WaveformDataset waveform is from and get the index in that dataset
            # idx //= self.samples_per_waveform
            # idx_offset = 0
            # for i_wf in range(self.n_wf_datasets):
            #     len_wf = len(self.wf_datasets[i_wf])
            #     idx_wf = idx - idx_offset
            #     if idx_wf < len_wf:
            #         break
            #     idx_offset += len_wf
            # else:
            #     raise IndexError("Index out of bounds")

            # n_more_wf = np.random.choice(len(self.num_wfs_probs), p=self.num_wfs_probs)
            # num_wfs += n_more_wf
            possible_wf_locations = np.arange(*self.loc_range)
            invalid_wf_locations = []

            # ind_wf = [(i_wf, idx_wf)]  # Each element is tuple of [index of self.wf_datasets, index of waveform in the waveform dataset]
            # ind_wf.extend((i, np.random.choice(len(self.wf_datasets[i])))
            #               for i in np.random.choice(len(self.wf_datasets), size=n_more_wf))

            # for i, (i_wf, idx_wf) in enumerate(ind_wf):
            wf_ind = [idx]
            wf_ind.extend(np.random.choice(len(self), size=num_wfs-1, replace=False))
            for i, w in enumerate(wf_ind):
                w //= self.samples_per_waveform
                wf_np, peak_idx, wf_len, alpha, _ = self.wfs[w].unravel()
                # wf = torch.tensor(wf_np, dtype=self.torch_dtype, device=self.device)
                wf_trace_loc = np.random.choice(possible_wf_locations)  # randint(*self.loc_range)
                self.add_wf_to_trace(trace, wf_np, peak_idx, wf_len, wf_trace_loc)

                wf_trace_locs[i] = wf_trace_loc
                wf_alphas[i] = alpha
                invalid_wf_locations = np.union1d(
                    invalid_wf_locations,
                    range(wf_trace_loc - self.isi_wf_min, wf_trace_loc + self.isi_wf_min + 1)
                )

                if self.isi_wf_max is not None:
                    new_possible_wf_locations = range(
                        max(self.loc_range[0], wf_trace_loc - self.isi_wf_max),
                        min(self.loc_range[1], wf_trace_loc + self.isi_wf_max + 1)
                    )
                    if i == 0:
                        possible_wf_locations = new_possible_wf_locations
                    else:
                        possible_wf_locations = np.union1d(possible_wf_locations, new_possible_wf_locations)
                possible_wf_locations = np.setdiff1d(possible_wf_locations, invalid_wf_locations)
        
        trace = trace - np.median(trace)
        # trace = trace - np.mean(trace)
        # trace = trace / np.std(trace)
        trace = torch.tensor(trace, dtype=self.torch_dtype, device=self.device)
        return (trace[None, :],
                torch.tensor(num_wfs, dtype=torch.int, device=self.device),
                torch.tensor(wf_trace_locs, dtype=torch.int, device=self.device),
                torch.tensor(wf_alphas, dtype=torch.int, device=self.device)
                )

    # def get_alpha_to_waveform_dict(self):
    #     """Return a dictionary that maps an alpha value to its corresponding waveform"""
    #     a_wf_dict = dict()
    #     for dataset in self.wf_datasets:
    #         for i in range(len(dataset)):
    #             wf, peak_idx, wf_len, alpha = dataset[i]
    #             a_wf_dict[alpha] = (wf, peak_idx, wf_len)
    #     return a_wf_dict  # type: dict

    def cat_full(self):
        """Concatenate all samples together"""

        inputs = []
        labels = []
        for i in range(len(self)):
            trace, label = self[i]
            inputs.append(trace)
            labels.append(label)
        return torch.stack(inputs), torch.stack(labels)

    def plot_sample(self, trace: torch.Tensor, num_wfs: torch.Tensor, wf_locs: torch.Tensor, wf_alphas: torch.Tensor):
        fig, subplots = plt.subplots(3, figsize=(6, 7), tight_layout=True)
        a0, a1, a2 = subplots

        trace = trace[0, :].cpu().numpy().copy()
        plot.set_ticks(subplots, trace)

        num_wfs, wf_locs, wf_alphas = num_wfs.item(), wf_locs.cpu().numpy(), wf_alphas.cpu().numpy()

        # Plot sample trace
        a2.plot(trace)
        a2.set_title("Sample")

        if num_wfs:
            # Plot boundary for possible spike locations
            a2.axvline(self.loc_range[0], color="#ff7070", linestyle="dashed", label="Waveform Boundary")
            a2.axvline(self.loc_range[1], color="#ff7070", linestyle="dashed")

            # Plot each waveform
            for i, (loc, alpha) in enumerate(zip(wf_locs, wf_alphas)):
                if alpha == -1:
                    continue
                alpha = int(alpha)

                loc = int(loc)  # Needs to be int for plotting waveform and removing waveform from trace
                a2.axvline(loc, alpha=0.7, color="blue", linestyle="dashed", label="Waveform" if i == 0 else None)

                wf, peak_idx, wf_len, _, _ = self.wfs[alpha].unravel()

                # Plot waveform
                plot.plot_waveform(wf, peak_idx, wf_len, alpha, loc,
                                   a1)

                # Remove waveform from trace to get underlying noise
                self.add_wf_to_trace(trace, -wf, peak_idx, wf_len, loc)

            # Plot noise
            a0.plot(trace)

            a1.legend()
            a2.legend()

        # Prevent jitter when showing plots
        a0.set_title("Noise")
        a1.set_title("Waveform")

        plt.show()

    def add_wf_to_trace(self, trace, wf, wf_peak_idx, wf_len, wf_loc):
        # No return because trace is either a np.array or torch.Tensor --> both are passed by reference
        wf_trace_left = max(0, wf_loc - wf_peak_idx)
        wf_trace_right = min(self.sample_size, wf_loc + (wf_len - wf_peak_idx))

        trace[wf_trace_left: wf_trace_right] += wf[wf_peak_idx - (wf_loc - wf_trace_left): wf_peak_idx + (wf_trace_right - wf_loc)]

    def get_means_and_stds(self, num_samples):
        """
        Get mean and std of individual samples 

        Args:
            num_samples (int): Number of samples to generate to caluclate mean and std

        Returns:
            float, float, list(float), list(float): mean of means, mean of stds, means, stds
        """
        
        means = []
        stds = []
        for i in range(num_samples):
            if isinstance(self, DataLoader):
                sample = self.dataset[i % len(self)]
            else:
                sample = self[i % len(self)]
            trace = sample[0].flatten().cpu().numpy().astype("float32")
            means.append(np.mean(trace))
            stds.append(np.std(trace))
        
        return np.mean(means), np.mean(stds), means, stds

    def get_mean_and_std_across(self, num_samples):
        """
        Get mean and std across samples

        Args:
            num_samples (int): Number of samples to generate to caluclate mean and std

        Returns:
            float: std
            float: mean
        """
        samples = []
        for i in range(num_samples):
            if isinstance(self, DataLoader):
                sample = self.dataset[i % len(self)]
            else:
                sample = self[i % len(self)]
            trace = sample[0].flatten().cpu().numpy().astype("float32")
            samples.append(trace)
        return np.mean(samples), np.std(samples)
            
    def get_ranges(self, num_samples):
        """
        Get range (max - min) of samples
        """
        
        ranges = []
        for i in range(num_samples):
            if isinstance(self, DataLoader):
                sample = self.dataset[i % len(self)]
            else:
                sample = self[i % len(self)]
            trace = sample[0].flatten().cpu().numpy().astype("float32")
            ranges.append(np.max(trace) - np.min(trace))
        return np.mean(ranges), ranges

    @staticmethod
    def load_single(path_folder,
                    samples_per_waveform, front_buffer, end_buffer,
                    num_wfs_probs, isi_wf_min, isi_wf_max,
                    sample_size,
                    thresh_amp, thresh_std, gain_to_uv, x_highest=None, use_positive_peaks=False, ms_before_after=None,
                    device="cuda:0", dtype=torch.float32, mmap_mode="r",
                    ):
        """
        Return a MultiRecordingDataset object that represents a single recording

        :param path_folder: str or Path
            Path of folder containing the .npy traces of a recording
        :return: MultiRecordingDataset
        """

        data_root = Path(path_folder)
        return MultiRecordingDataset(rec_paths=[data_root], 
                                     samples_per_waveform=samples_per_waveform, front_buffer=front_buffer, end_buffer=end_buffer,
                                     num_wfs_probs=num_wfs_probs, isi_wf_min=isi_wf_min, isi_wf_max=isi_wf_max,
                                     sample_size=sample_size,
                                     thresh_amp=thresh_amp, thresh_std=thresh_std, mmap_mode=mmap_mode, use_positive_peaks=use_positive_peaks, ms_before_after=ms_before_after,
                                     device=device, dtype=dtype,
                                     gain_to_uv=gain_to_uv, x_highest=x_highest,
                                     )


class SubMultiRecordingDataset(MultiRecordingDataset):
    """Class to represent a subset of a MultiRecordingDataset (only some indices)"""
    def __init__(self, dataset: MultiRecordingDataset, indices: list):
        self.dataset = dataset
        self.indices = indices

        self.wf_datasets = dataset.wf_datasets

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        return self.dataset[idx]


class RecordingDataloader(DataLoader, MultiRecordingDataset):
    """Same as PyTorch's DataLoader except that this class inherits members from MultiRecordingDataset (important since functions need :method alpha_to_waveform_dict:"""
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)

    def __getattr__(self, item):
        # __getattr__ is called when using dot notation to get an attribute that obj does not have
        # __getattribute__ is called when using dot notation to get an attribute that obj has
        return self.dataset.__getattribute__(item)

    def __setattr__(self, attr, val):
        return super().__setattr__(attr, val)


class TensorDataloader(MultiRecordingDataset):
    """Convert concatenated samples as Tensors to dataloader format"""
    def __init__(self, inputs, labels, dataset):
        # dataset is the dataset object that inputs and labels came from

        self.stop = 0

        self.inputs = inputs
        self.labels = labels

        self.wf_datasets = dataset.wf_datasets

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop:
            raise StopIteration

        self.stop += 1
        return self.inputs, self.labels


class RecordingCrossVal:
    """
    Iter class for leave-one-out recording cross validation (iter for more efficient RAM)
    Split data into train-val by leaving one recording out of each train and having that recording be the validation data
    (Basically a wrapper for MultiRecordingDataset)
    """

    def __init__(self, 
                 # MultiRecordingDataset
                 samples_per_waveform, front_buffer, end_buffer,
                 num_wfs_probs, isi_wf_min, isi_wf_max,
                 # Recording and WaveformDataset
                 rec_paths, 
                 # WaveformDataset
                 thresh_amp, thresh_std, 
                 # Recording
                 sample_size, start=0, ms_before=3, ms_after=3, 
                 # All
                 device="cuda", dtype=torch.float16,
                 mmap_mode="r",
                 
                 # WaveformDataset
                 x_highest=None, use_positive_peaks=False, ms_before_after=None,
                 
                 # Other
                 verbose=True,
                 
                 # RecordingCrossVal --> return pytorch dataloaders
                 as_datasets=False,
                 **dataloader_kwargs
                 ):
        """
        :param samples_per_waveform:
            Number of samples per waveform. The length of each dataset is num_waveforms * samples_per_waveform
            If list, first element is samples in train dataloader, and second is samples in val dataloader
        :param dtype:
        :param as_datasets: bool
            If True, return datasets instead of dataloaders
        :param verbose: bool
            Whether to print information about a fold when retrieving it (see method __getitem__)
        :param dataloader_kwargs:
            kwargs for pytorch DataLoader
        """

        self.rec_paths = [Path(p) for p in rec_paths]

        if isinstance(samples_per_waveform, list) or isinstance(samples_per_waveform, tuple):
            self.samples_per_waveform_train = samples_per_waveform[0]
            self.samples_per_waveform_val = samples_per_waveform[1]
        else:
            self.samples_per_waveform_train = self.samples_per_waveform_val = samples_per_waveform

        self.dataset_kwargs = {
            # No samples_per_waveform because set separately                       
            "front_buffer": front_buffer,
            "end_buffer": end_buffer,
            "num_wfs_probs": num_wfs_probs,
            "isi_wf_min": isi_wf_min,
            "isi_wf_max": isi_wf_max,
            # No rec_paths because set separately
            "thresh_amp": thresh_amp,
            "thresh_std": thresh_std,
            "x_highest": x_highest,
            "use_positive_peaks": use_positive_peaks,
            "ms_before_after": ms_before_after,
            "sample_size": sample_size,
            "start": start,
            "ms_before": ms_before,
            "ms_after": ms_after,
            # "gain_to_uv": gain_to_uv,
            "device": device,
            "dtype": dtype,
            "mmap_mode": mmap_mode
        }
        self.dataloader_kwargs = dataloader_kwargs

        self.as_datasets = as_datasets

        self.verbose = verbose

    def __getitem__(self, idx):
        # Get name_of_val_recording, train_val
        idx = self.name_to_idx(idx)

        path_val_rec = self.rec_paths[idx]
        
        if self.verbose:
            print("\n")
            print("="*150)
            print(f"Validation Recording: {path_val_rec}")
            print("Setting up datasets (this could take a few minutes) ...")

        # samples_per_waveform = max(int(np.ceil(self.val_min_samples / val_num_wf)), 2)
        val = MultiRecordingDataset(
            **self.dataset_kwargs,
            samples_per_waveform=self.samples_per_waveform_val,
            rec_paths=[path_val_rec]
            )
        train = MultiRecordingDataset(
             **self.dataset_kwargs,
             samples_per_waveform=self.samples_per_waveform_train,
             rec_paths=[p for p in self.rec_paths if p != path_val_rec]
        )
        if self.verbose:
            num_train = len(train)/self.samples_per_waveform_train
            num_val = len(val)/self.samples_per_waveform_val
            print(f"Train: {num_train:.0f} samples -- Validation: {num_val:.0f} -- Train/Total: {num_train/(num_train+num_val)*100:.1f}%")
        
        if not self.as_datasets:
            val = RecordingDataloader(val, **{key: value for key, value in self.dataloader_kwargs.items() if key != "batch_size"}, batch_size=1000)
            train = RecordingDataloader(train, **self.dataloader_kwargs)

        return path_val_rec, train, val

    def __iter__(self):
        self.rec_i = -1  # Which rec in self.rec_paths is the next val recording
        return self

    def __next__(self):
        self.rec_i += 1

        if self.rec_i == len(self.rec_paths):
            raise StopIteration

        # Get cross val for current self.rec_i (which specifies which recording is the validation recording)
        return self[self.rec_i]

    def __len__(self):
        return len(self.rec_paths)

    def name_to_idx(self, name):
        """
        Convert name of one of cross-val recordings to index in self.recordings

        :param name:
        """
        if isinstance(name, str):
            for i, rec in enumerate(self.rec_paths):
                if rec.name == name:
                    return i
            raise IndexError(f"{name} is not a valid name of a recording")
        else:
            return name

    @staticmethod
    def summarize(rec, train, val):
        num_samples_train = len(train.dataset)
        num_samples_val = len(val.dataset)
        print(f"Val Recording: {rec} - Train: {num_samples_train} samples - Val: {num_samples_val} - Train/Val: {num_samples_val/num_samples_train * 100:.1f}%")

    # Setup objects for training
    # cross_val = data.RecordingCrossVal(
    #     samples_per_waveform=samples_per_waveform, front_buffer=front_buffer, end_buffer=end_buffer,
    #     num_wfs_probs=num_wfs_probs,
    #     isi_wf_min=isi_wf_min, isi_wf_max=isi_wf_max,
    #     rec_paths=dl_folders, thresh_amp=thresh_amp,
    #     thresh_std=thresh_std,
    #     sample_size=sample_size, ms_before=recording_spike_before_ms, ms_after=recording_spike_after_ms,
    #     device=device, dtype=dtype, mmap_mode="r",
    #     as_datasets=False,
    #     num_workers=num_workers, shuffle=shuffle, batch_size=batch_size,
    #     verbose=True
    # )


class BandpassFilter:
    """
    From SpikeInterface

    Generic filter class based on:
      * scipy.signal.iirfilter
      * scipy.signal.filtfilt or scipy.signal.sosfilt
    BandpassFilterRecording is built on top of it.

    Parameters
    ----------
    band: int or tuple or list
        If int, cutoff frequency in Hz for 'highpass' filter type
        If list. band (low, high) in Hz for 'bandpass' filter type
    sf: int
        Sampling frequency of traces to be filtered (kHz)
    btype: str
        Type of the filter ('bandpass', 'highpass')
    margin_ms: float
        Margin in ms on border to avoid border effect
    filter_mode: str 'sos' or 'ba'
        Filter form of the filter coefficients:
        - second-order sections (default): 'sos'
        - numerator/denominator: 'ba'
    coeff: ndarray or None
        Filter coefficients in the filter_mode form.
    """

    def __init__(self, band=(300, 6000), sf=20000, btype="bandpass",
                 filter_order=5, ftype="butter", filter_mode="sos", margin_ms=5.0,
                 coeff=None):
        assert filter_mode in ("sos", "ba")
        if coeff is None:
            assert btype in ('bandpass', 'highpass')
            # coefficient
            if btype in ('bandpass', 'bandstop'):
                assert len(band) == 2
                Wn = [e / sf * 2 for e in band]
            else:
                Wn = float(band) / sf * 2
            N = filter_order
            # self.coeff is 'sos' or 'ab' style
            filter_coeff = signal.iirfilter(N, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)
        else:
            filter_coeff = coeff
            if not isinstance(coeff, list):
                if filter_mode == 'ba':
                    coeff = [c.tolist() for c in coeff]
                else:
                    coeff = coeff.tolist()

        margin = int(margin_ms * sf / 1000.)

        self.coeff = filter_coeff
        self.filter_mode = filter_mode
        self.margin = margin

    def __call__(self, trace):
        if self.filter_mode == "sos":
            filtered = signal.sosfiltfilt(self.coeff, trace, axis=-1)
        elif self.filter_mode == "ba":
            b, a = self.coeff
            filtered = signal.filtfilt(b, a, trace, axis=-1)
        return filtered


def setup_dl_folders(recording_files, dl_folders,
                     **run_kilsort2_kwargs):
    """
    Set up recordings and necessary files and folders to train DL model    
    """
    
    # Save curated kilosort2 results in sorted.npz
    inter_folders = [Path(folder) / "inter" for folder in dl_folders]
    run_kilsort2_kwargs['save_dl_data'] = True
    run_kilosort2(recording_files, inter_folders, dl_folders, **run_kilsort2_kwargs)
    
    # Save scaled traces 
    for rec_path, inter_folder in zip(recording_files, dl_folders):
        print(f"\nRecording: {rec_path}")
        inter_folder = Path(inter_folder)
        save_traces(rec_path, inter_folder)

    return dl_folders


def is_dl_folder(folder):
    # Check if folder has necessary files for training and testing DL model
    folder = Path(folder)
    return folder.is_dir() and (folder / "sorted.npz").exists() and (folder / "scaled_traces.npy").exists()


def main():
    setup_dl_folders(
        recording_files=[
            "/data/MEAprojects/organoid/intrinsic/200123/2954/network/data.raw.h5"
        ],
        dl_folders=[
            "/data/MEAprojects/organoid/intrinsic/200123/2954/network"
        ],
        kilosort_path="/home/mea/SpikeSorting/kilosort/Kilosort2"
        )
    
    multirec = MultiRecordingDataset(["/data/MEAprojects/organoid/intrinsic/200123/2954/network"])
    print(len(multirec))
    print(multirec[0])


if __name__ == "__main__":
    main()
