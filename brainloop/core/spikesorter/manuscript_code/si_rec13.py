from copy import deepcopy
import h5py
from math import ceil
from multiprocessing import Pool
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
from sklearn.mixture import GaussianMixture
import torch

from diptest import diptest
from spikeinterface.extractors import NwbRecordingExtractor, MaxwellRecordingExtractor
from tqdm import tqdm

# from src import utils
# from src.comparison import Comparison
# from src.sorters.base import Unit, SpikeSorter
# from src.run_alg.model import ModelSpikeSorter

from braindance.core.spikesorter.manuscript_code import utils
from braindance.core.spikesorter.manuscript_code.comparison import Comparison
from braindance.core.spikedetector.model2 import ModelSpikeSorter

# region Utils
def pickle_dump(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
        
def pickle_load(path):
    with open(path, "rb") as file:
        return pickle.load(file)

def order_sequences(sequences):
    # Order sequences based on root_elec idx
    ordered = sorted(sequences, key=lambda seq: seq.root_elec)
    for idx, seq in enumerate(ordered):
        seq.idx = idx
    return ordered
   
def get_nearby_clusters(clusters, x, y, dist):
    """
    Get clusters with root elects within :dist: of :x, y:
    """
    nearby_clusters = []
    for clust in clusters:
        if utils.calc_dist(x, y, *ELEC_LOCS[clust.root_elec]) <= dist:
            nearby_clusters.append(clust)
    return nearby_clusters
   
class MaxwellTracesWrapper:
    def __init__(self, rec_path, start_frame=0) -> None:        
        recording = h5py.File(rec_path, "r")
        chan_ind = []  # some chans in recording['sig'] contain meaningless values
        for mapping in recording['mapping']:  # (chan_idx, elec_id, x_cord, y_cord)
            chan_ind.append(mapping[0])
        self.chan_ind = chan_ind
        self.gain = recording['settings']['lsb'][0] * 1e6
        
        self.start_frame = start_frame
        self.traces = recording['sig']
        self.shape = (len(chan_ind), self.traces.shape[1])
        
    def __getitem__(self, idx):
        chans, frames = idx  # unpacking like this is okay since FILT_TRACES[:] is never called
        chans = self.chan_ind[chans]
        if isinstance(frames, slice):
            frames = slice(frames.start+self.start_frame, frames.stop+self.start_frame, frames.step)
        else:
            frames += self.start_frame
        return self.traces[chans, frames].astype("float16") * self.gain
# endregion

# region For plot_elec_probs
class Unit: 
    """Class to represent a detected unit from a spike sorter"""
    def __init__(self, idx: int, spike_train, channel_idx: int, recording=None):
        self.idx = idx
        self.spike_train = spike_train
        self.chan = int(channel_idx)
        self.recording = Recording(recording)

    def __len__(self):
        return len(self.spike_train)

    def plot_isi_dist(self, **hist_kwargs):
        isis = np.diff(self.spike_train)
        plot.hist(isis, **hist_kwargs)
        
        plt.title(f"ISI distribution for unit idx {self.idx}")
        
        plt.xlabel("ISI (ms)")
        
        xmax = None
        if "range" in hist_kwargs:
            xmax = hist_kwargs["range"][1]
        plt.xlim(0, xmax)
        
        plt.ylabel("Number of spikes")

    def get_isi_viol_f(self, isi_viol=1.5):
        """
        Get fraction of spikes that violate ISI

        :param isi_viol: Any consecutive spikes within isi_viol (ms) is considered an ISI violation
        """
        isis = np.diff(self.spike_train)
        violation_num = np.sum(isis < isi_viol)
        # violation_time = 2 * num_spikes * (isi_threshold_s - min_isi_s)
        #
        # total_rate = num_spikes / total_duration
        # violation_rate = violation_num / violation_time
        # violation_rate_ratio = violation_rate / total_rate

        return violation_num / len(self.spike_train)

    def get_waveforms(self, spike_time_idx=None, ms_before=2, ms_after=2, chans=None) -> np.ndarray:
        """
        Get waveforms at each constituent electrode at self.spike_train[spike_time_idx]

        :param spike_time_idx:
            Which spike time to choose
            If None, pick a random spike time
        :param ms_before:
            How many ms before spike time to extract
        :param ms_after:
            How many ms after spike time to extract
        :param chans:
            Which channels to select
            If None, select all

        :return:
            np.ndarray with shape [num_electrodes, num_frames]
            num_frames = ms_before (in frames) + 1 (for spike time) + ms_after (in frames)
        """

        sf = self.recording.get_sampling_frequency()

        if spike_time_idx is None:
            spike_time_idx = np.random.choice(len(self.spike_train))
        if chans is None:
            chans = np.arange(self.recording.get_num_channels())
        
        spike_time = int(self.spike_train[spike_time_idx] * sf)        
        n_before = int(ms_before * sf)
        n_after = int(ms_after * sf)
        
        start_frame = spike_time - n_before
        end_frame = spike_time + n_after + 1
        wf = self.recording.get_traces_filt(max(0, start_frame), min(end_frame, self.recording.get_total_samples()), chans)
        if start_frame < 0:
            pad = np.zeros((len(chans), -start_frame))
            wf = np.concatenate([pad, wf], axis=1)
        elif end_frame > self.recording.get_total_samples():
            pad = np.zeros((len(chans), end_frame-self.recording.get_total_samples()))
            wf = np.concatenate([wf, pad], axis=1)
        return wf

    def get_templates(self, num_wfs=300, ms_before=2, ms_after=2):
        sf = self.recording.get_sampling_frequency()
        templates = np.zeros((self.recording.get_num_channels(), int(ms_before * sf)+int(ms_after * sf)+1), dtype=float)

        # templates_median = []
    
        if num_wfs is None or num_wfs >= len(self):
            spike_ind = range(len(self))
        else:
            spike_ind = np.random.choice(len(self), replace=False, size=num_wfs)
        
        for idx in spike_ind:
            waveforms = self.get_waveforms(spike_time_idx=idx, ms_before=ms_before, ms_after=ms_after)

            
            large_window = self.get_waveforms(spike_time_idx=idx, ms_before=33, ms_after=33)
            medians = np.median(np.abs(large_window), axis=1, keepdims=True) / 0.6745
            waveforms = waveforms / medians
            templates += waveforms
            # templates_median.append(waveforms)
        return templates / len(spike_ind)
        # print("Using templates median")
        # return np.median(templates_median, axis=0)
    
    def set_templates(self, num_wfs=300, ms_before=2, ms_after=2):
        self.templates = self.get_templates(num_wfs, ms_before, ms_after)

    def plot(self, num_wfs=300, ms_before=2, ms_after=2,
             chans_rms=None, add_lines=[],
             wf=None, time=None, sub=None,
             mea=False,  
             save=False, 
             axis=None, fig=None,
             ylim=None, scale_h=None,
             window_half_size=120,# 80,
             wf_alpha=1,  
             wf_widths=1, wf_colors="black",
             min_c=None, max_c=None
             ): 
        """Plot waveforms at location of electrodes

        Args:
            num_wfs (int, optional): _description_. Defaults to 300.
            ms_before (int, optional): _description_. Defaults to 2.
            ms_after (int, optional): _description_. Defaults to 2.
            
            chans_rms (list, optional): 
                If None, do not plot
                Else, plot mean waveforms and chans_rms[c] = rms on channel c
            add_lines (list, optional):
                Should be a list of lists. Inner list has size (num_elecs,)
                For each inner lists, plot horizontal lines the same way as would be for chans_rms
                (Used for plotting DL detections so that there is one line for 10% and one line for 17.5%)
            
            wf (np.array, optional):
                If not None, plot this wf
            time (float, None, optional):
                If None, plot templates
                Else, plot at time (ms) 
            sub (np.ndarray, None, optional):
                If not None, subtract sub from wfs
            mea (bool, optional): _description_. Defaults to False.
            save (bool, optional): _description_. Defaults to False.
            
            axis (optional): 
                If not None, create axis
            fig (optional):
                If need colorbar, fig cannot be None
            
            return_steup (bool, optional): If True, return the following as kwargs:
            scale_h (float, optional): If not None, scale height of waveforms so waveforms on different electrodes won't overlap
            

            wf_widths: float, list
                If float, all waveforms given width of wf_widths
                
                If list, should be list of size (num_elecs,)
                num_elecs[c] gives thickness of waveform on elec c
            wf_colors:
                Same as wf_widths but for color
                
                If color is str, should be name ("green") or hex value
                Otherwise, should be float used for colormap
        """
        
        if chans_rms is not None:
            if wf is not None:
                wfs = wf
            elif time is None:
                if not hasattr(self, "templates"):
                    wfs = self.get_templates(num_wfs=num_wfs, ms_before=ms_before, ms_after=ms_after)
                else:
                    wfs = self.templates  # For coactivations.ipynb
            else:
                sf = self.recording.get_sampling_frequency()
                wfs = self.recording.get_traces_filt(
                    start_frame=round((time-ms_before)*sf), 
                    end_frame=round((time+ms_before)*sf+1)
                )
        
        if sub is not None:
            wfs = wfs - sub
        
        # Set self.chan
        if self.chan == -1:
            self.chan = np.argmin(np.min(wfs, axis=1))
        
        # Setup wf_widths
        try:
            wf_widths[0]
        except TypeError:
            wf_widths = [wf_widths] * wfs.shape[0]
        
        # Plot parameters        
        
        electrode_size = 20
        electrode_color = "#888888"
        
        xlabel = "x (µm)"
        ylabel = "y (µm)"

        if mea:
            # Plot parameters for MEA
            xlim = None
            window_half_size = 75
            
            if scale_h is None and chans_rms is not None:
                max_val = np.max(np.abs(wfs[max(0, self.chan-5):self.chan+5]))
                max_val = max(max_val, chans_rms[self.chan] * 5)  # max_val before this could be below 5RMS, so red line for 5RMS could be too high
                scale_h = 15 / max_val  # 17.5 is pitch of electrode

        else:
            # Plot parameters for neuroxpixels
            xlim = (2, 68)  # for SI rec
            # xlim = (-35, 35)  # for SI ground truth rec
            
            # window_half_size = 80 # 90
            
            if scale_h is None and chans_rms is not None:
                scale_h = 20 / np.max(np.abs(wfs[max(0, self.chan-5):self.chan+5]))  # 20 is dist between two rows of electrodes

        # scale_h *= 0.7

        # cmap_latency = plt.get_cmap(cmap_latency, max_latency)

        # Find which channels to plot
        chan_center = self.chan
        locs = self.recording.get_channel_locations().astype("float32")
        loc_center = locs[chan_center]

        chans, dists = self.recording.nearest_chan[chan_center]
                    
        # Create axis
        if axis is None:
            fig, axis = plt.subplots(1)
            
        # Setup axis
        axis.set_aspect("equal")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)

        if ylim is None:
            # ymin = max(0, loc_center[1] - window_half_size)
            ymin = max(-1950, loc_center[1] - window_half_size)  # SI ground truth
            ymax = ymin + window_half_size * 2
        else:
            ymin, ymax = ylim
        axis.set_ylim(ymin, ymax)

        if xlim is None:
            xmin = max(0, loc_center[0] - window_half_size)
            xmax = xmin + window_half_size * 2
        else:
            xmin, xmax = xlim
        axis.set_xlim(xmin, xmax)

        chans = [c for c in chans if xmin <= locs[c, 0] <= xmax and ymin <= locs[c, 1] <= ymax]
        # Setup wf_colors:
        try:
            if len(wf_colors) != wfs.shape[0]:  # wf_colors could be a str (e.g. "black")
                wf_colors = [wf_colors] * wfs.shape[0]
        except TypeError:
            wf_colors = [wf_colors] * wfs.shape[0]
        if not isinstance(wf_colors[0], str): # wf_colors are floats for color map
            cmap = plt.cm.ocean
            # Only normalize colormap based on elecs used in plot
            wf_colors = np.array(wf_colors)
            
            # Adaptive color bar. -1 and +1 for when plotting individual spikes, values above/below max/min seq's latency is at ends of colorbar
            if min_c is None:
                min_c = round(np.floor(np.min(wf_colors[chans]))) - 1
            if max_c is None:
                max_c = round(np.ceil(np.max(wf_colors[chans]))) + 1
            # min_c = max(-10, min_c)
            # max_c = min(max_c, 10)
            
            num_levels = (max_c - min_c + 1)
            cmap = mpl.colors.ListedColormap(cmap(np.linspace(0, 0.9, num_levels)))  # Only use up to 0.9 since ocean is white at end
            
            bounds = np.linspace(min_c, max_c, num_levels+1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            wf_colors = [cmap(norm(c)) for c in wf_colors]
            
            # region Figuring out how to use color map and color bar
            # fig, ax = plt.subplots(figsize=(6, 1))

            # min_c = -3
            # max_c = 3
            # num_levels = (max_c - min_c) * 2
            # cmap = mpl.colors.ListedColormap(plt.cm.gist_rainbow(np.linspace(0, 1, num_levels)))

            # bounds = np.linspace(min_c, max_c, num_levels+1) # range(min_c, max_c+1)
            # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

            # colorbar = fig.colorbar(
            #     mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
            #     cax=ax, orientation='horizontal',
            # )
            # colorbar.set_ticks(bounds)

            # plt.show()

            # # Test values
            # for v in np.arange(min_c, max_c+1):
            #     plt.axvline(v, color=cmap(norm(v)))
            #     v += 0.5
            #     plt.axvline(v, color=cmap(norm(v)))
            # plt.show()
            # endregion
        else:
            cmap = min_c = max_c = None

        # Plot each channel waveform
        for c in chans:
            loc = locs[c]

            # Check if channel is out of bounds
            # if mea:
            #     if np.sqrt(np.sum(np.square(loc - loc_center))) >= max_dist:
            #         break
            # if not (xlim[0] <= loc[0] <= xlim[1] and ymin <= loc[1] <= ymax):
            #     continue

            if chans_rms is None:
                axis.scatter(*loc, marker="s", color=electrode_color, s=electrode_size) # Mark electrodes with square
            else:
                # Plot waveform
                wf = wfs[c]  # shape is (num_channels, num_waveforms, num_frames)
                wf = np.atleast_2d(wf)
                for w in wf:
                    x_values = np.linspace(loc[0]-7, loc[0]+7, w.size)
                    y_values = w * scale_h + loc[1]
                    
                    # Plot 5RMS
                    if isinstance(chans_rms, np.ndarray):
                        # Horizontal line indicating 5RMS
                        rms_scaled = 5 * (-chans_rms[c] * scale_h) + loc[1]  # 5 for 5RMS
                        axis.plot(x_values, [rms_scaled] * len(x_values),
                                    linestyle="dashed",
                                    c="red", alpha=0.6, zorder=5)
                        # Vertical line connecting horizontal line to electrode position
                        axis.plot([loc[0]] * 10, np.linspace(rms_scaled, loc[1], 10),
                                    linestyle="dashed",
                                    c="red", alpha=0.3, zorder=5)
                    # Plot additional lines
                    for lines in add_lines:
                        y_value = lines[c] * scale_h + loc[1]
                        axis.plot(x_values, [y_value] * len(x_values),
                                  linestyle="dashed",
                                  c="red", alpha=0.6, zorder=5
                                  )
                    
                    # # If cross chans_rms, make black instead of gray
                    # wf_alpha_elec = wf_alpha
                    # wf_alpha_elec -= 0.4 * (np.max(np.abs(w)) < np.abs(5*chans_rms[c]))  # Make wfs less than 5RMS gray
                    # wf_alpha_elec = max(0.1, wf_alpha_elec)
                    
                    # If cross chans_rms, make line thicker
                    
                    
                # axis.plot(x_values, y_values, c="red" if c == chan_center else "black")
                # axis.plot(x_values, y_values, color=wf_color, alpha=wf_alpha_elec, zorder=15)  # If cross chans_rms, make black instead of gray
                axis.plot(x_values, y_values, c=wf_colors[c], alpha=wf_alpha, linewidth=wf_widths[c], zorder=15)  # If cross chans_rms, make line thicker

        if fig is not None and cmap is not None:
            colorbar = fig.colorbar(
                mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                ax=axis,
            )
            colorbar.set_ticks(bounds[:-1] + np.diff(bounds)/2)
            colorbar.set_ticklabels(range(min_c, max_c+1))

        axis.set_title(f"Index: {self.idx}, #spikes: {len(self.spike_train)}")

        if save:
            plt.savefig(save)
            
        # if return_kwargs:
        return {
            "chans_rms": chans_rms,
            "add_lines": add_lines,
            "ylim": (ymin, ymax), "scale_h": scale_h,
            "ms_before": ms_before,
            "ms_after": ms_after,
            "window_half_size": window_half_size,
            "min_c": min_c,
            "max_c": max_c
            }


class Recording:
    def __init__(self, path, freq_min=300, freq_max=3000, rec_name="", gain=True):
        # MaxwellRecordingExtractor is used for type hints because quacks like a duck
        # if isinstance(path, BaseRecording):
        #     self.rec_raw : MaxwellRecordingExtractor = path
        # elif str(path).endswith("h5"):
        #     self.rec_raw = MaxwellRecordingExtractor(path)
        # elif str(path).endswith("nwb"):
        #     self.rec_raw = NwbRecordingExtractor(path)
        # elif str(path).endswith("si"):  # Stored with {spikeinterface_extractor}.save_to_folder()
        #     self.rec_raw : MaxwellRecordingExtractor = load_extractor(path)
        # else:
        #     raise ValueError(f"Recording '{path}' is not in .h5 or .nwb format")
        self.rec_raw = path

        self.name = rec_name

        # if not gain:
        #     gain = 1
        #     offset = 0
        # elif self.rec_raw.has_scaled_traces():
        #     gain = self.rec_raw.get_channel_gains()
        #     offset = self.rec_raw.get_channel_offsets()
        # else:
        #     print("Recording does not have scaled traces. Setting gain to 0.195")
        #     gain = 0.195  # 1.0
        #     offset = 0.0
        # rec_filt = scale(self.rec_raw, gain=gain, offset=offset, dtype="float32")
        # self.rec_filt = bandpass_filter(rec_filt, freq_min=freq_min, freq_max=freq_max)

        """
        If i is the index of a channel, then the ith element of self.nearest_chan is a tuple
            0) Closest channels indices: An where the 0th element is i, 1st element is closest channel, 2nd element is second closest channel, etc.
            1) Distance of channels
        """
        nearest_chan = []
        locs = self.get_channel_locations()
        for i in range(len(locs)):
            loc = locs[i]
            dists = np.sqrt(np.sum(np.square(locs - loc), axis=1))
            dists_sorted = np.sort(dists)
            chans_sorted = np.argsort(dists)

            nearest_chan.append((chans_sorted, dists_sorted))
        self.nearest_chan = nearest_chan

    def get_total_duration(self):
        return self.rec_raw.get_total_duration()

    def get_total_samples(self):
        return self.rec_raw.get_total_samples()

    def get_channel_locations(self):
        return self.rec_raw.get_channel_locations()

    def get_num_channels(self):
        return self.rec_raw.get_num_channels()

    def get_sampling_frequency(self):
        return self.rec_raw.get_sampling_frequency() / 1000  # in kHz

    def get_traces_raw(self, start_frame=None, end_frame=None, channel_ind=None):
        return Recording._get_traces(self.rec_raw, start_frame, end_frame, channel_ind)

    def get_traces_filt(self, start_frame=None, end_frame=None, channel_ind=None):
        return Recording._get_traces(self.rec_filt, start_frame, end_frame, channel_ind)

    def plot_traces(self, start_frame, end_frame, channel_ind=None, show=True):
        """

        :param start_frame:
        :param end_frame:
        :param channel_ind:
        :param show:
            If True, show figure
            If False, don't show figure and return (fig, plots)
        """

        traces = (self.get_traces_raw(start_frame, end_frame, channel_ind),
                  self.get_traces_filt(start_frame, end_frame, channel_ind))

        fig, plots = plt.subplots(1, 2, figsize=(10, 3))
        plots[0].set_title("Raw traces")
        plots[1].set_title("Filtered traces")

        for i in range(2):
            for trace in traces[i]:
                plots[i].plot(trace)
            plots[i].set_xlim(0, end_frame-start_frame)

        if show:
            plt.show()
        else:
            return fig, plots

    def plot_waveform(self, st, chan_center=None, n_before=40, n_after=40,
                      subplots=None,
                      save_path=None, close_window=False):
        """
        Make waveform plot described in TJ's email "Additional information"

        :param st:
            Spike time
        :param chan_center:
            Which channel the center waveform comes from
            (usually the channel the spike is detected on)
            If None, choose channel with maximum negative amplitude
        :param n_before:
            Number of frames before st to include
        :param n_after:
            Number of frames after st to include
        :param subplots:
            If None, create subplots
            Else, plot on subplots which is (fig, a0, a1)
        :param save_path:
            If not None, save figure to save_path
        :param close_window:
            If True, close window and do not show figure
            If false, show figure
        """
        # Get waveforms
        # waveforms = self.get_traces_raw(st-n_before, st+n_after+1).astype(float)
        # waveforms -= np.mean(waveforms, axis=1, keepdims=True)
        waveforms = self.get_traces_filt(st-n_before, st+n_after+1).astype(float)

        if chan_center is None:
            chan_center = np.argmin(waveforms[:, n_before])  # np.argmax(np.abs(waveforms[:, n_before]))

        # Create plotting figure
        if subplots is None:
            fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 5))
        else:
            fig, (a0, a1) = subplots

        # Plot max waveform
        a0.set_xlabel("Rel. time (frames, 20kHz)")
        a0.set_ylabel("Voltage (µV)")
        set_ticks((a0,), waveforms[chan_center, :], center_xticks=True)
        a0.set_xlim(0, n_before+n_after)
        a0.plot(waveforms[chan_center, :])

        # Plot waveform channels
        WINDOW_HALF_SIZE = 90  # length and width of window will be WINDOW_HALF_SIZE * 2
        SCALE_W = 0.25  # Multiply width of waveform by this to scale it down
        SCALE_H = 0.01 #*6  # Multiple height of waveform by this to scale it down

        a1.set_aspect("equal")
        a1.set_xticks([])
        a1.set_yticks([])

        # Each grid space is PITCHxPITCH in area
        a1.set_xlim(-100, 80)
        a1.set_ylim(-WINDOW_HALF_SIZE, WINDOW_HALF_SIZE)

        # Get channel waveforms to plot
        chans, dists = self.nearest_chan[chan_center]
        locs = self.get_channel_locations()
        locs[:, 0] *= 1.5

        loc_center = locs[chan_center]  # Location of channel with max amp

        max_dist = np.sqrt(2) * WINDOW_HALF_SIZE # Distance from center of window to corner

        # Plot each channel waveform
        for c in chans:
            # Offset the location of the waveform to make max amplitude channel (0, 0)
            loc = locs[c] - loc_center
            if np.sqrt(np.sum(np.square(loc))) >= max_dist:
                break

            wf = waveforms[c]
            x_values = (np.arange(wf.size) - n_before) * SCALE_W + loc[0]
            y_values = wf * SCALE_H + loc[1]
            a1.plot(x_values, y_values, c="red" if c == chan_center else "black")

            # For prop signal adding rms (see end of run_alg_decomp_2.ipynb)
            rms_buffer = 500
            prop_buffer = int(1.5*30)
            traces = self.get_traces_filt(st - rms_buffer, st + rms_buffer, c).flatten()
            rms = np.sqrt(np.mean(np.square(traces)))
            factor = -np.min(traces[rms_buffer-prop_buffer:rms_buffer+prop_buffer+1]) / rms
            a1.text(x_values[x_values.size // 2], y_values[y_values.size // 2], c, c="blue", size=15)
            a1.text(x_values[x_values.size // 2]+20, y_values[y_values.size // 2], f"{factor:.1f}", c="brown", size=13)

            # a1.scatter(*loc)

        if save_path is not None:
            plt.savefig(save_path)

        fig.suptitle(f"Rec: {self.name}, c: {chan_center}, st: {st}")

        if close_window:
            plt.close()

        if subplots is None:
            plt.show()

    @staticmethod
    def _get_traces(rec, start_frame=None, end_frame=None, channel_ind=None):
        # Helper function for get_traces_raw and get_traces_filt
        if channel_ind is None:
            channel_ids = rec.get_channel_ids()
        else:
            channel_ind = np.atleast_1d(channel_ind)
            channel_ids = rec.get_channel_ids()[channel_ind]
        traces = rec.get_traces(start_frame=start_frame, end_frame=end_frame, channel_ids=channel_ids, return_scaled=False)  # (n_frames, n_channels)
        return traces.T  # (n_channels, n_frames)
# endregion


# region Plotting
class TracesFiltWrapper:
    """
    Wrapper for RECORDING so it can be used for FILT_TRACES
    """
    def __init__(self, recording):
        self.recording = recording
        self.chans = range(recording.get_num_channels())
        # self.frames = range(recording.get_total_samples()) 
        self.shape = (recording.get_num_channels(), recording.get_total_samples())
        
    def __getitem__(self, idx):
        chans, frames = idx  # unpacking like this is okay since FILT_TRACES[:] is never called
        chans = self.chans[chans]
        return self.recording.get_traces_filt(frames.start, frames.stop, chans)

def plot_elec_probs(seq, idx=None,
                    amp_kwargs=None, prob_kwargs=None,
                    use_filt=False,
                    use_formation_spike_train=False,
                    ms_before_after=2,
                    return_wf_data=False,
                    correct_spike_time_offset=False,
                    debug=True, show_colorbar=True,
                    num_spikes=300):        
    """
    TJ:
    For me to construct the final figure panels with the sequence footprints and single spikes, could you please make the following adjustments to your figures:

-X       Let the colorbar range from the smallest latency-1 to the largest latency+1 of the average sequence but only looking at electrodes that have a median detection score above the loose threshold (loose electrodes). Currently, the noisy electrodes among the footprint electrodes (any electrode within 50um of a loose electrode) might have high latency outliers that lead to very large latency ranges which make the actual latencies of the propagation hard to see. The bottom tick label of the colormap can then say <=X and the top can say >=Y where X and Y are the smallest-1 and largest+1 respectively.

-X       For the single spikes, please use the same colormap range as for the sequence average.

- X - Implement for MEAs:       If possible, it would be nice to have the size of the footprint subplot and the detection footprint subplot to be the same. This might require you to put the colorbar in for this detection footprint subplot too. That is fine since I can crop it out.

-X       Please remove the figure title and the subplot titles with the index and number of spikes

- X Implement for MEAs        Please let the colorbar not go higher and lower than the footprint plot (might be sufficient to adjust the figure dimensions)

- X       Please add a label to the colorbar with “Latency (frames)”

- X       Please remove all the x-labels, y-labels, x-ticks and y-ticks. I’ll manually add a scale bar instead.

    Params:
        return_wf_data
            If True, just return templates, wf_widths, and wf_colors. Used for plotting just waveforms in a different function
        debug:
            If True, include seq idx and #spikes in subplot titles, x and y labels and ticks, do not include label for colorbar
    """
    SI_SIM_REC = False  # If True, use plotting parameters specific for spikeinterface's ground-truth recording

    stringent_prob_thresh = STRINGENT_THRESH
    loose_prob_thresh = LOOSE_THRESH
    
    # Convert thresholds from percent to decimal
    # stringent_prob_thresh /= 100
    # loose_prob_thresh /= 100
    
    # if not isinstance(unit, Unit):
    if hasattr(seq, "root_elecs"):
        channel_idx = seq.root_elecs[0]
    else:
        channel_idx = seq.chan
    
    if idx is None:
        idx = getattr(seq, "idx", 0)     
    
    if use_formation_spike_train:
        spike_train = seq.formation_spike_train
    else:
        spike_train = seq.spike_train
    
    unit = Unit(idx=idx, spike_train=spike_train, channel_idx=channel_idx, recording=RECORDING)
    
    wfs = extract_waveforms(unit, use_filt=use_filt, ms_before=ms_before_after, ms_after=ms_before_after, num_cocs=num_spikes)
    temps = np.mean(wfs, axis=0)
    if channel_idx == -1:  # If no chan was specified, set chan to chan with largest negative peak
        unit.chan = np.argmin(np.min(temps, axis=1))
    
    # unit.set_templates()

    all_elec_probs = extract_detection_probs(unit, num_cocs=num_spikes)  # (num_spikes, num_elecs, num_samples)    
    elec_probs = np.mean(all_elec_probs, axis=0)  # for footprint

    if SI_SIM_REC:
        all_elec_probs = -wfs
        loose_prob_thresh = 3  # Loose elecs = loose_prob_thresh * SNR
        correct_spike_time_offset = True

    # mid = elec_probs.shape[1]//2
    # elec_probs = elec_probs[:, mid-6:mid+6]
    
    # Plot elec probs on top of each other
    # plt.plot(elec_probs[[0, 2, 4, 1, 3], :].T)
    # plt.show()
    
    CHANS_RMS = np.full(NUM_ELECS, 1)  # For when templates are np.mean(amp/median)

    # Width of each elec's waveforms and detections (pass refers to whether detection crosses loose_prob_threshold)
    root_width=2.75
    pass_width=1.5
    not_pass_width=0.85
    wf_widths = []  
    wf_colors = []  # Color of each elec's waveforms (it is actually the latency on that electrode
    wf_probs = []  # Median prob on each elec
    # For elec's waveform thickness (Mean/median of individual spikes is different than mean of footprint)
    root_elec_latencies = np.argmax(all_elec_probs[:, unit.chan, :], axis=1)
    for c in range(NUM_ELECS):
        # Widths (median probability)
        probs = all_elec_probs[:, c, :]  # (num_spikes, num_samples)
        peaks = np.max(probs, axis=1)  # (num_spikes,)
        median = np.median(peaks)
        wf_probs.append(median)
        if median >= loose_prob_thresh:
            wf_widths.append(pass_width)
        else:
            wf_widths.append(not_pass_width)
            
        # Colors (mean latency)
        peaks = np.argmax(probs, axis=1)
        if correct_spike_time_offset:  # Needed when spike time not centered on root elec trough
            peaks -= root_elec_latencies # (num_spikes,)
        else:
            peaks -= probs.shape[1] // 2
        mean = np.mean(peaks)
        wf_colors.append(mean)
    wf_colors[unit.chan] = 0  # Force 0 latency on root elec (in case average latency is not exactly 0)
    wf_widths[unit.chan] = root_width
    if return_wf_data:
        return temps, wf_widths, wf_colors 

    # Plot waveforms footprint
    amp_kwargs = {"chans_rms": CHANS_RMS} if amp_kwargs is None else amp_kwargs
    
    # Colorbar calibrated only to loose elecs
    loose_elecs_wf_colors = np.array(wf_colors)[[elec for elec, prob in enumerate(wf_probs) if prob >= loose_prob_thresh]]     
    # # if len(loose_elecs_wf_colors) > 0:
    if "min_c" not in amp_kwargs:
        amp_kwargs['min_c'] = round(np.floor(np.min(loose_elecs_wf_colors))) - 1
    if "max_c" not in amp_kwargs:
        amp_kwargs['max_c'] = round(np.ceil(np.max(loose_elecs_wf_colors))) + 1
    
    # For neuropixels
    if not MEA:
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(6.4, 5.5))
    else:
        # For MEA
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9, 6))
    
    amp_kwargs = unit.plot(axis=ax0, wf=temps, 
                           wf_widths=wf_widths, wf_colors=wf_colors,
                           fig=fig if show_colorbar else None, mea=MEA,
                           **amp_kwargs)
    
    # Plot detections footprint
    if prob_kwargs is None:
        # Horizontal lines at loose and stringent thresh
        prob_chans_rms = np.array([-loose_prob_thresh/5]*NUM_ELECS)  # Extra modifiers to loc_prob_thresh make thresh line visible in plot
        add_lines = [[stringent_prob_thresh]*NUM_ELECS]  # Add stringent thresh line

        # Set horizontal lines at specific probs (in percent)
        if SI_SIM_REC:
            lower_line = 0.01 
            higher_line = 0.05
            prob_chans_rms = np.array([-lower_line/5]*NUM_ELECS)  # Extra modifiers to loc_prob_thresh make thresh line visible in plot
            add_lines = [[higher_line]*NUM_ELECS]  # Add stringent thresh line

        prob_kwargs = {
            "chans_rms": prob_chans_rms,
            "add_lines": add_lines
        }
    prob_kwargs = unit.plot(axis=ax1, wf=elec_probs, 
                            wf_widths=wf_widths, mea=MEA,
                            **prob_kwargs)
    
    if not debug:
        for ax in (ax0, ax1):
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.set_ylabel("")
            ax.set_yticks([])
            
            # Hide top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Increase thickness of the bottom and left spines
            ax.spines["bottom"].set_linewidth(2.5)
            ax.spines["left"].set_linewidth(2.5)

            # Increase thickness of tick marks
            ax.tick_params(axis='both', direction='out', length=6, width=2.5, colors='black')
    
    return amp_kwargs, prob_kwargs

def plot_spikes(spike_train, channel_idx, **plot_elec_probs_kwargs):
    """
    Wrapper for plot_elec_probs to plot spike_train
    """
    unit = Unit(plot_elec_probs_kwargs.get("idx", 0), spike_train, channel_idx, RECORDING)
    amp_kwargs, prob_kwargs = plot_elec_probs(unit, **plot_elec_probs_kwargs)
    return amp_kwargs, prob_kwargs

def plot_amp_dist(cluster, ylim=None, **hist_kwargs):
    """
    Params
    cluster
        Can be obj with attr spike_train
        Or np.array of amplitudes
    """
    
    try:
        amplitudes = np.array(get_amp_dist(cluster))
    except AttributeError:
        amplitudes = cluster
    
    # Set default number of bins
    if "bins" not in hist_kwargs:
        hist_kwargs["bins"] = 40
    
    plt.hist(amplitudes, **hist_kwargs)
    plt.xlabel("Amplitude / (median/0.6745)")
    # plt.xlabel("Amplitude (µV)")
    plt.ylabel("#spikes")
    plt.ylim(ylim)

    # dip, pval = diptest.diptest(amplitudes)
    # print(f"p-value: {pval:.3f}")
    # print(f"ISI viol%: {get_isi_viol_p(cluster):.2f}%")
    
    return amplitudes

def plot_split_amp(cluster, thresh):
    """
    Divide cluster's spike train into spikes below and above amp thresh
    Then plot resulting footprints
    """

    # If CocCluster class (._spike_train)
    spike_train = cluster.spike_train
    
    amplitudes = plot_amp_dist(cluster, bins=40)
    amplitudes = np.array(amplitudes)
    plt.show()
    
    dip, pval = diptest.diptest(amplitudes)
    print(f"Dip test p-val: {pval}")

    cluster._spike_train = spike_train
    amp_kwargs, prob_kwargs = plot_elec_probs(cluster)
    plt.show()
    print(f"ISI viol %:", get_isi_viol_p(cluster))

    cluster._spike_train = spike_train[amplitudes < thresh]
    plot_elec_probs(cluster, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs)
    plt.show()
    
    cluster._spike_train = spike_train[amplitudes >= thresh]
    plot_elec_probs(cluster, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs)
    plt.show()
    print(f"ISI viol %:", get_isi_viol_p(cluster))
    
    print(f"ISI viol %:", get_isi_viol_p(cluster))


    cluster._spike_train = spike_train
    
    # If Unit class (.spike_train)
    # spike_train = cluster.spike_train
    
    # amplitudes = plot_amp_dist(cluster, bins=40)
    # plt.show()

    # cluster.spike_train = spike_train
    # amp_kwargs, prob_kwargs = plot_elec_probs(cluster)
    # plt.show()
    # print(get_isi_viol_p(cluster))

    # cluster.spike_train = spike_train[amplitudes < thresh]
    # plot_elec_probs(cluster, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs)
    # plt.show()
    # print(get_isi_viol_p(cluster))

    # cluster.spike_train = spike_train[amplitudes >= thresh]
    # plot_elec_probs(cluster, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs)
    # plt.show()
    # print(get_isi_viol_p(cluster))

    # cluster.spike_train = og_spike_train

def plot_gmm(gmm, data):    
    data = data.flatten()
    plt.hist(data, bins=30, density=True, alpha=0.5, color="gray")
    from scipy.stats import norm
    x_range = np.linspace(data.min(), data.max(), 1000)
    for i, (mean, cov, weight) in enumerate(zip(gmm.means_.flatten(), gmm.covariances_.flatten(), gmm.weights_)):
        print(f"cluster {i}, mean: {mean:.3f}, std: {np.sqrt(cov):.3f}")
        plt.plot(x_range, weight * norm.pdf(x_range, mean, np.sqrt(cov)), label=f'Component {i+1}')
        plt.xlabel("Amplitude")
        plt.ylabel("Fraction of spikes")

def plot_dip_p_values(coc_clusters):
    """
    Plot hist of each cluster's dip-test p-value on root elec
    """
    p_values = []
    for cluster in coc_clusters:
        dip, pval = diptest(cluster.every_amp_median[cluster.root_elec])
        p_values.append(pval)
    plt.hist(p_values, bins=20, range=(0, 1))
    plt.xlim(0, 1)
    plt.xlabel("p-value")
    plt.ylabel("#sequences")

def plot_isi_viols(cluster, ms_before_after=2,
                   isi_viol=1.5, max_count=5,
                   random_seed=101):
    """
    Plot ISI violations
    
    Params:
    ms_before_after
        Make it larger than duration of spike if you want to see if there are multiple spikes visible in plots
    max_count
        Max #viols to plot
    """
    isis = np.diff(cluster.spike_train)
    viols = isis <= isi_viol
    print(f"#viols: {np.sum(viols)}")
    ind = np.flatnonzero(viols)
    if len(ind) > max_count:
        np.random.seed(random_seed)
        ind = np.random.choice(ind, max_count, replace=False)
    
    for idx in ind:
        print("-"*100)
        time = cluster.spike_train[idx]
        print(time)
        plot_spikes([time], cluster.root_elec, idx=idx, ms_before_after=ms_before_after)
        plt.show()
        time = cluster.spike_train[idx+1]
        print(time)
        plot_spikes([time], cluster.root_elec, idx=idx+1, ms_before_after=ms_before_after)
        plt.show()

def _save_sequences_plots(args, n_spikes=5):
    sequence, root_path = args
    path = root_path / str(sequence.idx)
    path.mkdir(exist_ok=True, parents=True)

    unit = Unit(sequence.idx, sequence.formation_spike_train, sequence.root_elecs[0], None)  # Plot sequence's spike train in TRAINING_FRAMES
    amp_kwargs, prob_kwargs = plot_elec_probs(unit, idx=sequence.idx)
    # plot_elec_probs(sequence, idx=sequence.idx)
    plt.savefig(path / "average_footprint.jpg", format="jpg")
    plt.close()

    # Plot individual spikes 
    spike_train = sequence.spike_train
    np.random.seed(1)
    for idx in np.random.choice(spike_train.size, n_spikes):
        time = spike_train[idx]
        plot_seq_spike_overlap(sequence, time, idx, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs)
 
        plt.savefig(path / f"spike{idx}.jpg", format="jpg")
        plt.close()
    
    # Plot amp median distribution
    # amp_medians = get_amp_medians(sequence, use_formation_spike_train=True)
    amp_medians = sequence.every_amp_median[sequence.root_elec]
    plt.hist(amp_medians, bins=30)
    plt.ylabel("#spikes")
    plt.xlabel("amplitude/median on root electrode")
    plt.savefig(path / "root_amp_median_dist.jpg", format="jpg")
    plt.close()
    
    # Plot individual clusters from merge
    # if hasattr(sequence, "history"):
    #     for idx, cluster in enumerate(sequence.history):
    #         plot_elec_probs(cluster, idx=idx)
    #         plt.savefig(path / f"sequence{idx}.jpg", format="jpg")
    #         plt.close()

def save_sequences_plots(sequences, root_path):
    """
    For each sequence, save a folder in root_path/{idx} containing:
        1. average_footprint.png
        2. n_spikes .png of randomly selected spikes in the sequence
    """
        
    tasks = [(seq, root_path) for seq in sequences]    
    with Pool(processes=20) as pool:
        for _ in tqdm(pool.imap_unordered(_save_sequences_plots, tasks), total=len(tasks)):
            pass
# endregion


# region Electrode dists
def calc_elec_dist(elec1, elec2):
    # Calculate the spatial distance between two electrodes
    x1, y1 = ELEC_LOCS[elec1]
    x2, y2 = ELEC_LOCS[elec2]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_nearby_elecs(ref_elec, max_dist=100):
    nearby_elecs = []
    for elec in ALL_CLOSEST_ELECS[ref_elec]:
        if calc_elec_dist(ref_elec, elec) <= max_dist:
            nearby_elecs.append(elec)
    return nearby_elecs

def get_merge_elecs(ref_elec, max_dist=100):
    # [ref_elec] + get_nearby_elecs
    return [ref_elec] + get_nearby_elecs(ref_elec, max_dist)
# endregion


# region Model output utils
def rec_ms_to_output_frame(ms):
    # Convert time (in ms) in recording to frame in model's outputs
    return round(ms * SAMP_FREQ) - FRONT_BUFFER

def sigmoid(x):
    # return np.where(x>=0,
    #                 1 / (1 + np.exp(-x)),
    #                 np.exp(x) / (1+np.exp(x))
    #                 )
    x = np.clip(x, a_min=-9, a_max=10)  # Prevent overflow error
    return np.exp(x) / (1+np.exp(x))  # Positive overflow is not an issue because DL does not output large positive values (only large negative)

def sigmoid_inverse(y):
    return -np.log(1 / y - 1)

# endregion


# region Extract recording/output data
def extract_waveforms(prop, num_cocs=300, ms_before=2, ms_after=2,
                      use_filt=False):
    """
    Parameters
    ----------
    num_cocs: int
        Number of cocs to sample to extract detection probabilities
    ms_before and ms_after: float
        Window for extracting probabilities
    use_filt:
        Whether to use TRACES or FILT_TRACES
    """
    np.random.seed(231)

    # Load outputs 
    if use_filt:
        outputs = FILT_TRACES
    else:
        outputs = TRACES
        # assert False, "If not used for plotting, need to account that these waveforms have not been divided by median"
    num_chans, total_num_frames = outputs.shape

    # Load spike train
    spike_train = prop.spike_train
    if num_cocs is not None and len(spike_train) > num_cocs:
        spike_train = np.random.choice(spike_train, size=num_cocs, replace=False)

    # Extract waveforms
    n_before = round(ms_before * SAMP_FREQ)
    n_after = round(ms_after * SAMP_FREQ)
    all_waveforms = np.zeros((len(spike_train), num_chans, n_before+n_after+1), dtype="float32")  # (n_spikes, n_chans, n_samples)
    
    # for i, time_ms in enumerate(spike_train):
    for i, time_ms in enumerate(tqdm(spike_train, "Extracting waveforms")):
        time_frame = round(time_ms * SAMP_FREQ)
                
        if time_frame-n_before < 0 or time_frame+n_after+1 > total_num_frames :  # Easier and faster to ignore edge cases than to handle them
            continue
        
        window = outputs[:, time_frame-n_before:time_frame+n_after+1]    
        window = window / calc_pre_median(time_frame)[:, None]
            
        # if not use_filt:
        #     large_window = outputs[:, max(0, time_frame-500):time_frame+501]
        #     # means = np.mean(large_window, axis=1, keepdims=True)
        #     # large_window = large_window - means
        #     # medians = np.median(np.abs(large_window), axis=1, keepdims=True) / 0.6745
        #     # window = (window - means) / medians
        #     window = window - means
                
        all_waveforms[i] = window
        
    return all_waveforms

def extract_detection_probs(prop, num_cocs=300, ms_before=0.5, ms_after=0.5):
    """
    Parameters
    ----------
    num_cocs: int
        Number of cocs to sample to extract detection probabilities
    ms_before and ms_after: float
    """
    np.random.seed(231)

    # Load outputs 
    outputs = OUTPUTS # np.load(MODEL_OUTPUTS_PATH, mmap_mode="r")  # Load each time to allow for multiprocessing
    num_chans, total_num_frames = outputs.shape

    # Load spike train
    spike_train = prop.spike_train
    if num_cocs is not None and len(spike_train) > num_cocs:
        spike_train = np.random.choice(spike_train, size=num_cocs, replace=False)

    # Extract probabilities
    n_before = round(ms_before * SAMP_FREQ)
    n_after = round(ms_after * SAMP_FREQ)
    all_elec_probs = np.zeros((len(spike_train), num_chans, n_before+n_after+1), dtype="float16")  # (n_cocs, n_chans, n_samples) float16: Model's output is float16
    # for i, time_ms in enumerate(spike_train):
    for i, time_ms in enumerate(tqdm(spike_train, "Extracting detection probabilities")):
        time_frame = rec_ms_to_output_frame(time_ms)
        if time_frame-n_before < 0 or time_frame+n_after+1 > total_num_frames :  # Easier and faster to ignore edge cases than to handle them
            continue
        
        window = outputs[:, time_frame-n_before:time_frame+n_after+1]
        all_elec_probs[i] = sigmoid(window)
    # elec_probs /= len(spike_train)
    return all_elec_probs

def calc_pre_median(frame, elecs=slice(None)):
    """
    Get the preceeding mean and median of traces before :time:
    
    Says "mean" but is actually "median" (because it was changed to well after it was first implemented as mean)
    
    1/13/24 - median has already been extracted from window
    """    
    # frame = round(time * SAMP_FREQ)
    # pre_frames = round(pre_ms * SAMP_FREQ)
    window = TRACES[elecs, max(0, frame-PRE_MEDIAN_FRAMES):frame]
    
    # means = np.median(window, axis=1) # np.mean(window, axis=1)
    
    # median = np.median(np.abs(window - means[:, None]), axis=1) 
    median = np.median(np.abs(window), axis=1) 
    
    # median = np.abs(np.median(np.abs(window), axis=1) - means)  # Faster to subtract means after finding median, but this is inaccurate due to taking abs val of window
    # return means, np.clip(median / 0.6745, a_min=0.5, a_max=None) 
    return np.clip(median / 0.6745, a_min=0.5, a_max=None) 

# endregion


# region Run DL model
def _save_traces_si(task):
    start_frame, end_frame, channel_idx, save_path = task    
    # traces = RECORDING.get_traces_filt(start_frame=start_frame, end_frame=end_frame, channel_ind=channel_idx).flatten().astype("float16")
    traces = RECORDING.get_traces(start_frame=start_frame, 
                                  end_frame=end_frame, 
                                  channel_ids=[RECORDING.get_channel_ids()[channel_idx]],
                                  return_scaled=True).flatten().astype("float16")
    saved_traces = np.load(save_path, mmap_mode="r+")
    saved_traces[channel_idx] = traces

def save_traces_si(save_path, 
                   start_ms=0, end_ms=None,
                   num_proceses=16):
    """
    Save scaled traces (microvolts) using spikeinterface recording defined at RECORDING
    
    Params
    start_ms and end_ms
        Must be for one contiguous block
    """        
    samp_freq = RECORDING.get_sampling_frequency()  # Hz
    num_elecs = RECORDING.get_num_channels()
    
    start_frame = round(start_ms / 1000 * samp_freq)
    
    if end_ms is None:
        end_frame = RECORDING.get_total_samples()
    else:
        end_frame = round(end_ms / 1000 * samp_freq)

    print("Alllocating memory for traces ...")
    traces = np.zeros((num_elecs, end_frame-start_frame), dtype="float16")
    np.save(save_path, traces)
    del traces
    
    print("Extracting traces ...")
    tasks = [(start_frame, end_frame, channel_idx, save_path) for channel_idx in range(num_elecs)]
    # return tasks
    with Pool(processes=num_proceses) as pool:
        for _ in tqdm(pool.imap_unordered(_save_traces_si, tasks), total=len(tasks)):
            pass
        
def _save_traces_mea_old(task):
    rec_path, save_path, start_frame, chan_ind, chunk_start, chunk_size, gain = task
    sig = h5py.File(rec_path, 'r')['sig']
    traces = sig[chan_ind, chunk_start:chunk_start+chunk_size].astype("float16") * gain
    saved_traces = np.load(save_path, mmap_mode="r+")
    saved_traces[:, chunk_start-start_frame:chunk_start-start_frame+chunk_size] = traces
        
def save_traces_mea_old(rec_path, save_path,
                        start_ms=0, end_ms=None, samp_freq=20,  # kHz
                        default_gain=1,
                        chunk_size=100000,
                        num_processes=16):
    """
    This only works for the old format of Maxwell MEA .h5 files
    """
    
    start_frame = round(start_ms * samp_freq)

    recording = h5py.File(rec_path, 'r')

    if end_ms is None:
        end_frame = recording['sig'].shape[1]
    else:
        end_frame = round(end_ms * samp_freq)

    chan_ind = []
    for mapping in recording['mapping']:  # (chan_idx, elec_id, x_cord, y_cord)
        if mapping[1] != -1:
            chan_ind.append(mapping[0])
    if 'lsb' in recording['settings']:
        gain = recording['settings']['lsb'][0] * 1e6    
    else:
        gain = default_gain
        print(f"'lsb' not found in 'settings'. Setting gain to uV to {gain}")

    print("Alllocating memory for traces ...")
    traces = np.zeros((len(chan_ind), end_frame-start_frame), dtype="float16")
    np.save(save_path, traces)
    del traces
    
    print("Extracting traces ...")
    tasks = [(rec_path, save_path, start_frame, chan_ind, chunk_start, chunk_size, gain) 
             for chunk_start in range(start_frame, end_frame, chunk_size)]
    with Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(_save_traces_mea_old, tasks), total=len(tasks)):
            pass
        
def _save_traces_mea_new(task):
    rec_path, save_path, start_frame, chan_ind, chunk_start, chunk_size, gain = task
    sig = h5py.File(rec_path, 'r')['recordings']['rec0000']['well000']['groups']['routed']['raw']
    traces = sig[chan_ind, chunk_start:chunk_start+chunk_size].astype("float16") * gain
    saved_traces = np.load(save_path, mmap_mode="r+")
    saved_traces[:, chunk_start-start_frame:chunk_start-start_frame+chunk_size] = traces
        
def save_traces_mea_new(rec_path, save_path,
                       start_ms=0, end_ms=None, samp_freq=20,  # kHz
                      default_gain=1,
                      chunk_size=100000,
                      num_processes=16):
    """
    Only works for newer .h5 format
    
    Can't save traces with spikeinterface get_traces() because it is really slow
    """
    
    start_frame = round(start_ms * samp_freq)

    rec_si = MaxwellRecordingExtractor(rec_path)
    
    # Check that h5py matches rec_si
    rec_h5 = h5py.File(rec_path)
    assert rec_h5['recordings']['rec0000']['well000']['groups']['routed']['raw'].shape == (rec_si.get_num_channels(), rec_si.get_total_samples())

    if end_ms is None:
        end_frame = rec_si.get_total_samples()
    else:
        end_frame = round(end_ms * samp_freq)

    chan_ind = list(range(rec_si.get_num_channels())) # [int(id) for id in rec_si.get_channel_ids()]
    
    if rec_si.has_scaled_traces():
        gain = rec_si.get_channel_gains()   
    else:
        gain = np.full_like(chan_ind, default_gain, dtype="float16")
        print(f"Recording does not have channel gains. Setting gain to {gain}")
    gain = gain[:, None]

    print("Alllocating memory for traces ...")
    traces = np.zeros((len(chan_ind), end_frame-start_frame), dtype="float16")
    np.save(save_path, traces)
    del traces
    
    print("Extracting traces ...")
    tasks = [(rec_path, save_path, start_frame, chan_ind, chunk_start, chunk_size, gain) 
             for chunk_start in range(start_frame, end_frame, chunk_size)]
    _save_traces_mea_new(tasks[0])
    
    with Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(_save_traces_mea_new, tasks), total=len(tasks)):
            pass

def run_dl_model(model_path, scaled_traces_path,
                 model_traces_path, model_outputs_path,
                 device="cuda"):    
    """
    WARNING: [Torch-TensorRT] - Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors
        - This is nothing unless using model on a different GPU than what created it (https://github.com/dusty-nv/jetson-inference/issues/883#issuecomment-754106437)
    
    """

    torch.backends.cudnn.benchmark = True
    np_dtype = "float16"
    
    # region Load model
    print("Loading DL model ...")
    model = ModelSpikeSorter.load(model_path) 
    sample_size = model.sample_size
    num_output_locs = model.num_output_locs
    input_scale = model.input_scale
    model = ModelSpikeSorter.load_compiled(model_path)
    # endregion
    
    # region Prepare data
    scaled_traces = np.load(scaled_traces_path, mmap_mode="r")
    
    num_chans, rec_duration = scaled_traces.shape

    start_frames_all = np.arange(0, rec_duration-sample_size+1, num_output_locs)
    
    print("Allocating memory to save model traces and outputs ...")
    traces_all = np.zeros(scaled_traces.shape, dtype=np_dtype)
    np.save(model_traces_path, traces_all)
    traces_all = np.load(model_traces_path, mmap_mode="r+")

    outputs_all = np.zeros((num_chans, start_frames_all.size*num_output_locs), dtype=np_dtype)
    np.save(model_outputs_path, outputs_all)
    outputs_all = np.load(model_outputs_path, mmap_mode="r+")
    # endregion
    
    # region Calculating inference scaling
    if INFERENCE_SCALING_NUMERATOR is not None:
        window = scaled_traces[:, :PRE_MEDIAN_FRAMES]
        iqrs = scipy.stats.iqr(window, axis=1)
        median = np.median(iqrs)
        inference_scaling = INFERENCE_SCALING_NUMERATOR / median
    else:
        inference_scaling = 1
    print(f"Inference scaling: {inference_scaling}")
    # endregion
    
    # region Run model
    print("Running model ...")    
    with torch.no_grad():
        for start_frame in tqdm(start_frames_all):
            traces_torch = torch.tensor(scaled_traces[:, start_frame:start_frame+sample_size], device=device)
            traces_torch -= torch.median(traces_torch, dim=1, keepdim=True).values
            outputs = model(traces_torch[:, None, :] * input_scale * inference_scaling).cpu()

            traces_all[:, start_frame:start_frame+sample_size] = traces_torch.cpu()
            outputs_all[:, start_frame:start_frame+num_output_locs] = outputs[:, 0, :]
    # endregion
    
    # region Save traces and outputs
    # np.save(model_traces_path, traces_all)
    # np.save(model_outputs_path, outputs_all)
    # endregion

def extract_crossings(model_outputs_path, 
                      all_crossings_path, elec_crossings_ind_path,
                      end_ms=None,
                      window_size=1000,
                      device="cpu"):          
    outputs = np.load(model_outputs_path, mmap_mode="r")
    
    # Using end_ms (rel to recording traces) since only need crossings during pre-recording to form sequences, not entire duration  
    if end_ms is None:
        end_frame = outputs.shape[1]-window_size-1
    else: 
        end_frame = round(end_ms * SAMP_FREQ) - FRONT_BUFFER  # -FRONT_BUFFER to convert from rec. traces to model outputs
    
    all_crossings = []  # [(elec_idx, time, amp)]
    elec_crossings_ind = [[] for _ in range(NUM_ELECS)]  # ith element for elec idx i. Contains ind in all_crossings for elec idx i's crossings
    crossing_idx = 0
    for start_frame in tqdm(range(0, end_frame, window_size)):
        if start_frame >= end_frame: 
            break
        
        window = outputs[:, start_frame:start_frame+window_size+2]
        window = torch.tensor(window, device=device)
        
        main = window[:, 1:-1]
        greater_than_left = main > window[:, :-2]
        greater_than_right = main > window[:, 2:]
        peaks = greater_than_left & greater_than_right
        crosses = main >= STRINGENT_THRESH_LOGIT
        nonzeros = torch.nonzero((peaks & crosses).T)  # .T is so that outputs are ordered based on peak_ind first then elec_ind
        for peak, elec in nonzeros:
            peak = peak.item()
            elec = elec.item()
            time_ms = (FRONT_BUFFER + peak + start_frame + 1) / SAMP_FREQ  # +1 since rel. to main (which is +1 rel to window and window is rel. to start_frame)
            all_crossings.append((elec, time_ms, -1)) 
            elec_crossings_ind[elec].append(crossing_idx)
            crossing_idx += 1
            
        # if start_frame > 10000:
        #     times = [all_crossings[idx][1] for idx in elec_crossings_ind[17]]
        #     print(times)
        #     plot_spikes(times, 17)
        #     plt.show()
        #     return
    
    np.save(all_crossings_path, np.array(all_crossings, dtype=object))
    np.save(elec_crossings_ind_path, np.array(elec_crossings_ind, dtype=object))

# endregion


# region Form clusters
class Patience:
    def __init__(self, root_elec, patience_end):
        self.counter = 0
        self.root_elec = root_elec
        
        self.last_dist = 0
        
        self.patience_end = patience_end
        
    def reset(self):
        self.counter = 0
    
    def end(self, comp_elec) -> bool:
        """
        Increment counter and check if to end
        """
        
        dist = calc_elec_dist(self.root_elec, comp_elec)
        if dist != self.last_dist:
            self.counter += 1        
            if self.counter >= self.patience_end:
                return True
                
        self.last_dist = dist
        return False
        
    def verbose(self):
        print(f"Patience: {self.counter}/{self.patience_end}")
        
class CocCluster:
    def __init__(self, root_elec, split_elecs, spike_train):
        # Form 1 initial cluster on root elec
        self.root_elec = root_elec
        self.root_elecs = [root_elec]
        self.split_elecs = split_elecs
        self._spike_train = spike_train
        # self.latencies = []
        
    def split(self, split_elec, spike_train):
        # Split cluster using split_elec
        return CocCluster(self.root_elec, self.split_elecs.union({split_elec}), spike_train)
        
    @property
    def spike_train(self):
        return np.sort(self._spike_train)

def branch_coc_cluster(root_cluster, comp_elecs,
                       coc_dict, allowed_root_times,
                       
                       min_unimodal_p, max_n_components,
                       
                       max_latency_diff, 
                       min_coc_n, min_coc_p,
                       
                       min_extend_comp_p,
                       
                       patience,
                       
                       verbose=False):
    """
    Recursive function, first called in form_coc_clusters

    Params:
    allowed_root_times
        The root_times in the cluster being branched, i.e. new clusters can only be formed from the times in allowed_root_times
    max_latency_diff
        For coc to join cluster, the latency difference on the comp elec has to be at most max_latency_diff
        (Keep value as float (even if wanting value to be 0) to account for floating point rounding error)
    min_coc_p
        If #cluster_cocs/#total_cocs_on_root_elec_comp_pair < min_coc_p/100, cluster is discarded  
    """
    
    comp_elec = comp_elecs[0]
    
    if verbose:
        print(f"Comparing to elec {comp_elec}, loc: {ELEC_LOCS[comp_elec]}")
        
    # region Using absolute latency difference for finding clusters
    # # Form new elec clusters
    # elec_clusters = []
    # # Iterate through allowed root times
        
    # # all_latencies = []  # For plotting all latency distribution
    # for root_time in allowed_root_times:
    #     cocs = coc_dict[root_time]
        
    #     # Check each electrode that cooccurs
    #     for tar_elec, tar_time, tar_latency in cocs:
    #         if tar_elec == comp_elec:  # Comp elec found
    #             # Form new cluster for coc or add to existing cluster
    #             closest_cluster = None
    #             min_diff = max_latency_diff
    #             for cluster in elec_clusters:
    #                 diff = np.abs(cluster.mean_latency - tar_latency)
    #                 if diff <= min_diff:
    #                     closest_cluster = cluster
    #                     min_diff = diff
    #             if closest_cluster is not None:  # Add to existing cluster
    #                 closest_cluster.add_coc(tar_time, tar_latency, -1)
    #             else:  # Form new cluster
    #                 elec_clusters.append(CocCluster(root_elec, tar_time, tar_latency, -1))

    #             # all_latencies.append(tar_latency)  # For plotting all latency distribution 
                
    # # Due to moving averages with adding cocs to cluster, CocClusters may be within max_latency_diff, so they need to be merged
    # dead_clusters = set()
    # while True:
        # Find best merge
        # merge = None
        # min_diff = max_latency_diff
        # for i in range(len(elec_clusters)):        
        #     cluster_i = elec_clusters[i]
        #     if cluster_i in dead_clusters:
        #         continue
        #     for j in range(i+1, len(elec_clusters)):            
        #         cluster_j = elec_clusters[j]
        #         if cluster_j in dead_clusters:
        #             continue
        #         diff = np.abs(cluster_i.mean_latency - cluster_j.mean_latency) 
        #         if diff <= min_diff:
        #             merge = [cluster_i, cluster_j]
        #             min_diff = diff
                    
        # # If no merges are found, end loop
        # if merge is None:
        #     break
        
        # merge[0].merge(merge[1])
        # dead_clusters.add(merge[1])
    # endregion
    
    # Using GMM for finding clusters
    """
    Pseudocode:
        1. Find latencies on comparison electrode
        2. Split cluster based on latencise
            a. Fit GMM
            b. Form clusters based on GMM
            c. Add electrode to clusters's group of splitting elecs
        3. Pick next electrode
        4. Determine if group of upcoming comparison electrodes need to be extended
    """
    
    # 1.
    all_times = []
    all_latencies = []
    # all_amps = []
    for root_time in allowed_root_times:
        cocs = coc_dict[root_time]
        # Check each electrode that cooccurs
        for tar_elec, tar_time, tar_latency in cocs:
            if tar_elec == comp_elec:  # Comp elec found
                all_latencies.append(tar_latency)  # For plotting all latency distribution 
                all_times.append(root_time)
                
                # rec_frame = round(tar_time * SAMP_FREQ)
                # pre_means, pre_medians = calc_pre_median(max(0, rec_frame-N_BEFORE), [comp_elec])        
                # amps = np.abs(TRACES[comp_elec, rec_frame] - pre_means)
                # amp_medians = amps / pre_medians
                # all_amps.append(amp_medians[0])
                
                break
            
    # 2.
    min_cocs = max(min_coc_n, len(allowed_root_times) * min_coc_p/100)
    
    if len(all_times) <= min_cocs:  # Not enough codetections on comp_elec to split, so allowed_root_times stay together        
        coc_clusters = [root_cluster.split(comp_elec, list(allowed_root_times))]      
        
        if patience.end(comp_elec):
            return coc_clusters
    else:          
        all_latencies = np.array(all_latencies)        
        # dip, pval = diptest(all_latencies)
        # if pval >= min_unimodal_p:  # latencies are unimodal
        #     coc_clusters = [root_cluster.split(comp_elec, all_times)]  # TODO: If unimodal latency and enough cocs to be cluster, not sure if cocs with and without coc on comp_elec should be split
        #     # coc_clusters = [root_cluster.split(comp_elec, all_times)]
        
        # gmm_latencies = np.vstack([all_latencies, all_amps]).T
        
        gmm_latencies = all_latencies.reshape(-1, 1)  # Reshape to (n_samples, n_features) for GMM
        best_score = np.inf
        best_gmm = None
        max_n_components = min(max_n_components, len(set(all_latencies)))  # Prevents warning "ConvergenceWarning: Number of distinct clusters (3) found smaller than n_clusters (4). Possibly due to duplicate points in X."
        for n_components in range(1, max_n_components+1):
            gmm = GaussianMixture(n_components=n_components, random_state=1150, n_init=1, max_iter=1000)
            try:
                gmm.fit(gmm_latencies)
                score = gmm.bic(gmm_latencies)      
            except FloatingPointError:  # FloatingPointError: underflow encountered in exp, 1/8/24: guess that this is caused by probably of fitting being so low that it causes underflow error
                continue
        
            # print(f"{n_components} components, {score}")
                 
            if score < 0: # If negative, it fits data too well
                continue
                 
            if score < best_score:
                best_score = score
                best_gmm = gmm
            
            if score < 0:
                continue
                
        if best_gmm is None or best_gmm.n_components == 1:  # If all GMM are negative, use n_components=1
            coc_clusters = [root_cluster.split(comp_elec, all_times)]
            patience.reset()  # Can just reset since "len(all_times) <= min_cocs" ensures enough coc if n_components=1
        else:
            predictions = best_gmm.predict(gmm_latencies)
            coc_clusters = [root_cluster.split(comp_elec, []) for _ in range(best_gmm.n_components)]
            for cluster, time in zip(predictions, all_times):
                coc_clusters[cluster]._spike_train.append(time)  
            # for cluster, time, latency in zip(predictions, all_times, all_latencies):
            #     coc_clusters[cluster]._spike_train.append(time)  
            #     coc_clusters[cluster].latencies.append(latency)           
            
            # region For plotting
            # for i, (mean, cov, weight) in enumerate(zip(best_gmm.means_.flatten(), best_gmm.covariances_.flatten(), best_gmm.weights_)):
            #     coc_clusters[i].plot_data = (mean, cov, weight)
            # endregion
            
            coc_clusters = [c for c in coc_clusters if len(c._spike_train) >= min_cocs]
            
            if len(coc_clusters) == 0:  # If allowed_root_times were split into clusters with not enough spikes, allow original cluster to continue branching
                coc_clusters = [root_cluster.split(comp_elec, list(allowed_root_times))]
                if patience.end(comp_elec):
                    return coc_clusters
            else:
                patience.reset()
                
        # region Naive Plot clusters
        # bins = {}
        # cmap = plt.get_cmap("tab10")
        # test = set()
        # for latency in all_latencies:
        #     if latency not in bins:
        #         bins[latency] = 1
        #     else:
        #         bins[latency] += 1
        # for latency, count in bins.items():
        #     prediction = best_gmm.predict([[latency]])
        #     test.add(prediction[0])
        #     plt.scatter([latency]*count, range(count), color=cmap(prediction[0]))
        # plt.show()
        # endregion

        # region Plot GMM
        # # good example on real spikeinterface recording: clusters = F.form_coc_clusters(1, TRAINING_MS, verbose=True)
        # plt.hist(all_latencies, bins=range(-15, 16), density=True, alpha=0.5, color='gray')

        # # Plot the PDFs of individual components
        # from scipy.stats import norm
        # x_range = np.linspace(all_latencies.min(), all_latencies.max(), 1000)
        # for i, cluster in enumerate(coc_clusters):
        #     if hasattr(cluster, "plot_data"):
        #         mean, cov, weight = cluster.plot_data
        #         print(f"cluster {i}, mean: {mean:.3f}, std: {np.sqrt(cov):.3f}, #cocs: {len(cluster._spike_train)}")
        #         plt.plot(x_range, weight * norm.pdf(x_range, mean, np.sqrt(cov)), label=f'Component {i+1}')     

        # Plot the combined PDF (sum of individual components)
        # pdf_sum = np.sum([weight * norm.pdf(x_range, mean, np.sqrt(cov)) for mean, cov, weight in zip(gmm.means_.flatten(), gmm.covariances_.flatten(), gmm.weights_)], axis=0)
        # plt.plot(x_range, pdf_sum, color='red', linewidth=2, label='Combined PDF')
        # plt.xlim(-15, 15)
        # plt.xlabel("Latencies (frames)")
        # plt.ylabel("Fraction of codetections")
        # plt.show()
        # # np.save("/data/MEAprojects/dandi/000034/sub-mouse412804/rt_sort/240319/example_latency_distribution/latencies.npy", all_latencies)
        # assert False
        # endregion

        # region More plotting
        # print()
    
        # print(f"total #coc_clusters: {len(coc_clusters)}")
        # for c in coc_clusters:
        #     s = len(c._spike_train)
        #     print(f"{s} cocs", f"{round(s/len(allowed_root_times)*100)}%")
            
        # predictions = best_gmm.predict(gmm_latencies)
        # clusters = [[] for _ in range(best_gmm.n_components)]
        # for label, time in zip(predictions, all_times):
        #     clusters[label].append(time)
        
        # for i, cluster in enumerate(clusters):
        #     unit = Unit(i, cluster, root_cluster.root_elec, None)
        #     plot_elec_probs(unit)
        #     plt.show()
        
        # # Plot distribution of all latencies
        # # latency_min = min(all_latencies)
        # # latency_max = max(all_latencies)
        # plt.hist(all_latencies, bins=range(-15, 16, 1)) #, bins=range(round(latency_min), round(latency_max)+1))
        # plt.xlim(-15, 15)
        # plt.xticks(range(-15, 16, 2))
        # plt.show()
        # assert False  
        # endregion


    if len(comp_elecs) == 1:  # If no more elecs to compare to/branch
        return coc_clusters
    
    # Recursion branching
    min_extend_comp = len(allowed_root_times) * min_extend_comp_p/100
    comp_elecs_set = set(comp_elecs)
    new_coc_clusters = []
    for cluster in coc_clusters:
        # Check if enough cocs for further splitting       
        if len(cluster._spike_train) <= min_coc_n: 
            new_coc_clusters.append(cluster)
            continue 

        # Check whether to add more electrodes to comp_elecs
        if len(cluster._spike_train) >= min_extend_comp:
            cluster_comp_elecs = comp_elecs[1:]
            for elec in ALL_CLOSEST_ELECS[comp_elec]:
                if calc_elec_dist(comp_elec, elec) > INNER_RADIUS:
                    break
                if elec not in comp_elecs_set and elec not in cluster.split_elecs:  # Prevent 1) double counting elecs 2) splitting on an elec that has already been used for splitting
                    cluster_comp_elecs.append(elec)
        else:
            cluster_comp_elecs = comp_elecs[1:]      

        # Actually do recursion
        branches = branch_coc_cluster(
            cluster, cluster_comp_elecs,
            coc_dict, allowed_root_times=set(cluster._spike_train),
            min_unimodal_p=min_unimodal_p, max_n_components=max_n_components,
            max_latency_diff=max_latency_diff, 
            min_coc_n=min_coc_n, min_coc_p=min_coc_p,
            min_extend_comp_p=min_extend_comp_p,
            patience=patience,
            verbose=verbose
        )
    
        new_coc_clusters += branches
        
    return new_coc_clusters

def form_coc_clusters(root_elec, time_frame, 
                      max_latency_diff=None, 
                      min_coc_n=10, min_coc_p=10, 
                      min_extend_comp_p=50,
                      min_unimodal_p=0.100001, max_n_components_amp=4,
                      max_n_components_latency=4,
                      min_coc_prob=None, 
                      max_amp_elec_dist=None, comp_elec_dist=None,
                      elec_patience=6,
                      verbose=False):    
    """
    Params
        min_unimodal_p
            If diptest p-value is >= min_unimodal_p, the distribution (amp/median) is considered unimodal
        max_n_components
            If amp/median is multimodal, use BIC to determine best n_components to fit amp/median dist, ranging from 2 to max_n_components
        max_amp_elec_dist:
            Which elecs to compare to see if root_elec has max (largest) amp/median
        comp_elec_dist:
            Which cocs to grow tree on 
        
        n_before and n_after
            Coacs need to be within :n_before: and :n_after: frames of root spike
    
    """    
    n_before = N_BEFORE
    n_after = N_AFTER
    
    min_unimodal_p = MIN_AMP_DIST_P
    min_coc_prob = STRINGENT_THRESH
    
    comp_elec_dist = INNER_RADIUS
    max_amp_elec_dist = OUTER_RADIUS
    
    # Setup
    comp_elecs = get_nearby_elecs(root_elec, comp_elec_dist)
    max_amp_elecs = get_nearby_elecs(root_elec, max_amp_elec_dist)
    if len(max_amp_elecs) < 1:
        return []
    
    min_coc_prob = sigmoid_inverse(min_coc_prob)
    
    # Extract root_times and amp/median
    start_time, end_time = time_frame
    all_times = []
    for root_idx in ELEC_CROSSINGS_IND[root_elec]:
        time = ALL_CROSSINGS[root_idx][1]
        if time < start_time:
            continue
        elif time > end_time:
            break
        else:
            all_times.append(time)
    
    if verbose:
        print(f"Starting with elec {root_elec}, loc: {ELEC_LOCS[root_elec]}")
        print("\nFinding coocurrences")
        all_times = tqdm(all_times)
        
    allowed_root_times = set() 
    root_amp_medians = []
    coc_dict = {}  # root time to cocs [(elec, latency)]
    root_time_to_amp_median = {}
    num_activity_cocs = 0  # To see if elec contains activity
    for time in all_times:  # Two loops for this so tqdm is accurate
        # Check if time is largest amp/median NOTE: Amp here is defined as max value in traces. Amp in other areas is defined by location of DL prediction. (Probably doesn't make a difference since DL is pretty accurate. Also doing it differently here might be better since the max-amp threshold is slightly more stringent this way, which is better for forming sequence backbones)
        rec_frame = round(time * SAMP_FREQ)
        
        # Raw traces and mean and median of preceeding window
        this_n_before = n_before if rec_frame - n_before >= 0 else rec_frame  # Prevents indexing problems
        start_frame = rec_frame - this_n_before
        pre_medians = calc_pre_median(start_frame, [root_elec] + max_amp_elecs)   
        root_pre_median = pre_medians[0]
        pre_medians = pre_medians[1:]
             
        traces = TRACES[max_amp_elecs, start_frame:rec_frame+n_after+1]  # Use rec_frame here so its rec_frame-this_n_before:rec_frame+n_after+1
        amp_medians = np.abs(np.min(traces, axis=1)) / pre_medians
        
        root_amp_median = np.abs(TRACES[root_elec, rec_frame]) / root_pre_median
        
        if root_amp_median < np.max(amp_medians):
            continue
                
        # Check if not noise spike
        output_frame = rec_ms_to_output_frame(time)
        noise_probs = OUTPUTS[:, output_frame]
        if np.sum(noise_probs >= LOOSE_THRESH_LOGIT) >= MIN_ELECS_FOR_ARRAY_NOISE:
            continue 
        
        # Check if time has enough a coac with comp_elecs
        this_n_before = n_before if output_frame - n_before >= 0 else output_frame  # Prevents indexing problems
        cocs = []
        for elec in comp_elecs:
            # Check if elec coactivates
            output_window = OUTPUTS[elec, output_frame-this_n_before:output_frame+n_after+1]
            prob = np.max(output_window)
            if prob < min_coc_prob:
                continue
            
            # Add to coc_dict
            latency = np.argmax(output_window) - this_n_before # (np.argmax(output_window) - this_n_before) / SAMP_FREQ  # root_elec detects spike at this_n_before
            cocs.append((elec, time, latency))
        if len(cocs) > 0:
            allowed_root_times.add(time)
            coc_dict[time] = cocs
            root_amp_medians.append(amp_medians[0])    
            root_time_to_amp_median[time] = root_amp_median
        if len(cocs) >= MIN_ACTIVITY_ROOT_COCS:
            num_activity_cocs += 1
    
    if len(root_amp_medians) < min_coc_n or num_activity_cocs < MIN_ACTIVITY:
        return []

    # Determine whether root times need to be split into different roots based on amp_median modality
    root_amp_medians = np.array(root_amp_medians)
    dip, pval = diptest(root_amp_medians)
    if True or pval >= min_unimodal_p:  # amp_medians are unimodal
        amps_allowed_root_times = [allowed_root_times]
    else:  # amp_medians are not unimodal
        root_amp_medians = root_amp_medians.reshape(-1, 1)  # Reshape to (n_samples, n_features)
        best_score = np.inf
        best_gmm = None
        for n_components in range(1, max_n_components_amp+1):
            gmm = GaussianMixture(n_components=n_components, random_state=1150, n_init=1, max_iter=1000)
            gmm.fit(root_amp_medians)
            score = gmm.bic(root_amp_medians)
            if score < best_score:
                best_score = score
                best_gmm = gmm
            print(n_components, score)
            
        amps_allowed_root_times = [[] for _ in range(best_gmm.n_components)]
        amps_allowed_root_times_activities = [0] * best_gmm.n_components
        for label, time in zip(best_gmm.predict(root_amp_medians), coc_dict.keys()):
            amps_allowed_root_times[label].append(time)
            if len(coc_dict[time]) >= MIN_ACTIVITY_ROOT_COCS:
                amps_allowed_root_times_activities[label] += 1
        amps_allowed_root_times = [times for times, count in zip(amps_allowed_root_times, amps_allowed_root_times_activities) if count >= MIN_ACTIVITY]
        if len(amps_allowed_root_times) == 0:
            return []
            
    if verbose:
        print(f"{len(allowed_root_times)} cocs total")
        
        if len(amps_allowed_root_times) > 1:
            print(f"\nMultimodal amp/median with p-value: {pval:.3f}")
            print(f"Dividing root cocs into amp/median groups with #cocs:")
            print(f"{[len(t) for t in amps_allowed_root_times]}")
            # plt.hist(root_amp_medians, bins=30)
            # plt.show()
            # assert False
        else:
            print(f"\nUnimodal amp/median with p-value: {pval:.3f}")

    # print(f"root amps:")
    # plt.hist(root_amp_medians, bins=30)
    # plt.show()

    # # For faster testing
    # utils.pickle_dump(amps_allowed_root_times, "/data/MEAprojects/PropSignal/MAX_DELETE_ME/amps_allowed_root_times.pickle") 
    # utils.pickle_dump(coc_dict, "/data/MEAprojects/PropSignal/MAX_DELETE_ME/coc_dict.pickle")
    # amps_allowed_root_times = utils.pickle_load("/data/MEAprojects/PropSignal/MAX_DELETE_ME/amps_allowed_root_times.pickle")
    # coc_dict = utils.pickle_load("/data/MEAprojects/PropSignal/MAX_DELETE_ME/coc_dict.pickle")

    all_coc_clusters = []
    root_cluster = CocCluster(root_elec, {root_elec}, [])
    for allowed_root_times in amps_allowed_root_times:
        if verbose and len(amps_allowed_root_times) > 1:       
            print(f"-"*50)
            print(f"Starting on amp/median group with {len(allowed_root_times)} cocs")
        allowed_root_times = set(allowed_root_times)

        # patience_counter = 0
        # Compare root to each comp elec
        for c in range(len(comp_elecs)):    
            if verbose: 
                print(f"\nComparing to elec {comp_elecs[c]}, loc: {ELEC_LOCS[comp_elecs[c]]}")
                
            # Grow tree on root-comp elec pair
            coc_clusters = branch_coc_cluster(root_cluster, comp_elecs[c:],  # Elecs before c would have already been compared to root-comp elec pair
                                              coc_dict, allowed_root_times=allowed_root_times,
                                              min_unimodal_p=min_unimodal_p, max_n_components=max_n_components_latency,
                                              max_latency_diff=max_latency_diff, 
                                              min_coc_n=min_coc_n, min_coc_p=min_coc_p,
                                              min_extend_comp_p=min_extend_comp_p,
                                              patience=Patience(root_elec, elec_patience),
                                              verbose=False)            
            for cluster in coc_clusters:
                allowed_root_times.difference_update(cluster._spike_train)
                all_coc_clusters.append(cluster)
            
            if verbose:
                print(f"Found {len(coc_clusters)} clusters")
                print(f"{len(allowed_root_times)} cocs remaining")
                                
            if len(allowed_root_times) < min_coc_n:
                if verbose:
                    print(f"\nEnding early because too few cocs remaining")
                break
            
            # if len(coc_clusters) == 0:
            #     patience_counter += 1
            # else:
            #     patience_counter = 0

            # if verbose:
            #     print(f"Patience counter: {patience_counter}/{elec_patience}")
                
            # if patience_counter == elec_patience:
            #     if verbose:
            #         print(f"\nStopping early due to patience")
            #     break
        
    # region Split coc_clusters based on root amp medians
    if not SPLIT_ROOT_AMPS_AGAIN:
        if verbose:
            print(f"\nTotal: {len(all_coc_clusters)} clusters")
        
        return all_coc_clusters
    
    # # Save pval (in si_rec13.ipynb, see end of section Development/Figure finding)
    # all_p_values = []
    
    all_split_coc_clusters = []
    for i, cluster in enumerate(all_coc_clusters):
        # plot_elec_probs(cluster, idx=i)
        # plt.show()
        
        # amp_medians = get_amp_medians(cluster, n_cocs=None).flatten()
        # plt.hist(amp_medians, bins=30)
        # plt.show()
        
        # print(f"dip p-val: {diptest(amp_medians)[1]:.3f}")
        
        if len(cluster._spike_train) < min_coc_n:
            continue
        
        root_amp_medians = np.array([root_time_to_amp_median[time] for time in cluster._spike_train])
        dip, pval = diptest(root_amp_medians)
        
        # # Save pval (in si_rec13.ipynb, see end of section Development/Figure finding)
        # all_p_values.append(pval)
        
        if pval >= min_unimodal_p:  # root_amp_medians are unimodal
            all_split_coc_clusters.append(cluster)
            continue
        if verbose:
            print(f"\nCluster {i}: p-val={pval:.4f}")
        
        root_amp_medians = root_amp_medians.reshape(-1, 1)  # Reshape to (n_samples, n_features)
        best_score = np.inf
        best_gmm = None
        for n_components in range(2, max_n_components_amp+1):
            gmm = GaussianMixture(n_components=n_components, random_state=1150, n_init=1, max_iter=1000)
            gmm.fit(root_amp_medians)
            score = gmm.bic(root_amp_medians)
            if score < best_score:
                best_score = score
                best_gmm = gmm
            # print(n_components, score)
            
        # # Plot amplitude distribution
        # plot_gmm(best_gmm, root_amp_medians)
        # plt.show()
        # # np.save("/data/MEAprojects/dandi/000034/sub-mouse412804/rt_sort/240319/example_amplitude_distribution/amplitudes.npy", root_amp_medians.flatten())
            
        split_coc_clusters = [cluster.split(cluster.root_elec, []) for _ in range(best_gmm.n_components)]  # split() using cluster.root_elec so that no elec is added to splitting elecs
        for label, time in zip(best_gmm.predict(root_amp_medians), cluster._spike_train):
            split_coc_clusters[label]._spike_train.append(time)
        for cluster in split_coc_clusters:
            if len(cluster._spike_train) >= min_coc_n:
                all_split_coc_clusters.append(cluster)
        if verbose:
            print(f"Split cluster {i} into {len(split_coc_clusters)} clusters")
    # endregion
        
    # # Save pval (in si_rec13.ipynb, see end of section Development/Figure finding)
    # np.save(P_INTER_PATH / f"root_elec_{root_elec}.npy", all_p_values)
        
    if verbose:
        print(f"\nTotal: {len(all_coc_clusters)} clusters")
        
    return all_split_coc_clusters

    # Show all root times and remaining root times after forming clusters
    # unit = Unit(0, list({root_time for root_time, root_amp in coc_dict.keys()}), root_elec, RECORDING)
    # amp_kwargs, prob_kwargs = plot_elec_probs(unit)
    # plt.show()

    # unit = Unit(0, list(allowed_root_times), root_elec, RECORDING)
    # plot_elec_probs(unit, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs)
    # plt.show()

def _form_all_clusters(task):
    root_elec, traces_training_ms = task
    coc_clusters = form_coc_clusters(root_elec, traces_training_ms, verbose=False)
    setup_coc_clusters(coc_clusters)
    coc_clusters = filter_clusters(coc_clusters)
    # coc_clusters = merge_coc_clusters(coc_clusters, stop_at=6)
    return coc_clusters

def form_all_clusters(traces_training_ms):
    """
    Form all preliminary propagation sequences (essentially, call form_coc_clusters() on all electrodes)
    """
    
    np.random.seed(1150)
    all_clusters = []
    tasks = [(root_elec, traces_training_ms) for root_elec in range(NUM_ELECS)]
    with Pool(processes=20) as pool:
        for clusters in tqdm(pool.imap_unordered(_form_all_clusters, tasks), total=len(tasks)):
            all_clusters += clusters
    print(f"{len(all_clusters)} sequences before merging") 
    return all_clusters
    
def reassign_spikes(all_clusters, traces_training_ms, min_spikes=10):
    """
    Reassign spikes to preliminary propagation sequences
    """
    assign_spikes_torch(all_clusters, traces_training_ms)  
    torch.cuda.empty_cache()
    all_clusters = [cluster for cluster in all_clusters if len(cluster.spike_train) >= min_spikes]
    all_clusters = setup_coc_clusters_parallel(all_clusters) 
    all_clusters = filter_clusters(all_clusters)
    return all_clusters
# endregion


# region Setup coc clusters
def _setup_coc_clusters_parallel(cluster):
    # Job for setup_coc_clusters_parallel
    setup_cluster(cluster)
    return cluster

def setup_coc_clusters_parallel(coc_clusters):
    """
    Run setup_cluster on coc_clusters with parallel processing
    """
    new_coc_clusters = []
    with Pool(processes=20) as pool:
        for cluster in tqdm(pool.imap(_setup_coc_clusters_parallel, coc_clusters), total=len(coc_clusters)):
            new_coc_clusters.append(cluster)
    return new_coc_clusters

def setup_coc_clusters(coc_clusters, verbose=False):
    # Set important data needed for merging and other analyses
    if verbose:
        coc_clusters = tqdm(coc_clusters)
    
    for cluster in coc_clusters:        
        setup_cluster(cluster)

def setup_cluster(cluster, n_cocs=None): 
    """    
    Parameters:
    n_cocs:
        If not None, setup cluster using randomly selected n_cocs
    
    Previous version
    elec_prob_thresh=0.1:
        Prob on elec needs to cross this to count as part of trunk
    rel_to_closest_elecs=3:
        Set relative amplitudes relative to mean amp of rel_to_closet_elecs elecs
    """
    n_before = N_BEFORE
    n_after = N_AFTER
    
    root_elec = cluster.root_elec
    # all_elecs = [root_elec] + get_nearby_elecs(root_elec, max_elec_dist)
    array_elecs = range(NUM_ELECS)
    
    # Select random cocs
    spike_train = cluster.spike_train
    if n_cocs is not None and n_cocs < len(spike_train):
        spike_train = np.random.choice(spike_train, n_cocs, replace=False)

    # Start extracting stats
    # sum_elec_probs = np.zeros(NUM_ELECS, "float32")  # using mean # (n_elecs,)
    all_elec_probs = []  # (n_cocs, n_elecs)
    all_latencies = []  # (n_cocs, n_elecs)
    all_amp_medians = []  # (n_cocs, n_elecs)
    
    for time in spike_train:
        # Get elec probs
        output_frame = rec_ms_to_output_frame(time)
        this_n_before = n_before if output_frame - n_before >= 0 else output_frame  # Prevents indexing problems
        output_window = OUTPUTS[:, output_frame-this_n_before:output_frame+n_after+1]
        elec_probs = np.max(output_window, axis=1)
        elec_probs[root_elec] = output_window[root_elec, this_n_before]  # max value in window may not be at :time:
        
        all_elec_probs.append(sigmoid(elec_probs))
        
        # Get latencies
        latencies = np.argmax(output_window, axis=1) - this_n_before
        latencies[root_elec] = 0
        all_latencies.append(latencies)
        
        # Get amp/medians
        rec_frame = round(time * SAMP_FREQ)
        pre_medians = calc_pre_median(max(0, rec_frame-n_before))        
        amps = np.abs(TRACES[array_elecs, rec_frame + latencies])
        amp_medians = amps / pre_medians
        all_amp_medians.append(amp_medians)
    
    # Set stats (all for all electroes in array, but store values for self.comp_elecs for fast comparison for assigning spikes)    
    
    # When elecs are only based on inner and outer radius
    # elecs = [root_elec] + get_nearby_elecs(root_elec, max_elec_dist)  # Store sliced array for fast comparision with intraelec merging and assigning spikes
    # cluster.elecs = elecs
    
    # all_elec_probs = sum_elec_probs / len(spike_train) 
    cluster.every_elec_prob = np.array(all_elec_probs).T  # (n_elecs, n_cocs)
    all_elec_probs = np.median(all_elec_probs, axis=0)
    all_elec_probs[all_elec_probs < MIN_ELEC_PROB] = 0
    cluster.all_elec_probs = all_elec_probs  # (n_elecs)
    
    cluster.every_latency = np.array(all_latencies).T  # (n_elecs, n_cocs)
    cluster.all_latencies = np.mean(all_latencies, axis=0)
    # cluster.latencies = cluster.all_latencies[comp_elecs[1:]]  # Don't include root elec since always 0

    cluster.every_amp_median = np.array(all_amp_medians).T  # (n_elecs, n_cocs)
    cluster.all_amp_medians = np.mean(all_amp_medians, axis=0)

    # cluster.amp_medians = cluster.all_amp_medians[comp_elecs]
    
    cluster.formation_spike_train = cluster.spike_train
    
    setup_elec_stats(cluster)
    
    # region Previous version (stats are based on mean footprint, not mean of individual spikes)
    # # Set important data needed for merging, assigning spikes, and other analyses
    # all_elec_probs = extract_detection_probs(cluster)  # (n_cocs, n_chans, n_samples)
    # elec_probs = np.mean(all_elec_probs, axis=0)  # (n_chans, n_samples)
    
    # # Find probabilities used for elec weighting
    # elec_weight_probs = []
    # for probs in elec_probs:  # 1it for each elec. probs: (n_samples)
    #     peak = np.argmax(probs)
    #     elec_weight_probs.append(np.sum(probs[peak-1:peak+2]))
    
    # # Needed for latencies and amplitudes
    # waveforms = extract_waveforms(cluster)
    
    # latencies = np.argmax(elec_probs, axis=1)
    # # latencies = np.argmin(np.mean(waveforms, axis=0), axis=1)
    # cluster.latencies = latencies - elec_probs.shape[1] // 2 # in frames
    
    # # Save for plotting
    # cluster.all_elec_probs = all_elec_probs  
    # cluster.elec_probs = elec_probs
    
    # # Save for merging
    # cluster.elec_weight_probs = np.array(elec_weight_probs)  # (n_chans,)
    # cluster.amplitudes = get_amp_dist(cluster)
    
    # # Save for assigning spikes
    # elecs = get_merge_elecs(cluster.root_elecs[0])
    # elec_weight_probs = cluster.elec_weight_probs[elecs]
    # cluster.elec_weights = elec_weight_probs / np.sum(elec_weight_probs)
    # # cluster.main_elecs = np.flatnonzero(elec_weight_probs >= elec_prob_thresh)
    # cluster.main_elecs = np.flatnonzero(np.max(cluster.elec_probs[elecs], axis=1) >= elec_prob_thresh)
    # cluster.elecs = elecs  # Elecs to watch for comparing latencies and rel amps
    
    # # cluster.elecs = np.flatnonzero(np.max(elec_probs, axis=1)>=prob_thresh)  # No longer needed
    
    # wf_amps = waveforms[:, range(waveforms.shape[1]), (latencies).astype(int)]  # (n_wfs, n_chans)
    # mean_amps = np.abs(np.mean(wf_amps, axis=0))
    
    # cluster.waveforms = waveforms
    # cluster.mean_amps = mean_amps

    # # Save for assigning spikes to increase speed
    # cluster.rel_amps = mean_amps / np.mean(mean_amps[cluster.elecs[:rel_to_closest_elecs]])
    # cluster.latencies_elecs = cluster.latencies[cluster.elecs]
    # cluster.rel_amps_elecs = cluster.rel_amps[cluster.elecs]
    # endregion

def setup_elec_stats(cluster):
    """
    Set cluster.loose_elecs, cluster.inner_loose_elecs, cluster.root_to_stats
    and the root elec
    
    cluster.root_to_stats = {root_elec: [comp_elecs, elec_probs, latencies, amp_medians]}
        Stats are from all_stat[comp_elecs]. Each root_elec has same comp_elecs, but comp_elecs[0] = root_elec
    """
    # root_elec_amp_medians = cluster.all_amp_medians[cluster.root_elecs]
    # root_elec_idx = np.argmax(root_elec_amp_medians)
    # cluster.root_elec = cluster.root_elecs[root_elec_idx]
    # cluster.root_amp_median = root_elec_amp_medians[root_elec_idx]
    
    cluster.root_to_amp_median_std = {root: np.std(cluster.every_amp_median[root, :], ddof=1) for root in cluster.root_elecs}
    cluster.root_amp_median = cluster.all_amp_medians[cluster.root_elec]
    
    # Find elecs
    cluster.loose_elecs = []
    for elec in np.flatnonzero(cluster.all_elec_probs >= LOOSE_THRESH):
        for split_elec in cluster.split_elecs:
            if calc_elec_dist(elec, split_elec) <= INNER_RADIUS:
                cluster.loose_elecs.append(elec)
                break
    
    cluster.inner_loose_elecs = []
    comp_elecs = set(cluster.loose_elecs)  # set() to prevent an elec being added more than once
    # Find inner_loose_elecs and comp_elecs
    for loose_elec in cluster.loose_elecs:
        # Check if loose elec within INNER_RADIUS of any inner elec to be a inner_loose_elec
        for root_elec in cluster.root_elecs:
            if calc_elec_dist(root_elec, loose_elec) <= INNER_RADIUS:
                cluster.inner_loose_elecs.append(loose_elec)
                break  # Add loose_elec only once
        # Add elec's inner elecs to comp_elecs
        for elec in ALL_CLOSEST_ELECS[loose_elec]:
            if calc_elec_dist(elec, loose_elec) <= INNER_RADIUS and cluster.all_elec_probs[elec] > 0:
                comp_elecs.add(elec)
    cluster.min_loose_detections = max(MIN_LOOSE_DETECTIONS_N, MIN_LOOSE_DETECTIONS_R_SPIKES * len(cluster.loose_elecs))
                
    # For each root elec, make separate comp_elecs so that first elec is root_elec (needed for fast access to compare latencies since root elec should not be considered)
    # This is for fast indexing for assigning spikes
    cluster.root_to_stats = {}
    for root_elec in cluster.root_elecs[::-1]:  # Do it in reverse order so that comp_elecs will be set for root_elecs[0] for rest of function
        comp_elecs = [root_elec] + [elec for elec in comp_elecs if elec != root_elec]        
        latencies = cluster.all_latencies[comp_elecs]
        cluster.root_to_stats[root_elec] = (
            comp_elecs, 
            cluster.all_elec_probs[comp_elecs],
            latencies[1:] - latencies[0],  # For individual spikes due to variations in latency, offsetting latency like this may not be accurate. But averaged over hundreds of spikes, it should be fine
            cluster.all_amp_medians[comp_elecs],
            cluster.root_to_amp_median_std[root_elec]
        )
    cluster.comp_elecs = comp_elecs

def relocate_root(cluster, new_root):
    """
    Relocate cluster's root electrode
    """
    cluster.root_elec = new_root
    cluster.root_elecs = [new_root]
    # Adjust latencies
    cluster.every_latency -= cluster.every_latency[new_root, :]
    cluster.all_latencies = np.mean(cluster.every_latency, axis=1)
    # Adjust elecs
    setup_elec_stats(cluster)

def relocate_root_latency(cluster):
    """
    Change root electrode to most negative-latency electrode with:
        Mean latency of -2 frames or less
        Median detection above stringent threhsold
        Mean amplitude that is 80% or more of current root
    If no electrode meets requirements, keep current root electrode
    """
    possible_roots = np.flatnonzero((cluster.all_amp_medians/cluster.all_amp_medians[cluster.root_elec] >= RELOCATE_ROOT_MIN_AMP) & (cluster.all_elec_probs >= STRINGENT_THRESH))
    latencies = cluster.all_latencies[possible_roots]
    min_latency = np.min(latencies)
    if min_latency <= RELOCATE_ROOT_MAX_LATENCY:
        root_elec = possible_roots[np.argmin(latencies)]
        relocate_root(cluster, root_elec)
    
def relocate_root_prob(cluster):
    """
    Change root electrode to electrode with highest median detection score
    """
    new_root = np.argmax(cluster.all_elec_probs)
    # new_root = np.argmax(cluster.all_amp_medians)
    relocate_root(cluster, new_root)
    
def get_amp_medians(cluster, root_elec=None, n_cocs=None, use_formation_spike_train=False):
    """
    Only on root elec
    """
    assert not MEA, "Bug somewhere?"
    
    if root_elec is None:
        root_elec = cluster.root_elecs[0]
        
    if use_formation_spike_train:
        spike_train = cluster.formation_spike_train
    else:
        spike_train = cluster.spike_train
    
    all_amp_medians = []
    if n_cocs is not None and n_cocs < len(spike_train):
        spike_train = np.random.choice(spike_train, n_cocs, replace=False)
        
    for time in spike_train:
        rec_frame = round(time * SAMP_FREQ)
        amp = TRACES[root_elec, rec_frame]
        # window_medians = calc_window_medians(time, [root_elec])[0]
        pre_median = calc_pre_median(rec_frame, [root_elec])
        all_amp_medians.append((np.abs(amp)) / pre_median)
        
    return np.array(all_amp_medians)

def filter_clusters(coc_clusters):
    """
    Return coc_clusters that enough loose and inner loose electrodes
        But not too many to be considered a noise sequence
    """
    filtered_clusters = []
    for cluster in coc_clusters:
        if len(cluster.inner_loose_elecs) >= MIN_INNER_LOOSE_DETECTIONS and \
            len(cluster.loose_elecs) >= MIN_LOOSE_DETECTIONS_N and \
            np.sum(cluster.all_elec_probs >= LOOSE_THRESH) < MIN_ELECS_FOR_SEQ_NOISE:
                filtered_clusters.append(cluster)
    return filtered_clusters
# endregion


# region Merge
def combine_means(x, n, y, m):
    """
    Params:
    x: 
        Mean of sample 1
    n:
        #elements in sample 1
    y:
        Mean of sample 2
    m:
        #elements in sample 2
    
    Returns:
    Mean of combining two samples
    """
    return (x*n + y*m)/(n+m)

class Merge:
    # Represent a CocCluster merge
    def __init__(self, cluster_i, cluster_j) -> None:
        self.cluster_i = cluster_i
        self.cluster_j = cluster_j
        # self.closest_elecs = cluster_i.elecs  # Should not really matter whose elecs since clusters should be close together
        
        # i_probs = cluster_i.elec_weight_probs
        # j_probs = cluster_j.elec_weight_probs
        # # self.elec_probs = (i_probs + j_probs) / 2  # /2 to find average between two elecs
        # self.elec_probs = np.max(np.vstack((i_probs, j_probs)), axis=0)  # Max between two elecs
        
    # def score_latencies(self):

        
    #     return latency_diff
        
    def score_rel_amps(self):
        assert False, "Consider using amp/median"
        
        elecs = self.closest_elecs
        elec_weights = self.get_elec_weights(elecs)
        
        # Clusters' amps relative to different electrodes
        # i_amps = self.cluster_i.mean_amps[elecs]
        # i_rel_amps = i_amps / np.mean(-np.sort(-i_amps)[:3])
        # j_amps = self.cluster_j.mean_amps[elecs]
        # j_rel_amps = j_amps / np.mean(-np.sort(-j_amps)[:3])
            
        # To the same electrodes
        i_amps = self.cluster_i.mean_amps[elecs]
        i_rel_amps = i_amps / np.mean(i_amps[:3])
        j_amps = self.cluster_j.mean_amps[elecs]
        j_rel_amps = j_amps / np.mean(j_amps[:3])
        
        # rel_amp_div = np.min(np.vstack((i_rel_amps, j_rel_amps)), axis=0)
        rel_amp_div = np.mean(np.vstack((i_rel_amps, j_rel_amps)), axis=0)
        
        rel_amp_diff = np.abs(i_rel_amps - j_rel_amps) / rel_amp_div
        rel_amp_diff = elec_weights * rel_amp_diff
        rel_amp_diff = np.sum(rel_amp_diff)
        return rel_amp_diff
    
    def score_amp_dist(self):
        """
        Return p-value of Hartigan's dip test on amplitude distribution
        """
        # all_amps = get_amp_dist(self.cluster_i) + get_amp_dist(self.cluster_j)
        
        assert False, "Amplitudes need to be extracted on the same electrode"
        
        all_amps = self.cluster_i.amplitudes + self.cluster_j.amplitudes
        # Calculate the dip statistic and p-value
        dip, pval = diptest.diptest(np.array(all_amps))
        return pval
        
    def can_merge(self, max_latency_diff, max_rel_amp_diff, min_amp_dist_p):
        return (self.score_latencies() <= max_latency_diff) and (self.score_rel_amps() <= max_rel_amp_diff) and (self.score_amp_dist() >= min_amp_dist_p)
        
        # Incorporate % spike overlap to determine whether or not to merge
        # if not ((self.score_latencies() <= max_latency_diff) and (self.score_rel_amps() <= max_rel_amp_diff)):
        #     return False        
        
        # num_i = len(self.cluster_i.spike_train)
        # num_j = len(self.cluster_j.spike_train)
        # num_overlaps = len(set(self.cluster_i.spike_train).intersection(self.cluster_j.spike_train))
        # return num_overlaps / (num_i + num_j - num_overlaps) >= 0.3

    # def OLD_merge(self):
    #     # region Combine spike trains, but if both clusters detect same spike, only add once
    #     # spike_train_i = self.cluster_i.spike_train
    #     # spike_train_j = self.cluster_j.spike_train
        
    #     # # all_root_amp_medians_i = self.cluster_i.all_root_amp_medians
    #     # # all_root_amp_medians_j = self.cluster_j.all_root_amp_medians
        
    #     # spike_train = [spike_train_i[0]]
    #     # # all_root_amp_medians = [all_root_amp_medians_i[0]]
    #     # i, j = 1, 0
    #     # while i < len(spike_train_i) and j < len(spike_train_j):
    #     #     spike_i, spike_j = spike_train_i[i], spike_train_j[j]
    #     #     if spike_i < spike_j:  # i is next to be added
    #     #         if np.abs(spike_train[-1] - spike_i) >= 0.1: # 1/SAMP_FREQ:  # Ensure not adding same spikes twice (clusters being merged often detect the same spikes) (account for rounding error)
    #     #             spike_train.append(spike_i)
    #     #             # all_root_amp_medians.append(all_root_amp_medians_i[i])
    #     #         i += 1
    #     #     else:  # j is next to be added
    #     #         if np.abs(spike_train[-1] - spike_j) >= 0.1: # 1/SAMP_FREQ: # Ensure not adding same spikes twice (clusters being merged often detect the same spikes) (account for rounding error)
    #     #             spike_train.append(spike_j)
    #     #             # all_root_amp_medians.append(all_root_amp_medians_j[j])
    #     #         j += 1

    #     # # Append remaning elements (only one cluster's spike train can be appended due to while loop)
    #     # if i < len(spike_train_i):
    #     #     spike_train.extend(spike_train_i[i:])
    #     #     # all_root_amp_medians.extend(all_root_amp_medians_i[i:])
    #     # else:
    #     #     spike_train.extend(spike_train_j[j:])
    #     #     # all_root_amp_medians.extend(all_root_amp_medians_j[j:])
        
    #     # # Set new spike train
    #     # try:
    #     #     self.cluster_i._spike_train = spike_train
    #     # except AttributeError:
    #     #     self.cluster_i.spike_train = spike_train
    #     # self.cluster_i.all_root_amp_medians = all_root_amp_medians
    #     # endregion
        
    #     cluster_i = self.cluster_i
    #     cluster_j = self.cluster_j
        
    #     # region Update stats 
    #     n = len(cluster_i._spike_train)
    #     m = len(cluster_j._spike_train) 
        
    #     # Elec probs
    #     # all_elec_probs = combine_means(cluster_i.all_elec_probs, n, cluster_j.all_elec_probs, m)
    #     cluster_i.every_elec_prob = np.concatenate((cluster_i.every_elec_prob, cluster_j.every_elec_prob), axis=1)
    #     all_elec_probs = np.median(cluster_i.every_elec_prob, axis=1)
    #     all_elec_probs[all_elec_probs < MIN_ELEC_PROB] = 0
    #     cluster_i.all_elec_probs = all_elec_probs  # (n_elecs)
    
    #     # cluster_i.elec_probs = cluster_i.all_elec_probs[elecs]
        
    #     # Latencies 
    #     every_latency = cluster_j.every_latency
    #     # all_latencies = cluster_j.all_latencies 
    #     if cluster_i.root_elecs[0] != cluster_j.root_elecs[0]:  #  Need to adjust cluster_j latencies to cluster_i)
    #         every_latency -= every_latency[cluster_i.root_elecs[0], :]
    #         # all_latencies = np.mean(every_latency, axis=1)
    #     cluster_i.every_latency = np.concatenate((cluster_i.every_latency, every_latency), axis=1)
    #     # cluster_i.all_latencies = combine_means(cluster_i.all_latencies, n, all_latencies, m)
    #     cluster_i.all_latencies = np.median(cluster_i.every_latency, axis=1)
    #     # cluster_i.all_latencies = np.mean(cluster_i.every_latency, axis=1)
    #     # cluster_i.latencies = cluster_i.all_latencies[elecs[1:]]
        
    #     # Amp/medians
    #     cluster_i.every_amp_median = np.concatenate((cluster_i.every_amp_median, cluster_j.every_amp_median), axis=1)
    #     cluster_i.all_amp_medians = np.median(cluster_i.every_amp_median, axis=1)
    #     # cluster_i.all_amp_medians = combine_means(cluster_i.all_amp_medians, n, cluster_j.all_amp_medians, m)
    #     # luster_i.amp_medians = cluster_i.all_amp_medians[elecs]
    #     # endregion
        
    #     # try:
    #     cluster_i._spike_train.extend(cluster_j._spike_train)
    #     # except AttributeError:
    #     #     self.cluster_i.spike_train.extend(self.cluster_j.spike_train)
    #     #     self.cluster_i.spike_train = np.sort(self.cluster_i.spike_train)  # If accessing spike train this way, keep it sorted

    #     # Update root elecs
    #     cluster_i_elecs = set(cluster_i.root_elecs)
    #     for elec in cluster_j.root_elecs:
    #         if elec not in cluster_i_elecs:
    #             cluster_i.root_elecs.append(elec)
            
    #     setup_elec_stats(cluster_i)
    #     # setup_cluster(self.cluster_i)  # Update stats
        
    #     return cluster_j  # Return to update dead_clusters

    def merge(self):              
        # region Combine spike trains, but if both clusters detect same spike, only add once
        
        # Now handled in merge_coc_clusters
        # if self.cluster_i.root_amp_median >= self.cluster_j.root_amp_median:
        #     cluster_i = self.cluster_i
        #     cluster_j = self.cluster_j
        # else:
        #     cluster_i = self.cluster_j
        #     cluster_j = self.cluster_i
        #     self.cluster_i = cluster_i
        #     self.cluster_j = cluster_j
        cluster_i = self.cluster_i
        cluster_j = self.cluster_j

        spike_train_i = cluster_i.spike_train       
        spike_train_j = cluster_j.spike_train
        
        spike_train = [spike_train_i[0]]        
        every_elec_prob = [cluster_i.every_elec_prob[:, 0]]
        every_latency = [cluster_i.every_latency[:, 0]]
        every_amp_median = [cluster_i.every_amp_median[:, 0]]
        
        i, j = 1, 0
        while i < len(spike_train_i) and j < len(spike_train_j):
            spike_i, spike_j = spike_train_i[i], spike_train_j[j]
            if spike_i < spike_j:  # i is next to be added
                if spike_i - spike_train[-1] > OVERLAP_TIME: # 1/SAMP_FREQ:  # Ensure not adding same spikes twice (clusters being merged often detect the same spikes) (account for rounding error)
                    spike_train.append(spike_i)
                    every_elec_prob.append(cluster_i.every_elec_prob[:, i])
                    every_latency.append(cluster_i.every_latency[:, i])
                    every_amp_median.append(cluster_i.every_amp_median[:, i])
                i += 1
            else:  # j is next to be added
                if spike_j - spike_train[-1] > OVERLAP_TIME: # 1/SAMP_FREQ: # Ensure not adding same spikes twice (clusters being merged often detect the same spikes) (account for rounding error)
                    spike_train.append(spike_j)
                    every_elec_prob.append(cluster_j.every_elec_prob[:, j])
                    
                    latency = cluster_j.every_latency[:, j]
                    if cluster_i.root_elecs[0] != cluster_j.root_elecs[0]:  #  Need to adjust cluster_j latencies to cluster_i
                        latency = latency - latency[cluster_i.root_elecs[0]]
                    every_latency.append(latency)
                    
                    every_amp_median.append(cluster_j.every_amp_median[:, j])
                j += 1

        # Append remaning elements (only one cluster's spike train can be appended due to while loop)
        if i < len(spike_train_i):
            spike_train.extend(spike_train_i[i:])
            every_elec_prob.extend(cluster_i.every_elec_prob[:, i:].T)
            every_latency.extend(cluster_i.every_latency[:, i:].T)
            every_amp_median.extend(cluster_i.every_amp_median[:, i:].T)
        else:
            spike_train.extend(spike_train_j[j:])
            every_elec_prob.extend(cluster_j.every_elec_prob[:, j:].T)
            every_latency.extend(cluster_j.every_latency[:, j:].T)
            every_amp_median.extend(cluster_j.every_amp_median[:, j:].T)
        
        # Set new spike train
        # try:
        #     self.cluster_i._spike_train = spike_train
        # except AttributeError:
        #     self.cluster_i.spike_train = spike_train
        cluster_i._spike_train = spike_train
        # endregion
        
        # region Update stats 
        # n = len(cluster_i._spike_train)
        # m = len(cluster_j._spike_train) 
        
        # Update root elecs
        cluster_i_elecs = set(cluster_i.root_elecs)
        for elec in cluster_j.root_elecs:
            if elec not in cluster_i_elecs:
                cluster_i.root_elecs.append(elec)
                
        # Elec probs
        # all_elec_probs = combine_means(cluster_i.all_elec_probs, n, cluster_j.all_elec_probs, m)
        # cluster_i.every_elec_prob = np.concatenate((cluster_i.every_elec_prob, cluster_j.every_elec_prob), axis=1)
        cluster_i.every_elec_prob = np.vstack(every_elec_prob).T
        all_elec_probs = np.median(cluster_i.every_elec_prob, axis=1)
        all_elec_probs[all_elec_probs < MIN_ELEC_PROB] = 0
        cluster_i.all_elec_probs = all_elec_probs  # (n_elecs)
    
        # cluster_i.elec_probs = cluster_i.all_elec_probs[elecs]
        
        # Latencies 
        # every_latency = cluster_j.every_latency
        # # all_latencies = cluster_j.all_latencies 
        # if cluster_i.root_elecs[0] != cluster_j.root_elecs[0]:  #  Need to adjust cluster_j latencies to cluster_i)
        #     every_latency -= every_latency[cluster_i.root_elecs[0], :]
        #     # all_latencies = np.mean(every_latency, axis=1)
        # cluster_i.every_latency = np.concatenate((cluster_i.every_latency, every_latency), axis=1)
        # cluster_i.all_latencies = combine_means(cluster_i.all_latencies, n, all_latencies, m)
        cluster_i.every_latency = np.vstack(every_latency).T
        # cluster_i.all_latencies = np.median(cluster_i.every_latency, axis=1)
        cluster_i.all_latencies = np.mean(cluster_i.every_latency, axis=1)
        # cluster_i.latencies = cluster_i.all_latencies[elecs[1:]]
        
        # Amp/medians
        # cluster_i.every_amp_median = np.concatenate((cluster_i.every_amp_median, cluster_j.every_amp_median), axis=1)
        cluster_i.every_amp_median = np.vstack(every_amp_median).T
        cluster_i.all_amp_medians = np.mean(cluster_i.every_amp_median, axis=1)
        cluster_i.root_to_amp_median_std = {root: np.std(cluster_i.every_amp_median[root, :], ddof=1) for root in cluster_i.root_elecs}
        cluster_i.root_amp_median = cluster_i.all_amp_medians[cluster_i.root_elec]
        # cluster_i.all_amp_medians = combine_means(cluster_i.all_amp_medians, n, cluster_j.all_amp_medians, m)
        # luster_i.amp_medians = cluster_i.all_amp_medians[elecs]
        # endregion
                
        # try:
        # cluster_i._spike_train.extend(cluster_j._spike_train)
        # except AttributeError:
        #     self.cluster_i.spike_train.extend(self.cluster_j.spike_train)
        #     self.cluster_i.spike_train = np.sort(self.cluster_i.spike_train)  # If accessing spike train this way, keep it sorted
                
            
        setup_elec_stats(cluster_i)
        # setup_cluster(self.cluster_i)  # Update stats
        
        return cluster_j  # Return to update dead_clusters

    def is_better(self, other, max_latency_diff=1, max_rel_amp_diff=1):
        """
        Determine if self is a better merge than other
        
        Parameters
        ----------
        max_latency_diff:
            Scale latency diff by this to normalize it, so it can be compared to rel amp
        max_rel_amp_diff:
            Scale rel amp diff by this to normalize it, so it can be compared to latency 
        """
        
        self_diff = self.score_latencies() / max_latency_diff + self.score_rel_amps() / max_rel_amp_diff# + (1-self.score_amp_dist())
        other_diff = other.score_latencies() / max_latency_diff + other.score_rel_amps() / max_rel_amp_diff #+ (1-other.score_amp_dist())
        return self_diff < other_diff

    def summarize(self):
        """
        Print merge metrics
        """
        print(f"Latency diff: {self.score_latencies():.2f}. Rel amp diff: {self.score_rel_amps():.2f}")
        print(f"Amp dist p-value {self.score_amp_dist():.4f}")

        
def merge_verbose(merge, update_history=True):
    """
    Verbose for merge_coc_clusters
    
    Params:
    update_history:
        If True, history of clusters will be updated
        False is for when no merge is found, but still want verbose
    """
    cluster_i, cluster_j = merge.cluster_i, merge.cluster_j
    
    if hasattr(cluster_i, "merge_history"):
        message = f"\nMerged {cluster_i.merge_history} with "
        if update_history:
            cluster_i.merge_history.append(cluster_j.idx)                
    else:
        message = f"\nMerged {cluster_i.idx} with "
        if update_history:
            cluster_i.merge_history = [cluster_i.idx, cluster_j.idx]
    
    if hasattr(cluster_j, "merge_history"):
        message += str(cluster_j.merge_history)
        if update_history:
            cluster_i.merge_history += cluster_j.merge_history[1:]
    else:
        message += f"{cluster_j.idx}"
    print(message)
    print(f"Latency diff: {merge.latency_diff:.2f}. Amp median diff: {merge.amp_median_diff:.2f}")
    print(f"Amp dist p-value {merge.dip_p:.4f}")

    print(f"#spikes:")
    num_overlaps = Comparison.count_matching_events(cluster_i.spike_train, cluster_j.spike_train, delta=OVERLAP_TIME)
    # num_overlaps = len(set(cluster_i.spike_train).intersection(cluster_j.spike_train))
    print(f"Merge base: {len(cluster_i.spike_train)}, Add: {len(cluster_j.spike_train)}, Overlaps: {num_overlaps}")
    
    # Find ISI violations after merging
    # cat_spikes = np.sort(np.concatenate((cluster_i.spike_train, cluster_j.spike_train)))
    # diff = np.diff(cat_spikes)
    # num_viols = np.sum(diff <= 1.5)
    # print(f"ISI viols: {num_viols}")
    
    # Plot footprints
    # amp_kwargs, prob_kwargs = plot_elec_probs(cluster_i, idx=cluster_i.idx)
    # plt.show()
    # plot_elec_probs(cluster_j, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs, idx=cluster_j.idx)
    # plt.show()   
    
    # # Plot amp distribution
    # all_amps = get_amp_dist(cluster_i) + get_amp_dist(cluster_j)
    # plot_amp_dist(np.array(all_amps))
    # plt.show()

def merge_coc_clusters(coc_clusters,
                       auto_setup_coc_clusters=False, verbose=False,
                       stop_at=None):    
    """
    Params
        stop_at: None or int
            If remaining clusters that could be merged is equal to stop_at, stop merging even if more merging is possible
    
    """
    
    max_latency_diff = MAX_LATENCY_DIFF_SEQUENCES
    max_amp_median_diff = MAX_AMP_MEDIAN_DIFF_SEQUENCES
    min_amp_dist_p = MIN_AMP_DIST_P
        
    if auto_setup_coc_clusters:
        setup_coc_clusters(coc_clusters, verbose=verbose)

    for idx, cluster in enumerate(coc_clusters):
        cluster.idx = idx
        
    dead_clusters = set()
    while True:
        # Find best merge
        best_merge = None
        best_unmerge = None  # Best merge that cannot merge (for final verbose)
        
        if stop_at is not None and len(coc_clusters) - len(dead_clusters) == stop_at:
            break
        
        for i in range(len(coc_clusters)):              
            # Load cluster i     
            cluster_i = coc_clusters[i]
            if cluster_i in dead_clusters:
                continue
            
            for j in range(i+1, len(coc_clusters)):    
                # Load cluster j        
                cluster_j = coc_clusters[j]
                if cluster_j in dead_clusters:
                    continue
                
                # Check if root elecs are close enough (find max dist between i_root_elecs and j_root_elecs)
                max_dist = 0
                for root_i in cluster_i.root_elecs:
                    for root_j in cluster_j.root_elecs:
                        max_dist = max(max_dist, calc_elec_dist(root_i, root_j))
                        if max_dist >= INNER_RADIUS:
                            break
                    else:
                        continue
                    break
                if max_dist >= INNER_RADIUS:
                    continue
                    
                # Check if enough overlapping loose electrodes
                total_loose = len(set(cluster_i.loose_elecs).union(cluster_j.loose_elecs))
                num_loose_overlaps = len(set(cluster_i.loose_elecs).intersection(cluster_j.loose_elecs))
                # print(num_loose_overlaps, total_loose, num_loose_overlaps/total_loose)
                if num_loose_overlaps < MIN_LOOSE_DETECTIONS_N or num_loose_overlaps/total_loose < MIN_LOOSE_DETECTIONS_R_SEQUENCES:
                    continue
                num_inner_loose_overlaps = len(set(cluster_i.inner_loose_elecs).intersection(cluster_j.inner_loose_elecs))
                if num_inner_loose_overlaps < MIN_INNER_LOOSE_DETECTIONS:
                    continue
                    
                # if calc_elec_dist(cluster_i.root_elecs[0], cluster_j.root_elecs[0]) > max_root_elec_dist:
                #     continue
        
                # Get elecs for comparison (do it this way so comp_elecs[0] is root elec)
                if cluster_i.root_amp_median >= cluster_j.root_amp_median: ## Find which cluster's root amp to use (use one with higher amplitude)
                    i_comp_elecs = cluster_i.comp_elecs
                    i_comp_elecs_set = set(i_comp_elecs)
                    comp_elecs = i_comp_elecs + [elec for elec in cluster_j.comp_elecs if elec not in i_comp_elecs_set]
                else:
                    j_comp_elecs = cluster_j.comp_elecs
                    j_comp_elecs_set = set(j_comp_elecs)
                    comp_elecs = j_comp_elecs + [elec for elec in cluster_i.comp_elecs if elec not in j_comp_elecs_set]
                                
                # Get elec probs
                i_elec_probs = cluster_i.all_elec_probs[comp_elecs]
                j_elec_probs = cluster_j.all_elec_probs[comp_elecs]
                
                # Compare latencies                 
                i_latencies = cluster_i.all_latencies[comp_elecs][1:]
                j_latencies = cluster_j.all_latencies[comp_elecs][1:] - cluster_j.all_latencies[comp_elecs[0]]  # Relative to same electrode as cluster_i
                elec_weights = get_elec_weights(i_elec_probs, j_elec_probs, for_latencies=True)
                latency_diff = np.abs(i_latencies - j_latencies)
                latency_diff = np.clip(latency_diff, a_min=None, a_max=CLIP_LATENCY_DIFF)
                latency_diff = np.sum(latency_diff * elec_weights)
                # latency_diff = np.sum(np.abs(i_latencies - j_latencies) * elec_weights)
                if latency_diff > max_latency_diff:
                    continue
                
                # Compare amp/medians
                i_amp_medians = cluster_i.all_amp_medians[comp_elecs]
                j_amp_medians = cluster_j.all_amp_medians[comp_elecs]                
                elec_weights = get_elec_weights(i_elec_probs, j_elec_probs, for_latencies=False)
                amp_median_div = (i_amp_medians + j_amp_medians) / 2
                amp_median_diff = np.abs((i_amp_medians - j_amp_medians)) / amp_median_div
                amp_median_diff = np.clip(amp_median_diff, a_min=None, a_max=CLIP_AMP_MEDIAN_DIFF)
                amp_median_diff = np.sum(amp_median_diff * elec_weights)
                if amp_median_diff > max_amp_median_diff:
                    continue                
                
                # Test if merge is bimodal
                are_unimodal = True
                pval = np.inf  # Called p-value but is double z-score test now
                for root_elec in set(cluster_i.root_elecs + cluster_j.root_elecs):
                    # root_amps_i = cluster_i.every_amp_median[root_elec, :] # get_amp_medians(cluster_i, root_elec=root_elec)
                    # root_amps_j = cluster_j.every_amp_median[root_elec, :] # get_amp_medians(cluster_j, root_elec=root_elec)
                    # Skip if distribution is not unimodal before merging
                    # if diptest(root_amps_i)[1] < min_amp_dist_p or diptest(root_amps_j)[1] < min_amp_dist_p:
                    #     pval = -1
                    #     continue      
                    
                    # Dip test              
                    # dip, pval = diptest(np.concatenate([root_amps_i, root_amps_j]))
                    # if pval < min_amp_dist_p:
                    #     are_unimodal = False
                    #     break
                    
                    mean_i = cluster_i.all_amp_medians[root_elec]
                    std_i = cluster_i.root_to_amp_median_std[root_elec] if root_elec in cluster_i.root_elecs else np.std(cluster_i.every_amp_median[root_elec], ddof=1)
                    mean_j = cluster_j.all_amp_medians[root_elec]
                    std_j = cluster_j.root_to_amp_median_std[root_elec] if root_elec in cluster_j.root_elecs else np.std(cluster_j.every_amp_median[root_elec], ddof=1)
                    mean_diff = np.abs(mean_i - mean_j)
                    two_z_score = pval = max(mean_diff/std_i, mean_diff/std_j)
                    if two_z_score > MAX_ROOT_AMP_MEDIAN_STD_SEQUENCES:
                        are_unimodal = False
                        break
                    
                if not are_unimodal:
                    continue
                                
                # Calculate quality of merge
                cur_merge = Merge(cluster_i, cluster_j)
                score = latency_diff / max_latency_diff + amp_median_diff / max_amp_median_diff 
                if best_merge is None or score < best_merge.score:
                    best_merge = cur_merge
                    best_merge.score = score
                    best_merge.latency_diff = latency_diff
                    best_merge.amp_median_diff = amp_median_diff
                    best_merge.dip_p = pval
                
                # if not cur_merge.can_merge(max_latency_diff, max_rel_amp_diff, min_amp_dist_p):
                #     if verbose and (best_unmerge is None or cur_merge.is_better(best_unmerge)):
                #         best_unmerge = cur_merge
                #     continue
                # if best_merge is None or cur_merge.is_better(best_merge, max_latency_diff, max_rel_amp_diff):
                #     best_merge = cur_merge
                    
        # If no merges are good enough
        if best_merge is None:
        # if not best_merge.can_merge(max_latency_diff, max_rel_amp_diff):
            # if verbose:
            #     print(f"\nNo merge found. Next best merge:")
            #     merge_verbose(best_unmerge, update_history=False)
            break
        
        ## Merge best merge
        # Possibly switch cluster_i and cluster_j if cluster_j has larger root amp median
        if best_merge.cluster_j.root_amp_median > best_merge.cluster_i.root_amp_median:
            cluster_j = best_merge.cluster_j
            best_merge.cluster_j = best_merge.cluster_i
            best_merge.cluster_i = cluster_j
        
        if verbose:
            merge_verbose(best_merge)
        
        dead_clusters.add(best_merge.merge())
        if verbose:
            print(f"After merging: {len(best_merge.cluster_i.spike_train)}")
            
    merged_clusters = [cluster for cluster in coc_clusters if cluster not in dead_clusters]
    
    if verbose:       
        print(f"\nFormed {len(merged_clusters)} merged clusters:")
        for m, cluster in enumerate(merged_clusters):
            # message = f"cluster {m}: {cluster.idx}"
            # if hasattr(cluster, "merge_history"):
            #     message += f",{cluster.merge_history}"
            # print(message)
            
            # Without +[]
            if hasattr(cluster, "merge_history"):
                print(f"cluster {m}: {cluster.merge_history}")
            else:
                print(f"cluster {m}: {cluster.idx}")
        # print(f"Formed {len(merged_clusters)} merged clusters")  # Reprint this because jupyter notebook cuts of middle of long outputs
    return merged_clusters

def _intra_merge(clusters):
    clusters = merge_coc_clusters(clusters, auto_setup_coc_clusters=False, verbose=False)
    for cluster in clusters:
        relocate_root_latency(cluster) 
    return clusters

def intra_merge(all_clusters):
    """
    intra = merge clusters with same root electrode
    """
    
    root_elec_to_clusters = {}
    for cluster in all_clusters:
        if cluster.root_elec not in root_elec_to_clusters:
            root_elec_to_clusters[cluster.root_elec] = [cluster]
        else:
            root_elec_to_clusters[cluster.root_elec].append(cluster)
    tasks = root_elec_to_clusters.values()

    intra_merged_clusters = []
    with Pool(processes=20) as pool:
        for clusters in tqdm(pool.imap_unordered(_intra_merge, tasks), total=len(tasks)):
            intra_merged_clusters.extend(clusters)
    print(f"{len(intra_merged_clusters)} sequences after first merging") 
    return intra_merged_clusters

def inter_merge(intra_merged_clusters, min_spikes=10):
    """
    inter = merge clusters with different root electrodes
    """
    
    merged_sequences = merge_coc_clusters(intra_merged_clusters, auto_setup_coc_clusters=False, verbose=True)
    merged_sequences = [seq for seq in merged_sequences if len(seq.spike_train) >= min_spikes]
    merged_sequences = filter_clusters(merged_sequences)
    for seq in merged_sequences:
        relocate_root_prob(seq)
    merged_sequences = order_sequences(merged_sequences)
    
    for seq in merged_sequences:  # Set formation spike train
        seq.formation_spike_train = seq.spike_train
        
    print(f"{len(merged_sequences)} sequences after second merging") 
    return merged_sequences
# endregion


# region Assign spikes
def get_isi_viol_p(cluster, isi_viol=1.5):
    spike_train = cluster.spike_train
    diff = np.diff(spike_train)
    viols = np.sum(diff <= isi_viol)
    return viols / len(spike_train) * 100

def get_spike_match(cluster, root_time,
                    elec_prob_thresh=0.1, elec_prob_mask=0.03,
                    rel_to_closest_elecs=3,
                    min_coacs_r=0.5, max_latency_diff=2.51, max_rel_amp_diff=0.40):
    """
    Return how well spike at :param time: matches with :param unit:
    
    Params:
    elec_prob_thresh:
        Prob on elec needs to cross this to count as coactivation
    max_latency_diff
        Used for determining size of extraction window 
        
        
    Returns:
    ratio of elecs that have detection, latency diff, rel amp diff
    """
    
    # Load elecs
    main_elecs = cluster.main_elecs
    elecs = cluster.elecs
    
    # Load cluster stats
    cluster_latencies = cluster.latencies_elecs 
    cluster_rel_amps = cluster.rel_amps_elecs  
    
    # Calculate extraction window n_before and n_after
    #+1 for good measure
    n_before = round(np.min(cluster_latencies)) + ceil(max_latency_diff) + 1  # Use these signs so n_before is positive
    n_before = max(1, n_before)  # Ensure at least one frame n_before
    n_after = round(np.max(cluster_latencies)) + ceil(max_latency_diff) + 1
    n_after = max(1, n_after)  

    # Extract latencies
    output_frame = rec_ms_to_output_frame(root_time)
    output_window = OUTPUTS[elecs, max(0, output_frame-n_before):output_frame+n_after+1]
    latencies = np.argmax(output_window, axis=1) 
        
    # Extract probabilities    
    elec_probs = sigmoid(output_window[range(latencies.size), latencies])
    
    # Calculate coacs ratio
    coacs_r = sum(elec_probs >= elec_prob_thresh)/len(main_elecs)
    if coacs_r < min_coacs_r:
        return coacs_r, np.inf, np.inf
    
    # Calculate elec weights
    elec_weights = (cluster.elec_weights + elec_probs) / 2
    above_mask = elec_weights >= elec_prob_mask  # 11/19/23  This should be after finding weighted values
    
    new_elec_weights = elec_weights[above_mask]
    elec_weights[above_mask] = new_elec_weights / np.sum(new_elec_weights)
    elec_weights[~above_mask] = 0
    
    # Calculate latency diff
    latency_diff = np.sum(np.abs(cluster_latencies - (latencies - n_before)) * elec_weights)
    if latency_diff > max_latency_diff:
        return coacs_r, latency_diff, np.inf
    
    # Extract rel amps
    rec_frame = round(root_time * SAMP_FREQ)
    rec_window = TRACES[elecs, max(0, rec_frame-n_before):rec_frame+n_after+1]  # Not need max for end of slice since numpy ok with it
    amps = np.abs(rec_window[range(len(latencies)), latencies])
    rel_amps = amps / np.mean(amps[:rel_to_closest_elecs])     
    
    # Calculate rel amp diff
    rel_amp_diff = np.abs(cluster_rel_amps - rel_amps) / cluster_rel_amps
    rel_amp_diff = np.sum((rel_amp_diff * elec_weights))
    
    return coacs_r, latency_diff, rel_amp_diff

def get_elec_weights(elec_probs_i, elec_probs_j, for_latencies=False, min_prob=0.03):
    """
    Need to do this before scoring match with :other:
    
    for_latencies:
        If True, only include elec_probs[1:] 
        Else (if for amp medians), multiply elec_probs[0] by 2 to increase weight on root elec
    min_r:
        Elec probabilities below min_r are set to 0
    """
    if for_latencies:
        elec_probs = (elec_probs_i[1:] + elec_probs_j[1:]) / 2
    else:
        elec_probs = (elec_probs_i + elec_probs_j) / 2
        # elec_probs[0] *= 2  # To weight root elec more
    # elec_probs[elec_probs < min_prob] = 0  
    return elec_probs / np.sum(elec_probs)

def assign_spikes(all_units, time_frame, interelec=False,
                  max_latency_diff=2.51, max_amp_median_diff=0.35,                    
                  overlap_dist=50, 
                  
                  only_max_amps=True,                  
                  verbose=False):
    raise NotImplementedError("Use assign_spikes_torch()")
    
    """
    OLD 
        Spike splitting pseudocode:
        1. Have units watch root_elecs for DL detections
        2. When detection on elec, assign spike to unit with best footprint match
        If len(root_elecs)>1: unit watches multiple elecs, but ignores other elecs
        if detects a spike within Xms of another elec (prevent counting same spike multiple times)
        
        Leaves spikes in a buffer zone. Only split spikes within Xms and Xum
        Remove all in front of spike_buffer that occur too before new spike, leave the rest
        
        Attempted method: Fails. Hard to account for when a unit belongs to multiple split groups
            Rationale: Tries to account for edge case of spikes 0.1, 0.3, 0.45 when overlap_time=0.4. 
            Naive method would split 0.1 and 0.3, but perhaps 0.3 should be split with 0.45
        
    OLD
        Spike splitting pseudocode:
            Before running through recording:
        1. Determine which units are in the same split group
        2. Create dict[unit] = split group id
            While running through recording:
        1. When unit detects spike, fill in dict[split group id] = [(unit, spike)]
            a. If new spike is not within overlap_time, split spikes
        
            Actually split spikes:
        1. Form two spike clusters: 
            1) spikes closest to earliest spike
            2) spikes closest to latest spike (only if spikes are within overlap_time of latest spike)
        2. In cluster 1, assign spike to unit with best match
        3. Remove spikes in cluster 1 from dict[split group id]    
        
        At end, split again 
        
        Units can belong to more than split groups. unit.spike_train is a set, so if a unit
        is the best match for a spike in multiple groups, the spike is only added once
        
    CURRENT 
        Spike splitting pseudocode:
        1. Add detected spikes to buffer
        2. If newest spike added is farther than overlap_time with oldest spike, start splitting (when implemented in real time, this would be after 0.1ms pass):
            a. Select spikes that are not within overlap_time of newest spike or closer to oldest spike 
                (Unselected spikes remain in buffer)
            b. Find which spike has highest overlap score 
            c. Calculate the number of units (X) that are allowed to detect spike by extracting window of DL output probs around this spike (see code for details)
            d. For spikes within overlap_dist of best spike, assign spike to top X units with highest overlap scores and demove other units
            e. Repeat step b until no spikes remain
        
    Params:
    inter_elec:
        If True, assume all_units contains different root elecs (slower, but needed for interelec spike splitting)
    """    
    n_before = N_BEFORE
    n_after = N_AFTER
    
    max_latency_diff = MAX_LATENCY_DIFF_SPIKES
    max_amp_median_diff = MAX_AMP_MEDIAN_DIFF
    
    clip_latency_diff = CLIP_LATENCY_DIFF
    clip_amp_median_diff = CLIP_AMP_MEDIAN_DIFF
    
           
    for unit in all_units:        
        unit._spike_train = []
        # unit._spike_scores = []  # For spike splitting after assigning spikes
        unit._elec_train = []  # Indicates which root elec's detection led to spike in unit._spike_train
        
    # For ith elec, which units are watching
    elec_watchers = {}
    elec_to_seq_elecs = {}  # For each elec if stringent threshold crossing, only access data for electrodes that contain a sequence (seq_elecs)
    for unit in all_units:
        # unit.elec_to_stats = {}  # Now done in set_elec_stats - Original: Cache so don't have to keep slicing for spike match
        for elec in unit.root_elecs:
            if elec not in elec_watchers:
                elec_watchers[elec] = [unit]
            else:
                elec_watchers[elec].append(unit)                  
            
            if elec not in elec_to_seq_elecs:
                elec_to_seq_elecs[elec] = set(unit.comp_elecs)
            else:
                elec_to_seq_elecs[elec].update(unit.comp_elecs)
                
        unit.time_to_spike_match = {} # For testing with time_to_spike_match, {spike: [root_elec, num_inner_loose_elecs, num_loose_elecs, latency diff, amplitude diff, match score]}
                
    elec_to_seq_elecs_dict = {}  # For fast indexing elec_to_seq_elecs to form root_to_assign_spikes_stats for each unit
    for elec, seq_elecs in elec_to_seq_elecs.items():
        seq_elecs = list(seq_elecs)
        elec_to_seq_elecs[elec] = seq_elecs
        elec_to_seq_elecs_dict[elec] = {e: idx for idx, e in enumerate(seq_elecs)} 

    # For each unit, convert elec ind for entire array to elec ind in to elec_to_seq_elecs (comp_elecs)
    for unit in all_units:
        unit.assign_spikes_root_to_stats = {}  # root_to_stats for assigning spikes faster
        for root_elec in unit.root_elecs:
            elec_to_seq_idx = elec_to_seq_elecs_dict[root_elec]
            
            comp_elecs, elec_probs, latencies, amp_medians, amp_median_std = unit.root_to_stats[root_elec]
            comp_elecs = [elec_to_seq_idx[root_elec]] + [elec_to_seq_idx[e] for e in comp_elecs if e != root_elec]  # root_elec is first elec
            
            loose_elecs = [elec_to_seq_idx[e] for e in unit.loose_elecs]
            inner_loose_elecs = [elec_to_seq_idx[e] for e in unit.inner_loose_elecs]
            
            latency_elec_weights = elec_probs[1:] / np.sum(elec_probs[1:])
            amp_elec_weights = elec_probs / np.sum(elec_probs)
            
            # For max amp requirement            
            inner_loose_elec_to_comp_elec_idx = {e: idx for idx, e in enumerate(comp_elecs)}
            inner_loose_elecs_set = {inner_loose_elec_to_comp_elec_idx[e] for e in inner_loose_elecs}  # inner_loose_elecs relative to comp_elecs ind
            
            unit.assign_spikes_root_to_stats[root_elec] = (
                loose_elecs, inner_loose_elecs, comp_elecs, 
                inner_loose_elecs_set,
                latency_elec_weights, amp_elec_weights, 
                latencies, amp_medians, amp_median_std
            )
    
    elec_to_outer_elecs = {} 
    for elec in elec_watchers:        
        elec_to_outer_elecs[elec] = [elec] + get_nearby_elecs(elec, OUTER_RADIUS)

    # Start watching for spikes
    spike_buffer = []  # Store spikes before they have been assigned
    
    all_crossings_times = [c[1] for c in ALL_CROSSINGS]
    start_idx = np.searchsorted(all_crossings_times, time_frame[0], side="left")
    end_idx = np.searchsorted(all_crossings_times, time_frame[1], side="right")
    crossings = ALL_CROSSINGS[start_idx:end_idx]
    if verbose:
        crossings = tqdm(crossings)
    
    # Consider recalculating pre amps and medians every thresh ms (actually seems to be slower)
    # thresh = 1
    # prev_time = -np.inf
    
    for elec, time, amp in crossings:
        # if time not in time_to_spike_match:  # Handle multiple elecs detecting a spike at the same time
        #     time_to_spike_match[time] = []
        
        if elec not in elec_watchers:  # No units are watching elec
            continue
        
        # if time - prev_time >= thresh:
        #     rec_frame = round(time * SAMP_FREQ)
        #     pre_means, pre_medians = calc_pre_median(rec_frame-n_before)
        #     prev_time = time
        
        # inner_elecs, outer_elecs, elecs = elec_to_nearby_elecs[elec]
        seq_elecs = elec_to_seq_elecs[elec]   
        
        output_frame = rec_ms_to_output_frame(time)
        
        # region Check for all electrode array getting detection (noise spike)
        noise_probs = OUTPUTS[:, output_frame]
        if np.sum(noise_probs >= LOOSE_THRESH_LOGIT) >= MIN_ELECS_FOR_ARRAY_NOISE:
            continue
        # endregion
        
        this_n_before = n_before if output_frame - n_before >= 0 else output_frame  # Prevents indexing problems. TODO: Consider removing for speed and just not detecting spikes in first N_BEFORE frames
        # output_window = OUTPUTS[:, output_frame-this_n_before:output_frame+n_after+1]
        output_window = OUTPUTS[seq_elecs, output_frame-this_n_before:output_frame+n_after+1]
        all_elec_probs = sigmoid(np.max(output_window, axis=1))
         
        spike_num_elecs = np.sum(all_elec_probs >= LOOSE_THRESH)
         
        # Check if enough codetections
        # if np.sum(all_elec_probs[inner_elecs] >= min_coc_prob) < min_inner_cocs:
        #     continue
        
        # if np.sum(all_elec_probs[outer_elecs] >= min_coc_prob) < min_outer_cocs:
        #     continue
        
        # Intraelectrode spike splitting (all have same root_elec and therefore nearby elecs, so only extract data for this spike once)
        # elecs = elec_watchers[elec][0].elecs
        
        # Get elec probs
        # output_frame = rec_ms_to_output_frame(time)
        # this_n_before = n_before if output_frame - n_before >= 0 else output_frame  # Prevents indexing problems
        # output_window = OUTPUTS[elecs, output_frame-this_n_before:output_frame+n_after+1]
        # output_window = output_window[:]
        # elec_probs = sigmoid(np.max(output_window, axis=1))
        
        # Get latencies
        all_latencies = np.argmax(output_window, axis=1) - this_n_before 
        
        # Get amp/medians
        rec_frame = round(time * SAMP_FREQ)
        # pre_means, pre_medians = calc_pre_median(rec_frame-n_before)
        # amps = np.abs(TRACES[array_elecs, rec_frame + all_latencies] - pre_means)
        pre_medians = calc_pre_median(rec_frame-n_before, seq_elecs)
        amps = np.abs(TRACES[seq_elecs, rec_frame + all_latencies])
        all_amp_medians = amps / pre_medians
        
        if PRE_INTERELEC_ROOT_MAX_AMP_ONLY: # only_max_amps:  # 1/6/24 max-amp requirement is now unique to each sequence and is always required
            outer_elecs = elec_to_outer_elecs[elec]
            pre_medians = calc_pre_median(rec_frame-n_before, outer_elecs)
            outer_elecs_traces = TRACES[outer_elecs, rec_frame-this_n_before:rec_frame+n_after+1]
            amps = np.min(outer_elecs_traces, axis=1)
            amps = amps / pre_medians
            if np.argmin(amps) != 0:
                continue
        
        # Now done separately for each seq :  Don't include root elec in latencies since always 0  (Needed it earlier for getting amp/medians)
        # latencies = latencies[1:]
        
        best_unit = None
        best_score = np.inf
        for unit in elec_watchers[elec]:
            # region Previous code (more modular but slower)
            # elecs_r, latency_diff, rel_amp_diff = get_spike_match(unit, time,
            #                                                       max_latency_diff=max_latency_diff, min_coacs_r=min_coacs_r)
            
            # if elecs_r >= min_coacs_r and latency_diff <= max_latency_diff and rel_amp_diff <= max_rel_amp_diff:
            #     # Score spike match with footprint (lower score is better)
            #     match_score = (1-elecs_r) + latency_diff / max_latency_diff + rel_amp_diff / max_rel_amp_diff  # Need to normalize different metrics
            #     if match_score < best_score:
            #         best_unit = unit
            #         best_score = match_score
            # endregion

            # region unit.root_to_stats
            # Check if enough codetections for unit
            
            unit_loose_elecs, unit_inner_loose_elecs, unit_comp_elecs, \
            unit_inner_loose_elecs_set, \
            latency_elec_weights, amp_elec_weights, \
            unit_latencies, unit_amp_medians, unit_amp_median_std = unit.assign_spikes_root_to_stats[elec]
            
            num_inner_loose_elecs = np.sum(all_elec_probs[unit_inner_loose_elecs] >= LOOSE_THRESH)
            # if num_inner_loose_elecs < MIN_INNER_LOOSE_DETECTIONS:
            #     continue
            
            # max-amp requirement
            amp_medians = all_amp_medians[unit_comp_elecs]
            if np.argmax(amp_medians) not in unit_inner_loose_elecs_set:
                continue
        
            num_loose_elecs = np.sum(all_elec_probs[unit_loose_elecs] >= LOOSE_THRESH)
            # if num_loose_elecs < unit.min_loose_detections:
            if num_loose_elecs < 2:
                # time_to_spike_match[time].append((
                #     unit, elec, num_inner_loose_elecs, num_loose_elecs, np.nan, np.nan, np.nan
                # ))
                continue

            # region Faster method, TODO: return to this implementation in final version
            # if np.sum(all_elec_probs[unit_inner_loose_elecs] >= LOOSE_THRESH) < MIN_INNER_LOOSE_DETECTIONS:
            #     continue
        
            # num_loose_elecs = np.sum(all_elec_probs[unit_loose_elecs] >= LOOSE_THRESH)
            # if num_loose_elecs < unit.min_loose_detections:
            #     continue
            # endregion
                        
            # Get unit's stats
            # unit_comp_elecs, unit_elec_probs, unit_latencies, unit_amp_medians = unit.root_to_stats[elec]
            
            # Set spike's elec_probs
            # elec_probs = all_elec_probs[unit_comp_elecs]
            
            # Compare latencies
            latencies = all_latencies[unit_comp_elecs[1:]]
            # elec_weights = get_elec_weights(unit_elec_probs, elec_probs, for_latencies=True)
            latency_diff = np.abs(unit_latencies - latencies)
            latency_diff = np.clip(latency_diff, a_min=None, a_max=clip_latency_diff)
            latency_diff = np.sum(latency_diff * latency_elec_weights)
            # latency_diff = np.sum(np.abs(unit_latencies - latencies) * elec_weights)
            if latency_diff > max_latency_diff:
                continue
            
            # Compare amp/medians
            # amp_medians = all_amp_medians[unit_comp_elecs]
            # elec_weights = get_elec_weights(unit_elec_probs, elec_probs, for_latencies=False)
            # amp_median_div = (unit_amp_medians + amp_medians) / 2
            amp_median_diff = np.abs((unit_amp_medians - amp_medians)) / unit_amp_medians # / amp_median_div
            amp_median_diff = np.clip(amp_median_diff, a_min=None, a_max=clip_amp_median_diff)
            amp_median_diff = np.sum(amp_median_diff * amp_elec_weights)
            if amp_median_diff > max_amp_median_diff:
                continue
            
            amp_median_z = np.abs(amp_medians[0] - unit_amp_medians[0]) / unit_amp_median_std
            if amp_median_z > MAX_ROOT_AMP_MEDIAN_STD_SPIKES:
                continue
            # endregion
            
            # region unit.all_elec_probs[elecs]
            # # Compare latencies
            # unit_elec_probs = unit.all_elec_probs[elecs]
            
            # elec_weights = get_elec_weights(unit_elec_probs, elec_probs, for_latencies=True)
            # latency_diff = np.sum(np.abs(unit.all_latencies[elecs[1:]] - latencies) * elec_weights)
            # if latency_diff > max_latency_diff:
            #     continue
            
            # # Compare amp/medians
            # elec_weights = get_elec_weights(unit_elec_probs, elec_probs, for_latencies=False)
            # unit_amp_medians = unit.all_amp_medians[elecs]
            # amp_median_div = (unit_amp_medians + amp_medians) / 2
            # amp_median_diff = np.abs((unit_amp_medians - amp_medians)) / amp_median_div
            # amp_median_diff = np.sum(amp_median_diff * elec_weights)
            # if amp_median_diff > max_amp_median_diff:
            #     continue
            # endregion
            
            # Calc match score
            score = (latency_diff / max_latency_diff) + (amp_median_diff / max_amp_median_diff) + (1 - num_loose_elecs / spike_num_elecs)
            if num_inner_loose_elecs >= MIN_INNER_LOOSE_DETECTIONS and num_loose_elecs >= unit.min_loose_detections:
                if latency_diff <= max_latency_diff and amp_median_diff <= max_amp_median_diff:
                    if score < best_score:
                        best_unit = unit
                        best_score = score
                
            # spike_match = (elec, num_inner_loose_elecs, num_loose_elecs, num_loose_elecs/len(unit_loose_elecs), latency_diff, amp_median_diff, score)
            # if time not in unit.time_to_spike_match:  # Need to account for if unit has more than one root elec and detects spike on each root elec
            #     unit.time_to_spike_match[time] = spike_match
            # else:
            #     # Each root elec has same num_inner_loose_elecs, num_loose_elecs
            #     if score < unit.time_to_spike_match[time][-1]:
            #         unit.time_to_spike_match[time] = spike_match
                    
            # time_to_spike_match[time].append((
            #     unit, elec, num_inner_loose_elecs, num_loose_elecs, latency_diff, amp_median_diff, score
            #     ))
            
        if best_unit is None:
            continue
        
        if interelec:
            spike_buffer.append((best_unit, time, elec, best_score))
            if len(spike_buffer) > 1 and time - spike_buffer[0][1] > OVERLAP_TIME:
                split_interelec_spike(spike_buffer, time, overlap_dist, elec_to_outer_elecs)
        else:
            best_unit._spike_train.append(time)
            # best_unit._spike_scores.append(best_score)
            
    if interelec:     
        if len(spike_buffer) > 1:
            split_interelec_spike(spike_buffer, time, overlap_dist, elec_to_outer_elecs)
        elif len(spike_buffer) == 1:
            unit, time, elec, score = spike_buffer[0]
            unit._spike_train.append(time)
            unit._elec_train.append(elec)
          
def split_interelec_spike(spike_buffer, time,
                          overlap_dist, 
                          elec_to_outer_elecs):      
    """
    spike_buffer[i] = [unit, time, elec, score]
    """
    
    # Find which spikes overlap more with earliest spike than latest, split these
    first_time = spike_buffer[0][1]
    
    overlapping_spikes = []
    while len(spike_buffer) > 0:
        old_time = spike_buffer[0][1]
        if (old_time - first_time) > (time - old_time):  # old_time is closer to new time than first_time, so it should be split with new time
            break
        spike_data = spike_buffer.pop(0)
        overlapping_spikes.append(spike_data)
    
    overlapping_spikes = sorted(overlapping_spikes, key=lambda spike_data: spike_data[3])    
    
    # Split spikes
    while len(overlapping_spikes) > 0:
        # Find best score
        # best_data = [None, None, None, np.inf]  # (unit, time, elec, score)
        # for spike_data in overlapping_spikes:
        #     if spike_data[3] < best_data[3]:
        #         best_data = spike_data
                        
        best_unit, best_time, best_elec, best_score = overlapping_spikes.pop(0)
        # if len(unit._elec_train) == 0 or \
        # time - unit._spike_train[-1] > overlap_time or unit._elec_train[-1] == elec:  # If same spike is detected by a different root_elec, do not assign spike
        #     unit._spike_train.append(time)
        #     unit._elec_train.append(elec)
        add_spike_to_unit(best_unit, best_time, best_elec, OVERLAP_TIME)
        
        # Find number of spikes that can detect spike
        output_frame = rec_ms_to_output_frame(best_time)
        this_n_before = N_BEFORE if output_frame - N_BEFORE >= 0 else output_frame  # Prevents indexing problems. TODO: Consider removing for speed and just not detecting spikes in first N_BEFORE frames
        output_window = OUTPUTS[elec_to_outer_elecs[best_elec], output_frame-this_n_before:output_frame+N_AFTER+1]
        above_thresh = output_window >= LOOSE_THRESH_LOGIT
        # Actually do computation
        peaks = above_thresh[:, :-1] & ~above_thresh[:, 1:]
        num_spikes = np.sum(peaks, axis=1) + above_thresh[:, -1]  # above_thresh[:, -1] to account for a peak not going below LOOSE_THRESH due to window ending 
        num_spikes = np.max(num_spikes) - 1  # -1 since spike already assigned to best unit
        
        # Remove all spikes within overlap_dist of best spike
        # for s in range(len(overlapping_spikes)-1, -1, -1):
        #     if calc_elec_dist(elec, overlapping_spikes[s][2]) <= overlap_dist:
        #         overlapping_spikes.pop(s)
        
        # Add spikes to top units
        for s in range(len(overlapping_spikes)-1, -1, -1):
            if calc_elec_dist(best_elec, overlapping_spikes[s][2]) <= overlap_dist:
                cur_unit, cur_time, cur_elec, cur_score = overlapping_spikes.pop(s)
                if num_spikes > 0:
                    add_spike_to_unit(cur_unit, cur_time, cur_elec, OVERLAP_TIME)
                    num_spikes -= 1          

def add_spike_to_unit(unit, time, elec, overlap_time):
    if len(unit._elec_train) == 0 or \
    time - unit._spike_train[-1] > overlap_time or unit._elec_train[-1] == elec:  # If same spike is detected by a different root_elec, do not assign spike
        unit._spike_train.append(time)
        unit._elec_train.append(elec)
        
def get_seq_spike_overlap(seq, time):
    """
    Find overlap between seq and spike
        Used fror testing and _save_sequences_plots()
    """   
    
    # Get stats for all elecs
    n_before = N_BEFORE
    n_after = N_AFTER
        
    output_frame = rec_ms_to_output_frame(time)
    this_n_before = n_before if output_frame - n_before >= 0 else output_frame  # Prevents indexing problems
    output_window = OUTPUTS[:, output_frame-this_n_before:output_frame+n_after+1]
    all_elec_probs = sigmoid(np.max(output_window, axis=1))
    
    all_latencies = np.argmax(output_window, axis=1) - this_n_before 

    rec_frame = round(time * SAMP_FREQ)
    pre_medians = calc_pre_median(rec_frame-n_before)
    amps = np.abs(TRACES[np.arange(all_latencies.size), rec_frame + all_latencies])
    all_amp_medians = amps / pre_medians
    
    # If comparing a sequence to another sorter's spike, it would be good to adjust the other sorter's spike time to the trough
        
    # Get stats for seq elecs     
    unit_comp_elecs, unit_elec_probs, unit_latencies, unit_amp_medians, root_amp_std = seq.root_to_stats[seq.root_elec]
    
    num_inner_loose_elecs = np.sum(all_elec_probs[seq.inner_loose_elecs] >= LOOSE_THRESH)
    num_loose_elecs = np.sum(all_elec_probs[seq.loose_elecs] >= LOOSE_THRESH)
    
    latency_elec_weights = unit_elec_probs[1:] / np.sum(unit_elec_probs[1:])
    latencies = all_latencies[unit_comp_elecs[1:]]
    latency_diff = np.abs(unit_latencies - latencies)
    latency_diff = np.clip(latency_diff, a_min=None, a_max=CLIP_LATENCY_DIFF)
    latency_diff = np.sum(latency_diff * latency_elec_weights)
    
    amp_elec_weights = unit_elec_probs / np.sum(unit_elec_probs)
    amp_medians = all_amp_medians[unit_comp_elecs]
    amp_median_diff = np.abs((unit_amp_medians - amp_medians)) / unit_amp_medians # / amp_median_div
    amp_median_diff = np.clip(amp_median_diff, a_min=None, a_max=CLIP_AMP_MEDIAN_DIFF)
    amp_median_diff = np.sum(amp_median_diff * amp_elec_weights)
    
    root_amp_z = np.abs(unit_amp_medians[0] - amp_medians[0]) / root_amp_std
    
    return num_inner_loose_elecs, num_loose_elecs, num_loose_elecs / len(seq.loose_elecs), latency_diff, amp_median_diff, root_amp_z
    
def plot_seq_spike_overlap(seq, time, idx=0,
                           amp_kwargs=None, prob_kwargs=None):    
    num_inner_loose_elecs, num_loose_elecs, ratio_loose_elecs, latency_diff, amp_median_diff, root_amp_z = get_seq_spike_overlap(seq, time)
    unit = Unit(idx, [time], seq.root_elecs[0], RECORDING)
    amp_kwargs, prob_kwargs = plot_elec_probs(unit, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs)
    plt.suptitle(f"Inner: {num_inner_loose_elecs}/{MIN_INNER_LOOSE_DETECTIONS}. Loose: {num_loose_elecs}/{seq.min_loose_detections:.1f}. Latency: {latency_diff:.2f} frames. Amp: {amp_median_diff*100:.1f}%. Root amp STD: {root_amp_z:.1f}")
    return amp_kwargs, prob_kwargs
# endregion


# region Fast assign spikes
def assign_spikes_torch(merged_sequences, time_frame_ms=None,
                        return_spikes=False):
    """
    time_frams_ms:
        If None, it will be (PRE_MEDIAN_FRAMES * SAMP_FREQ, {end of recording})
        
    return_spikes:
        If False, spikes are assigned to merged_sequences._spike_train
        If True, spikes are not assigned and instead returned
        
    gpu_memory:
        The memory of the GPU to use (GB). This only includes loading TRACES and OUTPUTS onto GPU, nothing else (such as sequence tensors or intermediate computations)
    
    
    Test to measure speed of moving CPU data on to GPU
        import timeit
        import time

        def test():
            duration=200
            start_frame = np.random.randint(0, 1000000)
            
            test = torch.tensor(TRACES[:, start_frame:start_frame+duration], dtype=torch.float16, device="cuda")
            two = torch.tensor(OUTPUTS[:, start_frame:start_frame+duration], dtype=torch.float16, device="cuda")

            torch.cuda.synchronize()

        # Measure the execution time of test() by running it 1000 times
        num_iterations = 1000
        execution_time = timeit.timeit(test, number=num_iterations) / num_iterations
        print("Average execution time:", execution_time * 1000, "milliseconds")
        execution_time = timeit.timeit(test, number=num_iterations) / num_iterations
        print("Average execution time:", execution_time * 1000, "milliseconds")
        execution_time = timeit.timeit(test, number=num_iterations) / num_iterations
        print("Average execution time:", execution_time * 1000, "milliseconds")
    """
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    dtype = torch.float16

    # device = "cpu"
    # dtype = torch.float32

    # region Handle sequences with multiple root elecs (this is slower and barely changes spike assignment)
    # all_sequences = []
    # seq_ids = []
    # for id, seq in enumerate(merged_sequences):
    #     for root_elec in seq.root_elecs:
    #         seq_copy = deepcopy(seq)
    #         seq_copy.root_elec = root_elec
    #         all_sequences.append(seq_copy)
    #         seq_ids.append(id)
            
    all_sequences = merged_sequences
            
    # endregion

    # region Setup tensors
    num_seqs = len(all_sequences)

    all_comp_elecs = set()
    # seq_n_before = -np.inf
    # seq_n_after = -np.inf
    seq_no_overlap_mask = torch.full((num_seqs, num_seqs), 0, dtype=torch.bool, device=device)
    for a, seq in enumerate(all_sequences):
        all_comp_elecs.update(seq.comp_elecs)
        # seq_n_before = max(seq_n_before, ceil(np.abs(np.min(seq.all_latencies[seq.comp_elecs]))) + CLIP_LATENCY_DIFF)
        # seq_n_after = max(seq_n_after, ceil(np.abs(np.max(seq.all_latencies[seq.comp_elecs]))) + CLIP_LATENCY_DIFF)
        for b, seq_b in enumerate(all_sequences):
            if calc_elec_dist(seq.root_elec, seq_b.root_elec) > INNER_RADIUS:
                seq_no_overlap_mask[a, b] = 1

    all_comp_elecs = list(all_comp_elecs)
    seq_n_before = N_BEFORE
    seq_n_after = N_AFTER 
    spike_arange = torch.arange(0, seq_n_before+seq_n_after+1, device=device)

    seqs_root_elecs = set()
    seqs_root_elecs_rel_comp_elecs = []
    seqs_inner_loose_elecs = []
    seqs_min_loose_elecs = []
    seqs_loose_elecs = []
    seqs_latencies = []
    seqs_latency_weights = []
    seqs_amps = []
    seqs_root_amp_means = []
    seqs_root_amp_stds = []
    seqs_amp_weights = []
    for seq in all_sequences: 
        seqs_root_elecs.add(seq.root_elec)
        seqs_root_elecs_rel_comp_elecs.append(all_comp_elecs.index(seq.root_elec))

        # Binary arrays (1 for comp_elec in inner_loose_elecs/loose_elecs, else 0)
        seqs_inner_loose_elecs.append(torch.tensor([1 if elec in seq.inner_loose_elecs else 0 for elec in all_comp_elecs], dtype=torch.bool, device=device))
        seqs_loose_elecs.append(torch.tensor([1 if elec in seq.loose_elecs else 0 for elec in all_comp_elecs], dtype=torch.bool, device=device))
        
        seqs_min_loose_elecs.append(ceil(seq.min_loose_detections))
        
        seq_comp_elecs = set(seq.comp_elecs)
        seqs_latencies.append(torch.tensor([seq.all_latencies[elec] + seq_n_before if elec in seq_comp_elecs and elec != seq.root_elec else 0 for elec in all_comp_elecs], dtype=dtype, device=device))
        seqs_amps.append(torch.tensor([seq.all_amp_medians[elec] if elec in seq_comp_elecs else 1 for elec in all_comp_elecs], dtype=dtype, device=device))  # Needs to be 1 to prevent divide by zero
        
        elec_probs = torch.tensor([seq.all_elec_probs[elec] if elec in seq_comp_elecs and elec != seq.root_elec else 0 for elec in all_comp_elecs], dtype=dtype, device=device)
        seqs_latency_weights.append(elec_probs/torch.sum(elec_probs))
        
        elec_probs = torch.tensor([seq.all_elec_probs[elec] if elec in seq_comp_elecs else 0 for elec in all_comp_elecs], dtype=dtype, device=device)
        seqs_amp_weights.append(elec_probs/torch.sum(elec_probs))
        
        seqs_root_amp_means.append(seq.all_amp_medians[seq.root_elec])
        seqs_root_amp_stds.append(seq.root_to_amp_median_std[seq.root_elec])
        
    comp_elecs = torch.tensor(all_comp_elecs, dtype=torch.long, device=device)[:, None]
    comp_elecs_flattened = comp_elecs.flatten()
        
    seqs_root_elecs = list(seqs_root_elecs)
        
    seqs_inner_loose_elecs = torch.vstack(seqs_inner_loose_elecs)
    seqs_loose_elecs = torch.vstack(seqs_loose_elecs)
    seqs_min_loose_elecs = torch.tensor(seqs_min_loose_elecs, dtype=torch.int16, device=device)
    seqs_latencies = torch.vstack(seqs_latencies)
    seqs_amps = torch.vstack(seqs_amps)
    seqs_latency_weights = torch.vstack(seqs_latency_weights)
    seqs_amp_weights = torch.vstack(seqs_amp_weights)
    seqs_root_amp_means = torch.tensor(seqs_root_amp_means, dtype=dtype, device=device)
    seqs_root_amp_stds = torch.tensor(seqs_root_amp_stds, dtype=dtype, device=device)
        
    max_pool = torch.nn.MaxPool1d(seq_n_before+seq_n_after+1, return_indices=True)

    step_size = OUTPUT_WINDOW_HALF_SIZE * 2 - seq_n_after - seq_n_before
    # endregion

    # region Additional setup to port to this file    
    if time_frame_ms is not None:
        START_FRAME = round(time_frame_ms[0] * SAMP_FREQ)
        END_FRAME = round(time_frame_ms[1] * SAMP_FREQ)
    else:
        START_FRAME = 0
        END_FRAME = TRACES.shape[1]
    all_output_frames = range(START_FRAME+PRE_MEDIAN_FRAMES-FRONT_BUFFER, END_FRAME-OUTPUT_WINDOW_HALF_SIZE+seq_n_after, step_size)
    OVERLAP = round(OVERLAP_TIME * SAMP_FREQ)
    # endregion

    # region Get detections   
    detections = [[] for _ in range(num_seqs)]
    last_pre_median_output_frame = -np.inf
    last_detections = torch.full((num_seqs,), -PRE_MEDIAN_FRAMES, dtype=torch.int64, device=device)
        
    # # Measure time it takes to finish spike sorting
    # MODEL_PATH = "/data/MEAprojects/buzsaki/SiegleJ/AllenInstitute_744912849/session_766640955/dl_models/240318/c/240318_165245_967091" 
    # SCALED_TRACES_PATH = "/data/MEAprojects/dandi/000034/sub-MEAREC-250neuron-Neuropixels/rt_sort/dl_model/240318/scaled_traces.npy"
    # scaled_traces = np.load(SCALED_TRACES_PATH, mmap_mode="r")
    # model = ModelSpikeSorter.load(MODEL_PATH) 
    # front_buffer = model.buffer_front_sample
    # end_buffer = model.buffer_end_sample
    # if INFERENCE_SCALING_NUMERATOR is not None:
    #     window = scaled_traces[:, :PRE_MEDIAN_FRAMES]
    #     iqrs = scipy.stats.iqr(window, axis=1)
    #     median = np.median(iqrs)
    #     inference_scaling = INFERENCE_SCALING_NUMERATOR / median
    # else:
    #     inference_scaling = 1
    # input_scale = model.input_scale * inference_scaling 
    # model = ModelSpikeSorter.load_compiled(Path(MODEL_PATH))
    # sorting_computation_times = []  # Time to compute sorting
    # sorting_delays = []  # Delay from when spike happened to final sorting
    
    # # Test on specific frame (used when debugging when error at certain tqdm progress bar iter)
    # test = list(range(PRE_MEDIAN_FRAMES-FRONT_BUFFER, TORCH_OUTPUTS.shape[1]-OUTPUT_WINDOW_HALF_SIZE-seq_n_after, step_size))
    # for output_frame in tqdm(test[55795:]):
    for output_frame in tqdm(all_output_frames):
        # # Test at specific time
        # if not (output_frame-OUTPUT_WINDOW_HALF_SIZE +seq_n_before <= rec_ms_to_output_frame(302286.21875) < output_frame+OUTPUT_WINDOW_HALF_SIZE-seq_n_after):
        #     continue
        
        torch_window = torch.tensor(OUTPUTS[:, output_frame-OUTPUT_WINDOW_HALF_SIZE:output_frame+OUTPUT_WINDOW_HALF_SIZE], dtype=dtype, device=device)
        
        if output_frame - last_pre_median_output_frame >= PRE_MEDIAN_FRAMES:
            # start = perf_counter()
            rec_frame = output_frame + FRONT_BUFFER
            # pre_medians = TORCH_TRACES[comp_elecs_flattened, rec_frame-PRE_MEDIAN_FRAMES:rec_frame] 
            rec_window = torch.tensor(TRACES[:, rec_frame-PRE_MEDIAN_FRAMES:rec_frame], dtype=dtype, device=device)
            pre_medians = rec_window[comp_elecs_flattened, :] 
            pre_medians = torch.median(torch.abs(pre_medians), dim=1).values  # Pytorch median different than numpy median: https://stackoverflow.com/a/54310996
            pre_medians = torch.clip(pre_medians / 0.6745, min=0.5, max=None)  # a_min=0.5 to prevent dividing by zero when data is just 1s and 0s (median could equal 0) 
            # torch.cuda.synchronize()
            # end = perf_counter()
            # delays_pre_median.append((end-start)*1000)
        
        rec_start_frame = (output_frame - OUTPUT_WINDOW_HALF_SIZE) + FRONT_BUFFER
        rec_window = torch.tensor(TRACES[:, rec_start_frame:rec_start_frame+OUTPUT_WINDOW_HALF_SIZE*2], dtype=dtype, device=device)
        
        # # Include running DL model in sorting_delays
        # traces_torch = torch.tensor(scaled_traces[:, rec_start_frame-front_buffer:rec_start_frame+OUTPUT_WINDOW_HALF_SIZE*2+end_buffer], dtype=dtype, device=device)
        # start = perf_counter()
        # with torch.no_grad():
        #     traces_torch -= torch.median(traces_torch, dim=1, keepdim=True).values
        #     outputs = model(traces_torch[:, None, :] * input_scale)
        #     # assert torch.all(torch.isclose(outputs[:, 0, :], torch_window)) # outputs won't be exact same since traces_torch won't be exact window used to create model_outputs.npy
        #     torch.cuda.synchronize()
        # start_sorting = perf_counter()
        
        # output_window = OUTPUTS[:, output_frame-90:output_frame+90]
        # torch_window = torch.tensor(output_window, device=device)
        # start_sorting = perf_counter()
        # output_start_frame = output_frame - OUTPUT_WINDOW_HALF_SIZE  # Make slices in TORCH_OUTPUTS relative to torch_window (see spike_assignment2.ipynb)
        
        # region Find peaks    
        # start = perf_counter()
        # window = TORCH_OUTPUTS[seqs_root_elecs, output_start_frame+seq_n_before-1:output_frame+OUTPUT_WINDOW_HALF_SIZE-seq_n_after+1]  # torch_window[seqs_root_elecs, seq_n_before-1:-seq_n_after+1]  # -1 and +1 to identify peaks
        window = torch_window[seqs_root_elecs, seq_n_before-1:-seq_n_after+1]
        main = window[:, 1:-1]
        greater_than_left = main > window[:, :-2]
        greater_than_right = main > window[:, 2:]
        peaks = greater_than_left & greater_than_right
        crosses = main >= STRINGENT_THRESH_LOGIT
        peak_ind_flat = torch.nonzero(peaks & crosses, as_tuple=True)[1]
        # peak_ind_flat = peak_ind_flat[torch.sum(TORCH_OUTPUTS[:, output_start_frame+peak_ind_flat+seq_n_before] >= LOOSE_THRESH_LOGIT, dim=0) <= MIN_ELECS_FOR_ARRAY_NOISE]  # may be slow
        peak_ind_flat = peak_ind_flat[torch.sum(torch_window[:, peak_ind_flat+seq_n_before] >= LOOSE_THRESH_LOGIT, dim=0) <= MIN_ELECS_FOR_ARRAY_NOISE]  # may be slow
        # if torch.numel(peak_ind_flat) == 0:  # This is very rare (only happend once for 1/16/24 spikeinterface simulated recording)
        #     # end = perf_counter()
        #     # delays_find_peak.append((end-start)*1000)
        #     # delays_total.append(delays_find_peak[-1])
        #     # delays_total_spike_detected.append(False)
        #     continue
        # end = perf_counter()
        # delays_find_peak.append((end-start)*1000)
        # endregion
        
        # # Plot peak waveform and detection footprints
        # peak=peak_ind_flat[0].item()
        # elec=seqs_root_elecs[0]
        # ##
        # frame = peak + seq_n_before + rec_start_frame
        # ms = frame / SAMP_FREQ
        # plot_spikes([ms], elec)
        # plt.show()
        
        # start = perf_counter()
        peak_ind = peak_ind_flat[:, None, None]  # Relative to output_start_frame+seq_n_before
        spike_window = torch_window[comp_elecs, peak_ind + spike_arange] 

        elec_probs, latencies = max_pool(spike_window)  # Latencies are relative to peak-seq_n_before
        elec_crosses = (elec_probs >= LOOSE_THRESH_LOGIT).transpose(1, 2)
        num_inner_loose = torch.sum(elec_crosses & seqs_inner_loose_elecs, dim=2)
        pass_inner_loose = num_inner_loose >= MIN_INNER_LOOSE_DETECTIONS

        num_loose = torch.sum(elec_crosses & seqs_loose_elecs, dim=2)
        pass_loose = num_loose >= seqs_min_loose_elecs
        # end = perf_counter()
        # delays_elec_cross.append((end-start)*1000)

        # Slower to check this since so many DL detections
        # can_spike = pass_inner_loose & pass_loose
        # if not torch.any(can_spike):
        #     continue

        # start = perf_counter()
        latencies_float = latencies.transpose(1, 2).to(dtype)
        latency_diff = torch.abs(latencies_float - seqs_latencies)
        latency_diff = torch.clip(latency_diff, min=None, max=CLIP_LATENCY_DIFF)
        latency_diff = torch.sum(latency_diff * seqs_latency_weights, axis=2)
        pass_latency = latency_diff <= MAX_LATENCY_DIFF_SPIKES
        # end = perf_counter()
        # delays_latency.append((end-start)*1000)

        # # Getting rec_window was moved up in code to not include it in sorting computation time
        # # start = perf_counter()
        # # rec_start_frame = output_start_frame + FRONT_BUFFER
        # rec_start_frame = (output_frame - OUTPUT_WINDOW_HALF_SIZE) + FRONT_BUFFER
        # # amps = torch.abs(TORCH_TRACES[comp_elecs, rec_start_frame + peak_ind + latencies].transpose(1, 2)) / pre_medians  # peak_ind+seq_n_before = peak_ind_in_rec_window. latency-seq_n_before = latency_rel_peak_ind. --> (peak_ind+seq_n_before)+(latency-seq_n_before) = peak_ind+latency = spike index in rec_window
        # rec_window = torch.tensor(TRACES[:, rec_start_frame:rec_start_frame+OUTPUT_WINDOW_HALF_SIZE*2], dtype=dtype, device=device)
        amps = torch.abs(rec_window[comp_elecs, peak_ind + latencies].transpose(1, 2)) / pre_medians  # peak_ind+seq_n_before = peak_ind_in_rec_window. latency-seq_n_before = latency_rel_peak_ind. --> (peak_ind+seq_n_before)+(latency-seq_n_before) = peak_ind+latency = spike index in rec_window

        root_amp_z = torch.abs(amps[:, 0, seqs_root_elecs_rel_comp_elecs] - seqs_root_amp_means) / seqs_root_amp_stds
        pass_root_amp_z = root_amp_z <= MAX_ROOT_AMP_MEDIAN_STD_SPIKES
        # end = perf_counter()
        # delays_root_z.append((end-start)*1000)

        # start = perf_counter()
        amp_diff = torch.abs(amps - seqs_amps) / seqs_amps
        amp_diff = torch.clip(amp_diff, min=None, max=CLIP_AMP_MEDIAN_DIFF)
        amp_diff = torch.sum(amp_diff * seqs_amp_weights, axis=2)
        pass_amp_diff = amp_diff <= MAX_AMP_MEDIAN_DIFF_SPIKES
        # end = perf_counter()
        # delays_amp.append((end-start)*1000)

        # Since every seq is compared to every peak, this is needed (or a form of this) so (in extreme case) peak detected on one side of array is not assigned to seq on other side by coincidence
        strict_crosses_root = spike_window[:, seqs_root_elecs_rel_comp_elecs, seq_n_before] >= STRINGENT_THRESH_LOGIT

        # start = perf_counter()
        can_spike = strict_crosses_root & pass_inner_loose & pass_loose & pass_latency & pass_root_amp_z & pass_amp_diff
        # end = perf_counter()
        # delays_can_spike.append((end-start)*1000)

        # Slighty faster than the following due to only slicing once: # spike_scores = latency_diff[can_spike] / MAX_LATENCY_DIFF_SPIKES + amp_diff[can_spike] / MAX_AMP_MEDIAN_DIFF_SPIKES  - num_loose[can_spike] / torch.sum(elec_crosses, dim=2)
        # start = perf_counter()
        spike_scores = latency_diff / MAX_LATENCY_DIFF_SPIKES + amp_diff / MAX_AMP_MEDIAN_DIFF_SPIKES - (num_loose / torch.sum(elec_crosses, dim=2) * 0.5)
        spike_scores = 2.1 - spike_scores  # (additional 0.1 in case spike_scores=2)
        spike_scores *= can_spike

        # For debugging:
        # breakpoint: output_start_frame+seq_n_before <= rec_ms_to_output_frame(TIME) < output_frame+OUTPUT_WINDOW_HALF_SIZE-seq_n_after
        # time = (peak_ind.flatten()[2].item() + rec_start_frame + START_FRAME)/SAMP_FREQ
        # plot_spikes([time], all_sequences[20].root_elec)
        # plt.show()

        peak_ind_2d = peak_ind[:, 0]
        # next_can_spike = torch.full_like(can_spike, fill_value=1, dtype=torch.bool, device=device)   
        # end = perf_counter()
        # cur_delay_split_spike = (end-start)*1000
        
        # cur_delay_assign_spike = 0
        # spike_detected = False  # TODO: This is only needed for speed testing
        while torch.any(spike_scores):
            # start = perf_counter()
            spike_seq_idx = torch.argmax(spike_scores).item()
            spike_idx = spike_seq_idx // num_seqs
            seq_idx = spike_seq_idx % num_seqs
            offset_spike_time = peak_ind_flat[spike_idx].item()  # TODO: If spike assignment is now slow, remove .item() (which gets the spike time in python value from torch tensor)
            spike_time = offset_spike_time + seq_n_before + rec_start_frame # + START_FRAME
            # end = perf_counter()
            # cur_delay_assign_spike += ((end-start)*1000)
            
            # start = perf_counter()
            if spike_time - last_detections[seq_idx] <= OVERLAP:
                spike_scores[spike_idx, seq_idx] = 0
                continue
            
            detections[seq_idx].append(spike_time)
            
            # Spike splitting for current window
            # set score to 0 if (seq is spatially close enough) and (peak is temporally close enough)
            # keep score if (seq is NOT spatially close enough) or (peak is temporally far enough)
            spike_scores *= seq_no_overlap_mask[seq_idx] | (torch.abs(peak_ind_2d - offset_spike_time) > OVERLAP)
            # end = perf_counter()
            # cur_delay_split_spike += ((end-start)*1000)
            # spike_detected = True
            
            # Spike splitting for next window
            # FAST: just change seq_n_before for all seqs
            # Slower: last_spike[~no_overlap_mask] = max(last_spike[~no_overlap_mask], spike_time)
            # last_detections[(~no_overlap_mask) & (last_detections < spike_time)] = spike_time

        # delays_assign_spike.append(cur_delay_assign_spike)
        # delays_split_spike.append(cur_delay_split_spike)
        # delays_total.append(delays_find_peak[-1] + delays_elec_cross[-1] + delays_latency[-1] + delays_root_z[-1] + delays_amp[-1] + delays_can_spike[-1] + delays_assign_spike[-1] + delays_split_spike[-1])
        # delays_total_spike_detected.append(spike_detected)
        # torch.cuda.synchronize()  # Not needed because of while loop
        # end = perf_counter()
        # sorting_computation_times.append((end-start_sorting)*1000)
    # endregion
    # return sorting_computation_times

    all_spike_trains = []
    for seq, spikes in zip(all_sequences, detections):
        spike_train = np.sort([s / SAMP_FREQ for s in spikes])
        if not return_spikes:
            seq._spike_train = np.sort(spike_train)
        else:
            all_spike_trains.append(spike_train)
     
    if return_spikes:
        return all_spike_trains
     
    # all_spike_trains = [[] for _ in range(len(merged_sequences))]
    # for id, spike_train in zip(seq_ids, detections):
    #     all_spike_trains[id].extend(spike_train)
        
    # for seq, spike_train in zip(merged_sequences, all_spike_trains):
    #     seq._spike_train = np.sort(spike_train) / SAMP_FREQ
   
def measure_speed_assign_spikes_torch(merged_sequences, time_frame_ms=None,
                        return_spikes=False):
    """
    time_frams_ms:
        If None, it will be (PRE_MEDIAN_FRAMES * SAMP_FREQ, {end of recording})
        
    return_spikes:
        If False, spikes are assigned to merged_sequences._spike_train
        If True, spikes are not assigned and instead returned
        
    gpu_memory:
        The memory of the GPU to use (GB). This only includes loading TRACES and OUTPUTS onto GPU, nothing else (such as sequence tensors or intermediate computations)
    
    
    Test to measure speed of moving CPU data on to GPU
        import timeit
        import time

        def test():
            duration=200
            start_frame = np.random.randint(0, 1000000)
            
            test = torch.tensor(TRACES[:, start_frame:start_frame+duration], dtype=torch.float16, device="cuda")
            two = torch.tensor(OUTPUTS[:, start_frame:start_frame+duration], dtype=torch.float16, device="cuda")

            torch.cuda.synchronize()

        # Measure the execution time of test() by running it 1000 times
        num_iterations = 1000
        execution_time = timeit.timeit(test, number=num_iterations) / num_iterations
        print("Average execution time:", execution_time * 1000, "milliseconds")
        execution_time = timeit.timeit(test, number=num_iterations) / num_iterations
        print("Average execution time:", execution_time * 1000, "milliseconds")
        execution_time = timeit.timeit(test, number=num_iterations) / num_iterations
        print("Average execution time:", execution_time * 1000, "milliseconds")
    """
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    dtype = torch.float16

    # device = "cpu"
    # dtype = torch.float32

    # region Handle sequences with multiple root elecs (this is slower and barely changes spike assignment)
    # all_sequences = []
    # seq_ids = []
    # for id, seq in enumerate(merged_sequences):
    #     for root_elec in seq.root_elecs:
    #         seq_copy = deepcopy(seq)
    #         seq_copy.root_elec = root_elec
    #         all_sequences.append(seq_copy)
    #         seq_ids.append(id)
            
    all_sequences = merged_sequences
            
    # endregion

    # region Setup tensors
    num_seqs = len(all_sequences)

    all_comp_elecs = set()
    # seq_n_before = -np.inf
    # seq_n_after = -np.inf
    seq_no_overlap_mask = torch.full((num_seqs, num_seqs), 0, dtype=torch.bool, device=device)
    for a, seq in enumerate(all_sequences):
        all_comp_elecs.update(seq.comp_elecs)
        # seq_n_before = max(seq_n_before, ceil(np.abs(np.min(seq.all_latencies[seq.comp_elecs]))) + CLIP_LATENCY_DIFF)
        # seq_n_after = max(seq_n_after, ceil(np.abs(np.max(seq.all_latencies[seq.comp_elecs]))) + CLIP_LATENCY_DIFF)
        for b, seq_b in enumerate(all_sequences):
            if calc_elec_dist(seq.root_elec, seq_b.root_elec) > INNER_RADIUS:
                seq_no_overlap_mask[a, b] = 1

    all_comp_elecs = list(all_comp_elecs)
    seq_n_before = N_BEFORE
    seq_n_after = N_AFTER 
    spike_arange = torch.arange(0, seq_n_before+seq_n_after+1, device=device)

    seqs_root_elecs = set()
    seqs_root_elecs_rel_comp_elecs = []
    seqs_inner_loose_elecs = []
    seqs_min_loose_elecs = []
    seqs_loose_elecs = []
    seqs_latencies = []
    seqs_latency_weights = []
    seqs_amps = []
    seqs_root_amp_means = []
    seqs_root_amp_stds = []
    seqs_amp_weights = []
    for seq in all_sequences: 
        seqs_root_elecs.add(seq.root_elec)
        seqs_root_elecs_rel_comp_elecs.append(all_comp_elecs.index(seq.root_elec))

        # Binary arrays (1 for comp_elec in inner_loose_elecs/loose_elecs, else 0)
        seqs_inner_loose_elecs.append(torch.tensor([1 if elec in seq.inner_loose_elecs else 0 for elec in all_comp_elecs], dtype=torch.bool, device=device))
        seqs_loose_elecs.append(torch.tensor([1 if elec in seq.loose_elecs else 0 for elec in all_comp_elecs], dtype=torch.bool, device=device))
        
        seqs_min_loose_elecs.append(ceil(seq.min_loose_detections))
        
        seq_comp_elecs = set(seq.comp_elecs)
        seqs_latencies.append(torch.tensor([seq.all_latencies[elec] + seq_n_before if elec in seq_comp_elecs and elec != seq.root_elec else 0 for elec in all_comp_elecs], dtype=dtype, device=device))
        seqs_amps.append(torch.tensor([seq.all_amp_medians[elec] if elec in seq_comp_elecs else 1 for elec in all_comp_elecs], dtype=dtype, device=device))  # Needs to be 1 to prevent divide by zero
        
        elec_probs = torch.tensor([seq.all_elec_probs[elec] if elec in seq_comp_elecs and elec != seq.root_elec else 0 for elec in all_comp_elecs], dtype=dtype, device=device)
        seqs_latency_weights.append(elec_probs/torch.sum(elec_probs))
        
        elec_probs = torch.tensor([seq.all_elec_probs[elec] if elec in seq_comp_elecs else 0 for elec in all_comp_elecs], dtype=dtype, device=device)
        seqs_amp_weights.append(elec_probs/torch.sum(elec_probs))
        
        seqs_root_amp_means.append(seq.all_amp_medians[seq.root_elec])
        seqs_root_amp_stds.append(seq.root_to_amp_median_std[seq.root_elec])
        
    comp_elecs = torch.tensor(all_comp_elecs, dtype=torch.long, device=device)[:, None]
    comp_elecs_flattened = comp_elecs.flatten()
        
    seqs_root_elecs = list(seqs_root_elecs)
        
    seqs_inner_loose_elecs = torch.vstack(seqs_inner_loose_elecs)
    seqs_loose_elecs = torch.vstack(seqs_loose_elecs)
    seqs_min_loose_elecs = torch.tensor(seqs_min_loose_elecs, dtype=torch.int16, device=device)
    seqs_latencies = torch.vstack(seqs_latencies)
    seqs_amps = torch.vstack(seqs_amps)
    seqs_latency_weights = torch.vstack(seqs_latency_weights)
    seqs_amp_weights = torch.vstack(seqs_amp_weights)
    seqs_root_amp_means = torch.tensor(seqs_root_amp_means, dtype=dtype, device=device)
    seqs_root_amp_stds = torch.tensor(seqs_root_amp_stds, dtype=dtype, device=device)
        
    max_pool = torch.nn.MaxPool1d(seq_n_before+seq_n_after+1, return_indices=True)

    step_size = OUTPUT_WINDOW_HALF_SIZE * 2 - seq_n_after - seq_n_before
    # endregion

    # region Additional setup to port to this file    
    if time_frame_ms is not None:
        START_FRAME = round(time_frame_ms[0] * SAMP_FREQ)
        END_FRAME = round(time_frame_ms[1] * SAMP_FREQ)
    else:
        START_FRAME = 0
        END_FRAME = TRACES.shape[1]
    all_output_frames = range(START_FRAME+PRE_MEDIAN_FRAMES-FRONT_BUFFER, END_FRAME-OUTPUT_WINDOW_HALF_SIZE+seq_n_after, step_size)
    OVERLAP = round(OVERLAP_TIME * SAMP_FREQ)
    # endregion

    # region Get detections   
    detections = [[] for _ in range(num_seqs)]
    last_pre_median_output_frame = -np.inf
    last_detections = torch.full((num_seqs,), -PRE_MEDIAN_FRAMES, dtype=torch.int64, device=device)
        
    # Measure time it takes to finish spike sorting 
    # # F.MODEL_PATH = MODEL_PATH
    # # F.SCALED_TRACES_PATH = SCALED_TRACES_PATH
    scaled_traces = np.load(SCALED_TRACES_PATH, mmap_mode="r")
    model = ModelSpikeSorter.load(MODEL_PATH) 
    output_window_half_size = model.num_output_locs // 2
    front_buffer = model.buffer_front_sample
    end_buffer = model.buffer_end_sample
    if INFERENCE_SCALING_NUMERATOR is not None:
        window = scaled_traces[:, :PRE_MEDIAN_FRAMES]
        iqrs = scipy.stats.iqr(window, axis=1)
        median = np.median(iqrs)
        inference_scaling = INFERENCE_SCALING_NUMERATOR / median
    else:
        inference_scaling = 1
    input_scale = model.input_scale * inference_scaling 
    model = ModelSpikeSorter.load_compiled(Path(MODEL_PATH))
    sorting_computation_times = []  # Time to compute sorting
    sorting_delays = []  # Delay from when spike happened to final sorting
    
    # # Test on specific frame (used when debugging when error at certain tqdm progress bar iter)
    # test = list(range(PRE_MEDIAN_FRAMES-FRONT_BUFFER, TORCH_OUTPUTS.shape[1]-OUTPUT_WINDOW_HALF_SIZE-seq_n_after, step_size))
    # for output_frame in tqdm(test[55795:]):
    for output_frame in tqdm(all_output_frames):
        # For debugging
        # if not (output_frame-OUTPUT_WINDOW_HALF_SIZE +seq_n_before <= rec_ms_to_output_frame(19197.5) < output_frame+OUTPUT_WINDOW_HALF_SIZE-seq_n_after):
        #     continue
        
        torch_window = torch.tensor(OUTPUTS[:, output_frame-OUTPUT_WINDOW_HALF_SIZE:output_frame+OUTPUT_WINDOW_HALF_SIZE], dtype=dtype, device=device)
        
        if output_frame - last_pre_median_output_frame >= PRE_MEDIAN_FRAMES:
            # start = perf_counter()
            rec_frame = output_frame + FRONT_BUFFER
            # pre_medians = TORCH_TRACES[comp_elecs_flattened, rec_frame-PRE_MEDIAN_FRAMES:rec_frame] 
            rec_window = torch.tensor(TRACES[:, rec_frame-PRE_MEDIAN_FRAMES:rec_frame], dtype=dtype, device=device)
            pre_medians = rec_window[comp_elecs_flattened, :] 
            pre_medians = torch.median(torch.abs(pre_medians), dim=1).values  # Pytorch median different than numpy median: https://stackoverflow.com/a/54310996
            pre_medians = torch.clip(pre_medians / 0.6745, min=0.5, max=None)  # a_min=0.5 to prevent dividing by zero when data is just 1s and 0s (median could equal 0) 
            # torch.cuda.synchronize()
            # end = perf_counter()
            # delays_pre_median.append((end-start)*1000)
        
        rec_start_frame = (output_frame - OUTPUT_WINDOW_HALF_SIZE) + FRONT_BUFFER
        rec_window = torch.tensor(TRACES[:, rec_start_frame:rec_start_frame+OUTPUT_WINDOW_HALF_SIZE*2], dtype=dtype, device=device)
        
        detected_spikes = []  # Need to keep track of time of detected spikes for this recording chunk to measure detection latency
        # Include running DL model in sorting_delays
        traces_torch = torch.tensor(scaled_traces[:, rec_start_frame-front_buffer:rec_start_frame+output_window_half_size*2+end_buffer], dtype=dtype, device=device)
        start = perf_counter()
        with torch.no_grad():
            traces_torch -= torch.median(traces_torch, dim=1, keepdim=True).values
            outputs = model(traces_torch[:, None, :] * input_scale)
            # assert torch.all(torch.isclose(outputs[:, 0, :], torch_window)) # outputs won't be exact same since traces_torch won't be exact window used to create model_outputs.npy
            torch.cuda.synchronize()
        start_sorting = perf_counter()
        # sorting_computation_times.append(start_sorting - start)  # Test speed of just DL model
        
        # output_window = OUTPUTS[:, output_frame-90:output_frame+90]
        # torch_window = torch.tensor(output_window, device=device)
        # start_sorting = perf_counter()
        # output_start_frame = output_frame - OUTPUT_WINDOW_HALF_SIZE  # Make slices in TORCH_OUTPUTS relative to torch_window (see spike_assignment2.ipynb)
        
        # region Find peaks    
        # start = perf_counter()
        # window = TORCH_OUTPUTS[seqs_root_elecs, output_start_frame+seq_n_before-1:output_frame+OUTPUT_WINDOW_HALF_SIZE-seq_n_after+1]  # torch_window[seqs_root_elecs, seq_n_before-1:-seq_n_after+1]  # -1 and +1 to identify peaks
        window = torch_window[seqs_root_elecs, seq_n_before-1:-seq_n_after+1]
        main = window[:, 1:-1]
        greater_than_left = main > window[:, :-2]
        greater_than_right = main > window[:, 2:]
        peaks = greater_than_left & greater_than_right
        crosses = main >= STRINGENT_THRESH_LOGIT
        peak_ind_flat = torch.nonzero(peaks & crosses, as_tuple=True)[1]
        # peak_ind_flat = peak_ind_flat[torch.sum(TORCH_OUTPUTS[:, output_start_frame+peak_ind_flat+seq_n_before] >= LOOSE_THRESH_LOGIT, dim=0) <= MIN_ELECS_FOR_ARRAY_NOISE]  # may be slow
        peak_ind_flat = peak_ind_flat[torch.sum(torch_window[:, peak_ind_flat+seq_n_before] >= LOOSE_THRESH_LOGIT, dim=0) <= MIN_ELECS_FOR_ARRAY_NOISE]  # may be slow
        # if torch.numel(peak_ind_flat) == 0:  # This is very rare (only happend once for 1/16/24 spikeinterface simulated recording)
        #     # end = perf_counter()
        #     # delays_find_peak.append((end-start)*1000)
        #     # delays_total.append(delays_find_peak[-1])
        #     # delays_total_spike_detected.append(False)
        #     continue
        # end = perf_counter()
        # delays_find_peak.append((end-start)*1000)
        # endregion
        
        # start = perf_counter()
        peak_ind = peak_ind_flat[:, None, None]  # Relative to output_start_frame+seq_n_before
        spike_window = torch_window[comp_elecs, peak_ind + spike_arange] 

        elec_probs, latencies = max_pool(spike_window)  # Latencies are relative to peak-seq_n_before
        elec_crosses = (elec_probs >= LOOSE_THRESH_LOGIT).transpose(1, 2)
        num_inner_loose = torch.sum(elec_crosses & seqs_inner_loose_elecs, dim=2)
        pass_inner_loose = num_inner_loose >= MIN_INNER_LOOSE_DETECTIONS

        num_loose = torch.sum(elec_crosses & seqs_loose_elecs, dim=2)
        pass_loose = num_loose >= seqs_min_loose_elecs
        # end = perf_counter()
        # delays_elec_cross.append((end-start)*1000)

        # Slower to check this since so many DL detections
        # can_spike = pass_inner_loose & pass_loose
        # if not torch.any(can_spike):
        #     continue

        # start = perf_counter()
        latencies_float = latencies.transpose(1, 2).to(dtype)
        latency_diff = torch.abs(latencies_float - seqs_latencies)
        latency_diff = torch.clip(latency_diff, min=None, max=CLIP_LATENCY_DIFF)
        latency_diff = torch.sum(latency_diff * seqs_latency_weights, axis=2)
        pass_latency = latency_diff <= MAX_LATENCY_DIFF_SPIKES
        # end = perf_counter()
        # delays_latency.append((end-start)*1000)

        # # Getting rec_window was moved up in code to not include it in sorting computation time
        # # start = perf_counter()
        # # rec_start_frame = output_start_frame + FRONT_BUFFER
        # rec_start_frame = (output_frame - OUTPUT_WINDOW_HALF_SIZE) + FRONT_BUFFER
        # # amps = torch.abs(TORCH_TRACES[comp_elecs, rec_start_frame + peak_ind + latencies].transpose(1, 2)) / pre_medians  # peak_ind+seq_n_before = peak_ind_in_rec_window. latency-seq_n_before = latency_rel_peak_ind. --> (peak_ind+seq_n_before)+(latency-seq_n_before) = peak_ind+latency = spike index in rec_window
        # rec_window = torch.tensor(TRACES[:, rec_start_frame:rec_start_frame+OUTPUT_WINDOW_HALF_SIZE*2], dtype=dtype, device=device)
        amps = torch.abs(rec_window[comp_elecs, peak_ind + latencies].transpose(1, 2)) / pre_medians  # peak_ind+seq_n_before = peak_ind_in_rec_window. latency-seq_n_before = latency_rel_peak_ind. --> (peak_ind+seq_n_before)+(latency-seq_n_before) = peak_ind+latency = spike index in rec_window

        root_amp_z = torch.abs(amps[:, 0, seqs_root_elecs_rel_comp_elecs] - seqs_root_amp_means) / seqs_root_amp_stds
        pass_root_amp_z = root_amp_z <= MAX_ROOT_AMP_MEDIAN_STD_SPIKES
        # end = perf_counter()
        # delays_root_z.append((end-start)*1000)

        # start = perf_counter()
        amp_diff = torch.abs(amps - seqs_amps) / seqs_amps
        amp_diff = torch.clip(amp_diff, min=None, max=CLIP_AMP_MEDIAN_DIFF)
        amp_diff = torch.sum(amp_diff * seqs_amp_weights, axis=2)
        pass_amp_diff = amp_diff <= MAX_AMP_MEDIAN_DIFF_SPIKES
        # end = perf_counter()
        # delays_amp.append((end-start)*1000)

        # Since every seq is compared to every peak, this is needed (or a form of this) so (in extreme case) peak detected on one side of array is not assigned to seq on other side by coincidence
        strict_crosses_root = spike_window[:, seqs_root_elecs_rel_comp_elecs, seq_n_before] >= STRINGENT_THRESH_LOGIT

        # start = perf_counter()
        can_spike = strict_crosses_root & pass_inner_loose & pass_loose & pass_latency & pass_root_amp_z & pass_amp_diff
        # end = perf_counter()
        # delays_can_spike.append((end-start)*1000)

        # Slighty faster than the following due to only slicing once: # spike_scores = latency_diff[can_spike] / MAX_LATENCY_DIFF_SPIKES + amp_diff[can_spike] / MAX_AMP_MEDIAN_DIFF_SPIKES  - num_loose[can_spike] / torch.sum(elec_crosses, dim=2)
        # start = perf_counter()
        spike_scores = latency_diff / MAX_LATENCY_DIFF_SPIKES + amp_diff / MAX_AMP_MEDIAN_DIFF_SPIKES - (num_loose / torch.sum(elec_crosses, dim=2) * 0.5)
        spike_scores = 2.1 - spike_scores  # (additional 0.1 in case spike_scores=2)
        spike_scores *= can_spike

        # For debugging:
        # breakpoint: output_start_frame+seq_n_before <= rec_ms_to_output_frame(TIME) < output_frame+OUTPUT_WINDOW_HALF_SIZE-seq_n_after
        # time = (peak_ind.flatten()[2].item() + rec_start_frame + START_FRAME)/SAMP_FREQ
        # plot_spikes([time], all_sequences[20].root_elec)
        # plt.show()

        peak_ind_2d = peak_ind[:, 0]
        # next_can_spike = torch.full_like(can_spike, fill_value=1, dtype=torch.bool, device=device)   
        # end = perf_counter()
        # cur_delay_split_spike = (end-start)*1000
        
        # cur_delay_assign_spike = 0
        # spike_detected = False  # TODO: This is only needed for speed testing
        while torch.any(spike_scores):
            # start = perf_counter()
            spike_seq_idx = torch.argmax(spike_scores).item()
            spike_idx = spike_seq_idx // num_seqs
            seq_idx = spike_seq_idx % num_seqs
            offset_spike_time = peak_ind_flat[spike_idx].item()  # TODO: If spike assignment is now slow, remove .item() (which gets the spike time in python value from torch tensor)
            spike_time = offset_spike_time + seq_n_before + rec_start_frame # + START_FRAME
            # end = perf_counter()
            # cur_delay_assign_spike += ((end-start)*1000)
            
            # start = perf_counter()
            if spike_time - last_detections[seq_idx] <= OVERLAP:
                spike_scores[spike_idx, seq_idx] = 0
                continue
            
            # detections[seq_idx].append(spike_time)
            detected_spikes.append(offset_spike_time + seq_n_before)
            
            # Spike splitting for current window
            # set score to 0 if (seq is spatially close enough) and (peak is temporally close enough)
            # keep score if (seq is NOT spatially close enough) or (peak is temporally far enough)
            spike_scores *= seq_no_overlap_mask[seq_idx] | (torch.abs(peak_ind_2d - offset_spike_time) > OVERLAP)
            # end = perf_counter()
            # cur_delay_split_spike += ((end-start)*1000)
            # spike_detected = True
            
            # Spike splitting for next window
            # FAST: just change seq_n_before for all seqs
            # Slower: last_spike[~no_overlap_mask] = max(last_spike[~no_overlap_mask], spike_time)
            # last_detections[(~no_overlap_mask) & (last_detections < spike_time)] = spike_time

        # delays_assign_spike.append(cur_delay_assign_spike)
        # delays_split_spike.append(cur_delay_split_spike)
        # delays_total.append(delays_find_peak[-1] + delays_elec_cross[-1] + delays_latency[-1] + delays_root_z[-1] + delays_amp[-1] + delays_can_spike[-1] + delays_assign_spike[-1] + delays_split_spike[-1])
        # delays_total_spike_detected.append(spike_detected)
        # torch.cuda.synchronize()  # Not needed because of while loop
        end = perf_counter()
        sorting_computation_times.append((end-start_sorting)*1000)  # Time for just assigning spikes to sequences
        detection_sorting_time = (end - start) * 1000  # Time for detecting spikes with DL model and assigning spikes to sequences
        last_rec_frame = OUTPUT_WINDOW_HALF_SIZE*2+end_buffer  # This is the most recent rec frame (relative to rec_start_frame)
        for spike in detected_spikes:  # detected_spikes is in frames relative to rec_start_frame
            wait_time = (last_rec_frame - spike)/SAMP_FREQ  # Waiting time from spike being released until done collecting data for chunk
            sorting_delays.append(detection_sorting_time + wait_time)
            assert wait_time >= 0
        
    # endregion
    return sorting_computation_times, sorting_delays

    all_spike_trains = []
    for seq, spikes in zip(all_sequences, detections):
        spike_train = np.sort([s / SAMP_FREQ for s in spikes])
        if not return_spikes:
            seq._spike_train = np.sort(spike_train)
        else:
            all_spike_trains.append(spike_train)
     
    if return_spikes:
        return all_spike_trains
     
    # all_spike_trains = [[] for _ in range(len(merged_sequences))]
    # for id, spike_train in zip(seq_ids, detections):
    #     all_spike_trains[id].extend(spike_train)
        
    # for seq, spike_train in zip(merged_sequences, all_spike_trains):
    #     seq._spike_train = np.sort(spike_train) / SAMP_FREQ  
# endregion


# region Form full sequences
def form_from_root(root_elec, time_frame, 
                   verbose=False):
    raise NotImplementedError("Need to check .ipynb for how to fully form sequences from a root")
    
    # Form and merge propgations from root_elec
    coc_clusters = form_coc_clusters(root_elec, time_frame, verbose=verbose)

    if len(coc_clusters) == 0:
        return []

    setup_coc_clusters(coc_clusters)

    # Below curation is because if allowed_root_times stay together, it is possible to only have 1 loose elec (the root elec)
    coc_clusters = [cluster for cluster in coc_clusters if len(cluster.inner_loose_elecs) >= MIN_INNER_LOOSE_DETECTIONS and len(cluster.loose_elecs) >= MIN_LOOSE_DETECTIONS_N]

    # assign_spikes(coc_clusters, time_frame, only_max_amps=False, #True,
    #               max_latency_diff=max_latency_diff, max_amp_median_diff=max_amp_median_diff,
    #               verbose=verbose)

    # coc_clusters = [cluster for cluster in coc_clusters if len(cluster._spike_train) > 3]  # Need more than 3 spikes for dip test

    if len(coc_clusters) == 0:
        return []

    merges = merge_coc_clusters(coc_clusters, verbose=verbose)
    
    # assign_spikes(coc_clusters, time_frame, only_max_amps=True,
    #               max_latency_diff=max_latency_diff, max_amp_median_diff=max_amp_median_diff,
    #               min_inner_cocs=min_inner_coacs, min_outer_cocs=min_outer_cocs,
    #               verbose=verbose)
    
    # setup_coc_clusters(merges)
    
    return merges

# endregion


# region Using sorters.py
def clusters_to_clusters(clusters):
    """
    If si_rec6.py is changed, CocCluster objs created in si_rec6.ipynb before the change
    cannot be pickled (which is needed for parallel processing)
    
    This function will initialize new CocCluster objs
    """
    new_clusters = []
    for clust in clusters:
        new_clust = CocCluster(clust.root_elec, clust.split_elecs, clust.spike_train)
        if hasattr(clust, "idx"):
            new_clust.idx = clust.idx
        new_clusters.append(new_clust)
    return new_clusters


def clusters_to_units(clusters):
    # Convert CocCluster objs to Unit objs
    all_units = []
    for c, clust in enumerate(clusters):
        unit = Unit(c, clust.spike_train, clust.root_elecs[0], recording=None)  # recording=None for parallel processing
        unit.root_elecs = clust.root_elecs
        unit.mean_amps = clust.all_amp_medians
        all_units.append(unit)
    return all_units


def clusters_to_sorter(clusters):
    return SpikeSorter(RECORDING, "RT-Sort", units=clusters_to_units(clusters))
# endregion


# region Kilosort comparison
def select_prop_spikes_within_kilosort_spikes(prop_units, ks_units,
                                              max_ms_dist=0.4, max_micron_dist=100,
                                              return_ks_only_units=False):
    """
    For each prop unit, exclude spikes that occur when no spikes from kilosort are detected 
    within max_ms_dist and max_micron_dist
    
    Pseudocode:
    [sorted spike times]
    [xy position]
    for each prop unit
        for each spike
            np.serachsorted(sorted spike times, time-0.4ms)
            while loop  
                if any xy position and spike time close enough, break loop and count spike
                store which spikes are found by ks and which are not
    Returns:
        within_prop_units
            Prop units whose spike trains are detected by kilosort
        outside_prop_units
            Prop units whose spike trains are not detected by kilosort
    
        if return_ks_units: return ks units whose spikes trains are not detected by prop
    """
    
    # Get sorted spike times and corresponding xy-positions
    chan_locs = ELEC_LOCS
    
    all_ks_spike_times, all_ks_spike_locs, all_ks_ids = [], [], []
    for idx, unit in enumerate(ks_units):
        all_ks_spike_times.extend(unit.spike_train)
        all_ks_spike_locs += [chan_locs[unit.chan]] * len(unit.spike_train)
        all_ks_ids += [idx] * len(unit.spike_train)
        
    order = np.argsort(all_ks_spike_times)
    all_ks_spike_times, all_ks_spike_locs, all_ks_ids = np.array(all_ks_spike_times)[order], np.array(all_ks_spike_locs)[order], np.array(all_ks_ids)[order]
    matched_ks_spikes = set()  # (ks_id, ks_spike_time) in all_ks_spike_times that have already been matched to a prop spike
    
    # Start loop
    within_prop_units = []
    outside_prop_units = []
    for unit in prop_units:
        loc = chan_locs[unit.chan]
        within_spikes = []
        outside_spikes = []
        for spike in unit.spike_train:
            idx = np.searchsorted(all_ks_spike_times, spike-max_ms_dist, side="left")  # Find nearest spike in all_ks_spike_times
            while idx < len(all_ks_spike_times) and np.abs(all_ks_spike_times[idx] - spike) <= max_ms_dist:
                ks_spike = (all_ks_spike_times[idx], all_ks_ids[idx])
                if ks_spike not in matched_ks_spikes and utils.calc_dist(*all_ks_spike_locs[idx], *loc) <= max_micron_dist:
                    within_spikes.append(spike)
                    matched_ks_spikes.add(ks_spike)
                    break
                idx += 1
            else:
                outside_spikes.append(spike)
        
        within_prop_units.append(Unit(unit.idx, np.array(within_spikes), unit.chan, unit.recording))
        outside_prop_units.append(Unit(unit.idx, np.array(outside_spikes), unit.chan, unit.recording))
        # within_prop_units.append(PropUnit(unit.df, unit.idx, np.array(within_spikes), unit.recording))
        # outside_prop_units.append(PropUnit(unit.df, unit.idx, np.array(outside_spikes), unit.recording))

    if not return_ks_only_units:
        return within_prop_units, outside_prop_units
    
    ks_only_spikes = []
    for idx, unit in enumerate(ks_units):
        spike_train = []
        for spike in unit.spike_train:
            if (spike, idx) not in matched_ks_spikes:
                spike_train.append(spike)
        ks_only_spikes.append(Unit(idx, np.array(spike_train), unit.chan, unit.recording))
    
    return within_prop_units, outside_prop_units, ks_only_spikes
# endregion


# region Kilosort as sequences (see end of si_rec9.ipynb)
def set_ks_only_spike_match_scores(ks_only_units, all_ks_sequences
                                   ):
    undetectable_seqs = []  # No inner_loose_eles
    
    for unit, seq in zip(tqdm(ks_only_units), all_ks_sequences):
        elec = unit.chan
        unit.time_to_spike_match = {}
        
        loose_elecs = seq.loose_elecs
        inner_loose_elecs = seq.inner_loose_elecs
        
        if len(inner_loose_elecs) == 0:
            undetectable_seqs.append((seq))
            for time in unit.spike_train:   
                unit.time_to_spike_match[time] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            continue
        
        for time in unit.spike_train:          
            score = np.nan # latency_diff / max_latency_diff + amp_median_diff / max_amp_median_diff + new elec overlap score
            num_inner_loose_elecs, num_loose_elecs, ratio_loose_elecs, latency_diff, amp_median_diff = get_seq_spike_overlap(seq, time)
            if num_loose_elecs < 2:
                unit.time_to_spike_match[time] = (elec, num_inner_loose_elecs, num_loose_elecs, ratio_loose_elecs, np.nan, np.nan, np.nan)
            else:
                unit.time_to_spike_match[time] = (elec, num_inner_loose_elecs, num_loose_elecs, ratio_loose_elecs, latency_diff, amp_median_diff, score)

    print(f"Undetectable sequences: {[seq.idx for seq in undetectable_seqs]}")
    
def get_tp_fp_fn(metric_idx, all_ks_sequences, prop_and_ks_units, prop_only_units, ks_only_units):
    """
    Params
        include_nan
            Whether to include nan values in returned arrays
            (Needed when creating scatter plot when one metric on x-axis and another on y-axis)
    """
       
    true_positives, false_positives, false_negatives = [], [], []
    for spike_type, all_units, all_seqs in (
        (true_positives, prop_and_ks_units, all_ks_sequences),
        (false_positives, prop_only_units, all_ks_sequences),
        (false_negatives, ks_only_units, ks_only_units)
        ):
        for unit, seq in zip(all_units, all_seqs):
            for time in unit.spike_train:
                metric = seq.time_to_spike_match[time][metric_idx]
                if not np.isnan(metric):
                    spike_type.append(metric) 
    return true_positives, false_positives, false_negatives  

def get_spike_metrics(all_ks_sequences, prop_and_ks_units, prop_only_units, ks_only_units):
    """
    Same as get_tp_fp_fn but get all metrics for each spike instead of only 1
    
    Params
        include_nan
            Whether to include nan values in returned arrays
            (Needed when creating scatter plot when one metric on x-axis and another on y-axis)
    """
       
    true_positives, false_positives, false_negatives = [], [], []
    for spike_type, all_units, all_seqs in (
        (true_positives, prop_and_ks_units, all_ks_sequences),
        (false_positives, prop_only_units, all_ks_sequences),
        (false_negatives, ks_only_units, ks_only_units)
        ):
        for unit, seq in zip(all_units, all_seqs):
            for time in unit.spike_train:
                spike_type.append(seq.time_to_spike_match[time]) 
    return true_positives, false_positives, false_negatives  
# endregion


# region For BrainDance
class RTSort:
    def __init__(self, all_sequences, model:ModelSpikeSorter, scaled_traces_path, 
                 buffer_size=100,
                 device="cuda", dtype=torch.float16):        
        self.buffer_size = buffer_size  # Frames
        self.device = device
        self.dtype = dtype
        
        torch.backends.cudnn.benchmark = True
        
        self.model = model.compile(NUM_ELECS)
        self.front_buffer = model.buffer_front_sample
        self.end_buffer = model.buffer_end_sample
        
        # region Setup tensors
        num_seqs = len(all_sequences)

        all_comp_elecs = set()
        # seq_n_before = -np.inf
        # seq_n_after = -np.inf
        seq_no_overlap_mask = torch.full((num_seqs, num_seqs), 0, dtype=torch.bool, device=device)
        for a, seq in enumerate(all_sequences):
            all_comp_elecs.update(seq.comp_elecs)
            # seq_n_before = max(seq_n_before, ceil(np.abs(np.min(seq.all_latencies[seq.comp_elecs]))) + CLIP_LATENCY_DIFF)
            # seq_n_after = max(seq_n_after, ceil(np.abs(np.max(seq.all_latencies[seq.comp_elecs]))) + CLIP_LATENCY_DIFF)
            for b, seq_b in enumerate(all_sequences):
                if calc_elec_dist(seq.root_elec, seq_b.root_elec) > INNER_RADIUS:
                    seq_no_overlap_mask[a, b] = 1
        self.seq_no_overlap_mask = seq_no_overlap_mask

        all_comp_elecs = list(all_comp_elecs)
        seq_n_before = N_BEFORE
        seq_n_after = N_AFTER 
        self.spike_arange = torch.arange(0, seq_n_before+seq_n_after+1, device=device)

        seqs_root_elecs = set()
        seqs_root_elecs_rel_comp_elecs = []
        seqs_inner_loose_elecs = []
        seqs_min_loose_elecs = []
        seqs_loose_elecs = []
        seqs_latencies = []
        seqs_latency_weights = []
        seqs_amps = []
        seqs_root_amp_means = []
        seqs_root_amp_stds = []
        seqs_amp_weights = []
        for seq in all_sequences: 
            seqs_root_elecs.add(seq.root_elec)
            seqs_root_elecs_rel_comp_elecs.append(all_comp_elecs.index(seq.root_elec))

            # Binary arrays (1 for comp_elec in inner_loose_elecs/loose_elecs, else 0)
            seqs_inner_loose_elecs.append(torch.tensor([1 if elec in seq.inner_loose_elecs else 0 for elec in all_comp_elecs], dtype=torch.bool, device=device))
            seqs_loose_elecs.append(torch.tensor([1 if elec in seq.loose_elecs else 0 for elec in all_comp_elecs], dtype=torch.bool, device=device))
            
            seqs_min_loose_elecs.append(ceil(seq.min_loose_detections))
            
            seq_comp_elecs = set(seq.comp_elecs)
            seqs_latencies.append(torch.tensor([seq.all_latencies[elec] + seq_n_before if elec in seq_comp_elecs and elec != seq.root_elec else 0 for elec in all_comp_elecs], dtype=dtype, device=device))
            seqs_amps.append(torch.tensor([seq.all_amp_medians[elec] if elec in seq_comp_elecs else 1 for elec in all_comp_elecs], dtype=dtype, device=device))  # Needs to be 1 to prevent divide by zero
            
            elec_probs = torch.tensor([seq.all_elec_probs[elec] if elec in seq_comp_elecs and elec != seq.root_elec else 0 for elec in all_comp_elecs], dtype=dtype, device=device)
            seqs_latency_weights.append(elec_probs/torch.sum(elec_probs))
            
            elec_probs = torch.tensor([seq.all_elec_probs[elec] if elec in seq_comp_elecs else 0 for elec in all_comp_elecs], dtype=dtype, device=device)
            seqs_amp_weights.append(elec_probs/torch.sum(elec_probs))
            
            seqs_root_amp_means.append(seq.all_amp_medians[seq.root_elec])
            seqs_root_amp_stds.append(seq.root_to_amp_median_std[seq.root_elec])
            
        self.comp_elecs = torch.tensor(all_comp_elecs, dtype=torch.long, device=device)[:, None]
        self.comp_elecs_flattened = self.comp_elecs.flatten()
        
        self.seqs_root_elecs = list(seqs_root_elecs)
            
        self.seqs_inner_loose_elecs = torch.vstack(seqs_inner_loose_elecs)
        self.seqs_loose_elecs = torch.vstack(seqs_loose_elecs)
        self.seqs_min_loose_elecs = torch.tensor(seqs_min_loose_elecs, dtype=torch.int16, device=device)
        self.seqs_latencies = torch.vstack(seqs_latencies)
        self.seqs_amps = torch.vstack(seqs_amps)
        self.seqs_latency_weights = torch.vstack(seqs_latency_weights)
        self.seqs_amp_weights = torch.vstack(seqs_amp_weights)
        self.seqs_root_amp_means = torch.tensor(seqs_root_amp_means, dtype=dtype, device=device)
        self.seqs_root_amp_stds = torch.tensor(seqs_root_amp_stds, dtype=dtype, device=device)
            
        self.seqs_root_elecs_rel_comp_elecs = seqs_root_elecs_rel_comp_elecs
            
        self.max_pool = torch.nn.MaxPool1d(seq_n_before+seq_n_after+1, return_indices=True)

        # step_size = OUTPUT_WINDOW_HALF_SIZE * 2 - seq_n_after - seq_n_before
        # endregion
        
        # These variables need to be cached when call self.sort() multiple times
        self.overlap = round(OVERLAP_TIME * SAMP_FREQ)
        self.last_pre_median_output_frame = -np.inf
        
        self.input_scale = model.input_scale
        # self.inference_scaling_numerator = INFERENCE_SCALING_NUMERATOR
        # self.inference_scaling = None
        # Calculating this at start of experiment causes up to first second data to be lagged/frames missing
        # if INFERENCE_SCALING_NUMERATOR is not None:
        #     iqrs = scipy.stats.iqr(self.pre_median_frames, axis=1)
        #     median = np.median(iqrs)
        #     self.inference_scaling = self.inference_scaling_numerator / median
        # else:
        #     self.inference_scaling = 1  # self.inference_scaling=1 could be in __init__ BUT then another variable would be needed to wait for the first PRE_MEDIAN_FRAMES frames. Setting input_scale like this means another variable does not need to be set
        if INFERENCE_SCALING_NUMERATOR is not None:
            window = np.load(scaled_traces_path, mmap_mode="r")[:, :PRE_MEDIAN_FRAMES]
            iqrs = scipy.stats.iqr(window, axis=1)
            median = np.median(iqrs)
            inference_scaling = INFERENCE_SCALING_NUMERATOR / median
            self.input_scale *= inference_scaling
        
        # last_detections = torch.full((num_seqs,), -PRE_MEDIAN_FRAMES, dtype=torch.int64, device=device)
                
        # self.input_chunk = np.full((NUM_ELECS, model.sample_size), np.nan, dtype="float16")
        self.input_size = model.sample_size
        
        self.pre_median_frames = torch.full((NUM_ELECS, PRE_MEDIAN_FRAMES), torch.nan, dtype=dtype, device=device)
        self.pre_medians = None  # Used to divide traces
        self.cur_num_pre_median_frames = 0
        self.total_num_pre_median_frames = PRE_MEDIAN_FRAMES
        
        # self.pre_sub_medians = None # torch.full((NUM_ELECS, 1), 0, dtype=dtype, device=device)  # Used to subtract traces
        # self.cur_num_pre_sub_median_frames = 0
        # self.total_num_pre_sub_median_frames = model.sample_size  # Results are different (probably worse) if use different sized chunks
        
        self.num_seqs = len(all_sequences)
        self.num_elecs = NUM_ELECS
        self.seq_n_before = seq_n_before
        self.seq_n_after = seq_n_after
        self.stringent_thresh_logit = STRINGENT_THRESH_LOGIT
        self.loose_thresh_logit = LOOSE_THRESH_LOGIT
        self.min_elecs_for_array_noise = MIN_ELECS_FOR_ARRAY_NOISE
        self.min_inner_loose_detections = MIN_INNER_LOOSE_DETECTIONS
        self.clip_latency_diff = CLIP_LATENCY_DIFF
        self.max_latency_diff_spikes = MAX_LATENCY_DIFF_SPIKES
        self.max_root_amp_median_std_spikes = MAX_ROOT_AMP_MEDIAN_STD_SPIKES
        self.clip_amp_median_diff = CLIP_AMP_MEDIAN_DIFF
        self.max_amp_median_diff_spikes = MAX_AMP_MEDIAN_DIFF_SPIKES
        
        self.latest_frame = 0  # Spike times returned from self.sort() will be relative to the first frame of the first chunk received by RT-Sort
    
    def reset(self):
        # Reset variables like pre_medians and input_scale that need to be reset for each experiment
        # self.inference_scaling = None
        self.latest_frame = 0
        self.cur_num_pre_median_frames = 0
        self.pre_medians = None
        # self.pre_sub_medians = torch.full_like(self.pre_sub_medians, 0) 
        self.pre_median_frames = torch.full_like(self.pre_median_frames, torch.nan)
    
    def sort(self, obs):
        """
        Params
            obs:
                from env, obs = maxwell_env.step()
        """
        # TODO: Make sure that Maxwell MEA data is in uV
        
        detections = [] 
        
        # Update internal traces cache
        # TODO: May need to slice :obs: if maxwell sends data from unselected elecs
        # self.input_chunk[:, :-self.buffer_size] = self.input_chunk[:, self.buffer_size:]
        # self.input_chunk[:, -self.buffer_size:] = np.asarray(obs).T 
        self.pre_median_frames[:, :-self.buffer_size] = self.pre_median_frames[:, self.buffer_size:]
        # self.pre_median_frames[:, -self.buffer_size:] = torch.tensor(np.asarray(obs), device=self.device, dtype=self.dtype).T - self.pre_sub_medians  # Need to do "np.asarray(obs)" before converting to torch tensor because it is much faster (otherwise, frames are lost using 5ms buffer because it is so slow)
        
        obs = torch.tensor(np.asarray(obs), device=self.device, dtype=self.dtype).T
        obs -= torch.median(obs, dim=1, keepdim=True).values
        self.pre_median_frames[:, -self.buffer_size:] = obs
        self.latest_frame += obs.shape[1]

        self.cur_num_pre_median_frames += self.buffer_size
        if self.cur_num_pre_median_frames >= self.total_num_pre_median_frames:
            self.cur_num_pre_median_frames = 0
            # self.pre_sub_medians = torch.median(self.pre_median_frames, dim=1, keepdim=True).values   # TODO: uncomment this
            # if self.pre_medians is None:  # Need to subtract self.pre_sub_medians on first full self.pre_median_frames (when there was not enough data to calculate self.pre_sub_medians)
            #     self.pre_median_frames -= self.pre_sub_medians 
            
            # rec_window = torch.tensor(self.pre_median_frames, dtype=self.dtype, device=self.device)
            # pre_medians = rec_window[self.comp_elecs_flattened, :]
            pre_medians = self.pre_median_frames[self.comp_elecs_flattened]
            pre_medians = torch.median(torch.abs(pre_medians), dim=1).values  # Pytorch median different than numpy median: https://stackoverflow.com/a/54310996
            self.pre_medians = torch.clip(pre_medians / 0.6745, min=0.5, max=None)  # a_min=0.5 to prevent dividing by zero when data is just 1s and 0s (median could equal 0) 
            # if self.inference_scaling is None:
            #     self.inference_scaling = 1
                # if INFERENCE_SCALING_NUMERATOR is not None:
                #     iqrs = scipy.stats.iqr(self.pre_median_frames, axis=1)
                #     median = np.median(iqrs)
                #     self.inference_scaling = self.inference_scaling_numerator / median
                # else:
                #     self.inference_scaling = 1  # self.inference_scaling=1 could be in __init__ BUT then another variable would be needed to wait for the first PRE_MEDIAN_FRAMES frames. Setting input_scale like this means another variable does not need to be set
        
        #  if self.inference_scaling is None or self.pre_medians is None:
        if self.pre_medians is None:
            return detections
        
        # Run DL model
        # traces_torch = torch.tensor(self.pre_median_frames[:, -self.input_size:], device=self.device)
        # traces_torch -= torch.median(traces_torch, dim=1, keepdim=True).values
        traces_torch = self.pre_median_frames[:, -self.input_size:]
        torch_window = self.model(traces_torch[:, None, :] * self.input_scale)[:, 0, :]
        
        rec_window = traces_torch[:, self.front_buffer:-self.end_buffer]

        # region Find peaks    
        # start = perf_counter()
        window = torch_window[self.seqs_root_elecs, self.seq_n_before-1:-self.seq_n_after+1]
        main = window[:, 1:-1]
        greater_than_left = main > window[:, :-2]
        greater_than_right = main > window[:, 2:]
        peaks = greater_than_left & greater_than_right
        crosses = main >= self.stringent_thresh_logit
        peak_ind_flat = torch.nonzero(peaks & crosses, as_tuple=True)[1]
        # peak_ind_flat = peak_ind_flat[torch.sum(TORCH_OUTPUTS[:, output_start_frame+peak_ind_flat+seq_n_before] >= LOOSE_THRESH_LOGIT, dim=0) <= MIN_ELECS_FOR_ARRAY_NOISE]  # may be slow
        peak_ind_flat = peak_ind_flat[torch.sum(torch_window[:, peak_ind_flat+self.seq_n_before] >= self.loose_thresh_logit, dim=0) <= self.min_elecs_for_array_noise]  # may be slow
        # if torch.numel(peak_ind_flat) == 0:  # This is very rare (only happend once for 1/16/24 spikeinterface simulated recording)
        #     # end = perf_counter()
        #     # delays_find_peak.append((end-start)*1000)
        #     # delays_total.append(delays_find_peak[-1])
        #     # delays_total_spike_detected.append(False)
        #     continue
        # end = perf_counter()
        # delays_find_peak.append((end-start)*1000)
        # endregion
        
        # # Plot peak waveform and detection footprints
        # peak=peak_ind_flat[0].item()
        # elec=seqs_root_elecs[0]
        # ##
        # frame = peak + seq_n_before + rec_start_frame
        # ms = frame / SAMP_FREQ
        # plot_spikes([ms], elec)
        # plt.show()
            
        # start = perf_counter()
        peak_ind = peak_ind_flat[:, None, None]  # Relative to output_start_frame+seq_n_before
        spike_window = torch_window[self.comp_elecs, peak_ind + self.spike_arange] 

        elec_probs, latencies = self.max_pool(spike_window)  # Latencies are relative to peak-seq_n_before
        elec_crosses = (elec_probs >= self.loose_thresh_logit).transpose(1, 2)
        num_inner_loose = torch.sum(elec_crosses & self.seqs_inner_loose_elecs, dim=2)
        pass_inner_loose = num_inner_loose >= self.min_inner_loose_detections

        num_loose = torch.sum(elec_crosses & self.seqs_loose_elecs, dim=2)
        pass_loose = num_loose >= self.seqs_min_loose_elecs
        # end = perf_counter()
        # delays_elec_cross.append((end-start)*1000)

        # Slower to check this since so many DL detections
        # can_spike = pass_inner_loose & pass_loose
        # if not torch.any(can_spike):
        #     continue

        # start = perf_counter()
        latencies_float = latencies.transpose(1, 2).to(self.dtype)
        latency_diff = torch.abs(latencies_float - self.seqs_latencies)
        latency_diff = torch.clip(latency_diff, min=None, max=self.clip_latency_diff)
        latency_diff = torch.sum(latency_diff * self.seqs_latency_weights, axis=2)
        pass_latency = latency_diff <= self.max_latency_diff_spikes
        # end = perf_counter()
        # delays_latency.append((end-start)*1000)

        # # Getting rec_window was moved up in code to not include it in sorting computation time
        # # start = perf_counter()
        # # rec_start_frame = output_start_frame + FRONT_BUFFER
        # rec_start_frame = (output_frame - OUTPUT_WINDOW_HALF_SIZE) + FRONT_BUFFER
        # # amps = torch.abs(TORCH_TRACES[comp_elecs, rec_start_frame + peak_ind + latencies].transpose(1, 2)) / pre_medians  # peak_ind+seq_n_before = peak_ind_in_rec_window. latency-seq_n_before = latency_rel_peak_ind. --> (peak_ind+seq_n_before)+(latency-seq_n_before) = peak_ind+latency = spike index in rec_window
        # rec_window = torch.tensor(TRACES[:, rec_start_frame:rec_start_frame+OUTPUT_WINDOW_HALF_SIZE*2], dtype=dtype, device=device)
        amps = torch.abs(rec_window[self.comp_elecs, peak_ind + latencies].transpose(1, 2)) / self.pre_medians  # peak_ind+seq_n_before = peak_ind_in_rec_window. latency-seq_n_before = latency_rel_peak_ind. --> (peak_ind+seq_n_before)+(latency-seq_n_before) = peak_ind+latency = spike index in rec_window

        root_amp_z = torch.abs(amps[:, 0, self.seqs_root_elecs_rel_comp_elecs] - self.seqs_root_amp_means) / self.seqs_root_amp_stds
        pass_root_amp_z = root_amp_z <= self.max_root_amp_median_std_spikes 
        # end = perf_counter()
        # delays_root_z.append((end-start)*1000)

        # start = perf_counter()
        amp_diff = torch.abs(amps - self.seqs_amps) / self.seqs_amps
        amp_diff = torch.clip(amp_diff, min=None, max=self.clip_amp_median_diff)
        amp_diff = torch.sum(amp_diff * self.seqs_amp_weights, axis=2)
        pass_amp_diff = amp_diff <= self.max_amp_median_diff_spikes
        # end = perf_counter()
        # delays_amp.append((end-start)*1000)

        # Since every seq is compared to every peak, this is needed (or a form of this) so (in extreme case) peak detected on one side of array is not assigned to seq on other side by coincidence
        strict_crosses_root = spike_window[:, self.seqs_root_elecs_rel_comp_elecs, self.seq_n_before] >= self.stringent_thresh_logit

        # start = perf_counter()
        can_spike = strict_crosses_root & pass_inner_loose & pass_loose & pass_latency & pass_root_amp_z & pass_amp_diff
        # end = perf_counter()
        # delays_can_spike.append((end-start)*1000)

        # Slighty faster than the following due to only slicing once: # spike_scores = latency_diff[can_spike] / MAX_LATENCY_DIFF_SPIKES + amp_diff[can_spike] / MAX_AMP_MEDIAN_DIFF_SPIKES  - num_loose[can_spike] / torch.sum(elec_crosses, dim=2)
        # start = perf_counter()
        spike_scores = latency_diff / self.max_latency_diff_spikes + amp_diff / self.max_amp_median_diff_spikes - (num_loose / torch.sum(elec_crosses, dim=2) * 0.5)
        spike_scores = 2.1 - spike_scores  # (additional 0.1 in case spike_scores=2)
        spike_scores *= can_spike

        # For debugging:
        # breakpoint: output_start_frame+seq_n_before <= rec_ms_to_output_frame(TIME) < output_frame+OUTPUT_WINDOW_HALF_SIZE-seq_n_after
        # time = (peak_ind.flatten()[2].item() + rec_start_frame + START_FRAME)/SAMP_FREQ
        # plot_spikes([time], all_sequences[20].root_elec)
        # plt.show()

        peak_ind_2d = peak_ind[:, 0]
        # next_can_spike = torch.full_like(can_spike, fill_value=1, dtype=torch.bool, device=device)   
        # end = perf_counter()
        # cur_delay_split_spike = (end-start)*1000
        
        # cur_delay_assign_spike = 0
        # spike_detected = False  # TODO: This is only needed for speed testing
        while torch.any(spike_scores):
            # start = perf_counter()
            spike_seq_idx = torch.argmax(spike_scores).item()
            spike_idx = spike_seq_idx // self.num_seqs
            seq_idx = spike_seq_idx % self.num_seqs
            offset_spike_time = peak_ind_flat[spike_idx].item()  # TODO: If spike assignment is now slow, remove .item() (which gets the spike time in python value from torch tensor)
            spike_time = offset_spike_time + self.seq_n_before + (self.latest_frame - self.input_size) # + rec_start_frame 
            # (self.latest_frame - self.input_size) == rec_start_frame
            # end = perf_counter()
            # cur_delay_assign_spike += ((end-start)*1000)
            
            # start = perf_counter()
            # if spike_time - last_detections[seq_idx] <= OVERLAP:
                # spike_scores[spike_idx, seq_idx] = 0
                # continue
            
            detections.append((seq_idx, spike_time))
            # detections[seq_idx].append(spike_time)
            
            # Spike splitting for current window
            # set score to 0 if (seq is spatially close enough) and (peak is temporally close enough)
            # keep score if (seq is NOT spatially close enough) or (peak is temporally far enough)
            spike_scores *= self.seq_no_overlap_mask[seq_idx] | (torch.abs(peak_ind_2d - offset_spike_time) > self.overlap)
            # end = perf_counter()
            # cur_delay_split_spike += ((end-start)*1000)
            # spike_detected = True
            
            # Spike splitting for next window
            # FAST: just change seq_n_before for all seqs
            # Slower: last_spike[~no_overlap_mask] = max(last_spike[~no_overlap_mask], spike_time)
            # last_detections[(~no_overlap_mask) & (last_detections < spike_time)] = spike_time

            # delays_assign_spike.append(cur_delay_assign_spike)
            # delays_split_spike.append(cur_delay_split_spike)
            # delays_total.append(delays_find_peak[-1] + delays_elec_cross[-1] + delays_latency[-1] + delays_root_z[-1] + delays_amp[-1] + delays_can_spike[-1] + delays_assign_spike[-1] + delays_split_spike[-1])
            # delays_total_spike_detected.append(spike_detected)
            # torch.cuda.synchronize()  # Not needed because of while loop
            # end = perf_counter()
            # sorting_computation_times.append((end-start_sorting)*1000)
        return detections    

    def save(self, pickle_path):
        self.model = None
        utils.pickle_dump(self, pickle_path)

    @staticmethod
    def load_from_file(pickle_path, model):
        rt_sort = utils.pickle_load(pickle_path)
        rt_sort.model = model.compile(rt_sort.num_elecs)
        return rt_sort
# endregion


if __name__ == "__main__":
    # Parameters for recording
    MEA = False
    
    RECORDING = utils.rec_si()
    CHANS_RMS = utils.chans_rms_si()

    SAMP_FREQ = RECORDING.get_sampling_frequency()
    NUM_ELECS = RECORDING.get_num_channels()
    ELEC_LOCS = RECORDING.get_channel_locations()

    ALL_CLOSEST_ELECS = []
    for elec in range(NUM_ELECS):
        elec_ind = []
        dists = []
        x1, y1 = ELEC_LOCS[elec]
        for elec2 in range(RECORDING.get_num_channels()):
            if elec == elec2:
                continue
            x2, y2 = ELEC_LOCS[elec2]
            dists.append(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
            elec_ind.append(elec2)
        order = np.argsort(dists)
        ALL_CLOSEST_ELECS.append(np.array(elec_ind)[order])   
        
    ALL_CROSSINGS = np.load("/data/MEAprojects/dandi/000034/sub-mouse412804/dl_model/prop_signal/all_crossings.npy", allow_pickle=True)
    ALL_CROSSINGS = [tuple(cross) for cross in ALL_CROSSINGS]
    
    ELEC_CROSSINGS_IND = np.load("/data/MEAprojects/dandi/000034/sub-mouse412804/dl_model/prop_signal/elec_crossings_ind.npy", allow_pickle=True)
    ELEC_CROSSINGS_IND = [tuple(ind) for ind in ELEC_CROSSINGS_IND]  # [(elec's cross times ind in all_crossings)]
    
    TRACES_PATH = "/data/MEAprojects/dandi/000034/sub-mouse412804/traces.npy"
    FILT_TRACES_PATH = "/data/MEAprojects/dandi/000034/sub-mouse412804/FILT_TRACES.npy"
    MODEL_OUTPUTS_PATH = "/data/MEAprojects/dandi/000034/sub-mouse412804/dl_model/outputs.npy"
    
    TRACES = np.load(TRACES_PATH, mmap_mode="r")
    FILT_TRACES = np.load(FILT_TRACES_PATH, mmap_mode="r")
    OUTPUTS = np.load(MODEL_OUTPUTS_PATH, mmap_mode="r")
    
    # RT-Sort hyperparameters
    # (N_BEFORE and N_AFTER need to be set in .ipynb because they rely on SAMP_FREQ)
    N_BEFORE = N_AFTER = round(0.5 * SAMP_FREQ)  # Window for looking for electrode codetections
    
    # MIN_ELECS_FOR_ARRAY_NOISE = round(0.5 * NUM_ELECS)
    MIN_ELECS_FOR_ARRAY_NOISE = MIN_ELECS_FOR_SEQ_NOISE = max(100, round(0.15 * NUM_ELECS))  # see filter_clusters()
    
    FRONT_BUFFER = round(2*SAMP_FREQ)
    OUTPUT_WINDOW_HALF_SIZE = round(3*SAMP_FREQ)
    
    PRE_MEDIAN_FRAMES = round(50 * SAMP_FREQ)
    MIN_ACTIVITY = 30  # 0.05Hz in pre-recording
    
    
MIN_ACTIVITY_ROOT_COCS = 2
SPLIT_ROOT_AMPS_AGAIN = True

STRINGENT_THRESH = None # 0.175
STRINGENT_THRESH_LOGIT = None # sigmoid_inverse(STRINGENT_THRESH)
LOOSE_THRESH = None # 0.127
LOOSE_THRESH_LOGIT = None # sigmoid_inverse(LOOSE_THRESH)  # -2.1972245773362196  # For faster computation in spike splitting
INFERENCE_SCALING_NUMERATOR = None

INNER_RADIUS = 50
OUTER_RADIUS = 100

PRE_INTERELEC_ROOT_MAX_AMP_ONLY = True

MIN_AMP_DIST_P = 0.1  

RELOCATE_ROOT_MAX_LATENCY = -2  # Frames
RELOCATE_ROOT_MIN_AMP = 0.8  # Fraction of current root amp's

# #electrode codetections overlap for spike assignment and merging
MIN_LOOSE_DETECTIONS_N = 4
MIN_LOOSE_DETECTIONS_R_SPIKES = 1/3 
MIN_LOOSE_DETECTIONS_R_SEQUENCES = 1/3
MIN_INNER_LOOSE_DETECTIONS = 3

MAX_LATENCY_DIFF_SPIKES = 2.5
MAX_AMP_MEDIAN_DIFF_SPIKES = 0.45

MAX_LATENCY_DIFF_SEQUENCES = 2.5
MAX_AMP_MEDIAN_DIFF_SEQUENCES = 0.45
    
MIN_ELEC_PROB = 0.03  # If an elec's mean prob is less than MIN_ELEC_PROB, it is set to 0

CLIP_LATENCY_DIFF = 5
CLIP_AMP_MEDIAN_DIFF = 0.9

OVERLAP_TIME = 0.2  # For spike splitting and merging clusters with overlapping spikes

MAX_ROOT_AMP_MEDIAN_STD_SPIKES = 2.5
MAX_ROOT_AMP_MEDIAN_STD_SEQUENCES = 2.5