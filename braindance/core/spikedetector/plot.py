import numpy as np
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from matplotlib.axes._axes import Axes

from braindance.core.spikedetector import utils

TRACE_X_LABEL = "Time (ms)"


def plot_waveform(waveform, peak_idx, wf_len, wf_alpha, wf_trace_loc,
                  axis, xlim=None, ylim_diff=None, title="Waveform",
                  **wf_line_kwargs):
    # wf_trace_loc is the index of the waveform in the overall trace
    # axis is the subplot/axis to plot the waveform on
    # ylim_diff is the difference between the maximum ylim and the minimum ylim

    if title is not None:
        axis.set_title("Waveform")
    waveform_x = np.arange(wf_len) + wf_trace_loc - peak_idx
    axis.plot(waveform_x, waveform, **wf_line_kwargs, label=f"α: {wf_alpha}") # :.2f}")
    # Plot alpha
    # axis.plot((waveform_x[peak_idx], waveform_x[peak_idx]), (-wf_alpha, 0),
    #           linestyle="dotted", color="black")
    # label_alpha = Line2D([0], [0], label=f"Alpha = {alpha:.3f}", alpha=0)
    # axis.legend(handles=[label_alpha], frameon=False, loc="lower right")

    if xlim is not None:
        axis.set_xlim(xlim)
    if ylim_diff is not None:
        ylim_min, _ = axis.get_ylim()
        axis.set_ylim(ylim_min, ylim_min+ylim_diff)


def plot_hist_loc_mad(loc_deviations, n_bins=15):
    # Plot histogram of location MAD
    fig, ax = plt.subplots(1, tight_layout=True) 
    ax.set_title(f"Absolute deviation of location")
    bins = np.arange(n_bins + 1)
    ax.hist(loc_deviations)
    ax.set_xlabel("Milliseconds")
    # ax.set_xticks(bins)
    # ax.set_xlim(min(bins), max(bins))
    ax.set_ylabel("Count")
    ax.set_xlim(0, None)
    ax.scatter(0, 0, s=0, label=f"Mean = {np.mean(loc_deviations):.3f}")
    # ax.scatter(0, 0, s=0, label=f"{len([d for d in loc_deviations if d > n_bins])} outside")
    ax.legend(frameon=False)
    plt.show()


def plot_hist_percent_abs_error(percent_abs_errors, n_bins=10):
    # Plot histogram of alpha percent absolute error
    fig, ax = plt.subplots(1, tight_layout=True) 
    ax.set_title(f"Percent absolute error of trough amplitude")
    bins = np.arange((n_bins + 1) * 10, step=10)
    ax.hist(percent_abs_errors, bins=bins)
    ax.set_xlabel("Percent absolute error")
    ax.set_xticks(bins)
    ax.set_xlim(min(bins), max(bins))
    ax.set_ylabel("Count")
    ax.set_xlim(0, None)
    ax.scatter(0, 0, s=0, label=f"Mean = {np.mean(percent_abs_errors):.1f}%")
    ax.legend(frameon=False)
    plt.show()


def get_yticks_lim(trace, anchor=0, increment=5,
                   buffer_min=5, buffer_max=3):
    """
    Get lim and ticks for y-axis when trace is plotted

    :param trace: np.array
        Trace that will be plotted using the returned lim and ticks
    :param anchor: int or float
        The ticks will show anchor
    :param increment: int or float
        Increment between ticks
    :param buffer_min:
        Ticks will be within [min(trace) - buffer_min, max(trace) + buffer_max)]
    :param buffer_max:
        [min(trace) - buffer_min, max(trace) + buffer_max)]
    """
    trace_min = min(trace) - buffer_min
    trace_max = max(trace) + buffer_max

    ylim = (trace_min, trace_max)
    yticks = np.arange(
                anchor + np.floor(trace_min / increment) * increment,
                anchor + np.ceil(trace_max / increment) * increment + 1,
                increment
            )
    return yticks, ylim


def set_ticks(subplots: Tuple[Axes], trace: np.array, increment=10,
              buffer_min=10, buffer_max=10,
              center_xticks=False):
    """
    Set x and y ticks for subplots

    :param subplots
        Each element is a subplot
    :param trace
        The trace to calculate the appropatiate ticks for
    :param increment
    :param center_xticks
        Whether to set center of xticks to 0 (left is negative time and right is positive)
    """
    # samp_freq_khz=30  # Set this based on recording 
    
    yticks, ylim = get_yticks_lim(trace, 0, increment, buffer_min, buffer_max)

    sample_size = len(trace.flatten())
    
    xlim = (0, sample_size)
    xtick_locs = np.arange(0, sample_size + 1, samp_freq_khz)
    xtick_labels = xtick_locs / samp_freq_khz  # frames to milliseconds
    if center_xticks:
        xtick_labels -= (xtick_labels[-1] - xtick_labels[0]) / 2
    xtick_labels = xtick_labels.astype(int)
    
    for sub in subplots:
        sub.set_yticks(yticks)
        sub.set_ylim(ylim)

        sub.set_xticks(xtick_locs, xtick_labels)
        sub.set_xlim(xlim)

        sub.set_xlabel(TRACE_X_LABEL)


def set_dpi(dpi):
    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = dpi


def get_empty_line(label):
    # Get an invisible line that can be used to create a legend
    return Line2D([0], [0], alpha=0, label=label)


def display_prob_spike(spike_output, axis):
    # Display the model's probability of a spike occurring on axis
    # Plot model's probability of a spike
    spike_prob_legend = axis.legend(handles=[get_empty_line(f"ŷ = {spike_output * 100:.1f}%")],
                                    markerfirst=False,
                                    loc='upper left',
                                    handlelength=0,
                                    handletextpad=0)
    axis.add_artist(spike_prob_legend)


def unscaled_ticks_to_uv(subplot):
    # Convert yticks of :param subplot: from unscaled arbitrary units to microvolts
    yticks_uv = utils.round(subplot.get_yticks() * utils.FACTOR_UV)
    subplot.set_yticks(yticks_uv / utils.FACTOR_UV, yticks_uv)


def plot_hist_percents(data, ax=None, **hist_kwargs):
    # Plot a histogram with percents as y-axis
    # https://www.geeksforgeeks.org/matplotlib-ticker-percentformatter-class-in-python/
    # plt.hist(data, weights=np.ones(len(data)) / len(data), **hist_kwargs)
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=decimals))
    # plt.ylabel("Frequency")
    
    if ax is None:
        fig, ax = plt.subplots(1)

    # Create histogram
    n, bins, patches = ax.hist(data, **hist_kwargs)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks, [f'{y/len(data) * 100:.1f}%' for y in yticks])
    
    # ax.set_yticklabels([f'{x/len(data):.0f}' for x in ax.get_yticks()])
    ax.set_ylabel('Frequency')
    
    if "range" in hist_kwargs:
        ax.set_xlim(hist_kwargs["range"])

    return ax
    
