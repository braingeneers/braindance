import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter


def plot_heatmap_spikes(spikes, mapping, title=None, cmap='magma', save=False, filename=None):
    '''Plot heatmap of spikes, first converting to a list of positions, then calling
        plot_heatmap'''
    # Convert spikes to positions
    x,y = [],[]
    possible_chs = mapping['channel'].values
    for _,ch,amp in spikes:
        # Check if ch is in mapping
        if ch not in possible_chs:
            continue

        # Look up the x and y coordinates for this channel
        x_coord = mapping.loc[mapping['channel'] == ch, 'x'].values[0]
        y_coord = mapping.loc[mapping['channel'] == ch, 'y'].values[0]
        
        x.append(x_coord)
        y.append(y_coord)
        
    # Now that we have the x and y positions, we can plot the heatmap
    plot_heatmap(x, y, cmap=cmap)



def plot_heatmap(x, y, sigma=6, bins=1000, v_min=None, v_max=None, fig=None, ax=None, 
                 plt_show=True, cmap='viridis', conversion_factor=1, units='units',
                 cbar = True,log=False, alpha=1):
    def gauss_heatmap(x, y, s, bins=1000, conversion_factor=1):
        """Takes in scatter data x,y, along with smoothing factor s
        returns heatmap"""
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
        # print('Max: ', np.max(heatmap))
        heatmap = gaussian_filter(heatmap, sigma=s)
        # print('Max aft blur: ', np.max(heatmap))
        heatmap *= conversion_factor  # apply conversion factor
        # print('Max aft conv: ', np.max(heatmap))

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        return heatmap.T, extent

    if fig is None: 
        fig, ax = plt.subplots(1, 1)

    img, extent = gauss_heatmap(x, y, sigma, bins=bins, conversion_factor=conversion_factor)

    # Determine normalization
    if v_min is None:
        v_min = np.min(img)
    if v_max is None:
        v_max = np.max(img)
    
    if not log:
        norm = matplotlib.colors.Normalize(v_min, v_max)
    else:
        norm = matplotlib.colors.LogNorm(v_min, v_max)

    ax.imshow(img, extent=extent, origin='lower', cmap=cmap, norm=norm, alpha=alpha)

    

    # Apply the same normalization to the colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    if cbar:
        # Colorbar
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0)
        # cbar = fig.colorbar(mappable, cax=cax)
        # cbar.set_label('Hz')
        # # Set ticks to be 1 decimal place
        # cbar.set_ticks(np.linspace(v_min, v_max, 5))
        # cbar.set_ticklabels(['{:.1f}'.format(x) for x in np.linspace(v_min, v_max, 5)])
        # cbar = fig.colorbar(mappable, ax=ax)
        pass
    # Labels
    ax.set_xlabel('x ($\mu$m)')
    ax.set_ylabel('y ($\mu$m)')

    if plt_show:
        plt.show()
    return fig, ax


def plot_footprint(x,y,footprint,fig=None,ax=None, x_scale = .01, y_scale = .1,window_len=30, **kwargs):
    """
    Plot a waveform footprint at the corresponding channel
    
    Parameters
    ----------
    x : array_like
        x coordinate of channel
        
    y : array_like
        y coordinate of channel
    Footprint : array_like
        footprint of channel
    fig : matplotlib figure
        figure to plot on
    ax : matplotlib axis
        axis to plot on
    kwargs : dict
        kwargs to pass to plot
    """
    if fig is None:
        fig, ax = plt.subplots(1,1)

    # Make footprint the window length by taking the middle window_len/2 to the right and left
    len_footprint = len(footprint)
    if len_footprint < window_len:
        footprint = np.pad(footprint, (window_len - len_footprint, 0), 'constant', constant_values=0)
    else:
        footprint = footprint[len_footprint//2 - window_len//2:len_footprint//2 + window_len//2]
        
    t = np.arange(len(footprint)) * x_scale + x - len(footprint) * x_scale / 2
    footprint = footprint * y_scale + y
    ax.plot(t,footprint,**kwargs)
    return fig, ax

def plot_footprints(footprint_chs, footprint_waves, mapping, selected_channels, invert_y=True,
                    save_dir = None, annotate = True, **kwargs):
    fig, ax = plt.subplots(1,1)
    import seaborn as sns
    # colors the length of fp_ch
    # colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(footprint_chs)))
    # colors are seaborn pastel
    colors = sns.color_palette("pastel", len(footprint_chs))

    # Plot scatter square at every channel location
    xs,ys = mapping['x'].values, mapping['y'].values
    ax.scatter(xs,ys, s=2, color='k', marker='s', zorder=10, alpha=.5)

    # Color index
    c_i = 0
    for fp_ch, fp_wave, main_ch in zip(footprint_chs, footprint_waves, selected_channels):
        # Loop over each channel in the footprint
        for fp, wave in zip(fp_ch, fp_wave):
            x,y = mapping.loc[mapping['channel'] == fp, ['x', 'y']].values[0]
            # print(f'Channel: {fp}, x: {x}, y: {y}')
            if invert_y:
                wave = -wave
            fig, ax = plot_footprint(x,y,wave - np.median(wave), fig=fig, ax=ax, x_scale = .8, y_scale = 1, color=colors[c_i])
        # Plot circle around the main channel
        # ax.scatter(xs[main_ch],ys[main_ch], s=20, color=colors[c_i], marker='o', zorder=10, alpha=1)
        # Text label the 
        electrode = mapping.loc[mapping['channel'] == main_ch, 'electrode'].values[0]
        if annotate:
            ax.annotate(str(electrode), (xs[main_ch],ys[main_ch]), color='k', fontsize=12, zorder=10)

        c_i += 1

    #invert y axis
    if invert_y:
        ax.invert_yaxis()

    if save_dir is not None:
        plt.savefig(save_dir + '/footprints.png', dpi=300)

    return fig, ax


def plot_reactivity(analysis, file_path = None, fig=None, ax=None, **kwargs):

    fig, ax = plt.subplots(1,1)
    w = analysis.reactivity.sum(axis=2)
    plt.imshow(w)
    # set x ticks as stim_electrodes
    plt.ylabel('Stim Electrodes')
    plt.yticks(np.arange(0, len(analysis.stim_electrodes), 1.0))
    plt.gca().set_yticklabels(analysis.stim_electrodes)
    # set y ticks as channels of interest
    plt.xlabel('Channels of Interest')
        
    x_tick_labels = []
    
    for i in range(len(analysis.channels_of_interest)):
        # add the electrode conversion and orig_channel number
        ch = analysis.channels_of_interest[i]
        elec = analysis.get_electrodes([ch])[0]
        orig_ch = analysis.get_orig_channels(channels=[ch])[0]
        # Make str
        x_tick_labels.append(f'{elec}\n{orig_ch}')

    plt.xticks(np.arange(0, len(analysis.channels_of_interest), 1.0))
    plt.gca().set_xticklabels(x_tick_labels)
    # Change xtick font size
    plt.xticks(fontsize=8)
    plt.colorbar()

    return fig, ax


from scipy.signal import find_peaks
from tqdm import tqdm

def plot_reactivity_traces(analysis, save_dir = None, fig=None, ax=None, time_scatter=False,
                            **kwargs):

    pks_all = []

    

    cd = analysis.clean_data
    stim_electrodes = analysis.stim_electrodes
    channels_of_interest = analysis.channels_of_interest

    # Get colors for length of stim electrodes
    cmap = matplotlib.cm.get_cmap('winter')
    colors = cmap(np.linspace(0, 1, len(stim_electrodes)))

    # Plot reactive trace for each stim_electrode
    # for react_ind, react_ch in enumerate(channels_of_interest):
    for react_ind, react_ch in enumerate(tqdm(channels_of_interest)):
        pks_all = []
        fig, axs = plt.subplots(len(stim_electrodes) + 1,1, figsize=(10,10))
        
        react_elec = analysis.get_electrodes([react_ch])[0]
        for stim_ind, (stim_elec, ax) in enumerate(zip(stim_electrodes, axs)):
            
            cd_cur = cd[:,stim_ind,react_ind,:]
            cur_pks = []
            cur_amps = []
            cur_std = np.std(cd_cur)

            for stim_rep in range(cd_cur.shape[0]):
                pks, _ = find_peaks(-cd_cur[stim_rep,:], height=cur_std*5, distance=3*20)
                amps = [cd_cur[stim_rep,pk] for pk in pks]

                cur_pks.extend(pks)
                cur_amps.extend(amps)
                if time_scatter and len(pks) > 0:
                    ax.scatter(np.array(pks[0])/20, -stim_rep*np.ones_like(pks[0]), color='r', s=3, marker='|')
                    ax.scatter(np.array(pks)/20, -stim_rep*np.ones_like(pks), color='k', s=2,alpha=.5, marker='|')
            
                
            time_ms = np.arange(len(cd_cur[0,:]))/20 # 
            cur_pks = np.array(cur_pks) / 20 # convert to ms
            pks_all.append(cur_pks)

            if react_elec == stim_elec:
                continue

            if not time_scatter:
                # ax.plot(cd_cur[:,:].T, color=colors[stim_ind], alpha=.3)
                ax.plot(time_ms, cd_cur[:,:].T, color=colors[stim_ind], alpha=.3, linewidth=.5)
            
                ax.scatter(cur_pks, cur_amps, color='k', s=10, marker='x')

            else:
                # Make 2nd axis cumulative sum of all spikes
                ax2 = ax.twinx()
                ax2.hist(cur_pks, bins=50, color=colors[stim_ind], alpha=.5)
                ax2.set_ylim([0, 50])
                # if not last stim electrode
                if stim_ind != len(stim_electrodes) - 1:
                    ax2.set_yticks([])
            # ax.set_title(f"Stim El: {stim_elec}")
            # and rotate label 45
            ax.set_ylabel(f"{stim_elec}", rotation=60)

            

        for ax in axs[:-1]:
            ax.set_xticks([])
            ax.set_ylim([-60,10])
            ax.set_xlim([0, len(cd_cur[0,:])/20])
        
        for ax in axs[:-2]:
            # Remove ticks on left hand side
            ax.set_yticks([])



        # 3rd is histogram
        axs[-1].hist(pks_all, bins=15, color=colors)
        axs[-1].set_ylim([0, 50])

        plt.suptitle(f"React Ch: {react_ch}, react elec: {react_elec}")
        # plt.show()
        if save_dir is not None:
            if time_scatter:                
                title = f'/react_el_{react_elec}_sc.png'
            else:
                title = f'/react_el_{react_elec}.png'
            plt.savefig(save_dir + title, dpi=300)
            print("Saving to", save_dir + title)
            plt.close()
        else:
            plt.show()



import os

def plot_causal_connectivity(analysis, save_dir = None, fig=None, ax=None, first_order_ms = 30, multi_order_ms = 100,
                             save_mats = True, **kwargs):
    import copy
    r = copy.copy(analysis.reactivity_times)
    r_first_order = np.zeros(r.shape)
    r_multi = np.zeros(r.shape)


    first_order_frames = first_order_ms*20 # 30 * fs_ms
    multi_order_frames = multi_order_ms*20 # 100 * fs_ms
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            vals = []
            vals_multi = []
            for k in r[i,j]:
                vals.extend(k[k<=first_order_frames])
                vals_multi.extend(k[k<=multi_order_frames])

            r_first_order[i,j] = len(vals)
            r_multi[i,j] = len(vals_multi)
            if i == j:
                r_first_order[i,j] = 0
                r_multi[i,j] = 0
    # Reaction normalized to unit std of the columns
    # r_first_order = r_first_order/np.sum(r_first_order,axis=0)
    # r_all = r_all/np.sum(r_all,axis=0)
    first_order_mean = np.mean(r_first_order, axis=0)
    first_order_std = np.std(r_first_order, axis=0)
    multi_mean = np.mean(r_multi, axis=0)
    multi_std = np.std(r_multi, axis=0)

    # Ensure no divide by zero
    first_order_std[first_order_std == 0] = 1
    multi_std[multi_std == 0] = 1


    # Subtract out the mean from the reactivity, across the columns
    # print("STD",first_order_std)
    r_first_order_norm = (r_first_order - first_order_mean)/first_order_std
    r_multi_norm = (r_multi - multi_mean)/multi_std

    # r_first_order = (r_first_order - np.mean(r_first_order,axis=0))/np.std(r_first_order,axis=0)
    # r_multi = (r_multi - np.mean(r_multi,axis=0))/np.std(r_multi,axis=0)

    if save_mats:
        # create derived folder in save_dir
        if not os.path.exists(save_dir + '/derived'):
            os.makedirs(save_dir + '/derived')

        # Raw
        np.save(save_dir + '/derived/causal_connectivity_first.npy', r_first_order)
        np.save(save_dir + '/derived/causal_connectivity_multi.npy', r_multi)
        # Normalized
        np.save(save_dir + '/derived/causal_connectivity_first_norm.npy', r_first_order_norm)
        np.save(save_dir + '/derived/causal_connectivity_multi_norm.npy', r_multi_norm)
        # Mean and std
        np.save(save_dir + '/derived/causal_connectivity_first_mean.npy', first_order_mean)
        np.save(save_dir + '/derived/causal_connectivity_multi_mean.npy', multi_mean)
        np.save(save_dir + '/derived/causal_connectivity_first_std.npy', first_order_std)
        np.save(save_dir + '/derived/causal_connectivity_multi_std.npy', multi_std)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # we want subplots using gridspec. 2 on top, 1 on bottom
    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    axs = [ax1, ax2, ax3]

    # Normalized CC
    axs[0].imshow(r_first_order_norm, cmap='magma', vmin=-3, vmax=3)
    axs[0].set_title('First Order Causal Connectivity Normalized')
    axs[0].set_ylabel('Stim Electrodes')
    axs[0].set_yticks(np.arange(0, len(analysis.stim_electrodes), 1.0))
    axs[0].set_yticklabels(analysis.stim_electrodes)
    axs[0].set_xlabel('Channels of Interest')
    # x_tick_labels = []
    x_tick_labels = analysis.stim_electrodes

    axs[0].set_xticks(np.arange(0, len(analysis.channels_of_interest), 1.0))
    axs[0].set_xticklabels(x_tick_labels)
    axs[0].tick_params(axis='x', labelrotation=90)
    
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0)
    fig.colorbar(axs[0].get_images()[0], cax=cax, label='Normalized Z-Score')

    # Raw NC
    axs[1].imshow(r_first_order, cmap='magma')
    axs[1].set_title('First Order Causal Connectivity')
    axs[1].set_ylabel('Stim Electrodes')
    axs[1].set_yticks(np.arange(0, len(analysis.stim_electrodes), 1.0))
    axs[1].set_yticklabels(analysis.stim_electrodes)
    axs[1].set_xlabel('Channels of Interest')
    
    axs[1].set_xticks(np.arange(0, len(analysis.channels_of_interest), 1.0))
    axs[1].set_xticklabels(x_tick_labels)
    axs[1].tick_params(axis='x', labelrotation=90)
    
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0)
    fig.colorbar(axs[1].get_images()[0], cax=cax, label='Spike Count')

    # Should be 
    # axs[1].imshow(first_order_mean[None,:], cmap='magma')
    # Plot mean and std
    axs[2].errorbar(np.arange(len(analysis.channels_of_interest)), first_order_mean, yerr=first_order_std, fmt='o')
    axs[2].set_title('Mean Evoked Firing % per Reaction electrode')
    axs[2].set_ylabel('Mean')
    axs[2].set_xlabel('Electrodes of Interest')
    axs[2].set_xticks(np.arange(0, len(analysis.channels_of_interest), 1.0))
    axs[2].set_xticklabels(x_tick_labels)
    # Y log scale
    axs[2].set_yscale('log')
    axs[2].set_ylim([.1, 100])
    axs[2].yaxis.set_major_formatter(plt.ScalarFormatter())
    axs[2].tick_params(axis='x', labelrotation=90)
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir + f'/causal_connectivity_first.png', dpi=300)
        # as svg
        plt.savefig(save_dir + f'/causal_connectivity_first.svg')
        plt.close()
    

    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    axs = [ax1, ax2, ax3]


    axs[0].imshow(r_multi_norm, cmap='magma', vmin=-3, vmax=3)
    axs[0].set_title('Multi Order Causal Connectivity Normalized')
    axs[0].set_ylabel('Stim Electrodes')
    axs[0].set_yticks(np.arange(0, len(analysis.stim_electrodes), 1.0))
    axs[0].set_yticklabels(analysis.stim_electrodes)
    axs[0].set_xlabel('Channels of Interest')
    axs[0].set_xticks(np.arange(0, len(analysis.channels_of_interest), 1.0))
    axs[0].set_xticklabels(x_tick_labels)
    axs[0].tick_params(axis='x', labelrotation=90)

    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0)
    fig.colorbar(axs[0].get_images()[0], cax=cax, label='Normalized Z-Score')

    axs[1].imshow(r_multi, cmap='magma')
    axs[1].set_title('Multi Order Causal Connectivity')
    axs[1].set_ylabel('Stim Electrodes')
    axs[1].set_yticks(np.arange(0, len(analysis.stim_electrodes), 1.0))
    axs[1].set_yticklabels(analysis.stim_electrodes)
    axs[1].set_xlabel('Channels of Interest')
    axs[1].set_xticks(np.arange(0, len(analysis.channels_of_interest), 1.0))
    axs[1].set_xticklabels(x_tick_labels)
    axs[1].tick_params(axis='x', labelrotation=90)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0)
    fig.colorbar(axs[1].get_images()[0], cax=cax, label='Spike Count')






    axs[2].errorbar(np.arange(len(analysis.channels_of_interest)), multi_mean, yerr=multi_std, fmt='o')
    axs[2].set_title('Mean Evoked Firing % per Reaction electrode')
    axs[2].set_ylabel('Mean')
    axs[2].set_xlabel('Electrodes of Interest')
    axs[2].set_xticks(np.arange(0, len(analysis.channels_of_interest), 1.0))
    axs[2].set_xticklabels(x_tick_labels)
    
    # Y log scale
    axs[2].set_yscale('log')
    axs[2].set_ylim([0.1, 100])
    axs[2].yaxis.set_major_formatter(plt.ScalarFormatter())
    axs[2].tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    
    if save_dir is not None:
        plt.savefig(save_dir + f'/causal_connectivity_multi.png', dpi=300)
        # as svg
        plt.savefig(save_dir + f'/causal_connectivity_multi.svg')
        plt.close()

    return fig, axs


def plot_reactivity_traces_experiment(experiment, clean_data=None, stim_electrodes=None, channels_of_interest=None, save_dir=None, fig=None, ax=None, time_scatter=False, **kwargs):
    """
    Plot reactivity traces using data from an Experiment object. Similar to plot_reactivity_traces
    but uses experiment instead of analysis object.
    
    Parameters
    ----------
    experiment : Experiment
        Experiment object containing causal analysis results
    save_dir : str, optional
        Directory to save plots to
    fig : matplotlib figure, optional
        Figure to plot on
    ax : matplotlib axis, optional
        Axis to plot on
    time_scatter : bool, optional
        Whether to plot spike times as scatter points
    **kwargs : dict
        Additional kwargs to pass to plotting functions
    """
    from scipy.signal import find_peaks
    from tqdm import tqdm
    
    # Get data from the experiment
    if clean_data is None:
        clean_data = experiment.get_last('clean_data')
    if stim_electrodes is None:
        stim_electrodes = experiment.params['selected_electrodes']
    if channels_of_interest is None:
        channels_of_interest = experiment.get_last('channels_of_interest')
    
    if clean_data is None or stim_electrodes is None or channels_of_interest is None:
        missing_req = []
        if clean_data is None:
            missing_req.append('clean_data')
        if stim_electrodes is None:
            missing_req.append('selected_electrodes')
        if channels_of_interest is None:
            missing_req.append('channels_of_interest')
        print(f"Missing required data from experiment: {missing_req}. Ensure CausalAnalysis was run successfully.")
        return
    
    # Create save directory if it doesn't exist but was specified
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
    # If save_dir not provided but experiment has base_dir, use that
    if save_dir is None and hasattr(experiment, 'base_dir') and experiment.base_dir:
        filename = experiment.get_baseline_filename()
        if filename:
            save_dir = filename
    
    pks_all = []
    
    cd = clean_data
    
    # Get colors for length of stim electrodes
    cmap = matplotlib.cm.get_cmap('winter')
    colors = cmap(np.linspace(0, 1, len(stim_electrodes)))

    # Plot reactive trace for each stim_electrode
    for react_ind, react_ch in enumerate(tqdm(channels_of_interest)):
        pks_all = []
        fig, axs = plt.subplots(len(stim_electrodes) + 1, 1, figsize=(10, 10))
        
        axs = axs.flatten()

        react_elec = experiment.mapper.get_electrodes([react_ch])[0]
        
        for stim_ind, (stim_elec, ax) in enumerate(zip(stim_electrodes, axs)):
            
            cd_cur = cd[:, stim_ind, react_ind, :]
            cur_pks = []
            cur_amps = []
            cur_std = np.std(cd_cur)

            for stim_rep in range(cd_cur.shape[0]):
                pks, _ = find_peaks(-cd_cur[stim_rep, :], height=cur_std*5, distance=3*20)
                amps = [cd_cur[stim_rep, pk] for pk in pks]

                cur_pks.extend(pks)
                cur_amps.extend(amps)
                if time_scatter and len(pks) > 0:
                    ax.scatter(np.array(pks[0])/20, -stim_rep*np.ones_like(pks[0]), color='r', s=3, marker='|')
                    ax.scatter(np.array(pks)/20, -stim_rep*np.ones_like(pks), color='k', s=2, alpha=.5, marker='|')
            
            time_ms = np.arange(len(cd_cur[0, :]))/20
            cur_pks = np.array(cur_pks) / 20  # convert to ms
            pks_all.append(cur_pks)

            if react_elec == stim_elec:
                continue

            if not time_scatter:
                ax.plot(time_ms, cd_cur[:, :].T, color=colors[stim_ind], alpha=.3, linewidth=.5)
                ax.scatter(cur_pks, cur_amps, color='k', s=10, marker='x')
            else:
                # Make 2nd axis cumulative sum of all spikes
                ax2 = ax.twinx()
                ax2.hist(cur_pks, bins=50, color=colors[stim_ind], alpha=.5)
                ax2.set_ylim([0, 50])
                # if not last stim electrode
                if stim_ind != len(stim_electrodes) - 1:
                    ax2.set_yticks([])
            
            ax.set_ylabel(f"{stim_elec}", rotation=60)

        for ax in axs[:-1]:
            ax.set_xticks([])
            ax.set_ylim([-60, 10])
            ax.set_xlim([0, len(cd_cur[0, :])/20])
        
        for ax in axs[:-2]:
            # Remove ticks on left hand side
            ax.set_yticks([])

        # 3rd is histogram
        axs[-1].hist(pks_all, bins=15, color=colors)
        axs[-1].set_ylim([0, 50])

        plt.suptitle(f"React Ch: {react_ch}, react elec: {react_elec}")
        
        if save_dir is not None:
            if time_scatter:                
                title = f'/react_el_{react_elec}_sc.png'
            else:
                title = f'/react_el_{react_elec}.png'
            plt.savefig(str(save_dir) + title, dpi=300)
            print("Saving to", str(save_dir) + title)
            plt.close()
        else:
            plt.show()


def plot_causal_connectivity_experiment(experiment, info=None, save_dir=None, fig=None, ax=None, 
                                       first_order_ms=30, multi_order_ms=100, save_mats=True, **kwargs):
    """
    Plot causal connectivity using data from an Experiment object. Similar to plot_causal_connectivity
    but uses experiment instead of analysis object.
    
    Parameters
    ----------
    experiment : Experiment
        Experiment object containing causal analysis results
    info : dict, optional
        Dictionary containing pre-calculated causal connectivity metrics from causal_connectivity.py
    save_dir : str, optional
        Directory to save plots to
    fig : matplotlib figure, optional
        Figure to plot on
    ax : matplotlib axis, optional
        Axis to plot on
    first_order_ms : int, optional
        First order causality window in ms
    multi_order_ms : int, optional
        Multi-order causality window in ms
    save_mats : bool, optional
        Whether to save matrices to file
    **kwargs : dict
        Additional kwargs to pass to plotting functions
    """
    import os
    import copy
    import numpy as np
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # Get data from the experiment or info
    if info is None:
        # If no info dictionary provided, try to get data from experiment
        if hasattr(experiment, 'get_last') and callable(experiment.get_last):
            reactivity_times = experiment.get_last('reactivity_times')
            stim_electrodes = experiment.params.get('stim_electrodes', experiment.params.get('selected_electrodes'))
            channels_of_interest = experiment.mapper.get_channels(experiment.params['stim_electrodes'])
        else:
            print("No info dictionary provided and experiment doesn't have get_last method. Cannot proceed.")
            return None, None
    else:
        # Use data from info dictionary
        reactivity_times = info.get('reactivity_times')
        stim_electrodes = info.get('stim_electrodes', info.get('stim_patterns', []))
        if isinstance(stim_electrodes[0], (list, tuple)):
            # Extract just the first element if stim_patterns is used (which contains electrode, amplitude pairs)
            stim_electrodes = [p[0] for p in stim_electrodes]
        channels_of_interest = info.get('selected_channels')

    print("channels_of_interest", channels_of_interest)
    
    # Create save directory if it doesn't exist but was specified
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # If save_dir not provided but experiment has base_dir, use that
    if save_dir is None and hasattr(experiment, 'base_dir') and experiment.base_dir:
        filename = experiment.get_baseline_filename()
        if filename:
            save_dir = filename

    
    # Use pre-calculated metrics if available in info, otherwise calculate them
    if info and 'first_order_connectivity' in info and 'multi_order_connectivity' in info:
        r_first_order = info['first_order_connectivity']
        r_multi = info['multi_order_connectivity']
        burst_percent = info['burst_percent']

        # Normalize to by mean of the reaction electrode
        # r_first_order_norm = r_first_order / np.mean(r_first_order, axis=0)[:, None]
        # r_multi_norm = r_multi / np.mean(r_multi, axis=0)[:, None]
        # Actually make it z score of the reaction electrode
        r_first_order_norm = (r_first_order - np.mean(r_first_order, axis=0)) / np.std(r_first_order, axis=0)
        r_multi_norm = (r_multi - np.mean(r_multi, axis=0)) / np.std(r_multi, axis=0)
        
        first_order_mean = np.mean(r_first_order, axis=0)
        first_order_std = np.std(r_first_order, axis=0)
        multi_mean = np.mean(r_multi, axis=0)
        multi_std = np.std(r_multi, axis=0)
    else:
        raise ValueError("No pre-calculated metrics found in info. Please run causal_connectivity.py first.")

    
        
    # else:
    #     # Calculate metrics as in the original function
    #     r = copy.copy(reactivity_times)
    #     r_first_order = np.zeros(r.shape)
    #     r_multi = np.zeros(r.shape)

    #     first_order_frames = first_order_ms * 20  # Convert ms to frames
    #     multi_order_frames = multi_order_ms * 20

    #     for i in range(r.shape[0]):
    #         for j in range(r.shape[1]):
    #             vals = []
    #             vals_multi = []
    #             for k in r[i, j]:
    #                 vals.extend(k[k <= first_order_frames])
    #                 vals_multi.extend(k[k <= multi_order_frames])

    #             r_first_order[i, j] = len(vals)
    #             r_multi[i, j] = len(vals_multi)
    #             if i == j:
    #                 r_first_order[i, j] = 0
    #                 r_multi[i, j] = 0

    #     # Calculate statistics
    #     first_order_mean = np.mean(r_first_order, axis=0)
    #     first_order_std = np.std(r_first_order, axis=0)
    #     multi_mean = np.mean(r_multi, axis=0)
    #     multi_std = np.std(r_multi, axis=0)

    #     # Ensure no divide by zero
    #     first_order_std[first_order_std == 0] = 1
    #     multi_std[multi_std == 0] = 1

    #     # Normalize
    #     r_first_order_norm = (r_first_order - first_order_mean) / first_order_std
    #     r_multi_norm = (r_multi - multi_mean) / multi_std

    # Save matrices if requested
    if save_mats and save_dir:
        # Create derived folder in save_dir
        derived_dir = os.path.join(save_dir, 'derived')
        os.makedirs(derived_dir, exist_ok=True)

        # Save raw matrices
        np.save(os.path.join(derived_dir, 'causal_connectivity_first.npy'), r_first_order)
        np.save(os.path.join(derived_dir, 'causal_connectivity_multi.npy'), r_multi)
        
        # Save normalized matrices
        np.save(os.path.join(derived_dir, 'causal_connectivity_first_norm.npy'), r_first_order_norm)
        np.save(os.path.join(derived_dir, 'causal_connectivity_multi_norm.npy'), r_multi_norm)
        
        # Save statistics
        np.save(os.path.join(derived_dir, 'causal_connectivity_first_mean.npy'), first_order_mean)
        np.save(os.path.join(derived_dir, 'causal_connectivity_first_std.npy'), first_order_std)
        np.save(os.path.join(derived_dir, 'causal_connectivity_multi_mean.npy'), multi_mean)
        np.save(os.path.join(derived_dir, 'causal_connectivity_multi_std.npy'), multi_std)

    # Generate first-order plots
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    axs = [ax1, ax2, ax3]

    # Plot normalized first-order connectivity
    im1 = axs[0].imshow(r_first_order_norm, cmap='magma', vmin=-3, vmax=3)
    axs[0].set_title('First Order Causal Connectivity Normalized')
    axs[0].set_ylabel('Stim Electrodes')
    axs[0].set_yticks(np.arange(0, len(stim_electrodes), 1.0))
    axs[0].set_yticklabels(stim_electrodes)
    axs[0].set_xlabel('Channels of Interest')
    
    # Get x-axis labels - use channels or electrodes depending on what's available
    if hasattr(experiment, 'mapper') and callable(getattr(experiment.mapper, 'get_electrodes', None)):
        x_tick_labels = [experiment.mapper.get_electrodes([ch])[0] for ch in channels_of_interest]
    else:
        x_tick_labels = channels_of_interest
    
    axs[0].set_xticks(np.arange(0, len(channels_of_interest), 1.0))
    axs[0].set_xticklabels(x_tick_labels)
    axs[0].tick_params(axis='x', labelrotation=90)
    
    # Add colorbar for normalized plot
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0)
    plt.colorbar(im1, cax=cax, label='Normalized Z-Score')

    # Plot raw first-order connectivity
    im2 = axs[1].imshow(r_first_order, cmap='magma')
    axs[1].set_title('First Order Causal Connectivity')
    axs[1].set_ylabel('Stim Electrodes')
    axs[1].set_yticks(np.arange(0, len(stim_electrodes), 1.0))
    axs[1].set_yticklabels(stim_electrodes)
    axs[1].set_xlabel('Channels of Interest')
    
    axs[1].set_xticks(np.arange(0, len(channels_of_interest), 1.0))
    axs[1].set_xticklabels(x_tick_labels)
    axs[1].tick_params(axis='x', labelrotation=90)
    
    # Add colorbar for raw plot
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0)
    plt.colorbar(im2, cax=cax, label='Spike Count')

    # Plot mean and standard deviation
    axs[2].errorbar(np.arange(len(channels_of_interest)), first_order_mean, yerr=first_order_std, fmt='o')
    axs[2].set_title('Mean Evoked Firing per Reaction Electrode')
    axs[2].set_ylabel('Mean Spike Count')
    axs[2].set_xlabel('Electrodes of Interest')
    axs[2].set_xticks(np.arange(0, len(channels_of_interest), 1.0))
    axs[2].set_xticklabels(x_tick_labels)
    
    # Set y-axis to log scale
    axs[2].set_yscale('log')
    axs[2].set_ylim([0.1, 100])
    axs[2].yaxis.set_major_formatter(plt.ScalarFormatter())
    axs[2].tick_params(axis='x', labelrotation=90)
    
    plt.tight_layout()

    # Save first-order plot
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'causal_connectivity_first.png'), dpi=300)
        plt.savefig(os.path.join(save_dir, 'causal_connectivity_first.svg'))
        plt.close()

    # Generate multi-order plots
    fig2 = plt.figure(figsize=(10, 10))
    gs = fig2.add_gridspec(2, 2)
    ax1 = fig2.add_subplot(gs[0, 0])
    ax2 = fig2.add_subplot(gs[0, 1])
    ax3 = fig2.add_subplot(gs[1, :])
    axs = [ax1, ax2, ax3]

    # Plot normalized multi-order connectivity
    im1 = axs[0].imshow(r_multi_norm, cmap='magma', vmin=-3, vmax=3)
    axs[0].set_title('Multi Order Causal Connectivity Normalized')
    axs[0].set_ylabel('Stim Electrodes')
    axs[0].set_yticks(np.arange(0, len(stim_electrodes), 1.0))
    axs[0].set_yticklabels(stim_electrodes)
    axs[0].set_xlabel('Channels of Interest')
    axs[0].set_xticks(np.arange(0, len(channels_of_interest), 1.0))
    axs[0].set_xticklabels(x_tick_labels)
    axs[0].tick_params(axis='x', labelrotation=90)

    # Add colorbar for normalized plot
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0)
    plt.colorbar(im1, cax=cax, label='Normalized Z-Score')

    # Plot raw multi-order connectivity
    im2 = axs[1].imshow(r_multi, cmap='magma')
    axs[1].set_title('Multi Order Causal Connectivity')
    axs[1].set_ylabel('Stim Electrodes')
    axs[1].set_yticks(np.arange(0, len(stim_electrodes), 1.0))
    axs[1].set_yticklabels(stim_electrodes)
    axs[1].set_xlabel('Channels of Interest')
    axs[1].set_xticks(np.arange(0, len(channels_of_interest), 1.0))
    axs[1].set_xticklabels(x_tick_labels)
    axs[1].tick_params(axis='x', labelrotation=90)
    
    # Add colorbar for raw plot
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0)
    plt.colorbar(im2, cax=cax, label='Spike Count')

    # Plot mean and standard deviation
    axs[2].errorbar(np.arange(len(channels_of_interest)), multi_mean, yerr=multi_std, fmt='o')
    axs[2].set_title('Mean Evoked Firing per Reaction Electrode')
    axs[2].set_ylabel('Mean Spike Count')
    axs[2].set_xlabel('Electrodes of Interest')
    axs[2].set_xticks(np.arange(0, len(channels_of_interest), 1.0))
    axs[2].set_xticklabels(x_tick_labels)
    
    # Set y-axis to log scale
    axs[2].set_yscale('log')
    axs[2].set_ylim([0.1, 100])
    axs[2].yaxis.set_major_formatter(plt.ScalarFormatter())
    axs[2].tick_params(axis='x', labelrotation=90)
    
    plt.tight_layout()

    # Save multi-order plot
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'causal_connectivity_multi.png'), dpi=300)
        plt.savefig(os.path.join(save_dir, 'causal_connectivity_multi.svg'))
        plt.close()

    return fig, axs




def get_footprints(analysis, selected_electrodes=None, file_path=None):
    """Return the footprint channels and waveforms for the given electrodes."""
    from braindance.core.phases_analysis import FootprintPhase

    if file_path is None:
        file_path = analysis.file_path
    
    if selected_electrodes is None:
        selected_electrodes = analysis.selected_electrodes

    assert file_path is not None, "Must provide file_path or assign to analysis"
    assert selected_electrodes is not None, "Must provide selected_electrodes or assign to analysis"
    
    footprint_phase = FootprintPhase(verbose=True, rms_mult=1, wind=60, load_whole_recording=False, 
                                num_channel_thresh=120, remove_bad=True, remove_redundant=True,
                                similarity_thresh=.65)
    
    analysis.select_electrodes(selected_electrodes)
    analysis = footprint_phase.run(analysis)
    footprint_chs = analysis.selected_footprint_chans
    footprint_waves = analysis.selected_footprint_waves
    mapping = analysis.mapping
    return footprint_chs, footprint_waves, mapping
    
    