import braindance
from braindance.core.phases import Phase
from braindance import analysis
from braindance.analysis import data_loader, plot_helper
# from braindance.analysis.data_helper import butterworth_filter
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm


import os
from pathlib import Path
import time


class AnalysisPhase(Phase):
    def __init__(self):
        super().__init__(None)
        self.predicted_time = 0

    def run(self, filename):
        raise NotImplementedError


class ActivityPhase(AnalysisPhase):
    '''
    Identifies active electrodes in the recording by analyzing spike activity.
    
    This phase calculates a custom activity metric for each electrode by combining
    the spike count and mean amplitude. The metric helps identify electrodes that 
    show significant neural activity, which can be used for stimulation or recording.
    
    Parameters
    ----------
    name : str, optional
        Custom name for this phase instance
    verbose : bool, optional
        Whether to print detailed progress information
    make_plots : bool, optional
        Whether to generate activity heatmaps
    make_gif : bool, optional
        Whether to create a 3D rotating animation of activity
    save_plots : bool, optional
        Whether to save plots to disk instead of displaying them
    experiment : Experiment, optional
        Experiment object to associate with this phase
        
    Requires
    --------
    Either a filename set with set_filename() or an experiment with a baseline recording
    
    Provides
    --------
    basename : str
        Base filename of the recording
    mapping : pd.DataFrame
        Electrode mapping with added activity metrics
    active_electrodes : list
        List of electrodes identified as active, sorted by activity level
    spikes : pd.DataFrame
        Spike data from the recording
    
    Modifies Experiment Parameters
    -----------------------------
    active_electrodes : list
        Updates the experiment with the list of active electrodes
    '''
    def __init__(self, name = 'ActivityPhase', verbose=False, make_plots=True, make_gif=False, save_plots=False,
                 experiment=None):
        super().__init__()
        
        self.predicted_time = 30 # < 30 seconds
        self.verbose = verbose
        self.name = name
        self.make_plots = make_plots
        self.make_gif = make_gif
        self.save_plots = save_plots
        self.filename = None
        self.experiment = experiment

        self.provides = ['basename','mapping', 'active_electrodes', 'spikes']

    def set_filename(self, filename):
        self.filename = filename
    
    def run(self, experiment=None):
        results = {}
        if experiment is None and self.filename is None:
            raise ValueError("Must set experiment or filename before running phase")            
            
        if self.filename is None:
            self.filename = experiment.get_baseline_filename()
            if self.verbose:
                print(f"Using filename {self.filename} from experiment")
        
        assert self.filename is not None, "Must have a filename to run phase"
        
        results['basename'] = self.filename

        # Read in data
        if self.verbose:
            print(f"Loading mapping from {self.filename}")
        if experiment:
            mapping = experiment.get_mapping(filename=self.filename)
        else:
            mapping = data_loader.load_mapping_maxwell(self.filename)


        if self.verbose:
            print("Mapping is:")
            print("Loading data from", self.filename)

        # This should be t,ch,amp
        
        spikes_df = data_loader.load_data_maxwell(self.filename, spikes=True)

        results['spikes'] = spikes_df


        # print('Loading raw data')
        # data = data_loader.load_data_maxwell(self.filename)
        # print('Complete')

        # Convert spikes to a DataFrame
        spike_counts = spikes_df.groupby('channel').size()
        
        # Group by channel and calculate the mean amplitude
        mean_amps = spikes_df.groupby('channel')['amplitude'].mean()

        # mapping.set_index('channel', inplace=True)
        # mapping = mapping.reset_index().set_index('channel')

        # Assign the mean amplitudes to the mapping DataFrame
        mapping['mean_amp'] = mapping.index.map(mean_amps.to_dict())
        # Negate the mean amplitudes
        mapping['mean_amp'] = -mapping['mean_amp']

        # Continue with the rest of the function...
        mapping['spike_count'] = mapping.index.map(spike_counts.to_dict())
        
        # Apply a logarithmic function to the spike counts
        mapping['spike_count'] = np.log1p(mapping['spike_count'])

        # Scale the spike counts to the range [0, 1] manually
        mapping['spike_count'] = (mapping['spike_count'] - mapping['spike_count'].min()) / (mapping['spike_count'].max() - mapping['spike_count'].min())

        # Increase the weight of the amplitudes
        mapping['mean_amp'] *= .1  # Adjust the factor as needed

        # Recalculate the spike_amp_product
        mapping['spike_amp_product'] = (1 + mapping['spike_count']) * (1 + mapping['mean_amp'])

        results['mapping'] = mapping
        

    
        if self.make_plots:

            # Ensure the directory exists
            Path(self.filename).mkdir(parents=True, exist_ok=True)

            # Create a figure with three subplots
            fig, axs = plt.subplots(3, 1, figsize=(6, 18))

            # Plot spike_count
            sc = axs[0].scatter(mapping['x'], mapping['y'], c=mapping['spike_count'], cmap='magma')
            axs[0].set_title('Spike Count')
            fig.colorbar(sc, ax=axs[0])

            # Plot mean_amp
            ma = axs[1].scatter(mapping['x'], mapping['y'], c=mapping['mean_amp'], cmap='magma')
            axs[1].set_title('Mean Amplitude')
            fig.colorbar(ma, ax=axs[1])

            # Plot spike_amp_product
            sap = axs[2].scatter(mapping['x'], mapping['y'], c=mapping['spike_amp_product'], cmap='magma')
            axs[2].set_title('Spike Count * Mean Amplitude')
            fig.colorbar(sap, ax=axs[2])

            if self.save_plots:
                plt.savefig(f"{self.filename}heatmap.png")
            else:
                plt.show()


        # Now we create a matrix 
        elec_vals = np.zeros(26400) # Full array scale
        electrodes = np.arange(26400)
        # Set the values as the spike_amp_product
        elec_vals[mapping['electrode']] = mapping['spike_amp_product']
        # Reshape the array into a 2D array
        
        elec_vals = elec_vals.reshape(26400 // 220, 220)
        electrodes = electrodes.reshape(26400 // 220, 220)

        # Make NaNs zero
        elec_vals = np.nan_to_num(elec_vals)

        if self.make_plots:

            # Plot the heatmap
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            plt.imshow(elec_vals, cmap='magma', extent=[0, 220*17.5, 0, 120*17.5])
            cbar = plt.colorbar()
            cbar.set_label('Firing rate and amplitude')
            plt.title('Heatmap of Firing Rate and Amplitude')
            # plt.axis('off')

            # plt.plot(mapping['x'], 120*17.6 - mapping['y'], 'x', markersize=2)




            if self.save_plots:
                plt.savefig(f"{self.filename}/heatmap2.png")
            else:
                # analysis.fig = plt.gcf()
                # analysis.ax = plt.gca()
                plt.show()


        # Reduce the size by removing all rows and columns that are all zeros on the edges
        #here
        # Find indices where there are non-zero values
        rows = np.any(elec_vals, axis=1)
        cols = np.any(elec_vals, axis=0)

        # Get the bounding box of non-zero values
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        # Remove rows and columns that are all zeros on the edges
        elec_vals_cropped = elec_vals[ymin:ymax+1, xmin:xmax+1]
        electrodes_cropped = electrodes[ymin:ymax+1, xmin:xmax+1]


        # Create a grid of x, y coordinates
        x = np.arange(elec_vals_cropped.shape[1])
        y = np.arange(elec_vals_cropped.shape[0])
        x, y = np.meshgrid(x, y)

        # Find local maxima
        from scipy.ndimage import maximum_filter, minimum_filter

        # Define a size for the local neighborhood
        size = 5

        # Find local maxima
        local_max = maximum_filter(elec_vals_cropped, size=size) == elec_vals_cropped
        background = (elec_vals_cropped == 0)
        eroded_background = minimum_filter(background, size=size)
        is_local_max = local_max ^ eroded_background
        y_max, x_max = np.where(is_local_max)

        if self.make_plots:
            # Create a 3D plot
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d',computed_zorder=False)
            ax.plot_surface(x, y, elec_vals_cropped, cmap='magma', cstride=1, rstride=1, linewidth=0,
                            antialiased=False, zorder=0)

            # Plot the local maxima with a small offset
            z_max = elec_vals_cropped[is_local_max] + 0.01  # Adjust the offset as needed
            ax.scatter(x_max, y_max, z_max, color='white', s=50, edgecolors='red', zorder=10)

            # Set the labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Value')
            ax.set_zlim(0, 20)

            # Change the background color to black
            ax.set_facecolor('black')
            # Change the text color to white
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.zaxis.label.set_color('white')

            # Change tick labels and colors to white
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.tick_params(axis='z', colors='white')

            # change grid to white
            ax.xaxis._axinfo["grid"]['color'] = "white"
            ax.yaxis._axinfo["grid"]['color'] = "white"
            ax.zaxis._axinfo["grid"]['color'] = "white"
            # Turn off the grid
            # ax.grid(False)

            # Change the pane colors to black
            ax.xaxis.pane.set_color('black')
            ax.yaxis.pane.set_color('black')
            ax.zaxis.pane.set_color('black')

            if self.save_plots:
                plt.savefig(f"{self.filename}_heatmap3d.png")
            else:
                plt.show()


        if self.make_gif:
            import matplotlib.animation as animation
            from mpl_toolkits.mplot3d import Axes3D

            # Create a 3D plot
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d',computed_zorder=False)
            z_max = elec_vals_cropped[is_local_max] + 0.03  # Adjust the offset as needed
            
            ax.plot_surface(x, y, elec_vals_cropped, cmap='magma', cstride=1, rstride=1, linewidth=0,
                            antialiased=False, zorder=0)

            # Plot the local maxima again with a small size and the desired color
            ax.scatter(x_max, y_max, z_max, color='white', s=50, edgecolors='red', zorder=10)

            

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Activity Measure')
            ax.set_zlim(0, 20)

            # Change the background color to black
            ax.set_facecolor('black')
            # Change the text color to white
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.zaxis.label.set_color('white')

            # Change tick labels and colors to white
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.tick_params(axis='z', colors='white')

            # change grid to white
            ax.xaxis._axinfo["grid"]['color'] = "white"
            ax.yaxis._axinfo["grid"]['color'] = "white"
            ax.zaxis._axinfo["grid"]['color'] = "white"
            # Turn off the grid
            # ax.grid(False)

            # Change the pane colors to black
            ax.xaxis.pane.set_color('black')
            ax.yaxis.pane.set_color('black')
            ax.zaxis.pane.set_color('black')

            # Create an animation
            def update(num):
                if num < 180:  # First rotation
                    ax.view_init(elev=20., azim=num)
                else:  # After first rotation
                    ax.view_init(elev=min(20. + (num - 180),40), azim=num)  # Increase elevation
                    ax.dist = 10 - (num - 180) / 180 * 2  # Zoom in

            ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=100)

            # Continue with the rest of the code...
            # Save the animation as a GIF
            ani.save(f"{self.filename}_heatmap3d.gif", writer='imagemagick', fps=30)

            plt.show()

        active_electrodes = electrodes_cropped[is_local_max]
        # Sort the electrodes by their activity measure
        active_electrodes = list(reversed(active_electrodes[np.argsort(elec_vals_cropped[is_local_max])]))

        results['active_electrodes'] = active_electrodes
        if experiment:
            experiment.update_params({'active_electrodes':active_electrodes})

        return results

    
    def info(self):
        info = {}
        info['name'] = self.name
        info['description'] = "Determines the activity of each electrode in the recording using a custom " \
        "metric that combines the spike count and mean amplitude of each electrode. Spike counts used are from" \
        "the maxwell spike data in the recording."
        info['results_keys'] = ['basename','mapping', 'active_electrodes', 'spikes']
        return info


class FootprintPhase(AnalysisPhase):
    '''
    Analyzes the spatial footprint of neural activity for each active electrode.
    
    This phase identifies which neighboring electrodes show correlated activity
    when an active electrode detects a spike. These spatial footprints help identify
    unique neurons and remove redundant electrodes that are recording from the
    same neuron.
    
    Parameters
    ----------
    wind : int, optional
        Window size in frames (samples) for spike-triggered average
    rms_mult : float, optional
        Multiplier for RMS threshold for footprint detection
    spikes_per_channel : int, optional
        Maximum number of spikes to use per channel for footprint calculation
    num_channel_thresh : int, optional
        Maximum number of channels allowed in a footprint (for filtering)
    num_spikes_min : int, optional
        Minimum number of spikes needed for reliable footprint calculation
    remove_bad : bool, optional
        Whether to remove channels with too few spikes or too large footprints
    similarity_thresh : float, optional
        Threshold for considering two footprints as redundant
    pk_to_pk_thresh : float, optional
        Minimum peak-to-peak amplitude for including a channel in a footprint
    remove_redundant : bool, optional
        Whether to remove redundant footprints (likely same neuron)
    load_whole_recording : bool, optional
        Whether to load the entire recording into memory
    dao : object, optional
        Data access object for loading data
    verbose : bool, optional
        Whether to print detailed progress information
    experiment : Experiment, optional
        Experiment object to associate with this phase
    name : str, optional
        Custom name for this phase instance
        
    Requires
    --------
    An experiment with active_electrodes from a previous ActivityPhase
    
    Provides
    --------
    selected_electrodes : list
        List of electrodes selected after footprint analysis
    footprint_chans : list of lists
        For each selected electrode, a list of channels in its footprint
    selected_channels : list
        List of channels corresponding to selected electrodes
    
    Modifies Experiment Parameters
    -----------------------------
    selected_electrodes : list
        Updates the experiment with the list of selected electrodes
    stim_electrodes : list
        Sets the stimulation electrodes based on selected electrodes
    '''
    def __init__(self, wind=100, rms_mult=7, spikes_per_channel = 300,
                 num_channel_thresh = 60, num_spikes_min = 20, remove_bad = True,
                 similarity_thresh = .7, pk_to_pk_thresh=9,
                 remove_redundant = True,
                load_whole_recording = False, dao = None, verbose=False, experiment=None, name="FootprintPhase"):
        """
        Loads a fixed window size from the list of starts
        :param channel_oi: int
            Channel number to get footprint for
        :param wind: int
            Window size in frames (a window is taken on each side of the spike)
        :param rms_mult: float
            Multiplication factor of average footprint RMS for footprint channel detection
        :param spikes_per_channel: int
            Number of spikes to use per channel, randomly selected
        :param num_channel_thresh: int
            Number of channels that must be in the footprint for it to be considered valid
        :param remove_bad: bool
            Whether to remove bad channels from the footprint, does this in place
        :param load_whole_recording: bool
            Whether to load the whole recording into memory
        :param dao: AnalysisDAO
            DAO object to use for loading, using and moving data
        :param verbose: bool
            Whether to print out progress
        :return:
        footprint_chan_i: list of int
            Indices of all channels that are part of the footprint
        footprint_wave_i: list of 1D numpy arrays
        """
        super().__init__()
        self.wind = wind
        self.rms_mult = rms_mult
        self.load_whole_recording = load_whole_recording
        self.verbose = verbose
        
        self.spikes_per_channel = spikes_per_channel
        self.num_channel_thresh = num_channel_thresh
        self.num_spikes_min = num_spikes_min

        self.remove_bad = remove_bad
        self.remove_redundant = remove_redundant
        self.similarity_thresh = similarity_thresh
        self.pk_to_pk_thresh = pk_to_pk_thresh
        self.experiment = experiment
        self.provides = ['selected_electrodes']
        self.name = name

    def run(self, experiment=None):
        # TODO: Should this generate an experiment if there is none? Should you be able to call
        # this without an experiment or running a previous phase.. Why isn't it just one phase then...

        results = {}

        if experiment is not None:
            self.experiment = experiment
        else:
            raise ValueError("Experiment is currently required for FootprintPhase")

        if self.verbose:
            print("Loading spikes")
        
        all_spikes = experiment.get_spikes()
        filename = experiment.get_baseline_filename()
  
        results['basename'] = filename

        
        selected_channels = experiment.mapper.get_channels(experiment.get_last("active_electrodes"))

        assert all_spikes is not None, "Must load spikes before running footprint phase"
        assert filename is not None or filename != '', "Must set filename before running footprint phase"
        assert selected_channels is not None, "Must select channels before running footprint phase"

        if self.verbose:
            print("Calculating RMS from a 10s snippet")
        all_data_rms = data_loader.load_data_maxwell(filename, start=20000*50, length=20000*10,
                                                    spikes=False)
        all_data_rms = np.std(all_data_rms, axis=1)
        if self.load_whole_recording:
            if self.verbose:
                print("Loading whole recording")
            # This is ch, t
            all_windows_loaded = data_loader.load_data_maxwell(filename)
            all_data_rms = np.std(all_windows_loaded, axis=1)

        
            
        
        footprint_chans = []
        footprint_waves = []
        footprint_pk_to_pks = []
        bad_chans = []
        main_pk_to_pks = []

        # For each selected channels of interest
        for channel_oi in selected_channels:
            

            # Get frames where channel_oi spikes
            chan_spikes = np.asarray(all_spikes['frame'][all_spikes['channel'] == channel_oi])

            if self.verbose:
                print(f"Processing channel {channel_oi} with {len(chan_spikes)} spikes")

            if len(chan_spikes) > self.spikes_per_channel:
                chan_spikes = np.random.choice(chan_spikes, self.spikes_per_channel, replace=False)

            if len(chan_spikes) < self.num_spikes_min:
                print(f"Channel {channel_oi} has too few spikes ({len(chan_spikes)}), skipping.")
                # Remove selected channels from analysis
                bad_chans.append(channel_oi)

            

            # Load the raw traces around the frames where channel_oi spikes
            if self.load_whole_recording:
                # Change ch, t to chan_spikes, ch, t
                all_windows = np.empty((len(chan_spikes), all_windows_loaded.shape[0], self.wind*2))
                for i, spike in enumerate(chan_spikes):
                    all_windows[i] = all_windows_loaded[:, spike-self.wind : spike+self.wind]
            else:
                all_windows = data_loader.load_windows_maxwell(filename, chan_spikes-self.wind, window_sz=self.wind*2)
            
            # Filter all of the data
            # butterworth_filter(all_windows,lowcut=300,highcut=7000)

            # Make empty result list for footprint channels
            footprint_chan_i = []
            footprint_wave_i = []
            footprint_pk_to_pk_i = []

            # Get pk to pk on the channel of interest
            main_avg = np.mean(all_windows[:,channel_oi,:],axis=0)
            main_pk_to_pk = np.max(main_avg) - np.min(main_avg)
            main_pk_to_pks.append(main_pk_to_pk)

            # For each channel
            for chan_i in range(all_windows.shape[1]):
                # Average results over all spikes
                spk_trig_av = np.mean(all_windows[:,chan_i,:],axis=0)

                # Obtain maximum and minimum of averaged results
                max_av = np.max(spk_trig_av)
                min_av = np.min(spk_trig_av)
                pk_to_pk = max_av - min_av

                # Compute RMS of averaged results
                rms_sig = np.sqrt(np.mean(spk_trig_av**2))

                # If the difference between maximum and minimum is larger than RMS threshold
                # if max_av-min_av > self.rms_mult*rms_sig:
                #     # Store channel index
                #     footprint_chan_i.append(chan_i)
                #     footprint_wave_i.append(spk_trig_av)

                # if max_av-min_av > self.rms_mult*all_data_rms[chan_i]:
                #     # Store channel index
                #     footprint_chan_i.append(chan_i)
                #     footprint_wave_i.append(spk_trig_av)

                # print("RMS mult:",self.rms_mult*all_data_rms[channel_oi], "Peak to peak:", max_av-min_av, "Main peak to peak:", main_pk_to_pk)

                # if max_av - min_av > .5*main_pk_to_pk:
                if pk_to_pk > self.pk_to_pk_thresh: # 9 is a magic number
                    if pk_to_pk < self.rms_mult*all_data_rms[channel_oi]:
                        # print("RMS BAD CHANNEL")
                        pass
                    else:
                        footprint_chan_i.append(chan_i)
                        footprint_wave_i.append(spk_trig_av)
                        footprint_pk_to_pk_i.append(pk_to_pk)

                if len(footprint_chan_i) > self.num_channel_thresh or channel_oi in bad_chans:
                    # Add channel to remove list
                    bad_chans.append(channel_oi)
                    break

            if channel_oi not in bad_chans:
                footprint_chans.append(footprint_chan_i)
                footprint_waves.append(footprint_wave_i)
                footprint_pk_to_pks.append(footprint_pk_to_pk_i)
            elif channel_oi in bad_chans and self.remove_bad:
                print(f"Channel {channel_oi} is bad, skipping.")
                # Remove selected channels from analysis
                continue
            else:
                footprint_chans.append([])
                footprint_waves.append([])
                footprint_pk_to_pks.append([])

        if self.remove_bad:
            experiment.mapper.select_channels([c for c in selected_channels if c not in bad_chans])

        if self.remove_redundant:
            # Check to see if the union of any 2 footprints shares more than 50% of the channels
            # If so, remove the smaller one
            to_remove = []
            selected_channels = experiment.mapper.selected_channels
            for i in range(len(selected_channels)):
                for j in range(len(selected_channels)):
                    if selected_channels[i] in to_remove or selected_channels[j] in to_remove:
                        continue
                    if i == j:
                        continue
                    
                    if (len(set(footprint_chans[i])& set(footprint_chans[j])) > self.similarity_thresh*len(set(footprint_chans[i]))
                        and len(set(footprint_chans[i])& set(footprint_chans[j])) > self.similarity_thresh*len(set(footprint_chans[j]))):
                        # Remove the one with the smaller main peak to peak
                        if main_pk_to_pks[i] < main_pk_to_pks[j]:
                            print(f"Removing channel {selected_channels[i]}: Redundant with {selected_channels[j]}")   
                            to_remove.append(selected_channels[i])
                        else:
                            print(f"Removing channel {selected_channels[j]}: Redundant with {selected_channels[i]}")
                            to_remove.append(selected_channels[j])

            idces_to_remove = [selected_channels.index(c) for c in to_remove]
            footprint_chans = [c for i, c in enumerate(footprint_chans) if i not in idces_to_remove]
            footprint_waves = [c for i, c in enumerate(footprint_waves) if i not in idces_to_remove]
            footprint_pk_to_pks = [c for i, c in enumerate(footprint_pk_to_pks) if i not in idces_to_remove]

            experiment.mapper.select_channels([c for c in selected_channels if c not in to_remove])


        # analysis.set_footprints(footprint_chans, footprint_waves, footprint_pk_to_pks)
        results['footprint_chans'] = footprint_chans
        # results['footprint_waves'] = footprint_waves
        results['selected_channels'] = experiment.mapper.selected_channels
        results['selected_electrodes'] = experiment.mapper.get_electrodes(results['selected_channels'])

        
        selected_channels = experiment.mapper.selected_channels

        # Make footprint plot
        from braindance.analysis.plot_helper import plot_footprints
        print('Plotting footprints')
        fig, ax = plot_footprints(footprint_chans, footprint_waves, experiment.mapping, selected_channels)
        plt.savefig(f'{filename}/footprints.png')

        experiment.update_params({'selected_electrodes':results['selected_electrodes']})
        experiment.update_params({'selected_channels':results['selected_channels']})        
        experiment.update_params({'stim_electrodes':results['selected_electrodes']})
        experiment.maxwell_clean_electrodes() # Also sets valid_electrodes

        return results
    
    def info(self):
        info = {}
        info['name'] = 'FootprintPhase'
        info['description'] = "Determines the footprint of each channel in the recording using a custom " \
        "metric that combines the spike count and mean amplitude of each electrode. Spike counts used are from" \
        "the maxwell spike data in the recording."
        info['results_keys'] = ['basename','footprint_chans', 'footprint_waves', 'selected_channels']

        return info


class CausalAnalysis(AnalysisPhase):
    '''
    Analyzes causal connectivity between selected electrodes, seeing the reaction to stimulations.
    
    This analyzes stimulation of electrodes and records their evoked responses to map
    causal connectivity. It measures how neural activity propagates through
    the network and calculates various connectivity metrics like first-order and
    multi-order connectivity, defined as the percentage of spikes that occur within
    a certain time window after stimulation. 
    
    Parameters
    ----------
    wind_ms : int, optional
        Time window in milliseconds for response analysis after stimulation
    channels_of_interest : list, optional
        List of channels to analyze for causal connectivity
    file_path : str, optional
        Path to the recording file
    stim_log_path : str, optional
        Path to stimulation log file
    clean_data_path : str, optional
        Path to save cleaned data
    tag : str, optional
        Tag to identify this analysis in filenames
    remove_start_frames : int, optional
        Number of frames to remove from the start of each recording chunk
    art_rem_N : int, optional
        Number of samples for artifact removal window
    save_clean_data : bool, optional
        Whether to save cleaned data to disk
    verbose : bool, optional
        Whether to print detailed progress information
        
    Requires
    --------
    A stimulation experiment which used stim_electrodes to stimulate the neural network, with
    the corresponding log saved as the same name as the recording file but with a '_log.csv' extension.
    (ex: exp_stim.raw.h5 and exp_stim_log.csv)
    
    Provides
    --------
    spikes : numpy.ndarray
        Detected spikes after artifact removal
        shape (n_reps, n_stims, n_react_chans) -> gives you list of spikes at that index
    spike_amps : numpy.ndarray
        Amplitudes of the detected, same shape as spikes
    clean_data : numpy.ndarray
        Raw data after artifact removal.
        shape (n_reps, n_stims, n_react_chans, n_samples)
    reactivity : dict
        Reactivity metrics for each electrode
    stim_electrodes : list
        List of electrodes used for stimulation
    channels_of_interest : list
        Channels analyzed for connectivity
    
    Modifies Experiment Parameters
    -----------------------------
    connectivity_metrics : dict
        Updates the experiment with connectivity metrics
    '''
    def __init__(self, wind_ms=150, channels_of_interest = None,  file_path=None,
                 stim_log_path=None, clean_data_path=None,
                tag='causal', remove_start_frames=40, art_rem_N = 60,
                save_clean_data=True,
                verbose=False):
        """
        Loads a fixed window size from the list of starts
        """
        super().__init__()
        self.wind_ms = wind_ms
        self.verbose = verbose
        self.channels_of_interest = channels_of_interest
        self.file_path = file_path
        self.stim_log_path = stim_log_path
        self.clean_data_path = clean_data_path
        self.tag = tag
        self.remove_start_frames = remove_start_frames
        self.save_clean_data = save_clean_data
        self.art_rem_N = art_rem_N

    def set_from_experiment(self, experiment):
        self.experiment = experiment
        self.file_path = experiment.get_last('filename')
        # self.channels_of_interest = experiment.mapper.get_channels(experiment.get_last('stim_electrodes'))
        # self.channels_of_interest = experiment.get_last('selected_channels')
        # self.channels_of_interest = experiment.mapper.selected_channels
        self.channels_of_interest = experiment.mapper.get_channels(experiment.params['stim_electrodes'])
        self.stim_electrodes = experiment.params['stim_electrodes']

    def run(self, experiment, file_path = None, stim_electrode_ind = None):
        from braindance.analysis import causal_connectivity
        from braindance.analysis.plot_helper import plot_reactivity_traces_experiment, \
                                                    plot_causal_connectivity_experiment

        results = {}

        if experiment is not None:
            self.experiment = experiment
            self.set_from_experiment(experiment)

        if self.verbose:
            print("Running Causal Analysis")
        
        if file_path is None:
            file_path = self.file_path
            if file_path is None:
                raise ValueError("File path is not set")

        clean_data, info = causal_connectivity.causal_reactions(
            file_path,
            react_channels=self.channels_of_interest,
            wind_ms=self.wind_ms,
            stim_log=None,
            save_dir=file_path,
            tag=self.tag,
            save_clean_data=self.save_clean_data,
            verbose=self.verbose
        )

        # Get clean data if re-calculating
        # Check in info for clean_data_paths
        clean_data_paths = info.get('clean_data_paths', None)
        if self.verbose:
            print("Clean data paths:", clean_data_paths)

        if clean_data_paths is not None:
            results['clean_data_paths'] = clean_data_paths
            # Load clean data
            print("Loading clean data, assuming only one clean data path (not split in causal_reactions)")
            clean_data = np.load(clean_data_paths[0])


        # Get spikes:
        spikes,amps = causal_connectivity.get_spikes(
            clean_data,
            sorter="std_thresh",
            sorter_params={'std':3.5},
            save_dir=file_path,
            spikes_offset_ms=0
        )


        # Get metrics:
        first_order_connectivity, multi_order_connectivity, burst_percent = causal_connectivity.causal_connectivity_metrics(
                                spikes,
                                first_order_ms=20,
                                multi_order_ms=(20, 200),
                                save_dir=file_path,
                                count_bursts=False
        )

        info['first_order_connectivity'] = first_order_connectivity
        info['multi_order_connectivity'] = multi_order_connectivity
        info['burst_percent'] = burst_percent
        info['selected_channels'] = self.channels_of_interest

        results['spikes'] = spikes
        results['spike_amps'] = amps
        

        # Get stim electrodes from the info
        stim_patterns = info.get('stim_patterns', None)
        if stim_patterns is not None:
            stim_electrodes = [p[0] for p in stim_patterns]
            results['stim_electrodes'] = stim_electrodes


        if self.channels_of_interest is not None:
            results['channels_of_interest'] = self.channels_of_interest
        
        results['clean_data'] = clean_data
        results['reactivity'] = info.get('reactivity', None)
        results['reactivity_times'] = info.get('reactivity_times', None)

        if self.clean_data_path:
            print("Saving clean data")
        
        # Generate reactivity plots using the experiment-based function
        if self.verbose:
            print("Generating reactivity plots at", file_path)


        plot_causal_connectivity_experiment(
            experiment,
            info=info,
            save_dir=file_path,
            first_order_ms=self.wind_ms,
            multi_order_ms=self.wind_ms*2,
            save_mats=True
        )

        plot_reactivity_traces_experiment(
            experiment, 
            clean_data=clean_data,
            stim_electrodes=results['stim_electrodes'],
            channels_of_interest=results['channels_of_interest'],
            save_dir=file_path,
            time_scatter=False
        )

        plot_reactivity_traces_experiment(
            experiment, 
            clean_data=clean_data,
            stim_electrodes=results['stim_electrodes'],
            channels_of_interest=results['channels_of_interest'],
            save_dir=file_path,
            time_scatter=True
        )

        
        # Update experiment parameters with connectivity metrics
        if experiment is not None:
            experiment.update_params({
                'connectivity_metrics': {
                    'first_order': first_order_connectivity,
                    'multi_order': multi_order_connectivity,
                    'burst_percent': burst_percent
                }
            })

        return results


class RTSortPhase(AnalysisPhase):
    '''
    Performs real-time spike sorting on a baseline recording.
    
    This phase identifies individual neurons based on their spike waveforms and
    spatial distribution across the electrode array. It uses a machine learning model
    to detect spike sequences and group them into distinct neural units.
    
    Parameters
    ----------
    name : str, optional
        Custom name for this phase instance
    detection_model_path : str, optional
        Path to the trained spike detection model
    inter_path : str, optional
        Path for storing intermediate files during processing
    recording_window_ms : tuple, optional
        Start and end time (in ms) of the recording window to analyze
    wind_ms : int, optional
        Window size in milliseconds for spike detection
    artifact_removal_params : dict, optional
        Parameters for artifact removal algorithm
    art_rem_N : int, optional
        Number of samples for artifact removal window
    force_redo : bool, optional
        Whether to force reprocessing even if results exist
    make_plots : bool, optional
        Whether to generate visualization plots
    save_plots : bool, optional
        Whether to save plots to disk
    delete_inter : bool, optional
        Whether to delete intermediate files after processing
    verbose : bool, optional
        Whether to print detailed progress information
    file_path : str, optional
        Path to the recording file
        
    Requires
    --------
    Either an experiment with a baseline recording or a file_path
    
    Provides
    --------
    rt_sort : object
        The RT-Sort object containing sorting results
    sequence_spike_trains : list
        List of spike trains for each detected neuron
    channels_per_neuron : list of lists
        For each neuron, a list of channels it appears on
    electrodes_per_neuron : list of lists
        For each neuron, a list of electrodes it appears on
    positions_per_neuron : list of lists
        For each neuron, a list of (x,y) positions
    
    Modifies Experiment Parameters
    -----------------------------
    rt_sort_neurons : int
        Number of neurons detected
    rt_sort_channels : list of lists
        Channels associated with each neuron
    rt_sort_electrodes : list of lists
        Electrodes associated with each neuron
    '''
    def __init__(self, name="RTSortPhase", detection_model_path=None, inter_path=None,
                 recording_window_ms=(0, 1*60*1000),
                 artifact_removal_params=None, art_rem_N=60,
                 force_redo=True, make_plots=True, save_plots=True,
                 delete_inter=True, verbose=True, file_path=None, save_rt_sort=True):
        super().__init__()
        
        self.name = name
        self.predicted_time = 300  # 5 minutes estimate
        if detection_model_path is None:
            self.detection_model_path = braindance.get_rt_sort_path()
        else:
            self.detection_model_path = detection_model_path
        if inter_path is None:
            self.inter_path = Path(os.path.dirname(file_path)) / 'rt_sort_tmp'
        else:
            self.inter_path = inter_path
        self.recording_window_ms = recording_window_ms
        # TODO: Add artifact removal params
        # self.artifact_removal_params = artifact_removal_params or {
        #     "N": art_rem_N, 
        #     "nc_start": art_rem_N,
        #     "artifact_width": art_rem_N,
        #     "remove_frames_before": art_rem_N,
        #     "n_stds": 0.5
        # }
        self.force_redo = force_redo
        self.make_plots = make_plots
        self.save_plots = save_plots
        self.delete_inter = delete_inter
        self.verbose = verbose
        self.file_path = file_path
        self.save_rt_sort = save_rt_sort

        self.provides = [
            'rt_sort',
            'sequence_spike_trains', 
            'channels_per_neuron',
            'electrodes_per_neuron',
            'positions_per_neuron'
        ]

        # Verify imports so we can check for errors before running
        from braindance.core.spikesorter.rt_sort import detect_sequences
        from spikeinterface.extractors import MaxwellRecordingExtractor
        from braindance.core.spikedetector.model2 import ModelSpikeSorter
        
    def run(self, experiment=None, file_path=None):
        """
        Run RT-Sort on the baseline recording from experiment or file path
        
        Parameters
        ----------
        experiment : Experiment object, optional
            Experiment object with baseline recording
        file_path : str, optional
            Path to the recording file
            
        Returns
        -------
        results : dict
            Dictionary with RT-Sort results
        """
        from braindance.core.spikesorter.rt_sort import detect_sequences
        from spikeinterface.extractors import MaxwellRecordingExtractor
        from braindance.core.spikedetector.model2 import ModelSpikeSorter
        from braindance.analysis import data_loader

        results = {}
        
        start_time = time.time()
        # Get baseline recording path
        if file_path is not None:
            baseline_path = file_path
        elif self.file_path is not None:
            baseline_path = self.file_path
        elif experiment is not None:
            baseline_path = experiment.get_baseline_filename()
        else:
            raise ValueError("Either experiment or file_path must be provided")
            
        if self.verbose:
            print(f"Processing baseline recording: {baseline_path}")
            
        # Set up paths
        if self.inter_path is None:
            self.inter_path = Path(os.path.dirname(baseline_path)) / 'rt_sort_tmp'
        inter_path = Path(self.inter_path)
        inter_path.mkdir(exist_ok=True, parents=True)
        
        # Set up detection model
        if self.detection_model_path is None:
            self.detection_model_path = braindance.get_rt_sort_path()
        detection_model = ModelSpikeSorter.load(self.detection_model_path)
        
        # Run the RT-Sort
        if self.verbose:
            print("Loading recording...")
        
        # Extract raw_h5 file path
        if not baseline_path.endswith('.raw.h5'):
            recording_file = baseline_path + '.raw.h5'
        else:
            recording_file = baseline_path
            
        recording = MaxwellRecordingExtractor(recording_file)
        
        if self.verbose:
            print("Running RT-Sort detection...")
            
        rt_sort = detect_sequences(
            recording, inter_path, detection_model, 
            recording_window_ms=self.recording_window_ms,
            return_spikes=False, delete_inter=self.delete_inter,
            verbose=self.verbose
        )

        # import IPython; IPython.embed()
        
        if rt_sort is None or len(rt_sort.seq_spike_trains) == 0:
            if self.verbose:
                print("No sequences detected!")
            return {}
            
        if self.verbose:
            print(f"Detected {len(rt_sort.seq_spike_trains)} sequences")
            
        sequence_spike_trains = rt_sort.seq_spike_trains
        
        # Get channels per neuron
        channels_per_neuron = rt_sort.seq_comp_elecs
        
        # Convert to electrodes using mapping
        electrodes_per_neuron = []
        positions_per_neuron = []
        
        # If we have an experiment, use its mapper
        if experiment is not None:
            for i, channels in enumerate(channels_per_neuron):
                experiment.get_mapping(file_path)
                electrodes = experiment.mapper.get_electrodes(channels)
                positions = experiment.mapper.get_positions(electrodes=electrodes)
                electrodes_per_neuron.append(electrodes)
                positions_per_neuron.append(positions)
                
            # Update experiment parameters
            experiment.update_params({
                'rt_sort_neurons': len(channels_per_neuron),
                'rt_sort_channels': channels_per_neuron,
                'rt_sort_electrodes': electrodes_per_neuron
            })
        else:
            # If no experiment, load mapping from file
            try:
                mapping = data_loader.load_mapping_maxwell(baseline_path)
                for i, channels in enumerate(channels_per_neuron):
                    # Filter mapping to only include channels in the sequence
                    ch_mapping = mapping[mapping.index.isin(channels)]
                    electrodes = ch_mapping['electrode'].tolist()
                    positions = ch_mapping[['x', 'y']].values.tolist()
                    electrodes_per_neuron.append(electrodes)
                    positions_per_neuron.append(positions)
            except Exception as e:
                if self.verbose:
                    print(f"Error loading mapping: {e}")
                    print("Unable to convert channels to electrodes and positions")
                electrodes_per_neuron = [[] for _ in channels_per_neuron]
                positions_per_neuron = [[] for _ in channels_per_neuron]
        
        # Store results
        results['rt_sort'] = rt_sort
        results['sequence_spike_trains'] = sequence_spike_trains
        results['channels_per_neuron'] = channels_per_neuron
        results['electrodes_per_neuron'] = electrodes_per_neuron
        results['positions_per_neuron'] = positions_per_neuron
        
        # Plot results if requested
        if self.make_plots:
            # Plot neuron positions
            plt.figure(figsize=(10, 10))
            for i, positions in enumerate(positions_per_neuron):
                if len(positions) > 0:  # Check if positions list is not empty
                    positions = np.array(positions)
                    plt.scatter(positions[:, 0], positions[:, 1], label=f'Neuron {i}', alpha=0.6, s=3)
            
            plt.xlabel('X Position (μm)')
            plt.ylabel('Y Position (μm)')
            plt.title('Spatial Distribution of Electrodes per Neuron')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            
            if self.save_plots:
                plt.savefig(f"{baseline_path}_neurons.png")
            else:
                plt.show()
                
            # Plot spike trains
            plt.figure(figsize=(12, 8))
            for i, spike_train in enumerate(sequence_spike_trains):
                plt.plot(spike_train, i*np.ones_like(spike_train), 'k|')
            
            plt.xlim(0, min(10000, max([max(st) if len(st) > 0 else 0 for st in sequence_spike_trains]) + 1000))
            plt.xlabel('Time (ms)')
            plt.ylabel('Neuron')
            plt.title('Spike Trains')
            
            if self.save_plots:
                plt.savefig(f"{baseline_path}_spike_trains.png")
            else:
                plt.show()
        
        if self.verbose:
            print(f"RT-Sort completed in {time.time() - start_time:.1f} seconds")

        if self.save_rt_sort:
            # Check if save_rt_sort is a path or just a boolean
            if isinstance(self.save_rt_sort, (str, Path)):
                # Use the provided path directly
                rt_sort_path = self.save_rt_sort
                rt_sort.save(rt_sort_path)
                if experiment is not None:
                    experiment.update_params({'rt_sort_path': str(rt_sort_path)})
            else:
                # Use default path (original behavior)
                file_dir = os.path.dirname(baseline_path)
                rt_sort_path = os.path.join(file_dir, os.path.basename(baseline_path) + '_rt_sort.pkl')
                rt_sort.save(rt_sort_path)
                if experiment is not None:
                    experiment.update_params({'rt_sort_path': rt_sort_path})

        # Update experiment parameters
        if experiment is not None:
            experiment.update_params({
                'rt_sort_neurons': len(channels_per_neuron),
                'rt_sort_channels': channels_per_neuron,
                'rt_sort_electrodes': electrodes_per_neuron
            })

        return results
        
    def info(self):
        """Return information about the phase"""
        return {
            'name': self.name,
            'description': "Run RT-Sort on baseline recording and extract neural information",
            'detection_model': str(self.detection_model_path),
            'recording_window': str(self.recording_window_ms)
        }