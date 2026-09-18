# maxwell_env.py
import sys
import zmq
import struct
import array
import time
import numpy as np
import os
import warnings

from pathlib import Path
from multiprocessing import Process, Queue

try:
    from tqdm import trange
except ImportError:
    trange = None

try:
    import maxlab
    import maxlab.system
    import maxlab.chip
    import maxlab.util
    import maxlab.saving
except ImportError:
    print("No maxlab found, instead using dummy maxlab module!")
    
    import braindance.core.dummy_maxlab as maxlab
    

# stop_process_using_port(7204) to stop dummy raw data if not stopped automatically (useful for Jupyter notebooks)
import psutil
def sleep_with_progress(seconds, desc, verbose=1):
    if not verbose:
        time.sleep(seconds)
        return

    if trange is None:
        print(f"{desc} ({seconds}s)")
        time.sleep(seconds)
        return

    for _ in trange(seconds, desc=desc, unit="s"):
        time.sleep(1)


def find_process_by_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections():
                if conn.laddr.port == port:
                    return proc
        except psutil.AccessDenied:
            pass  # Ignore processes we don't have permission to access
    return None
def stop_process_using_port(port):
    proc = find_process_by_port(port)
    if proc:
        print(
            f"Found process {proc.pid} ({proc.name()}) using port {port}. Stopping...")
        proc.terminate()
        proc.wait()  # Wait for the process to terminate
    else:
        print(f"No process found using port {port}")


from collections import namedtuple
from csv import writer

from braindance.core.base_env import BaseEnv

# SpikeEvent is a named tuple that represents a single spike event
SpikeEvent = namedtuple('SpikeEvent', 'frame channel amplitude')

_spike_struct = '@Lfhc0L'
_spike_struct_size = struct.calcsize(_spike_struct)
_spike_struct_formats = (
    # Padded layout: frame, channel, amplitude.
    ('padded', '8xLif', struct.calcsize('8xLif')),
    # wellId layout: frame, amplitude, channel, wellId.
    ('well_id', '@Lfhc0L', struct.calcsize('@Lfhc0L')),
)
_spike_struct_format = None
fs_ms = 20 # sampling rate in kHz


class MaxwellEnv(BaseEnv):
    """
    Maxwell acquisition environment with hardware and local replay sources.


    Attributes
    ----------
    config : str
        Stores the config filepath in order to easily reload the array.
    name : str
        Stores the name of the environment instance.
    max_time_sec : int
        Stores the maximum experiment time.
    save_file : str
        The file where the data will be saved.
    stim_electrodes : list
        Stores the list of electrodes for stimulation.
    verbose : int
        Controls the verbosity of the environment's operations.
    array : None
        Initialized as None, to be updated in sub-classes as needed.
    subscriber : None
        Initialized as None, to be updated in sub-classes as needed.
    save_dir : str
        Stores the directory where the simulation data will be saved.
    is_stimulation : bool
        A flag that indicates whether a stimulation is going to occur.
    stim_log_file : str or None
        The file where the log of the stimulation is saved. If no stimulation is going to occur, this is None.
    stim_units : None
        Initialized as None, to be updated in sub-classes as needed.
    stim_electrodes_dict : None
        Initialized as None, to be updated in sub-classes as needed.
    start_time : float
        The time when the environment is initialized.
    cur_time : float
        The current time, updated at each step.
    last_stim_time : float
        The time when the last stimulation occurred.
    smoke_test : bool
        A flag that indicates whether the environment is being used for a smoke test
    skip_offset : bool
        A flag that indicates whether to skip the offset calculation.
        
    """

    def __init__(self, config=None, name="", stim_electrodes=None, max_time_sec=60,
                save_dir="data", multiprocess=False, render=False,
                filt=False, observation_type='spikes', verbose = 1, 
                smoke_test=False, dummy=None, start=True, skip_offset=False,
                replay=None):
        """
        Parameters
        ----------
        config : str
            A path to the maxwell config file. This is usually made by the Maxwell GUI, 
            and contains the information about the array.
        name : str
            The name of the environment instance. This is used for saving data.
        stim_electrodes : list
            A list of electrodes for stimulation. If no electrodes are specified, no stimulation will occur.
        max_time_sec : int
            The maximum experiment time in seconds.
        save_dir : str
            The directory where the stimulation data will be saved.
        filt : bool
            A flag that indicates whether a filter should be applied to the data. The filter is onboard the chip,
            and is applied to the data before it is sent to the computer. It adds ~100ms of latency.
        observation_type : str
            A string that indicates the type of observation that the environment should return.
                'spikes' returns a list of spike events
                'raw' returns the raw datastream frame with shape (ch,1) 
        verbose : int
            An integer that controls the verbosity of the environment's operations. 0 is silent, 1 is verbose.
        smoke_test : bool
            A flag that indicates whether the environment is being used for a smoke test. If True, the environment
            will not save any data, will use dummy logic, and no hardware will be used.
        dummy : str
            A flag that will indicate whether to use a dummy maxwell server.
                'sine' will use a sine wave for the data
                '{filepath}' will use the first 30 seconds of data from the filepath
                None will use the real maxwell server

        """
        super().__init__(max_time_sec=max_time_sec, verbose=verbose)

        if stim_electrodes is None:
            stim_electrodes = []
        if replay is None and dummy is not None:
            warnings.warn(
                "MaxwellEnv(dummy=...) is deprecated; use MaxwellEnv(replay={'source': ...})",
                DeprecationWarning,
                stacklevel=2,
            )
            replay = {'source': dummy}
        elif replay is not None and dummy is not None:
            raise ValueError("Specify replay or dummy, not both")

        self.is_replay = replay is not None
        self._maxlab = maxlab
        if self.is_replay:
            # Local acquisition must remain local on machines with maxlab installed.
            from braindance.core import dummy_maxlab
            self._maxlab = dummy_maxlab
        self.replay = dict(replay or {})
        self.replay_source = None
        self.replay_writer = None
        self._replay_done = False
        self._closed = False
        self.latest_frame = None
        self.latest_batch = None
        if self.is_replay and (multiprocess or render):
            raise ValueError("Replay mode does not support multiprocess or render")
        if not self.is_replay and config is None:
            raise ValueError("config is required when using Maxwell hardware")

        self.config = config
        self.config_data = None if self.is_replay else Config(config)
        self.base_name = name
        self.name = name
        self.multiprocess = multiprocess

        self.stim_electrodes = [int(e) for e in stim_electrodes]
        self.active_units = []
        self.observation_type = observation_type
        
        self.array = None
        self.subscriber = None

        self.worker = None
        self.plot_worker = None
        self.dummy = dummy

        if self.is_replay:
            from braindance.core.replay import H5ReplaySource

            transport = self.replay.pop('transport', 'direct')
            if transport != 'direct':
                raise ValueError("MaxwellEnv replay transport must be 'direct'; use dummy_zmq_np for ZMQ tests")
            self.replay_write_output = bool(self.replay.pop('write_output', True))
            if 'source' not in self.replay:
                raise ValueError("Replay configuration requires a source")
            source = self.replay.pop('source')
            if callable(getattr(source, 'read', None)):
                required = ('num_channels', 'mapping', 'sampling_hz', 'lsb',
                            'gain', 'hpf', 'finished', 'elapsed_s', 'close')
                missing = [key for key in required if not hasattr(source, key)]
                if missing or not callable(getattr(source, 'close', None)) or self.replay:
                    raise ValueError(f"Custom source needs {missing}; configure source options on the source itself")
                self.replay_source = source
            else:
                self.replay_source = H5ReplaySource(source=source, **self.replay)
            self.num_channels = self.replay_source.num_channels
            self.stim_units = [self._maxlab.chip.StimulationUnit(i) for i in range(len(stim_electrodes))]
            self.stim_electrodes_dict = {
                unit: electrode for unit, electrode in zip(self.stim_units, self.stim_electrodes)
            }
        else:
            self.num_channels = self.config_data.get_num_channels()
            # Setup Maxwell hardware and subscriber.
            self.subscriber, self.stim_units, self.stim_electrodes_dict = init_maxone(
                    config, stim_electrodes, filt=filt,
                    verbose=1, gain=1024, cutoff='1Hz',
                    spike_thresh=5, dummy=None, skip_offset=skip_offset
            )

        # Setup saving
        self.save_dir = str(Path(save_dir).resolve())
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self._validate_name()
        self.save_file = os.path.join(self.save_dir, f'{self.name}')


        

        # Check whether stimulation will occur
        if len(stim_electrodes) == 0:
            self.is_stimulation = False
            self.stim_log_file = None
        else:
            self.is_stimulation = True
            self.stim_num = 0
            self._init_log_stimulation()

        

        if self.multiprocess:
            subscriber_args = (filt, verbose)
            self.data_queue = Queue()
            self.event_queue = Queue()
            self.worker = Process(target=socket_worker, args=(self.data_queue, self.event_queue, subscriber_args))
            self.worker.start()

            if render:
                self.plot_worker = Process(target=plot_worker, args=(self.data_queue,))
                self.plot_worker.start()

        if not self.is_replay:
            # Let hardware settle, then drop packets accumulated during the wait.
            sleep_with_progress(10, "Settling after Maxwell init", verbose=verbose)
            ignore_remaining_packets(self.subscriber, verbose=verbose)
        

        

        

        # Time management
        self._init_time_management()
        if self.is_replay:
            self.cur_time = 0.0
        self.last_stim_time = 0
        self.last_stim_times = np.zeros(len(stim_electrodes))

        if start:
            # Flush the buffer
            # ignore_first_packet(self.subscriber)
            # ignore_remaining_packets(self.subscriber) # New, should wipe data

            # Start recording
            self._init_save()
            if verbose:
                print(f'Recording from {self.num_channels} channels and {len(stim_electrodes)} stim electrodes')
            print("===================== Beginning experiment =====================")
        
    def start(self):
        # Time management
        self._init_time_management()
        self.last_stim_time = 0
        self.last_stim_times = np.zeros_like(self.last_stim_times)

        # Flush the buffer
        # ignore_first_packet(self.subscriber)
        # ignore_remaining_packets(self.subscriber) # New, should wipe data

        # Start recording
        self._init_save()
        if self.is_stimulation:
            self._init_log_stimulation()
        print(f'Recording from {self.num_channels} channels and {len(self.last_stim_times)} stim electrodes')
        print("===================== Beginning experiment =====================")


    def reset(self):
        """
        Reset the environment
        """
        if self.is_replay:
            raise NotImplementedError("Create a new MaxwellEnv to restart replay")
        print("===================== Resetting experiment =====================")
        self._cleanup()
        self._closed = False

        # Change name
        self._validate_name()
        self.save_file = os.path.join(self.save_dir, f'{self.name}')
        # self.subscriber.setsockopt_string(zmq.SUBSCRIBE, '')

        
        self._init_save()
        self._init_log_stimulation()

        self.start_time = self.cur_time = time.perf_counter()
        self.last_stim_time = 0
        self.last_stim_times = np.zeros(len(self.stim_electrodes))



        # Close previous save files
        # self.stim_log_file.close()
        
        # self._validate_name()

        # if self.is_stimulation:
        #     self._init_log_stimulation()

        # # init_maxwell() #TODO: Reset the array
        # self.start_time = self.cur_time = time.perf_counter()
        # self.last_stim_time = 0

        # Reset the environment with the same parameters

    def clear_buffer(self, num_successive_waits=10, min_wait_f=0.5, buffer_size=10,
                     samp_freq_hz=20000, timeout_s=30):
        """
        Clear the ZMQ socket buffer, so self.step() returns latest data
        This is done by waiting until the time to receive buffer_size frames is at least to min_wait_f*buffer_size
        for num_successive_waits successive method calls
            There are two buffers: the ZMQ socket buffer and buffer_size used in self.step(buffer_size=buffer_size)
        
        Parameters
        ----------
        timeout_s : float
            Maximum time in seconds to spend clearing the buffer before giving up.
            Default 30s. Set to None to disable timeout (original behavior).
        """
        if self.is_replay:
            return
        print("Clearing buffer")
        
        from time import perf_counter
        
        total_start = perf_counter()
        cur_count = 0
        iteration = 0
        while True:
            start = perf_counter()
            self.step(buffer_size=buffer_size)
            end = perf_counter()
            elapsed_step = end - start
            frames = elapsed_step * samp_freq_hz
            
            if frames/buffer_size >= min_wait_f:
                cur_count += 1
            else:
                cur_count = 0
            
            iteration += 1
            
            # Progress logging every 1000 iterations
            if iteration % 1000 == 0:
                total_elapsed = perf_counter() - total_start
                print(f"  clear_buffer: {iteration} iterations, {total_elapsed:.1f}s elapsed, "
                      f"cur_count={cur_count}/{num_successive_waits}, "
                      f"last step: {elapsed_step*1000:.2f}ms ({frames:.1f} frames)")
                
            if cur_count == num_successive_waits:
                break
            
            # Timeout check
            if timeout_s is not None:
                total_elapsed = perf_counter() - total_start
                if total_elapsed > timeout_s:
                    print(f"WARNING: clear_buffer timed out after {total_elapsed:.1f}s "
                          f"({iteration} iterations, cur_count={cur_count}/{num_successive_waits}). "
                          f"Proceeding anyway.")
                    break

        total_end = perf_counter()
        print(f"Time to clear buffer: {total_end-total_start:.2f}s ({iteration} iterations)")
        

    def get_observation(self, buffer_size=None):
        '''
        Create the observation from the electrodes or spike events
        '''
        if self.is_replay:
            count = 1 if buffer_size is None else int(buffer_size)
            batch = self.replay_source.read(
                count=count,
                convert_raw=self.observation_type == 'raw',
            )
            if batch is None:
                self._replay_done = True
                return [] if self.observation_type == 'spikes' else None
            self.latest_batch = batch
            if self.replay_writer is not None:
                self.replay_writer.append(batch)
            self.latest_frame = int(batch['frame_numbers'][-1])
            self._replay_done = self.replay_source.finished

            if self.observation_type == 'spikes':
                return [event for frame_events in batch['events'] for event in frame_events]
            if self.observation_type == 'raw':
                # Packed buffers avoid a Python conversion for every channel/sample.
                frames = [array.array('f', frame.tobytes()) for frame in batch['raw_float32']]
                return frames[0] if buffer_size is None else frames
            raise ValueError(f"Unsupported observation_type: {self.observation_type}")

        if self.observation_type == 'spikes':
            # Receive data
            if self.multiprocess:
                # frame_number, frame_data, events_data = 
                obs = self.event_queue.get()

            else:
                frame_number, frame_data, events_data = receive_packet(self.subscriber) #TODO: Get all frames, populate buffer
                if frame_number is not None:
                    self.latest_frame = frame_number
                # frame = self._parse_frame(frame_data) # Raw datastream

                # If events on >15% of channels, then we assume that the data is bad -> stim artifact
                n_events = np.nan if events_data is None else len(events_data)//get_spike_struct_size(events_data)
                if n_events < self.num_channels * .15:
                    obs = parse_events_list(events_data) # Spike events
                else:
                    obs = []
                
            return obs
        
        elif self.observation_type == 'raw':
            # Receive data with a buffer
            # ---------------------------
            if buffer_size is not None:
                frame_numbers, frames, event = receive_packet(self.subscriber, buffer_size=buffer_size)
                if not frame_numbers:
                    return None
                obs_list = []
                for frame in frames:
                    obs_list.append(parse_frame(frame))
                self.latest_frame = frame_numbers[-1]
                return obs_list

            if self.multiprocess:
                # frame_number, frame_data, events_data = 
                pass # TODO implement

            else:
                frame_number, frame_data, events_data = receive_packet(self.subscriber)
                obs = parse_frame(frame_data)
                if frame_number is not None:
                    self.latest_frame = frame_number
            return obs 


    def step(self, action=None, tag=None, buffer_size=None):
        '''
        Recieve events published since last time step() was called.
        This includes spike events and raw datastream.

        Parameters
        ----------
        action : list 
            Either
                1) a tuple of the form (maxlab sequence, [electrode_inds])
                2) a tuple of the form (stim_command, electrode_inds)
                3) a tuple of the form (stim_command, electrode_inds, tag) # TODO: Update
            where stim_command is a list of tuples of the form ('stim', [neuron inds], mv, us per phase)
            electrode_inds is a list of electrode indices to stimulate
        
        '''
        self.cur_time = self._clock_now()

        # Receive data
        if self.multiprocess:
            # frame_number, frame_data, events_data = 
            obs = self.event_queue.get()

        else:
            obs = self.get_observation(buffer_size=buffer_size)
            # frame_number, frame_data, events_data = receive_packet(self.subscriber)#TODO: Get all frames, populate buffer
            # # frame = self._parse_frame(frame_data) # Raw datastream

            # # If events on >15% of channels, then we assume that the data is bad
            # n_events = np.nan if events_data is None else len(events_data)//_spike_struct_size
            # if n_events < self.num_channels * .15:
            #     obs = parse_events_list(events_data) # Spike events
            # else:
            #     obs = []


        if action is not None:
            self.stimulate(action, tag=tag)
        done = self._replay_done or self._check_if_done()
        return obs, done

    def stimulate(self, action, tag=None):
        """Dispatch after acquisition without consuming another batch.

        Replay logs the command and notifies an optional source callback. It
        never sends a hardware sequence, even when real maxlab is installed.
        """
        if self._closed:
            raise RuntimeError("Cannot stimulate a closed environment")
        if not self.is_stimulation:
            raise ValueError("No stimulation electrodes configured")
        manual = isinstance(action[0], str) and action[0] == "manual"
        if manual and self.is_replay:
            raise ValueError("Replay requires explicit pulse commands, not manual sequences")
        if manual:
            self.seq, electrode_inds = action[1], action[2]
        elif isinstance(action[0][0], str):
            electrode_inds = sorted({i for cmd in action if cmd[0] == 'stim' for i in cmd[1]})
            if any(not 0 <= i < len(self.stim_electrodes) for i in electrode_inds):
                raise ValueError("Stimulation index outside configured electrodes")
            self._create_stim_pulse_sequence(action)
        else:
            electrode_inds = action[0]
            if any(not 0 <= i < len(self.stim_electrodes) for i in electrode_inds):
                raise ValueError("Stimulation index outside configured electrodes")
            self._create_stim_pulse(action)
        if self.is_replay:
            callback = getattr(self.replay_source, 'on_stimulation', None)
            if callback is not None:
                callback(action, frame=self.latest_frame, stim_electrodes=self.stim_electrodes)
        else:
            self.seq.send()
        self._log_stimulation(action, tag=tag)
        stim_time = self._clock_now()
        self.last_stim_time = stim_time
        self.last_stim_times[electrode_inds] = stim_time
        if self.verbose >= 2:
            print(f'Stimulating at t={stim_time} with command:', self.seq.token)

    def _clock_now(self):
        if self.is_replay and self.replay_source is not None:
            return self.replay_source.elapsed_s
        return time.perf_counter()

    def time_elapsed(self):
        """Experiment time: source-frame time in replay, wall time on hardware."""
        if self.is_replay and self.replay_source is not None:
            return self.replay_source.elapsed_s
        return super().time_elapsed()

    def wall_time_elapsed(self):
        """Wall time since environment initialization, for diagnostics."""
        return super().time_elapsed()

    @property
    def dt(self):
        return self._clock_now() - self.cur_time
    
    @property
    def stim_dt(self):
        '''Returns time since last stimulation.'''
        return self._clock_now() - self.last_stim_time
    
    @property
    def stim_dts(self):
        '''Returns time since last stimulation.'''
        return self._clock_now() - self.last_stim_times
    
    def close(self):
        '''Shuts down the environment and saves the data.'''
        print("===================== Ending experiment =====================")
        self._cleanup()

    def _validate_name(self):
        '''
        Validate the name of the environment.
        If the name already exists, increment the name until it doesn't exist.
        '''
        name = self.base_name
        i = 0
        # Check if savefile path exists already and increment the name until it doesn't exist
        while os.path.exists(os.path.join(self.save_dir, f'{name}.raw.h5')):
            i += 1
            name = f'{self.base_name}_{i}'

        if self.verbose:
            print('Name of experiment: ', name)
            print('At ', os.path.join(self.save_dir, f'{name}.raw.h5'))
        self.name = name


    

    def _create_stim_pulse(self, stim_command):
        '''
        Create a pulse sequence that just sets the DAC amplitude in a pulse shape for a brief
        period. This should be combined with code that connects electrodes to the DAC in order
        to actually generate stimulus behavior.

        Parameters
        ----------
        stim_command : tuple
            A tuple of the form (stim_electrodes, amplitude_mV, phase_length_us)
                stim_electrodes : list
                    A list of electrode numbers to stimulate.
                amplitude_mV : float
                    The amplitude of the square wave, in mV.
                phase_length_us : float
                    The length of each phase of the square wave, in us.
        '''
        self.seq = self._maxlab.Sequence()
        self.active_units = []
        
        neurons, amplitude_mV, phase_length_us = stim_command # phase length in us
        
        # Append each neuron that needs to be stimmed in order
        for n in neurons:
            # print(f'Adding neuron {n} to stim sequence')
            unit = self.stim_units[n]
            self.active_units.append(unit)
            self.seq.append(unit.power_up(True))
        
        # Create pulse
        self._insert_square_wave(amplitude_mV, phase_length_us)

        # Power down all units
        for unit in self.active_units:
            self.seq.append(unit.power_up(False))

        return self.seq

    def _create_stim_pulse_sequence(self, stim_commands):
        '''
        Create a pulse sequence that just sets the DAC amplitude in a pulse shape for a brief
        period. This should be combined with code that connects electrodes to the DAC in order
        to actually generate stimulus behavior.

        Parameters
        ----------
        stim_commands : list of tuples
            A tuple of the form (command, stim_electrodes, amplitude_mV, phase_length_us)
                stim_electrodes : list
                    A list of electrode numbers to stimulate.
                amplitude_mV : float
                    The amplitude of the square wave, in mV.
                phase_length_us : float
                    The length of each phase of the square wave, in us.

        ------------------------------------------------
        For 'stim' command:
        ('stim', [neuron inds], mv, us per phase)

        For 'delay'
        ('delay', frames_delay)
        
        For 'next'
        ('next', None)
        This command acts as a placeholder to move to the next timepoint in the time_arr or the next
        period triggered by the freq_Hz
        -------------------------------------------------
        '''
        self.seq = self._maxlab.Sequence()
        self.active_units = []
        stim_commands = stim_commands.copy()

        # Build the sequence
        command = None
        while len(stim_commands) > 0:
            command, *params = stim_commands.pop(0)
            
            # ----------------- stim --------------------
            if command == 'stim':
                neurons, amplitude_mV, phase_length_us = params # phase length in us
                
                # Append each neuron that needs to be stimmed in order
                for n in neurons:
                    unit = self.stim_units[n]
                    self.active_units.append(unit)
                    self.seq.append(unit.power_up(True))
                
                # Create pulse
                self._insert_square_wave(amplitude_mV, phase_length_us)

                # Power down all units
                for unit in self.active_units:
                    self.seq.append(unit.power_up(False))
            
            # ----------------- delay --------------------
            if command == 'delay':
                self.seq.append( self._maxlab.system.DelaySamples(params[0]*fs_ms))
                
            # ----------------- next --------------------
            if command == 'next':
                break 
        
        self.stim_num += 1
        return self.seq
        



    def _insert_square_wave(self, amplitude_mV = 150, phase_length_us = 100):
        '''
        Adds a square wave to the sequence with a set amplitude and duty cycle.

        Parameters
        ----------
        amplitude_mV : float
            The amplitude of the square wave, in mV.

        duty_time_us : float
            The duty cycle of the square wave, in us.
            We can only set the duty cycle in increments of 50us, so this will be rounded.

        '''
        amplitude_lsbs = round(amplitude_mV / 2.9) # scaling factor given by Maxwell
        duty_time_samp = round(phase_length_us * .02)

        # Not sure what the 3rd argument should be -- user_id?
        self.seq.append(self._maxlab.system.Event(0, 1, 0, f"custom_id {self.stim_num}"))
        
        self.seq.append( self._maxlab.chip.DAC(0, 512 - amplitude_lsbs) )
        self.seq.append( self._maxlab.system.DelaySamples(duty_time_samp) )
        self.seq.append( self._maxlab.chip.DAC(0, 512 + amplitude_lsbs) )
        self.seq.append( self._maxlab.system.DelaySamples(duty_time_samp) )
        self.seq.append( self._maxlab.chip.DAC(0, 512) )
        self.seq.append( self._maxlab.system.DelaySamples(2) )


    def _insert_sine_wave(self, amplitude_mV=50, frequency_Hz=1):
        '''
        Adds a sine wave to the sequence with a set amplitude and frequency.

        Parameters
        ----------
        amplitude_mV : float
            The amplitude of the sine wave, in mV.

        phase_length_us : float
            The duration of one phase (half-period) of the sine wave, in us.

        frequency_Hz : float
            The frequency of the sine wave, in Hz.
        '''
        amplitude_lsbs = round(amplitude_mV / 2.9)  # scaling factor given by Maxwell
        num_samples = int((1 / frequency_Hz) * 20000)  # number of samples per full sine wave cycle

        # Create sine wave samples
        t = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        sine_wave = 512 + amplitude_lsbs * np.sin(t)

        for sample in sine_wave:
            self.seq.append(self._maxlab.chip.DAC(0, int(sample)))
            self.seq.append(self._maxlab.system.DelaySamples(1))
    
    #==========================================================================
    #=========================  Saving Functions  =============================
    #==========================================================================
    def _init_save(self):
        '''
        Initialize the save file for the environment.
        Saved in self.save_dir with name self.name.
        '''
        if self.is_replay:
            if not self.replay_write_output:
                return
            from braindance.core.replay import H5ReplayWriter

            self.replay_writer = H5ReplayWriter(
                self.save_file + '.raw.h5',
                mapping=self.replay_source.mapping,
                sampling_hz=self.replay_source.sampling_hz,
                lsb=self.replay_source.lsb,
                gain=self.replay_source.gain,
                hpf=self.replay_source.hpf,
            )
            return

        self.saver = self._maxlab.saving.Saving()
        self.saver.open_directory(self.save_dir)
        self.saver.set_legacy_format(False)
        self.saver.group_delete_all()
        self.saver.group_define(0, "routed")
        self.saver.start_file(self.name)
        self.saver.start_recording([0])


    def _init_log_stimulation(self):
        self.stim_log_file = os.path.join(self.save_dir, self.name + '_log.csv')

        if self.stim_log_file is not None:
            self.stim_log_file = open(self.stim_log_file, 'a+', newline='')
            self.stim_log_writer = writer(self.stim_log_file)
            # write first row: stim time, amplitude
            columns = ['time', 'amplitude', 'duty_time_ms', 'stim_electrodes', 'tag']
            if self.is_replay:
                columns.append('replay_frame')
            self.stim_log_writer.writerow(columns)

    def _log_stimulation(self, stim_command, tag=None):
        '''
        Log the stimulation command to a csv file.
        Stim command is a tuple of the form (stim_electrodes, amplitude_mV, phase_length_us)
        '''
        if self.stim_log_file is not None:
            if tag is None:
                tag = ''
            if isinstance(stim_command[0],str) and stim_command[0] == "manual":
                elecs = [self.stim_electrodes[i] for i in stim_command[2]]
                row = [self.time_elapsed(), "custom", 'manual', elecs, tag]
            elif type(stim_command[0][0]) == str:
                # We just write the first one since it becomes too complicated to write all of them
                elecs = []
                for cmd in stim_command:
                    if cmd[0] == 'stim':
                        elecs.append([self.stim_electrodes[i] for i in cmd[1]])
                row = [self.time_elapsed(), stim_command[0][1], stim_command[0][2], elecs, tag]
            else:
                elecs = [self.stim_electrodes[i] for i in stim_command[0]]
                row = [self.time_elapsed(), stim_command[1], stim_command[2], elecs, tag]
            if self.is_replay:
                row.append(self.latest_frame)
            self.stim_log_writer.writerow(row)
            self.stim_log_file.flush()

    def _cleanup(self):
        '''Shuts down the environment and saves the data.'''
        if self._closed:
            return
        self._closed = True
        if self.is_replay:
            if self.replay_writer is not None:
                self.replay_writer.close()
                self.replay_writer = None
            if self.replay_source is not None:
                self.replay_source.close()
        else:
            self.saver.stop_recording()
            self.saver.stop_file()
            self.saver.group_delete_all()
        if self.stim_log_file is not None:
            self.stim_log_file.close()
            self.stim_log_file = None

        # self.subscriber.setsockopt_string(zmq.UNSUBSCRIBE, '')

    #==========================================================================
    #=========================  Helper Functions  =============================
    #==========================================================================

    def disconnect_all(self):
        '''Disconnect all stimulation units.'''

        seq = self._maxlab.Sequence()
        for unit in self.stim_units:
            seq.append(unit.power_up(False).connect(False))
        seq.send()

    def connect_units(self, units=None, inds=None):
        '''Connect the specified stimulation units.'''
        if units is None:
            if inds is None:
                raise Exception('Must specify either units or inds')
            units = [self.stim_units[i] for i in inds]
            
        seq = self._maxlab.Sequence()
        for unit in units:
            seq.append(unit.power_up(False).connect(True))
        seq.send()
        

class MaxwellStim:
    """
    Used for stimulating electrodes in a parallel process
    """
    def __init__(self, stim_units):
        self.stim_units = stim_units
        self.stim_num = 0  # Not sure why needed, but prevents error in self._insert_square_wave()
        
    def _create_stim_pulse(self, stim_command):
        '''
        Create a pulse sequence that just sets the DAC amplitude in a pulse shape for a brief
        period. This should be combined with code that connects electrodes to the DAC in order
        to actually generate stimulus behavior.

        Parameters
        ----------
        stim_command : tuple
            A tuple of the form (stim_electrodes, amplitude_mV, phase_length_us)
                stim_electrodes : list
                    A list of electrode numbers to stimulate.
                amplitude_mV : float
                    The amplitude of the square wave, in mV.
                phase_length_us : float
                    The length of each phase of the square wave, in us.
        '''
        self.seq = maxlab.Sequence()
        self.active_units = []
        
        neurons, amplitude_mV, phase_length_us = stim_command # phase length in us
        
        # Append each neuron that needs to be stimmed in order
        for n in neurons:
            # print(f'Adding neuron {n} to stim sequence')
            unit = self.stim_units[n]
            self.active_units.append(unit)
            self.seq.append(unit.power_up(True))
        
        # Create pulse
        self._insert_square_wave(amplitude_mV, phase_length_us)

        # Power down all units
        for unit in self.active_units:
            self.seq.append(unit.power_up(False))

        return self.seq

    def _insert_square_wave(self, amplitude_mV = 150, phase_length_us = 100):
        '''
        Adds a square wave to the sequence with a set amplitude and duty cycle.

        Parameters
        ----------
        amplitude_mV : float
            The amplitude of the square wave, in mV.

        duty_time_us : float
            The duty cycle of the square wave, in us.
            We can only set the duty cycle in increments of 50us, so this will be rounded.

        '''
        amplitude_lsbs = round(amplitude_mV / 2.9) # scaling factor given by Maxwell
        duty_time_samp = round(phase_length_us * .02)

        # Not sure what the 3rd argument should be -- user_id?
        self.seq.append(maxlab.system.Event(0, 1, 0, f"custom_id {self.stim_num}"))
        
        self.seq.append( maxlab.chip.DAC(0, 512 - amplitude_lsbs) )
        self.seq.append( maxlab.system.DelaySamples(duty_time_samp) )
        self.seq.append( maxlab.chip.DAC(0, 512 + amplitude_lsbs) )
        self.seq.append( maxlab.system.DelaySamples(duty_time_samp) )
        self.seq.append( maxlab.chip.DAC(0, 512) )
        self.seq.append( maxlab.system.DelaySamples(2) )


def stim_process(
                 stim_process_ready,
                 stim_units, stim_ready,
                 stim_shm_name,
                 env_done,
                 stim_amp=400, stim_length=100, stim_tag="stim"):
    """
    Params
        stim_amp
            mV of square wave
        stim_length
            us for each phase of the square wave
    """
    
    from multiprocessing import shared_memory

    stim_shm = shared_memory.SharedMemory(name=stim_shm_name)
    stim_array = np.ndarray((len(stim_units),), dtype=bool, buffer=stim_shm.buf)

    mea_stim = MaxwellStim(stim_units)
    stim_process_ready.set()
    
    while not env_done.is_set():
        if not stim_ready.is_set():
            continue
        
        stim_ind = np.flatnonzero(stim_array)
        stim_array[:] = False
        
        print(stim_ind)
        
        # Stim MEA
        # start = perf_counter()
        # print(f"Stimulating at {env.time_elapsed():.4f}s")     
        stim_action = (stim_ind, stim_amp, stim_length)
        mea_stim._create_stim_pulse(stim_action)
        mea_stim.seq.send()
        # stim_actions.append((stim_action, 'test'))
        # env._log_stimulation(stim_action, tag="test")
        # env.last_stim_time = env.cur_time
        # env.last_stim_times[[0]] = env.cur_time
        # end = perf_counter()
        # print((end-start)*1000)
        # process_stim_times.append((end - start) * 1000)
        
        stim_ready.clear()

    stim_shm.close()



#==========================================================================
#=========================  MAXWELL FUNCTIONS  ============================
#==========================================================================

def init_maxone(config, stim_electrodes,filt=True, verbose=1, gain=512, cutoff='1Hz',
                spike_thresh=5, dummy=False, skip_offset=False):
    """
    Initialize MaxOne, set electrodes, and setup subscribers

    Parameters
    ----------
    config : str
        Path to the config file for the electrodes

    stim_electrodes : list  
        List of electrode numbers to stimulate

    filt : bool
        Whether to use the high-pass filter

    verbose : int   
        0: No print statements
        1: Print initialization statements
        2: Print all statements

    gain : int, {512, 1024, 2048}   
        Gain of the amplifier

    cutoff : str, {'1Hz', '300Hz'}  
        Cutoff frequency of the high-pass filter

    spike_thresh : int  
        Threshold for spike detection, in units of standard deviations
    """

    init_maxone_settings(gain=gain, cutoff=cutoff, spike_thresh=spike_thresh, verbose=verbose)
    subscriber = setup_subscribers(filt=filt, verbose=verbose)
    stim_units, stim_electrodes_dict = select_electrodes(config,
                                                        stim_electrodes, verbose=verbose,
                                                        dummy=dummy, skip_offset=skip_offset)
    ignore_first_packet(subscriber)
    ignore_remaining_packets(subscriber)
    return subscriber, stim_units, stim_electrodes_dict
        
        



def init_maxone_settings(gain=512,amp_gain=512,cutoff='1Hz', spike_thresh=5, verbose=1):
    """
    Initialize MaxOne and set gain and high-pass filter

    Parameters
    ----------
    gain : int, {512, 1024, 2048}

    amp_gain : int

    cutoff : str, {'1Hz', '300Hz'}
    """
    if verbose >= 1:
        print("Initializing MaxOne")

    maxlab.util.initialize()
    maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))
    maxlab.send(maxlab.chip.Amplifier().set_gain(amp_gain))
    
    # if cutoff == '1Hz':
    #     maxlab.send_raw("system_hpf " + str(4095))
    # elif cutoff == '300Hz':
    #     maxlab.send_raw("system_hpf " + str(1100))
    # maxlab.util.set_gain(gain)

    maxlab.send_raw(f"stream_set_event_threshold {spike_thresh}")
    # maxlab.util.set_gain(gain)
    # maxlab.util.hpf(cutoff)
    if verbose >=1:
        print('MaxOne initialized')

        



def setup_subscribers(filt, verbose=1):
    """
    Setup subscribers for events from MaxOne, this 
    allows us to read the data from the chip.
    """
    if verbose >= 1:
        print("Setting up subscribers")
    subscriber = zmq.Context.instance().socket(zmq.SUB)
    subscriber.setsockopt(zmq.RCVHWM, 100)
    subscriber.setsockopt(zmq.RCVBUF, 10*20000*1030)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
    subscriber.setsockopt(zmq.RCVTIMEO, 100)
    if filt:
        subscriber.connect('tcp://localhost:7205')
    else:
        subscriber.connect('tcp://localhost:7204')
    return subscriber


def select_electrodes(config, stim_electrodes, verbose=1, dummy=False, skip_offset=False):
    # Electrode selection logic
    array = maxlab.chip.Array('stimulation')
    array.reset() #delete previous array

    array.load_config(config)

    if verbose >= 1:
        print(f'Recording electrodes initialized from: {config}')

    array.select_stimulation_electrodes(stim_electrodes)

    if len(stim_electrodes) > 32:
        raise Exception('Too many stimulation electrodes.')

    stim_units = []
    stim_electrodes_dict = {}
    unit_ids = []
    for e in stim_electrodes:
        array.connect_electrode_to_stimulation(e)
        unit_id = array.query_stimulation_at_electrode(e)
        
        if not unit_id:
            # print(f"Error: electrode: {e} cannot be stimulated")
            RuntimeError(f"Error: electrode: {e} cannot be stimulated")

        if unit_id in unit_ids:
            RuntimeError(f"Error: electrode: {e} has a stim buffer conflict for buffer: {unit_id}")
        
        unit_ids.append(unit_id)

        unit = maxlab.chip.StimulationUnit(unit_id)

        stim_units.append(unit)
        stim_electrodes_dict[unit] = e

        if verbose >= 2:
            print(f'Connected Electrode # {e}')
    array.download()

    if verbose >= 1:
        print(f'Electrodes selected for stimulation: {stim_electrodes}')

    if not skip_offset:
        power_cycle_stim_electrodes(stim_units)
        if not dummy:
            # Let stimulation units settle after power cycling before recalculating offsets.
            sleep_with_progress(15, "Settling before offset", verbose=verbose)
        maxlab.util.offset()
        if verbose:
            print("Offseting")
    else:
        if verbose:
            print("Skipping power cycle and offset")
    return stim_units, stim_electrodes_dict




def power_cycle_stim_electrodes(stim_units):
    ''' "Power up and down again all the stimulation units.
    It appears this is needed to equilibrate the electrodes"
    - from maxwell code'''

    seq = maxlab.Sequence()
    for unit in stim_units:
        seq.append(
                unit.power_up(True).connect(True)
                    .set_voltage_mode().dac_source(0))
    for unit in stim_units:
        seq.append(unit.power_up(False).connect(False))
    seq.send()
    print('Power cycled')
    del seq
    seq = maxlab.Sequence()
    for unit in stim_units:
        seq.append(
                unit.power_up(True).connect(True))
    seq.send()


def disconnect_stim_electrodes(stim_units):
    """Disconnect each stimulation unit in the list."""
    seq = maxlab.Sequence()
    for unit in stim_units:
        seq.append(unit.power_up(False).connect(False))
    seq.send()

def connect_stim_electrodes(stim_units):
    """Connect each stimulation unit in the list."""
    seq = maxlab.Sequence()
    for unit in stim_units:
        seq.append(unit.power_up(True).connect(True))
    seq.send()


def _unpack_spike_event(name, fmt, event_data):
    unpacked = struct.unpack(fmt, event_data)
    if name == 'well_id':
        frame, amplitude, channel, _well_id = unpacked
    else:
        frame, channel, amplitude = unpacked

    return SpikeEvent(frame, channel, amplitude)


def _score_spike_events(events):
    score = 0
    for ev in events:
        if ev.frame >= 0:
            score += 1
        if 0 <= int(ev.channel) < 4096:
            score += 1
        if np.isfinite(ev.amplitude) and abs(ev.amplitude) < 10000:
            score += 1
    return score


def _detect_spike_struct_format(events_data):
    global _spike_struct_format

    if (_spike_struct_format is not None
            and len(events_data) % _spike_struct_format[2] == 0):
        return _spike_struct_format

    candidates = [
        (name, fmt, size)
        for name, fmt, size in _spike_struct_formats
        if len(events_data) % size == 0
    ]

    if not candidates:
        expected_sizes = ', '.join(str(size) for _, _, size in _spike_struct_formats)
        print(f'Events has {len(events_data)} bytes,',
            f'not divisible by expected spike sizes ({expected_sizes})', file=sys.stderr)
        return None

    best_format = None
    best_score = -1
    for name, fmt, size in candidates:
        try:
            parsed_events = [
                _unpack_spike_event(name, fmt, events_data[i:i+size])
                for i in range(0, len(events_data), size)
            ]
        except (struct.error, ValueError, OverflowError):
            continue

        score = _score_spike_events(parsed_events)
        if score > best_score:
            best_format = (name, fmt, size)
            best_score = score

    _spike_struct_format = best_format
    return _spike_struct_format


def get_spike_struct_size(events_data=None):
    spike_format = None if events_data is None else _detect_spike_struct_format(events_data)
    return _spike_struct_size if spike_format is None else spike_format[2]


def parse_events_list(events_data):
    '''
    Parse the raw binary events data into a list of SpikeEvent objects.
    '''
    events = []

    if events_data is not None:
        spike_format = _detect_spike_struct_format(events_data)
        if spike_format is None:
            return events

        name, fmt, size = spike_format
        if len(events_data) % size != 0:
            print(f'Events has {len(events_data)} bytes,',
                f'not divisible by detected spike size {size}', file=sys.stderr)
            return events

        for i in range(0, len(events_data), size):
            events.append(_unpack_spike_event(name, fmt, events_data[i:i+size]))

    return events
    
    
def parse_frame(frame_data):
    '''
    Parse the binary frame data into an array of floating-point voltages.
    '''
    return None if frame_data is None else array.array('f',frame_data)


def receive_packet(subscriber, buffer_size=None):
    '''
    Use the subscriber to capture the frame and event data from the server.
    Returns an integer frame_number as well as data buffers for the voltage
    data frame and the spike events. Also sets the current time.
    '''
    frame_number = frame_data = events_data = None

    if buffer_size is not None:
        frame_nums = []
        frames = []
        events = []
        for i in range(buffer_size):
            try:
                # The first component of each message is the frame number as
                # a long long, so unpack that.
                frame_number = struct.unpack('Q', subscriber.recv())[0]
                frame_nums.append(frame_number)

                # We're still ignoring the frame data, but we have to get it
                # out from the data stream in order to skip it.
                if subscriber.getsockopt(zmq.RCVMORE):
                    frame_data = subscriber.recv()
                    frames.append(frame_data)

                # This is the one that stores all the spikes.
                if subscriber.getsockopt(zmq.RCVMORE):
                    events_data = subscriber.recv()
                    events.append(events_data)

            except zmq.Again:
                pass
            except Exception as e:
                print(e)
            
        return frame_nums, frames, events
            
    # Sometimes the publisher will be interrupted, so fail cleanly by
    # terminating this run of the environment, returning done = True.
    try:
        # The first component of each message is the frame number as
        # a long long, so unpack that.
        frame_number = struct.unpack('Q', subscriber.recv())[0]

        # We're still ignoring the frame data, but we have to get it
        # out from the data stream in order to skip it.
        if subscriber.getsockopt(zmq.RCVMORE):
            frame_data = subscriber.recv()

        # This is the one that stores all the spikes.
        if subscriber.getsockopt(zmq.RCVMORE):
            events_data = subscriber.recv()

    except zmq.Again:
        pass
    except Exception as e:
        print("Error receiving packet:", e)

    return frame_number, frame_data, events_data


def socket_worker(data_queue, event_queue, subscriber_args):
    """Worker function that reads from the ZeroMQ socket."""
    subscriber = setup_subscribers(*subscriber_args)
    while True:
        frame_number, frame_data, events_data = receive_packet(subscriber)
        
        if events_data is not None:
            event_queue.put(parse_events_list(events_data))
        if frame_data is not None:
            data_queue.put(parse_frame(frame_data))


def plot_worker(queue):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib
    matplotlib.use('TkAgg')
    # Set up plot
    fig, ax = plt.subplots()
    cur_data = np.zeros((2,20000))# 1s
    line, = ax.plot(cur_data[0,:])  # Adjust as needed
    ax.set_ylim([-1,1])

    # Animation update function
    def update(i):
        nonlocal cur_data
        for i in range(2000): #100ms
            if not queue.empty():
                data = np.array(queue.get())
                cur_data = np.roll(cur_data, -1, axis=1)
                cur_data[:2,-1] = data[:2]
            # filtered_data = butter_bandpass_filter(data, lowcut=1, highcut=50, fs=20000)  # Filter parameters to be adjusted
        line.set_ydata(cur_data[0,:])
        # ax.relim()  # Recalculate limits
        # ax.autoscale_view(True, True, True)  # Autoscale the plot
        ax.set_title(f'Frame {i}')
        return line,

    # Create animation
    ani = animation.FuncAnimation(fig, update, interval=20, blit=True)

    # Show plot
    plt.show()



def ignore_first_packet(subscriber, verbose=1):
    '''
    This first loop ignores any partial packets to make sure the real
    loop gets aligned to an actual frame. First it spins for as long
    as recv() fails, then it waits for the RCVMORE flag to be False to
    check that the last partial frame is over.
    '''
    more = True
    t = time.perf_counter()
    while more:
        try:
            _ = subscriber.recv()
        except zmq.ZMQError:
            if time.perf_counter() - t >= 3:
                raise TimeoutError("Make sure the Maxwell Server is on.")
            continue
        more = subscriber.getsockopt(zmq.RCVMORE)

    if verbose >=1:
        print('Successfully ignored first packet')



def ignore_remaining_packets(subscriber, verbose=1):
    '''
    This function reads and discards all remaining data in the buffer. It uses
    a non-blocking recv() to ensure that it only reads available data and
    stops when there are no more packets.
    '''
    while True:
        try:
            # Using NOBLOCK to ensure non-blocking operation
            _ = subscriber.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            # zmq.Again is raised when there's no more data to read
            break

    if verbose >= 1:
        print('Successfully ignored remaining packets')


def launch_dummy_server(dummy):
    print("====================================")
    print("Using dummy data: \n\t", dummy)
    print("====================================")
    
    import braindance.core.dummy_zmq_np
    import atexit
    from multiprocessing import Process
    
    # Directly use the run function from the module
    dummy_process = Process(target=braindance.core.dummy_zmq_np.run, args=(dummy,))
    dummy_process.start()
    atexit.register(dummy_process.terminate)

    # Rest of the function remains the same
    ready = False
    subscriber = setup_subscribers(False, verbose=False)
    while not ready:
        try:
            ignore_first_packet(subscriber)
        except TimeoutError:
            print("Dummy server not ready. Waiting 5 seconds before trying again")
            time.sleep(5)
            continue
        ready = True
    print("Dummy server ready")



class Config:
    def __init__(self, filename):

        self.config = []
        self.mappings = []
        

        if filename is None:
            print('No config file specified')
            return
        with open(filename, 'r') as file:
            txt = file.read()
        self.config = [m.replace('(', ' ').replace(')', ' ').replace('/', ' ').split() for m in txt.split()[0].split(';')[:-1]]
        self.mappings = [self.Mapping(*m) for m in self.config]
        self.config = [(int(m[0]), int(m[1]), float(m[2]), float(m[3])) for m in self.config]

    def get_channels(self):
        return [m.channel for m in self.mappings]
    
    def get_electrodes(self):
        return [m.electrode for m in self.mappings]

    def get_channels_for_electrodes(self, electrodes):
        return [m.channel for m in self.mappings if m.electrode in electrodes]
    
    def get_num_channels(self):
        return len(self.get_channels())

    class Mapping:
        def __init__(self, channel, electrode, x, y):
            self.channel = int(channel)
            self.electrode = int(electrode)
            self.x = float(x)
            self.y = float(y)
