'''
Experiment phases
-----------------
Phases act as parts of experiments which have a specific purpose, such as
    - spontaneous recording
    - amplitude sweep
        --- over a neuron to find the stimulus amplitude that elicits a spike
    - frequency stimulation 
        ---to stimulate one or more neurons at a certain frequency
    - custom
        ---where users can build exact logic for stimulation
'''
import braindance.core.base_env as base_env
from braindance.core.artifact_removal import ArtifactRemoval

import numpy as np
import time
import csv
from collections import deque



class Phase:
    '''
    Base class for all phases
    '''
    def __init__(self, env: base_env.BaseEnv):
        self.env = env
        self.start_time = time.perf_counter()

    def run(self):
        raise NotImplementedError
    
    def time_elapsed(self):
        return time.perf_counter() - self.start_time
    
    def predicted_time(self):
        '''Returns the predicted time for the phase to run in seconds'''
        raise NotImplementedError
    
    def info(self):
        '''Returns a dictionary of information about the phase'''
        raise NotImplementedError


class PhaseManager:
    '''
    Manages phases of an experiment
    '''
    def __init__(self, env: base_env.BaseEnv, verbose=False):
        
        self.env = env
        self.phases = []
        self.filenames = []
        self.verbose = verbose
        self.analysis_dao = None

        if self.env:
            self.save_dir = self.env.save_dir

    def add_phase(self, phase: Phase):
        self.phases.append(phase)

    def add_phase_group(self, phase_group: list):
        '''
        Adds a group of phases to the manager, 
        each group will belong to the same save file
        '''
        self.phases.append(phase_group)

    def log_summary(self):
        '''Logs the summary of the experiment to a text file'''
        summary_str = self.summary()
        with open(self.save_dir + '/summary.txt', 'a') as f:
            f.write(summary_str)
            f.write("\n\n\n")
            f.write("="*20)
            
    def log_phase(self, phase):
        '''Appends the phase and filename to the log file'''
        with open(self.save_dir + '/phase_log.csv', 'a') as f:
            writer = csv.writer(f)
            if isinstance(phase, list):
                i = 0
                for sub_phase in phase:
                    writer.writerow([sub_phase.__class__.__name__, self.filenames[-i]])
                    i += 1
            else:
                writer.writerow([phase.__class__.__name__,
                                    self.filenames[-1]])

    def run(self):
        # from braindance.phases_analysis import AnalysisPhase, HeatmapPhase

        try:
            cur_filename = self.env.save_file
            self.filenames.append(cur_filename)

            self.log_summary()


            for phase in self.phases:

                # Reset environment only if it not an analysis phase
                if phase != self.phases[0]: #and not isinstance(phase, AnalysisPhase):
                    self.env.reset()
                    cur_filename = self.env.save_file
                    self.filenames.append(cur_filename)

                if self.verbose:
                    print("~"*20)
                    print("Save file:", self.env.save_file)
                    if isinstance(phase, list):
                        print("Running phase group of:", end=" ")
                        for sub_phase in phase:
                            print(sub_phase.__class__.__name__, end=" ")
                        print()
                    else:
                        print("Running phase:", phase.__class__.__name__)
                    print("~"*20)

                # Run the phase
                if isinstance(phase, list):
                    for sub_phase in phase:
                        print("Running sub phase:", sub_phase.__class__.__name__)
                        print("="*20)
                        sub_phase.run()
                elif isinstance(phase, AnalysisPhase):
                    # Make sure to close/save any previous phases
                    # phase.run(cur_filename)
                    # For testing
                    if isinstance(phase, HeatmapPhase):
                        self.analysis_dao = phase.run(self.analysis_dao, cur_filename)
                    else:
                        self.analysis_dao = phase.run(self.analysis_dao)
                elif isinstance(phase, Phase):
                    phase.run()
                self.env.close()
                self.log_phase(phase)
                

                

        except Exception as e:
            self.env.close()
            raise e
        finally:
            self.env.close()


    def summary(self):
        '''Returns a summary of the experiment as a string'''
        def summarize_phase(phase):
            summary_str = phase.__class__.__name__ + "\n"
            predicted_time = phase.predicted_time
            if predicted_time > 60:
                summary_str += "\tPredicted Time: {:.0f}m {:.0f}s\n".format(predicted_time // 60, predicted_time % 60)
            else:
                summary_str += "\tPredicted Time: {:.1f} seconds\n".format(predicted_time)
            
            try:
                phase_info = phase.info()
                for key in phase_info:
                    summary_str += "\t\t" + key + " : " + str(phase_info[key]) + "\n"
                summary_str += "\n"
            except NotImplementedError:
                summary_str += "\t\tNo info available\n"
            return summary_str

        total_time = 0
        summary_str = "Phase Summary\n-------------\n"
        for phase in self.phases:
            if isinstance(phase, list):
                summary_str += "Phase Group\n"
                group_time = 0
                for sub_phase in phase:
                    summary_str += summarize_phase(sub_phase)
                    group_time += sub_phase.predicted_time
                summary_str += "Group Total Time: {:.0f}m {:.0f}s\n\n".format(group_time // 60, group_time % 60)
                total_time += group_time
            else:
                summary_str += summarize_phase(phase)
                total_time += phase.predicted_time
        
        summary_str += "Total Experiment Time: {:.0f}m {:.0f}s\n".format(total_time // 60, total_time % 60)
        summary_str += "-------------"
        return summary_str


class RecordPhase(Phase):
    '''
    Phase for recording spontaneous activity

    Parameters
    ----------
    env : base_env.BaseEnv
        The environment to run the phase in
    duration : int
        The duration of the recording in seconds, by default 10
    '''
    def __init__(self, env: base_env.BaseEnv = None, duration: int = 10, ):
        self.duration = duration
        self.predicted_time = duration
        super().__init__(env)

    def run(self, env=None):
        self.start_time = time.perf_counter()

        
        
        done = False
        while not done:
            obs, done = self.env.step()
            if self.time_elapsed() > self.duration:
                done = True

    def info(self):
        return {
            'duration': self.duration
        }

    
class NeuralSweepPhase(Phase):
    '''
    Sweep the amplitude of a stimulation to find the minimum amplitude that
    elicits a spike

    Parameters
    ----------
    env : base_env.BaseEnv
        The environment to run the phase in
    neuron_list : list
        The neurons to stimulate
    amp_bounds : tuple
        The bounds of the amplitude sweep
        The step size defaults to 10% of the range, but 
        if a third element is provided, it will be used as the number of steps 
        (start, end, n_step)
    stim_freq : float, optional
        The frequency of the stimulation, by default 1
    replicates : int
        The number of times to repeat the stimulation, by default 30
    phase_length : int, optional
        The length of one of the phases in the stimulation pulse, by default 100
    type : str, optional
        The type of amplitude sweep to perform, by default 'ran', determined by the 
        order of the following characters
            - 'r': Iterate through replicates
            - 'a': Iterate through the amplitudes
            - 's': Iterate through the neurons
        If you put 'random', it will randomly iterate through everything
        Options:
            - 'ran': First iterate through the replicates, then the amplitudes, then the neurons
                --- ex: (r1, a1, n1), (r2, a1, n1),...,(r1, a2, n1), (r2, a2, n1),... 
            - 'rna': First iterate through the replicates, then the neurons, then the amplitudes
                --- ex: (r1, a1, n1), (r2, a1, n1),...,(r1, a1, n2), (r2, a1, n2),...
            - 'arn': First iterate through the amplitudes, then the replicates, then the neurons
                --- ex: (r1, a1, n1), (r1, a2, n1),...,(r2, a1, n1), (r2, a2, n1),...
            - etc.
    '''
    def __init__(self, env: base_env.BaseEnv, neuron_list: list, amp_bounds = [150,150,1], stim_freq:float = 1,
                replicates = 30, phase_length: int = 100, order='ran', single_connect = False,
                verbose=False, tag='neural_sweep'):
    
        assert len(neuron_list) > 0, "Must have at least one neuron to stimulate"
        assert order[0] in ['r', 'a', 'n'], "First character of type must be 'r', 'a', or 'n'"
        assert order[1] in ['r', 'a', 'n'], "Second character of type must be 'r', 'a', or 'n'"
        assert order[2] in ['r', 'a', 'n'], "Third character of type must be 'r', 'a', or 'n'"
        self.neuron_list = neuron_list

        # If int, we only have one value
        if isinstance(amp_bounds, int):
            amp_bounds = [amp_bounds, amp_bounds, 1]
        assert len(amp_bounds) >= 2, "Amplitude bounds must have at least a start and end value"
        assert amp_bounds[0] <= amp_bounds[1], "Amplitude bounds must be same or increasing"

        self.amplitude_start = amp_bounds[0]
        self.amplitude_end = amp_bounds[1]
        if amp_bounds[2]:
            self.n_amplitudes = amp_bounds[2]
        else:
            self.n_amplitudes = int((self.amplitude_end - self.amplitude_start) / 10)

        self.stim_freq = stim_freq
        self.phase_length = phase_length
        self.replicates = replicates

        self.single_connect = single_connect
        if self.single_connect and (order[2] != 'n' or order[1] != 'n') or order == 'random':
            print("Warning: single_connect should only be used with *n* or **n in the order")
        
        self.order = order
        self.tag = tag
        self.verbose = verbose

        self.predicted_time = self.n_amplitudes * len(neuron_list) * 1/stim_freq * replicates

        super().__init__(env)


    def generate_stim_commands(self):
        '''
        Generates the stimulation commands for the amplitude sweep
        '''
        amplitudes = np.linspace(self.amplitude_start, self.amplitude_end, self.n_amplitudes)

        stim_commands = []
        if self.order[2] == 'r':
            for r in range(self.replicates):
                if self.order[1] == 'a':
                    for a in amplitudes:
                        for n in self.neuron_list:
                            stim_commands.append(([n], a, self.phase_length))

                elif self.order[1] == 'n':
                    for n in self.neuron_list:
                        for a in amplitudes:
                            stim_commands.append(([n], a, self.phase_length))

        elif self.order[2] == 'a':
            for a in amplitudes:
                if self.order[1] == 'r':
                    for r in range(self.replicates):
                        for n in self.neuron_list:
                            stim_commands.append(([n], a, self.phase_length))

                elif self.order[1] == 'n':
                    for n in self.neuron_list:
                        for r in range(self.replicates):
                            stim_commands.append(([n], a, self.phase_length))

        elif self.order[2] == 'n':
            for n in self.neuron_list:
                if self.order[1] == 'r':
                    for r in range(self.replicates):
                        for a in amplitudes:
                            stim_commands.append(([n], a, self.phase_length))

                elif self.order[1] == 'a':
                    for a in amplitudes:
                        for r in range(self.replicates):
                            stim_commands.append(([n], a, self.phase_length))

        if self.order == 'random':
            np.random.shuffle(stim_commands)
                
        return stim_commands


    def run(self):
        done = False

        time_between_stims = 1 / self.stim_freq
        stim_count = 0

        stim_commands = self.generate_stim_commands()

        # Set as the first
        self.last_neuron = stim_commands[0][0][0]

        self.start_time = time.perf_counter()

        if self.single_connect:
            self.env.disconnect_all()
            # Connect the new neuron
            self.env.connect_units([self.env.stim_units[self.last_neuron]])

        

        while not done:
            if self.time_elapsed() > time_between_stims * stim_count:
                
                if len(stim_commands) == 0:
                    done = True
                    break

                if self.verbose:
                    print("Stimulating neuron", stim_commands[0][0], "at amplitude", stim_commands[0][1],
                          "at time: {:.3f}".format(self.time_elapsed()))
                
                stim_command = stim_commands.pop(0)

                # If we are only connecting one stimulation electrode at a time,
                # we need to disconnect the previous one
                if self.single_connect and stim_command[0][0] != self.last_neuron:
                    # Disconnect all connected channels
                    self.last_neuron = stim_command[0][0]
                    self.env.disconnect_all()
                    # Connect the new neuron
                    self.env.connect_units([self.env.stim_units[self.last_neuron]])

                self.env.step(action=stim_command, tag=self.tag)
                stim_count += 1
            else:
                self.env.step()

    def info(self):
        return {
            'neuron_list': self.neuron_list,
            'amplitude_start': self.amplitude_start,
            'amplitude_end': self.amplitude_end,
            'n_amplitudes': self.n_amplitudes,
            'stim_freq': self.stim_freq,
            'replicates': self.replicates,
            'phase_length': self.phase_length,
            'type': self.order,
            'tag': self.tag

        }


class FrequencyStimPhase(Phase):
    '''
    Phase for stimulating a command at a certain frequency

    Parameters
    ----------
    env : base_env.BaseEnv
        The environment to run the phase in
    stim_command : stim command or list of stim commands
        If a single stim command, it will be repeated at the given frequency
        If a list of stim commands, each stim command will be run sequentially
    stim_freq : float, optional
        The frequency of the stimulation, by default 1
    duration : int, optional
        The duration of the stimulation in seconds, by default 10
    tag : str, list of str, optional
        The tag to use for the stimulation, by default 'frequency_stim'
        If a list of strings, each tag will be used for the corresponding stim command
            --- Must be the same length as stim_command
    verbose : bool, optional
        Whether to print out information about the stimulation, by default False
    connect_units : list, optional
        The units to connect to the stimulation electrodes, by default None, leaves as is
        If a list is input, the stimulation units of the corresponding indexes in the environmen
        will be connected.
    '''
    def __init__(self, env: base_env.BaseEnv, stim_command: list, stim_freq: float = 1, duration: int = 10,
                tag = 'frequency_stim', verbose=False, connect_units = None):
        self.stim_freq = stim_freq
        self.stim_command = stim_command
        # Check if stim_command is a single stim command or a list of stim commands
        if type(stim_command[0][0]) == list:
            self.single_command = False
        else:
            self.single_command = True
        
        self.duration = duration
        self.predicted_time = min(duration, len(stim_command) / stim_freq)
        
        self.tag = tag
        self.single_tag = type(self.tag) == str
        if not self.single_tag and self.single_command:
            raise ValueError("Tag must be a string if stim_command is a single command")
        # Or if not single commands, tag must be string or list of strings the same length as stim_command
        if not self.single_tag and not self.single_command and len(self.tag) != len(self.stim_command):
            raise ValueError("Tag must be a string or list of strings the same length as stim_command")
        
        self.start_time = time.perf_counter()
        self.verbose = verbose
        self.connect_units = connect_units
        super().__init__(env)

    def run(self):
        if self.connect_units:
            self.env.disconnect_all()
            # Connect the new neuron
            self.env.connect_units(inds=self.connect_units)
            if self.verbose:
                print("Connecting stim units:", self.connect_units)
            time.sleep(3)

        self.start_time = time.perf_counter()
        done = False
        time_between_stims = 1 / self.stim_freq
        stim_count = 0
        while not done:
            if self.time_elapsed() > time_between_stims * stim_count:
                
                if not self.single_command:
                    # Pop
                    stim_command = self.stim_command.pop(0)
                else:
                    stim_command = self.stim_command
                if not self.single_tag:
                    tag = self.tag.pop(0)
                else:
                    tag = self.tag
                
                self.env.step(action=stim_command, tag=tag)

                if self.verbose:
                    print("Stimulating neuron(s) {} at time: {:.3f}".format(stim_command[0],self.time_elapsed()))


                stim_count += 1
            else:
                self.env.step()
            if self.time_elapsed() > self.duration:
                done = True

            # If no m
            if not self.single_command and len(self.stim_command) == 0:
                done = True

    def info(self):
        return {
            'stim_command': self.stim_command,
            'stim_freq': self.stim_freq,
            'duration': self.duration,
            'tag': self.tag
        }

