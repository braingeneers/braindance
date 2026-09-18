'''
Experiment phases
-----------------
Phases act as parts of experiments which have a specific purpose, such as
    - sponatneous recording
    - amplitude sweep
        --- over a neuron to find the stimulus amplitude that elicits a spike
    - frequency stimulation 
        ---to stimulate one or more neurons at a certain frequency
    - preset reactive
        ---where stimulations will respond to certain defined activity
    - custom reactive
        ---where users can build exact logic for stimulation
'''
import braindance.core.base_env as base_env

import numpy as np
import time
import csv
from braindance.core.artifact_removal import ArtifactRemoval
from collections import deque



class Phase:
    '''
    Base class for all phases
    '''
    def __init__(self, env: base_env.BaseEnv):
        self.env = env
        self.start_time = time.perf_counter()
        self.requires = []  # List of data requirements
        self.provides = []  # List of data this phase provides

    def run(self):
        raise NotImplementedError
    
    def validate(self, experiment = None):
        '''Validates the phase to make sure it can run'''
        return True
    
    def time_elapsed(self):
        return time.perf_counter() - self.start_time
    
    def predicted_time(self):
        '''Returns the predicted time for the phase to run in seconds'''
        raise NotImplementedError
    
    def info(self):
        '''Returns a dictionary of information about the phase'''
        raise NotImplementedError
    
    def cleanup(self):
        '''Cleans up the phase after it is done'''
        pass


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
            try:
                writer = csv.writer(f)
                if isinstance(phase, list):
                    i = 0
                    for sub_phase in phase:
                        writer.writerow([sub_phase.__class__.__name__, self.filenames[-1]])
                        i += 1
                else:
                    writer.writerow([phase.__class__.__name__,
                                        self.filenames[-1]])
            except Exception as e:
                print("Error logging phase:", e)
                print("Phase:", phase.__class__.__name__)

    def run(self):
        from .phases_analysis import AnalysisPhase, HeatmapPhase

        try:
            cur_filename = self.env.save_file
            self.filenames.append(cur_filename)

            self.log_summary()


            for phase in self.phases:

                # Reset environment only if it not an analysis phase
                if phase != self.phases[0] and not isinstance(phase, AnalysisPhase):
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
        self.predicted_time = duration
        
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


class CartPolePhase(Phase):
    '''
    Phase for the cartpole game
    
    Parameters
    ----------
    env : base_env.BaseEnv
        The environment to run the phase in
    sensory_neurons = list of sensory neurons respond to stimulation 
            (this is in indices of stim electrodes)
    motor_neurons = list of motor neurons to read out from 
            (this is in channels)
    training_neurons = list of training neurons to stimulate
            (this is in indices of stim electrodes)
    read_period_ms = period of time to read out from the motor neurons
    training_period_ms = period of time to stimulate the training neurons
    n_episodes = number of episodes to run the game for
    trainer = Trainer object to use for training
    artifact_removal = whether to use artifact removal
    continuous = whether to use continuous vs discrete estimations of motor signals from
            the read neurons
    assistive = how much to assist the game, 0 is no assistance, 1 is full assistance
            this is equivalent to the amount of stimulation to the sensory neurons
    minibatch_size = size of the myinibatch for the trainer, effectively should account for how
            many reward steps we care about after a training stimulus
    spike_thresh = thresholds for spike detection
            list of two values, first is the min value for a spike, second is the max value
    verbose = whether to print out information about the phase
    max_time = maximum time to run the phase for
    normalization = value to normalize spike difference by (2 would be spike_diff/2)

    '''
    
    def __init__(self, env: base_env.BaseEnv, 
                 sensory_neurons:list, motor_neurons:list, training_neurons:list,
                 amp_mv:int = 400, phase_width:int = 100,
                 read_period_ms:int = 200,
                 train_period_ms:int = 200, 
                 wait_period_ms:int = 400,
                 n_episodes:int = 10,
                 trainer=None, artifact_removal=False,
                 continuous=True, assistive=0.0,
                 minibatch_size=5, spike_thresh=[-3.1,-20],
                 verbose=False, max_time = np.inf,
                 normalization = 1,
                 force_train = False
                 ):
        
        import gymnasium as gym
        from braindance.games import cartpole_continuous

        self.continuous = continuous

        if continuous:
            self.game_env = cartpole_continuous.CartPoleContinuousEnv(render_mode='human')
            pass
        else:
            self.game_env = gym.make('CartPole-v1', render_mode='human')
        
        self.env = env
        self.read_period_ms = read_period_ms
        self.train_period_ms = train_period_ms
        self.wait_period_ms = wait_period_ms
        self.n_episodes = n_episodes
        self.episode = 0
        self.max_time = max_time
        self.minibatch_size = minibatch_size
        self.force_train = force_train
        self.normalization = normalization

        # IO
        self.sensory_neurons = np.array(sensory_neurons)
        self.motor_neurons = np.array(motor_neurons)
        self.training_neurons = np.array(training_neurons)

        self.amp_mv, self.phase_width = amp_mv, phase_width

        # Artifact removal
        # Ensure env.observation_type is 'raw'
        if artifact_removal:
            self.artifact_remover = []
            for ch in self.motor_neurons:
                # self.artifact_remover.append(ArtifactRemoval(N=20, nc_start=20, min_val=-13, max_val=13,
                #                                             spike_thresh = [-3.2,-15]))
                self.artifact_remover.append(ArtifactRemoval(N=20, nc_start=20, min_val=-25, max_val=25,
                                                            spike_thresh = spike_thresh))
        else:
            self.artifact_remover = None
        
        self.sensory_stim_Hz = np.zeros(len(self.sensory_neurons))
        self.trainer = trainer
        if self.trainer:
            self.trainer.phase = self
            
        self.last_action_ind = None
        self.last_action_inds = deque(maxlen=self.minibatch_size)
        self.game_obs = None
        self.episode_reward = None
        self.episode_reward_change = None

        self.assistive = assistive

        self.verbose = verbose



        self.motor_spike_count = np.zeros(len(self.motor_neurons)) # Stores the spike count for each motor neuron
        self.motor_spike_rate = np.zeros(len(self.motor_neurons))
        self.state = 'read' # 'read', 'train', 'game', or 'wait'

        self.predicted_time = (read_period_ms + train_period_ms)*20 * n_episodes / 1000 # x20 due to 20 steps of the game avg



        super().__init__(env)

    
    def set_sensory_signal(self, game_env_obs):
        '''
        Maps the game observation to the sensory neurons.
        This observation is of the form:
        ndarray with shape (4,): 
            - Cart Position, Cart Velocity, Pole Angle, Pole Velocity At Tip

        We will map the pole angle to the 0th and 1st sensory neurons

        '''
        pole_angle = game_env_obs[2]
        stim_bias = .15#.15#.2 # was .4, .1
        stim_scale = 7#9#15#8# was 7,  20

        self.sensory_stim_Hz = np.zeros(len(self.sensory_neurons))

        self.sensory_stim_Hz[0] = ((-np.sin(pole_angle) + stim_bias)*stim_scale)**2
        self.sensory_stim_Hz[1] = ((np.sin(pole_angle) + stim_bias)*stim_scale)**2
        if self.verbose:
            print(f'Stim:Hz: \t{self.sensory_stim_Hz}')



    def get_motor_signal(self, moving_avg=.2):
        '''
        Readout from the read/motor neurons set
        The action returned should be an integer {0,1} corresponding to the action

        Recommended to use self.spike_count, which is a list of the spike counts
        for each motor neuron 

        '''
        # if moving_avg:
        self.motor_spike_rate = moving_avg*self.motor_spike_rate + (1-moving_avg)*self.motor_spike_count
        
            
        # spike_diff = self.motor_spike_count[0] - self.motor_spike_count[1]
        
        if type(self.normalization) == np.ndarray:
            # m0 mean, m1 mean, m0 std, m1 std
            # spike_diff /= self.normalization
            spike_diff = (self.motor_spike_rate[0]/self.normalization[0]) - (self.motor_spike_rate[1]/self.normalization[1])
            spike_diff *= 10
        else:
            spike_diff = self.motor_spike_rate[0] - self.motor_spike_rate[1]
            spike_diff /= self.normalization


        if self.assistive:
            assistive_factor = self.assistive*(self.sensory_stim_Hz[0] - self.sensory_stim_Hz[1])
            print(f'Assistive factor: {assistive_factor:.2f} \t Spike diff: {spike_diff:.2f}')
            
            spike_diff += assistive_factor

        
        if self.continuous:
            print('[Action]: :',-spike_diff)
            return max(min(-spike_diff,1),-1)
        if spike_diff > 0:
            action = 0
            print('Going left')
        else:
            action = 1
            print('Going right')

        return action
        

    def get_training_signal(self, train_Hz=30, stim_count=3, delay_ms = 5, use_change=True):
        '''
        The stimulation on the training neurons 
        
        '''
        if self.trainer:
            # Update last reward
            if self.episode_reward != None and len(self.last_action_inds) == self.minibatch_size:
                # self.trainer.update_values(self.episode_reward, self.last_action_ind)
                if use_change:
                    self.trainer.update_values(self.episode_reward_change, self.last_action_inds[-1])
                else:
                    self.trainer.update_values(self.episode_reward, self.last_action_inds[-1])

            train_action, train_Hz, self.last_action_ind = self.trainer.get_action()
            self.last_action_inds.append(self.last_action_ind)
            return train_action, train_Hz
        
        neuron_order =  np.random.choice(self.training_neurons, size=stim_count, replace=False)
        train_action =[]
        for n in neuron_order:
            train_action.append(('stim',[n], self.amp_mv, self.phase_width))
            if n != neuron_order[-1]:
                train_action.append(('delay',delay_ms))

        return train_action, train_Hz


    def run(self):
        done = False
        
        assert self.env.observation_type == 'raw', 'Observation type must be raw'
        
        state, inf = self.game_env.reset()

        
        just_entered = True
        action = None
        reward = 0
        total_reward = 0
        rewards = []

        # Game log
        game_log = open(self.env.save_file + '_game_log.csv','w')
        # Header for csv
        game_logger = csv.writer(game_log)
        game_logger.writerow(['time','pole_angle','reward','action','spike_count_l', 'spike_count_r', 'state'])

        reward_log = open(self.env.save_file + '_reward_log.csv','w')
        # Header for csv
        reward_logger = csv.writer(reward_log)
        reward_logger.writerow(['time','episode', 'reward'])


        # Pattern log
        pattern_log = open(self.env.save_file + '_pattern_log.csv','w')
        # Header for csv
        pattern_logger = csv.writer(pattern_log)
        pattern_logger.writerow(['time','pattern','reward', 'probs', 'vals'])

        self.start_time = time.perf_counter()
        state_timer = self.time_elapsed()

        while not done:
            # ~~~~~~~~~~~ Read phase Logic ~~~~~~~~~~~
            if self.state == 'read':

                if just_entered:
                    state_timer = self.time_elapsed()
                    just_entered = False
                    self.motor_spike_count = np.zeros(len(self.motor_neurons))
                    
                
                if self.time_elapsed() - state_timer > self.read_period_ms / 1000:
                    self.state = 'game'
                    just_entered = True
                    continue

                # Neurons to stimulate
                active_sensory_neurons = np.where(self.env.stim_dts[self.sensory_neurons] > 1/self.sensory_stim_Hz)[0]
                # Stim the neurons at the correct frequency
                if len(active_sensory_neurons) > 0:
                    obs, done = self.env.step(action=(self.sensory_neurons[active_sensory_neurons], self.amp_mv, self.phase_width), tag='sensory')
                else:
                    # No action
                    obs, done = self.env.step()


                # Faster with artifact removal
                if self.artifact_remover:
                    # obs = np.array(obs)[self.motor_neurons]
                    if (obs == None):
                        continue
                    _,_, spike_0 = self.artifact_remover[0].fit_step(obs[self.motor_neurons[0]])
                    _,_, spike_1 = self.artifact_remover[1].fit_step(obs[self.motor_neurons[1]])
                    if spike_0:
                        self.motor_spike_count[0] += 1
                    if spike_1:
                        self.motor_spike_count[1] += 1
                else:
                    # Process the observation
                    for frame, ch, amp in obs:
                        if ch in self.motor_neurons:
                            self.motor_spike_count[np.where(self.motor_neurons==ch)] += 1

            # ~~~~~~~~~~~ Game phase Logic ~~~~~~~~~~~
            elif self.state == 'game':
                if just_entered:
                    state_timer = self.time_elapsed()

                    game_action = self.get_motor_signal()

                    self.game_obs,reward,game_done,trunc,inf = self.game_env.step(game_action)
                    total_reward += reward

                    game_logger.writerow([self.time_elapsed(),self.game_obs[2],reward,game_action,self.motor_spike_count, self.game_obs])

                    self.set_sensory_signal(self.game_obs) # Set the stim frequency rates

                    if self.verbose:
                        print('Spike count: ', self.motor_spike_count, f'Rate {self.motor_spike_rate[0]:.2f} || {self.motor_spike_rate[1]:.2f}')
                    
                
                    if game_done:
                        self.state = 'train'
                        just_entered = True
                        continue
                    else:
                        self.state = 'read'
                        just_entered = True
                        continue

            elif self.state == 'train':
                if just_entered:
                    state_timer = self.time_elapsed()
                    train_timer = self.time_elapsed()
                    just_entered = False
                    rewards.append(total_reward)
                    self.episode_reward = np.mean(rewards[-self.minibatch_size:])
                    self.episode_reward_change = np.mean(rewards[-self.minibatch_size:]) - np.mean(rewards[-20:])

                    # reward_increased = total_reward > max(np.mean(rewards[-5:]),15)
                    reward_increased = self.episode_reward_change > 0
                    # reward_increased = np.mean(rewards[-self.minibatch_size:]) >= np.mean(rewards[-20:])
                    do_training = not reward_increased or self.force_train
                    
                    reward_logger.writerow([self.time_elapsed(),self.episode,total_reward])
                    self.episode += 1
                    if self.episode >= self.n_episodes or self.time_elapsed() > self.max_time:
                        done = True
                        return

                    if self.verbose:
                        print('Reward:', total_reward, 'increased:', reward_increased)
                        print("Reward change:", self.episode_reward_change)
                        print(f'Last 20 : {np.mean(rewards[-20:]):.2f}, last 5: {np.mean(rewards[-5:]):.2f}')
                        print('-'*20)                        
                        print('Rewards:', rewards)

                    if do_training:
                        # Get pattern/update
                        train_pulse, train_freq = self.get_training_signal()
                        # Log pattern
                        pattern_logger.writerow([self.time_elapsed(),
                                                self.trainer.get_action(self.last_action_ind),
                                                total_reward, self.trainer.get_probs(), self.trainer.get_values()])
                        
                        obs, done = self.env.step(action=train_pulse, tag='train') 
                        train_timer = self.time_elapsed()
                        continue
                    else:
                        # Update even if it isn't currently training
                        self.trainer.update_values(self.episode_reward_change, self.last_action_inds[-1])
                        self.last_action_inds.append(None)

                
                if self.time_elapsed() - state_timer > self.train_period_ms / 1000:
                    self.state = 'wait'
                    just_entered = True
                    self.game_obs, inf = self.game_env.reset()
                    self.set_sensory_signal(self.game_obs)
                    total_reward = 0

                    continue
                
                if not (do_training):
                    self.game_obs, inf = self.game_env.reset()
                    self.set_sensory_signal(self.game_obs)
                    total_reward = 0
                    self.state = 'wait'
                    just_entered = True
                    continue

                # Training stimulation pulse -- If reward has not increased
                if do_training and (self.time_elapsed() - train_timer > 1/train_freq):
                    # Get pattern/update
                    # train_pulse, train_freq = self.get_training_signal()
                    # Log pattern
                    pattern_logger.writerow([self.time_elapsed(),
                                             self.trainer.get_action(self.last_action_ind),
                                             total_reward, self.trainer.get_probs(), self.trainer.get_values()])
                    
                    obs, done = self.env.step(action=train_pulse, tag='train') 
                    train_timer = self.time_elapsed()
                    

                else:
                    obs, done = self.env.step()

            elif self.state == 'wait':
                if just_entered:
                    state_timer = self.time_elapsed()
                    just_entered = False

                if self.time_elapsed() - state_timer > self.wait_period_ms / 1000:
                    self.state = 'read'
                    just_entered = True
                    continue

                obs, done = self.env.step()

        # Close the logs
        game_log.close()
        reward_log.close()
        pattern_log.close()
        return True


    def time_elapsed(self):
        return time.perf_counter() - self.start_time
    
    def info(self):
        return {
            'sensory_neurons': self.sensory_neurons,
            'motor_neurons': self.motor_neurons,
            'training_neurons': self.training_neurons,
            'read_period_ms': self.read_period_ms,
            'train_period_ms': self.train_period_ms,
            'n_episodes': self.n_episodes
        }

    def predicted_time(self):
        '''Returns the predicted time for the phase to run in seconds'''
        return self.predicted_time
    