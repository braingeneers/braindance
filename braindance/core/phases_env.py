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
import copy
from braindance.core.artifact_removal import ArtifactRemoval
from collections import deque
from braindance.core.phases import Phase

class FoodLandPhase(Phase):
    '''
    Phase for the FoodLand game
    
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
    minibatch_size = size of the myinibatch for the trainer, effectively should account for how
            many reward steps we care about after a training stimulus
    spike_thresh = thresholds for spike detection
            list of two values, first is the min value for a spike, second is the max value
    verbose = whether to print out information about the phase
    max_time = maximum time to run the phase for

    '''
    
    def __init__(self, env: base_env.BaseEnv, 
                 sensory_neurons:list, motor_neurons:list, training_neurons:list,
                 amp_mv:int = 400, phase_width:int = 100,
                 read_period_ms:int = 100,
                 train_period_ms:int = 100, 
                 wait_period_ms:int = 200,
                 n_episodes:int = 10,
                 trainer=None, artifact_removal=True,
                 spike_thresh=[-3.1,-20],
                 verbose=False, max_time = np.inf,
                 minibatch_size=3
                 ):
        
        import gymnasium as gym
        from braindance.games.food_land import FoodLandEnv


        self.game_env = FoodLandEnv(render_mode='human', reward_type='dense', max_steps=500)
        
        self.env = env
        self.read_period_ms = read_period_ms
        self.train_period_ms = train_period_ms
        self.wait_period_ms = wait_period_ms
        self.n_episodes = n_episodes
        self.episode = 0
        self.max_time = max_time

        self.minibatch_size = minibatch_size

        # IO
        self.sensory_neurons = np.array(sensory_neurons)
        self.motor_neurons = np.array(motor_neurons)
        self.training_neurons = np.array(training_neurons)

        self.amp_mv, self.phase_width = amp_mv, phase_width

        # Artifact removal
        # Ensure env.observation_type is 'raw'
        self.artifact_removal = artifact_removal
        if artifact_removal:
            #assert self.env.observation_type == 'raw', 'Observation type must be raw'
            self.artifact_remover = []
            for ch in self.motor_neurons:
                # self.artifact_remover.append(ArtifactRemoval(N=20, nc_start=20, min_val=-13, max_val=13,
                #                                             spike_thresh = [-3.2,-15]))
                self.artifact_remover.append(ArtifactRemoval(N=40, nc_start=40, min_val=-25, max_val=25,
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
        
        self.rewards = []
        self.food_got = []
        self.current_food = 0
        self.episode_reward = None
        self.episode_reward_change = None


        self.verbose = verbose

        self.motor_spike_count = np.zeros(len(self.motor_neurons)) # Stores the spike count for each motor neuron
        self.motor_spike_rate = np.zeros(len(self.motor_neurons))
        self.state = 'read' # 'read', 'train', 'game', or 'wait'

        self.predicted_time = (read_period_ms + train_period_ms)*20 * n_episodes / 1000 # x20 due to 20 steps of the game avg



        super().__init__(env)

    def set_env(self, env):
        self.env = env

    
    def set_sensory_function(self, func):
        '''
        Set the function to use to map the game observation to the sensory neurons
        '''
        self.set_sensory_signal = lambda game_env_obs: func(self, game_env_obs)

    def set_sensory_signal(self, game_env_obs):
        '''
        Maps the game observation to the sensory neurons.
        This observation is of the form:
        ndarray with shape (4,): 
            - Cart Position, Cart Velocity, Pole Angle, Pole Velocity At Tip

        We will map the pole angle to the 0th and 1st sensory neurons

        '''
        raise NotImplementedError
        # pole_angle = game_env_obs[2]
        # stim_bias = .15#.15#.2 # was .4, .1
        # stim_scale = 7#9#15#8# was 7,  20

        # self.sensory_stim_Hz = np.zeros(len(self.sensory_neurons))

        # self.sensory_stim_Hz[0] = ((-np.sin(pole_angle) + stim_bias)*stim_scale)**2
        # self.sensory_stim_Hz[1] = ((np.sin(pole_angle) + stim_bias)*stim_scale)**2
        # if self.verbose:
        #     print(f'Stim:Hz: \t{self.sensory_stim_Hz}')


    def set_motor_function(self, func):
        '''
        Set the function to use to map the motor neurons to the action
        '''
        self.get_motor_signal = lambda : func(self)


    def get_motor_signal(self, moving_avg=.2):
        '''
        Readout from the read/motor neurons set
        The action returned should be an integer {0,1} corresponding to the action

        Recommended to use self.spike_count, which is a list of the spike counts
        for each motor neuron 

        '''
        # if moving_avg:
        # self.motor_spike_rate = moving_avg*self.motor_spike_rate + (1-moving_avg)*self.motor_spike_count
        
            
        # # spike_diff = self.motor_spike_count[0] - self.motor_spike_count[1]
        # spike_diff = self.motor_spike_rate[0] - self.motor_spike_rate[1]
        # if self.assistive:
        #     assistive_factor = self.assistive*(self.sensory_stim_Hz[0] - self.sensory_stim_Hz[1])
        #     print(f'Assistive factor: {assistive_factor:.2f} \t Spike diff: {spike_diff:.2f}')
            
        #     spike_diff += assistive_factor

        # if self.continuous:
        #     print('Going with :',-spike_diff)
        #     return max(min(-spike_diff,1),-1)
        # if spike_diff > 0:
        #     action = 0
        #     print('Going left')
        # else:
        #     action = 1
        #     print('Going right')

        # return action
        raise NotImplementedError
    

    def set_training_function(self, func):
        '''
        Set the function to use to map the training neurons to the action
        '''
        self.get_training_signal = lambda : func(self)
        

    def get_training_signal(self, train_Hz=30, stim_count=3, delay_ms = 5, use_change=True):
        '''
        The stimulation on the training neurons 
        
        '''
        # if self.trainer:
        #     # Update last reward
        #     if self.episode_reward != None and len(self.last_action_inds) == self.minibatch_size:
        #         # self.trainer.update_values(self.episode_reward, self.last_action_ind)
        #         if use_change:
        #             self.trainer.update_values(self.episode_reward_change, self.last_action_inds[-1])
        #         else:
        #             self.trainer.update_values(self.episode_reward, self.last_action_inds[-1])

        #     train_action, train_Hz, self.last_action_ind = self.trainer.get_action()
        #     self.last_action_inds.append(self.last_action_ind)
        #     return train_action, train_Hz
        
        # neuron_order =  np.random.choice(self.training_neurons, size=stim_count, replace=False)
        # train_action =[]
        # for n in neuron_order:
        #     train_action.append(('stim',[n], self.amp_mv, self.phase_width))
        #     if n != neuron_order[-1]:
        #         train_action.append(('delay',delay_ms))

        # return train_action, train_Hz
        raise NotImplementedError

    def run(self):
        done = False

        
        state= self.game_env.reset()

        
        just_entered = True
        action = None
        reward = 0
        total_reward = 0
        self.rewards = []

        # Game log
        game_log = open(self.env.save_file + '_game_log.csv','w')
        # Header for csv
        game_logger = csv.writer(game_log)
        # time, obs, reward, action
        # time.time(), *self.game_obs, reward, self.episode, game_action[0], game_action[1]]
        # Obs is: 
            #  self.food_signal, self.spike_signal,
            #  self.agent_pos[0], self.agent_pos[1],
            #  self.agent_dir, self.food_got, self.spike_hit, 
            #  self.is_contacting_wall, self.food_signal_grad
        game_logger.writerow(['time','motor_spikes','food_signal', 'spike_signal', 'agent_pos_x', 'agent_pos_y',
                                'agent_dir', 'food_got', 'spike_hit', 'is_contacting_wall', 'food_signal_grad',
                                'reward','episode', 'moving_speed', 'turning_speed'])

        # ---- Training log ----
        train_log = open(self.env.save_file + '_train_log.csv','w')
        # Header for csv
        train_logger = csv.writer(train_log)
        train_logger.writerow(['time','pattern','reward', 'episode'])

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
                    if self.verbose:
                        print('Spike count: ', self.motor_spike_count, f'Rate {self.motor_spike_rate[0]:.2f} || {self.motor_spike_rate[1]:.2f}')
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
                    _,_, spike_0 = self.artifact_remover[0].fit_step(obs[self.motor_neurons[0]])
                    _,_, spike_1 = self.artifact_remover[1].fit_step(obs[self.motor_neurons[1]])
                    if len(self.artifact_remover) > 2:
                        _,_, spike_2 = self.artifact_remover[2].fit_step(obs[self.motor_neurons[2]])
                    else:
                        spike_2 = 0
                    if spike_0:
                        self.motor_spike_count[0] += 1
                    if spike_1:
                        self.motor_spike_count[1] += 1
                    if spike_2:
                        self.motor_spike_count[2] += 1
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
                    print('GAME ACTION:', game_action)

                    self.game_obs,reward,game_done,inf = self.game_env.step(game_action)
                    total_reward += reward
                    self.current_food += self.game_obs[5]

                    game_logger.writerow([self.time_elapsed(),self.motor_spike_count, *self.game_obs, reward, 
                                            self.episode, game_action[0], game_action[1]])

                    self.set_sensory_signal(self.game_obs) # Set the stim frequency rates
                
                    if game_done:
                        self.state = 'train'
                        just_entered = True
                        continue
                    else:
                        self.state = 'read'
                        just_entered = True
                        continue

            elif self.state == 'train':
                # ================== Enter train phase ==================
                if just_entered:
                    state_timer = self.time_elapsed()
                    train_timer = self.time_elapsed()
                    just_entered = False
                    self.rewards.append(total_reward)
                    self.food_got.append(copy.copy(self.current_food))
                    self.episode_reward = np.mean(self.rewards[-self.minibatch_size:])
                    self.episode_reward_change = np.mean(self.rewards[-self.minibatch_size:]) - np.mean(self.rewards[-20:])
                    self.current_food = 0
                    # reward_increased = total_reward > max(np.mean(rewards[-5:]),15)
                    reward_increased = self.episode_reward_change > 0
                    # reward_increased = np.mean(rewards[-self.minibatch_size:]) >= np.mean(rewards[-20:])
                    do_training = True #not reward_increased or self.force_train
                    
                    self.episode += 1
                    if self.episode >= self.n_episodes or self.time_elapsed() > self.max_time:
                        done = True
                        continue

                        # return

                    if self.verbose:
                        print('Reward:', total_reward, 'increased:', reward_increased)
                        print("Reward change:", self.episode_reward_change)
                        # print(f'Last 20 : {np.mean(self.rewards[-20:]):.2f}, last 5: {np.mean(self.rewards[-5:]):.2f}')
                        print('-'*20)                        
                        print('Rewards:', self.rewards)
                        print(f' ~~~~ Food got {self.food_got}~~~~')

                    # ================== Set Training pulses ==================
                    if do_training:
                        # Get pattern/update
                        train_pulse, train_freq = self.get_training_signal()
                        # Log pattern
                        train_logger.writerow([self.time_elapsed(), train_pulse, total_reward, self.episode])
                        obs, done = self.env.step(action=train_pulse, tag='train') 
                        train_timer = self.time_elapsed()
                        continue
                       

                
                if self.time_elapsed() - state_timer > self.train_period_ms / 1000:
                    self.state = 'wait'
                    just_entered = True
                    self.game_obs = self.game_env.reset()
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
        train_log.close()
        self.env.close()
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
    