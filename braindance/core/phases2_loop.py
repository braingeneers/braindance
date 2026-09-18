'''
Experiment phases
-----------------
Closed loop phases which map observations to stimulation commands

'''
import braindance.core.base_env as base_env
from braindance.core.phases2 import Phase
from proj.cartpole_v2 import experiment
from braindance.core.stim_commands import generate_stimulations

import numpy as np
import time
import csv
from braindance.utils.rt_linear_art_removal import LinearArtifactRemoval



class ClosedLoopPhase(Phase):
    '''
    Phase for a base closed-loop experiment
    
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
    experiment: experiment.Experiment, optional
        The experiment to run the phase in
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
                 input_neurons:list, output_neurons:list,
                 experiment:experiment.Experiment = None,
                 amp_mv:int = 400, phase_width:int = 100,
                 read_period_ms:int = 200,
                 train_period_ms:int = 200, 
                 wait_period_ms:int = 400,
                 n_episodes:int = 10,
                 trainer=None, artifact_removal=False,
                 minibatch_size=5, spike_thresh=[-3.1,-20],
                 verbose=False, max_time = np.inf,
                 suffix: str = "_closed_loop"):
        

        
        self.env = env
        self.read_period_ms = read_period_ms
        self.train_period_ms = train_period_ms
        self.wait_period_ms = wait_period_ms
        self.n_episodes = n_episodes
        self.episode = 0
        self.max_time = max_time
        self.minibatch_size = minibatch_size

        self.suffix = suffix

        # IO
        self.input_neurons = np.array(input_neurons)
        self.output_neurons = np.array(output_neurons)

        self.amp_mv, self.phase_width = amp_mv, phase_width

        # Artifact removal
        if artifact_removal:
            self.artifact_remover = LinearArtifactRemoval(
                n_channels=len(self.motor_neurons),
                N=60, nc_start=60, min_val=-85, max_val=85,
                spike_thresh=[-3.1, -25]
            )
        else:
            self.artifact_remover = None
        
        self.input_stim_Hz = np.zeros(len(self.input_neurons))


        self.verbose = verbose



        self.output_spike_count = np.zeros(len(self.output_neurons)) # Stores the spike count for each motor neuron
        self.output_spike_rate = np.zeros(len(self.output_neurons))
        self.state = 'read' # 'read', 'train', 'game', or 'wait'

        self.predicted_time = 0#(read_period_ms + train_period_ms)*20 * n_episodes / 1000 # x20 due to 20 steps of the game avg



        super().__init__(env, experiment=experiment)



    
    def get_input_signal(self, obs):
        '''
        Maps the obs to a stimulation command
        We use the input_output_map to find which output electrodes to stimulate


        '''
        active_input_neurons = np.where(obs[self.input_neurons] > 0)[0]
        if len(active_input_neurons) == 0:
            return None
        outputs_to_stim = self.input_output_map[active_input_neurons]
        stim_commands = generate_stimulations(outputs_to_stim, self.amp_mv, self.phase_width)
        return stim_commands




    def get_motor_signal(self, moving_avg=.2):
        '''
        Readout from the read/motor neurons set
        The action returned should be an integer {0,1} corresponding to the action

        Recommended to use self.spike_count, which is a list of the spike counts
        for each motor neuron 

        '''
        # if moving_avg:
        self.motor_spike_rate = moving_avg*self.motor_spike_rate + (1-moving_avg)*self.motor_spike_count
        
        action = 0

        return action
        



    def run(self):
        done = False
        
        assert self.env.observation_type == 'raw', 'Observation type must be raw'


        
        just_entered = True



        self.start_time = time.perf_counter()
        self.spike_count = np.zeros(len(self.output_neurons))
        # state_timer = self.time_elapsed()

        stim_command = None

        while not done:
            obs, done = self.env.step(action=stim_command)
            obs = self.process_observation(obs)
            stim_command = self.get_input_signal(obs)



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

        # Close the logs
        # game_log.close()
        # reward_log.close()
        # pattern_log.close()
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
    

class AntPhase(Phase):
    '''
    Phase for the Ant game
    
    Parameters
    ----------
    env : base_env.BaseEnv
        The environment to run the phase in
    motor_neurons : list
        List of motor neurons to read out from (this is in channels)
    training_neurons : list
        List of training neurons to stimulate (this is in indices of stim electrodes)
    experiment: experiment.Experiment, optional
        The experiment to run the phase in
    read_period_ms : int
        Period of time to read out from the motor neurons
    training_period_ms : int
        Period of time to stimulate the training neurons
    n_episodes : int
        Number of episodes to run the game for
    trainer : Trainer object
        Trainer object to use for training
    artifact_removal : bool
        Whether to use artifact removal
    verbose : bool
        Whether to print out information about the phase
    max_time : float
        Maximum time to run the phase for
    dummy_mode : str, optional
        If set to 'zeros', will use all zeros for motor spike counts when using dummy environment
        If set to 'random', will use random values for motor spike counts when using dummy environment
        Default is None, which will use actual spike counts from the environment
    '''
    
    def __init__(self, env: base_env.BaseEnv, 
                 motor_neurons:list, training_neurons:list,
                 experiment:experiment.Experiment = None,
                 amp_mv:int = 400, phase_width:int = 100,
                 read_period_ms:int = 200,
                 train_period_ms:int = 200, 
                 wait_period_ms:int = 400,
                 n_episodes:int = 10,
                 trainer=None, artifact_removal=False,
                 verbose=False, max_time = np.inf,
                 dummy_mode:str = None,
                 suffix: str = "_ant"):
        
        import gymnasium as gym
        from braindance.utils.rt_linear_art_removal import LinearArtifactRemoval
        
        self.game_env = gym.make('Ant-v5', render_mode='human')
        
        self.env = env
        self.read_period_ms = read_period_ms
        self.train_period_ms = train_period_ms
        self.wait_period_ms = wait_period_ms
        self.n_episodes = n_episodes
        self.episode = 0
        self.max_time = max_time
        self.suffix = suffix
        self.dummy_mode = dummy_mode


        # IO
        self.motor_neurons = np.array(motor_neurons)
        self.training_neurons = np.array(training_neurons)

        self.amp_mv, self.phase_width = amp_mv, phase_width

        # Artifact removal
        if artifact_removal:
            self.artifact_remover = LinearArtifactRemoval(
                n_channels=len(self.motor_neurons),
                N=60, nc_start=60, min_val=-85, max_val=85,
                spike_thresh=[-3.1, -25]
            )
        else:
            self.artifact_remover = None
        
        self.trainer = trainer
        if self.trainer:
            self.trainer.phase = self
            
        self.last_action_ind = None
        self.game_obs = None
        self.episode_reward = None

        self.verbose = verbose

        self.motor_spike_count = np.zeros(len(self.motor_neurons))
        self.motor_spike_rate = np.zeros(len(self.motor_neurons))
        self.state = 'read' # 'read', 'train', 'game', or 'wait'

        self.predicted_time = (read_period_ms + train_period_ms)*20 * n_episodes / 1000

        super().__init__(env, experiment=experiment)

    def get_motor_signal(self, moving_avg=.3):
        '''
        Readout from the read/motor neurons set
        The action returned should be a numpy array of shape (8,) with values between -1 and 1
        '''
        # Handle dummy mode - generate synthetic spike rates instead of using real ones
        if self.dummy_mode:
            if self.dummy_mode == 'zeros':
                self.motor_spike_count = np.zeros(len(self.motor_neurons))
            elif self.dummy_mode == 'random':
                self.motor_spike_count = np.random.rand(len(self.motor_neurons)) * 0.5  # Random values between 0 and 0.5
                
        self.motor_spike_rate = moving_avg*self.motor_spike_rate + (1-moving_avg)*self.motor_spike_count
        
        # Simple mapping: use first 8 neurons to control the 8 action dimensions
        # Scale the spike rates to be between -1 and 1
        action = np.zeros(8)
        for i in range(min(8, len(self.motor_spike_rate))):
            # Scale between 0 and 1 to between -1 and 1
            action[i] = np.clip(self.motor_spike_rate[i]*3 - 1, -1, 1)
            # action[i] = np.clip(self.motor_spike_rate[i]*5, -1, 1)  # Scale factor of 10 is arbitrary
            
        if self.verbose:
            print(f'Spike rates: {self.motor_spike_rate[:8]}')
            print(f'Action: {action}')
            
        return action.flatten()

    def get_training_signal(self, train_Hz=30, stim_count=3, delay_ms = 5):
        '''
        The stimulation on the training neurons 
        '''
        if self.trainer:
            # Update last reward
            if self.episode_reward != None:
                self.trainer.update_values(self.episode_reward, self.last_action_ind)

            train_action, train_Hz, self.last_action_ind = self.trainer.get_action()
            return train_action, train_Hz
        
        neuron_order = np.random.choice(self.training_neurons, size=stim_count, replace=False)
        train_action = []
        for n in neuron_order:
            train_action.append(('stim',[n], self.amp_mv, self.phase_width))
            if n != neuron_order[-1]:
                train_action.append(('delay',delay_ms))

        return train_action, train_Hz

    def run(self):
        done = False
        
        state, inf = self.game_env.reset()
        
        just_entered = True
        action = None
        reward = 0
        total_reward = 0
        rewards = []
        max_episode_steps = 1000  # Standard timeout for Ant environment
        episode_steps = 0

        # Game log
        game_log = open(self.env.save_file + '_game_log.csv','w')
        game_logger = csv.writer(game_log)
        game_logger.writerow(['time','reward','action','spike_count', 'state', 'z_pos'])

        reward_log = open(self.env.save_file + '_reward_log.csv','w')
        reward_logger = csv.writer(reward_log)
        reward_logger.writerow(['time','episode', 'reward', 'episode_steps'])

        pattern_log = open(self.env.save_file + '_pattern_log.csv','w')
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

                obs, done = self.env.step()

                if self.artifact_remover:
                    if obs is None:
                        continue
                    # Process all channels at once
                    obs_arr = np.array([obs[ch] for ch in self.motor_neurons]).reshape(1, -1).T
                    clean_values, artifacts, spikes = self.artifact_remover.fit_step(
                        obs_arr
                    )
                    self.motor_spike_count += spikes.flatten()
                else:
                    # Process the observation
                    for frame, ch, amp in obs:
                        if ch in self.motor_neurons:
                            self.motor_spike_count[np.where(self.motor_neurons==ch)] += 1

            # ~~~~~~~~~~~ Game phase Logic ~~~~~~~~~~~
            elif self.state == 'game':
                if just_entered:
                    state_timer = self.time_elapsed()
                    episode_steps += 1

                    game_action = self.get_motor_signal()

                    self.game_obs, reward, game_done, trunc, inf = self.game_env.step(game_action)
                    game_done = game_done or trunc  # Handle both termination and truncation
                    
                    # Check if episode timeout or ant flipped over
                    z_pos = self.game_obs[0] if len(self.game_obs) > 0 else 0  # z-coordinate of torso
                    
                    total_reward += reward


                    game_logger.writerow([self.time_elapsed(), reward, game_action, self.motor_spike_count, self.game_obs, z_pos])

                    if self.verbose:
                        print('Spike count: ', self.motor_spike_count, f'\nRate {self.motor_spike_rate[0]:.2f} || {self.motor_spike_rate[1]:.2f}',end='')
                        print(f'|| {self.motor_spike_rate[2]:.2f} || {self.motor_spike_rate[3]:.2f} || {self.motor_spike_rate[4]:.2f} || {self.motor_spike_rate[5]:.2f} || {self.motor_spike_rate[6]:.2f} || {self.motor_spike_rate[7]:.2f}')
                        
                    if game_done or episode_steps >= max_episode_steps:
                        if self.verbose and episode_steps >= max_episode_steps:
                            print(f"Episode truncated after {episode_steps} steps")
                        self.state = 'train'
                        just_entered = True
                        # Update networks at end of episode
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
                    self.episode_reward = np.mean(rewards[-5:])  # Use last 5 rewards

                    reward_logger.writerow([self.time_elapsed(), self.episode, total_reward, episode_steps])
                    self.episode += 1
                    episode_steps = 0  # Reset episode steps
                    
                    if self.episode >= self.n_episodes or self.time_elapsed() > self.max_time:
                        done = True
                        return
                    
                    if self.verbose:
                        print('Reward:', total_reward)
                        print('-'*20)                        
                        print('Rewards:', rewards)

                    # Get pattern/update
                    train_pulse, train_freq = self.get_training_signal()
                    # Log pattern
                    pattern_logger.writerow([self.time_elapsed(),
                                            self.trainer.get_action(self.last_action_ind) if self.trainer else None,
                                            total_reward, 
                                            self.trainer.get_probs() if self.trainer else None, 
                                            self.trainer.get_values() if self.trainer else None])
                    
                    obs, done = self.env.step(action=train_pulse, tag='train') 
                    train_timer = self.time_elapsed()
                    continue
                
                if self.time_elapsed() - state_timer > self.train_period_ms / 1000:
                    self.state = 'wait'
                    just_entered = True
                    self.game_obs, inf = self.game_env.reset()
                    total_reward = 0
                    continue

                # Training stimulation pulse
                if self.time_elapsed() - train_timer > 1/train_freq:
                    # Get pattern/update
                    train_pulse, train_freq = self.get_training_signal()
                    # Log pattern
                    pattern_logger.writerow([self.time_elapsed(),
                                            self.trainer.get_action(self.last_action_ind) if self.trainer else None,
                                            total_reward, 
                                            self.trainer.get_probs() if self.trainer else None, 
                                            self.trainer.get_values() if self.trainer else None])
                    
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
            'motor_neurons': self.motor_neurons,
            'training_neurons': self.training_neurons,
            'read_period_ms': self.read_period_ms,
            'train_period_ms': self.train_period_ms,
            'n_episodes': self.n_episodes
        }

    def predicted_time(self):
        '''Returns the predicted time for the phase to run in seconds'''
        return self.predicted_time
    

class NeuralAntPhase(AntPhase):
    '''
    Neural network version of AntPhase that uses a neural network for action selection.
    The network maps from motor spike rates to action outputs (between -1 and 1).
    
    Parameters
    ----------
    env : base_env.BaseEnv
        The environment to run the phase in
    motor_neurons : list
        List of motor neurons to read out from (this is in channels)
    training_neurons : list
        List of training neurons to stimulate (this is in indices of stim electrodes)
    network_arch : list, optional
        List of integers specifying the number of neurons in each hidden layer
        Default is [64, 64] for a two-layer network
    learning_rate : float, optional
        Learning rate for the neural network, default is 0.001
    gamma : float, optional
        Discount factor for future rewards, default is 0.99
    gae_lambda : float, optional
        Lambda parameter for Generalized Advantage Estimation, default is 0.95
    clip_ratio : float, optional
        PPO clip ratio, default is 0.2
    train_iters : int, optional
        Number of training iterations per update, default is 10
    batch_size : int, optional
        Batch size for training, default is 64
    dummy_mode : str, optional
        If set to 'zeros', will use all zeros for motor spike counts when using dummy environment
        If set to 'random', will use random values for motor spike counts when using dummy environment
        Default is None, which will use actual spike counts from the environment
    '''
    def __init__(self, env: base_env.BaseEnv, 
                 motor_neurons:list, training_neurons:list,
                 network_arch:list = [16],
                 learning_rate:float = 0.001,
                 gamma:float = 0.99,
                 gae_lambda:float = 0.95,
                 clip_ratio:float = 0.1,
                 train_iters:int = 10,
                 batch_size:int = 64,
                 dummy_mode:str = None,
                 **kwargs):
        
        super().__init__(env, motor_neurons, training_neurons, **kwargs)
        
        self.dummy_mode = dummy_mode
        
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.distributions import Normal
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network architecture
        self.network_arch = network_arch
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.batch_size = batch_size
        
        # Initialize neural network
        self.actor = self._build_actor().to(self.device)
        self.critic = self._build_critic().to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Initialize memory buffer
        self.memory = []
        self.max_memory_size = 10000
        
    def _build_actor(self):
        """Build a more stable actor network"""
        import torch.nn as nn
        import torch
        
        class StableActorNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dims):
                super().__init__()
                
                # Layer normalization for input stabilization
                self.input_norm = nn.LayerNorm(input_dim)
                
                # Shared network
                layers = []
                prev_dim = input_dim
                
                for dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, dim))
                    layers.append(nn.Tanh())  # Tanh is more stable than ReLU here
                    layers.append(nn.LayerNorm(dim))  # Add normalization after activation
                    prev_dim = dim
                    
                self.shared = nn.Sequential(*layers)
                
                # Separate heads for mean and log_std
                self.mean_head = nn.Sequential(
                    nn.Linear(prev_dim, 8),
                    nn.Tanh()  # Keep means in [-1, 1] range
                )
                
                self.log_std_head = nn.Sequential(
                    nn.Linear(prev_dim, 8),
                    nn.Hardtanh(-2, 0)  # Constrain log_std to keep std in [0.13, 1.0]
                )
                
                # Initialize weights with small values
                self.apply(self._init_weights)
                
            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=0.1)  # Small initial weights
                    nn.init.constant_(module.bias, 0)
                    
            def forward(self, x):
                x = self.input_norm(x)
                shared_features = self.shared(x)
                means = self.mean_head(shared_features)
                log_stds = self.log_std_head(shared_features)
                
                # Concatenate for expected output format
                return torch.cat([means, log_stds], dim=-1)
        
        # Use a deeper network with [32, 32] neurons in hidden layers
        return StableActorNetwork(len(self.motor_neurons), [16])
    
    def _build_critic(self):
        """Build a more stable critic network"""
        import torch.nn as nn
        import torch
        
        class StableCriticNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dims):
                super().__init__()
                
                # Layer normalization for input stabilization
                self.input_norm = nn.LayerNorm(input_dim)
                
                # Hidden layers
                layers = []
                prev_dim = input_dim
                
                for dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, dim))
                    layers.append(nn.Tanh())  # Tanh is more stable than ReLU
                    layers.append(nn.LayerNorm(dim))
                    prev_dim = dim
                    
                self.hidden = nn.Sequential(*layers)
                
                # Output layer
                self.output = nn.Linear(prev_dim, 1)
                
                # Initialize weights
                self.apply(self._init_weights)
                
            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=0.1)
                    nn.init.constant_(module.bias, 0)
                    
            def forward(self, x):
                x = self.input_norm(x)
                x = self.hidden(x)
                return self.output(x)
        
        # Use same architecture as actor for consistency
        return StableCriticNetwork(len(self.motor_neurons), [16])
    
    
    def get_motor_signal(self, moving_avg=.3):
        '''
        Readout from the read/motor neurons set using neural network
        The action returned should be a numpy array of shape (8,) with values between -1 and 1
        '''
        import torch
        from torch.distributions import Normal
        import numpy as np
        
        # Handle dummy mode - generate synthetic spike rates instead of using real ones
        if self.dummy_mode:
            if self.dummy_mode == 'zeros':
                self.motor_spike_count = np.zeros(len(self.motor_neurons))
            elif self.dummy_mode == 'random':
                self.motor_spike_count = np.random.rand(len(self.motor_neurons)) * 0.5  # Random values between 0 and 0.5
        
        # Update spike rates
        self.motor_spike_rate = moving_avg*self.motor_spike_rate + (1-moving_avg)*self.motor_spike_count
        
        # Handle zero and near-zero values
        epsilon = 1e-5
        
        # Apply log transform and clipping to handle extreme values
        # This is critical for numerical stability
        log_states = np.log(self.motor_spike_rate + epsilon)
        
        # Clip extreme values - limit the range to prevent explosion
        log_states = np.clip(log_states, -10, 10)
        
        # Convert to tensor
        state = torch.FloatTensor(log_states).to(self.device)
        
        # Forward pass through network
        try:
            with torch.no_grad():  # Prevent gradient tracking for inference
                action_params = self.actor(state)
                
                # Check for NaNs before proceeding
                if torch.isnan(action_params).any() or torch.isinf(action_params).any():
                    print("Warning: NaN or Inf in action_params, returning zero actions")
                    return np.zeros(8)
                    
                action_params = action_params.view(-1, 8, 2)
                means = action_params[..., 0]
                log_stds = action_params[..., 1]
                
                # Tight clamping of log_stds - very important for stability
                log_stds = torch.clamp(log_stds, -1.5, 0)  # Constrain std to (0.22, 1.0)
                
                # Sample action directly using the reparameterization trick
                # This is safer than using Normal distribution
                noise = torch.randn_like(means)
                stds = torch.exp(log_stds)
                action = means + noise * stds
                action = torch.tanh(action)  # Bound to [-1, 1]
                
                # Compute log_prob safely
                log_prob = -0.5 * (noise**2 + 2*log_stds + np.log(2*np.pi))
                log_prob = torch.sum(log_prob, dim=-1)
                
                # Account for the tanh squashing
                log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1)
                
                # Get value estimate
                value = self.critic(state)
                
                # Store in memory - detach tensors before converting to numpy
                self.memory.append({
                    'state': state.detach().cpu().numpy(),
                    'action': action.detach().cpu().numpy().flatten(),
                    'log_prob': log_prob.detach().cpu().numpy(),
                    'value': value.detach().cpu().numpy(),
                    'reward': 0,  # Will be updated later
                    'next_value': 0,  # Will be updated later
                    'done': False  # Will be updated later
                })
                
                # Keep memory size limited
                if len(self.memory) > self.max_memory_size:
                    self.memory.pop(0)
                    
                if self.verbose:
                    print(f'Action: {action.detach().cpu().numpy().flatten()}')
                    
                return action.detach().cpu().numpy().flatten()
                
        except Exception as e:
            print(f"Error in get_motor_signal: {e}")
            return np.zeros(8)  # Return safe action in case of any error
    
    def _compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation"""
        import torch
        import numpy as np
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
            
        returns = advantages + values
        return advantages, returns
    
    def _update_networks(self):
        """Update actor and critic networks using PPO with robust safeguards"""
        import torch
        import numpy as np
        
        if len(self.memory) < self.batch_size:
            return
        
        # Print memory for debugging
        if self.verbose:
            print(f"Update with {len(self.memory)} experiences")
        
        try:
            # Convert memory to tensors
            states = torch.FloatTensor(np.array([m['state'] for m in self.memory])).to(self.device)
            actions = torch.FloatTensor(np.array([m['action'] for m in self.memory])).to(self.device)
            old_log_probs = torch.FloatTensor(np.array([m['log_prob'] for m in self.memory])).to(self.device)
            rewards = torch.FloatTensor(np.array([m['reward'] for m in self.memory])).to(self.device)
            next_values = torch.FloatTensor(np.array([m['next_value'] for m in self.memory])).to(self.device)
            dones = torch.FloatTensor(np.array([m['done'] for m in self.memory])).to(self.device)
            
            # Handle extreme values in rewards
            rewards = torch.clamp(rewards, -10, 10)  # Clip large reward values
            
            # Compute advantages
            with torch.no_grad():
                values = self.critic(states).squeeze()
                advantages, returns = self._compute_gae(rewards.cpu().numpy(), 
                                                    values.detach().cpu().numpy(),
                                                    next_values.cpu().numpy(),
                                                    dones.cpu().numpy())
                advantages = torch.FloatTensor(advantages).to(self.device)
                returns = torch.FloatTensor(returns).to(self.device)
                
                # Normalize advantages - important for stable updates
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Use a smaller learning rate for the first few episodes
            if self.episode < 5:
                lr_scale = 0.1
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = self.learning_rate * lr_scale
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = self.learning_rate * lr_scale
            
            # PPO update
            for _ in range(self.train_iters):
                # Forward pass
                action_params = self.actor(states)
                
                # Check for NaN values in action_params and skip update if found
                if torch.isnan(action_params).any() or torch.isinf(action_params).any():
                    print("Warning: NaN values detected in action parameters, skipping update")
                    return
                    
                action_params = action_params.view(-1, 8, 2)
                means = action_params[..., 0]
                log_stds = action_params[..., 1]
                log_stds = torch.clamp(log_stds, -2, 0)  # Constrain log_std
                
                # Calculate new log probabilities
                try:
                    # Sample using reparameterization trick for numerical stability
                    noise = torch.randn_like(means)
                    stds = torch.exp(log_stds)
                    action_sample = means + noise * stds
                    
                    # Calculate new log probs manually for stability
                    new_log_probs = -0.5 * ((actions - means)**2 / (stds**2 + 1e-8) + 
                                        2 * log_stds + np.log(2 * np.pi))
                    new_log_probs = torch.sum(new_log_probs, dim=-1)
                    
                    # Account for tanh squashing (optional in this safe version)
                    # new_log_probs -= torch.sum(torch.log(1 - actions.pow(2) + 1e-8), dim=-1)
                    
                    # Compute PPO ratio safely
                    ratio = torch.exp(torch.clamp(new_log_probs - old_log_probs, -20, 20))
                    ratio = torch.clamp(ratio, 0.0, 5.0)  # Strict clipping of ratio
                    
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Compute value loss with clipping
                    value_pred = self.critic(states).squeeze()
                    value_loss = torch.nn.functional.mse_loss(value_pred, returns)
                    
                    # Update networks with careful gradient clipping
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.1)  # Strict clipping
                    self.actor_optimizer.step()
                    
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.1)
                    self.critic_optimizer.step()
                    
                except Exception as e:
                    print(f"Error in PPO update: {e}")
                    return
                    
            # Clear memory after update
            self.memory = []
            
        except Exception as e:
            print(f"Error in network update: {e}")
            self.memory = []  # Clear memory to avoid re-processing problematic data
        
    def run(self):
        """Override run method to include neural network updates"""
        done = False
        
        state, inf = self.game_env.reset()
        
        just_entered = True
        action = None
        reward = 0
        total_reward = 0
        rewards = []
        max_episode_steps = 100  # Standard timeout for Ant environment
        episode_steps = 0

        # Game log
        game_log = open(self.env.save_file + '_game_log.csv','w')
        game_logger = csv.writer(game_log)
        game_logger.writerow(['time','reward','action','spike_count', 'state', 'z_pos'])

        reward_log = open(self.env.save_file + '_reward_log.csv','w')
        reward_logger = csv.writer(reward_log)
        reward_logger.writerow(['time','episode', 'reward', 'episode_steps'])

        pattern_log = open(self.env.save_file + '_pattern_log.csv','w')
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

                obs, done = self.env.step()

                if self.artifact_remover:
                    if obs is None:
                        continue
                    # Process all channels at once
                    obs_arr = np.array([obs[ch] for ch in self.motor_neurons]).reshape(1, -1).T
                    clean_values, artifacts, spikes = self.artifact_remover.fit_step(
                        obs_arr
                    )
                    self.motor_spike_count += spikes.flatten()
                else:
                    # Process the observation
                    for frame, ch, amp in obs:
                        if ch in self.motor_neurons:
                            self.motor_spike_count[np.where(self.motor_neurons==ch)] += 1

            # ~~~~~~~~~~~ Game phase Logic ~~~~~~~~~~~
            elif self.state == 'game':
                if just_entered:
                    state_timer = self.time_elapsed()
                    episode_steps += 1

                    game_action = self.get_motor_signal()

                    self.game_obs, reward, game_done, trunc, inf = self.game_env.step(game_action)
                    game_done = game_done or trunc  # Handle both termination and truncation
                    if episode_steps >= max_episode_steps:
                        game_done = True
                    
                    # Check if episode timeout or ant flipped over
                    z_pos = self.game_obs[0] if len(self.game_obs) > 0 else 0  # z-coordinate of torso
                    
                    total_reward += reward

                    # Update memory with reward
                    if len(self.memory) > 0:
                        self.memory[-1]['reward'] = reward
                        self.memory[-1]['done'] = game_done

                    game_logger.writerow([self.time_elapsed(), reward, game_action, self.motor_spike_count, self.game_obs, z_pos])

                    if self.verbose:
                        print(f'E:{self.episode} S:{episode_steps} R:{reward} Z:{z_pos} ', self.motor_spike_count, f'\nRate {self.motor_spike_rate[0]:.2f} || {self.motor_spike_rate[1]:.2f}',end='')
                        print(f'|| {self.motor_spike_rate[2]:.2f} || {self.motor_spike_rate[3]:.2f} || {self.motor_spike_rate[4]:.2f} || {self.motor_spike_rate[5]:.2f} || {self.motor_spike_rate[6]:.2f} || {self.motor_spike_rate[7]:.2f}')
                        
                    if game_done or episode_steps >= max_episode_steps:
                        if self.verbose and episode_steps >= max_episode_steps:
                            print(f"Episode truncated after {episode_steps} steps")
                        self.state = 'train'
                        just_entered = True
                        # Update networks at end of episode
                        self._update_networks()
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
                    self.episode_reward = np.mean(rewards[-5:])  # Use last 5 rewards

                    reward_logger.writerow([self.time_elapsed(), self.episode, total_reward, episode_steps])
                    self.episode += 1
                    episode_steps = 0  # Reset episode steps
                    
                    if self.episode >= self.n_episodes or self.time_elapsed() > self.max_time:
                        done = True
                        return
                    
                    if self.verbose:
                        print('Reward:', total_reward)
                        print('-'*20)                        
                        print('Rewards:', rewards)

                    # Get pattern/update
                    train_pulse, train_freq = self.get_training_signal()
                    # Log pattern
                    pattern_logger.writerow([self.time_elapsed(),
                                            self.trainer.get_action(self.last_action_ind) if self.trainer else None,
                                            total_reward, 
                                            self.trainer.get_probs() if self.trainer else None, 
                                            self.trainer.get_values() if self.trainer else None])
                    
                    obs, done = self.env.step(action=train_pulse, tag='train') 
                    train_timer = self.time_elapsed()
                    continue
                
                if self.time_elapsed() - state_timer > self.train_period_ms / 1000:
                    self.state = 'wait'
                    just_entered = True
                    self.game_obs, inf = self.game_env.reset()
                    total_reward = 0
                    continue

                # Training stimulation pulse
                if self.time_elapsed() - train_timer > 1/train_freq:
                    # Get pattern/update
                    train_pulse, train_freq = self.get_training_signal()
                    # Log pattern
                    pattern_logger.writerow([self.time_elapsed(),
                                            self.trainer.get_action(self.last_action_ind) if self.trainer else None,
                                            total_reward, 
                                            self.trainer.get_probs() if self.trainer else None, 
                                            self.trainer.get_values() if self.trainer else None])
                    
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
            'motor_neurons': self.motor_neurons,
            'training_neurons': self.training_neurons,
            'read_period_ms': self.read_period_ms,
            'train_period_ms': self.train_period_ms,
            'n_episodes': self.n_episodes
        }

    def predicted_time(self):
        '''Returns the predicted time for the phase to run in seconds'''
        return self.predicted_time
    