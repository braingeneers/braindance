"""
Upgraded Phases for Experiment Framework V3

These phases use the enhanced base classes with automatic data passing
and configuration.
"""
import numpy as np
import time
import csv
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .phase_base_v3 import PhaseV3, AnalysisPhaseV3
from braindance.core.base_env import BaseEnv


class RecordPhaseV3(PhaseV3):
    """
    Phase for recording spontaneous activity.
    
    Provides:
        - recording_file: Path to the saved recording
        - recording_duration: Actual duration of recording
    """
    
    outputs = ['recording_file', 'recording_duration']
    
    def __init__(self, duration: int = 60, name: str = None, suffix: str = "_recording",
                 recording_tag: str = "rec"):
        super().__init__(name or "RecordPhase", suffix, recording_tag=recording_tag)
        self.duration = duration
        self.verbose = False
        
    def configure_from_experiment(self, experiment):
        """Auto-configure from experiment config."""
        super().configure_from_experiment(experiment)
        
        # Override duration if specified in config
        self.duration = experiment.get_param('record_duration', self.duration)
        self.verbose = experiment.get_param('verbose', False)
    
    def run(self, experiment) -> Dict[str, Any]:
        """Execute recording."""
        if self.verbose:
            print(f"   Recording for {self.duration} seconds...")
        
        done = False
        self.start_time = time.perf_counter()
        
        while not done:
            buffer_size = None
            if getattr(self.env, "is_replay", False):
                remaining_s = max(0.0, self.duration - self.time_elapsed())
                remaining_frames = int(np.ceil(
                    remaining_s * self.env.replay_source.sampling_hz - 1e-12
                ))
                buffer_size = max(
                    1,
                    min(self.env.replay_source.chunk_frames, remaining_frames),
                )
            obs, done = self.env.step(buffer_size=buffer_size)
            if self.time_elapsed() >= self.duration:
                done = True
        
        # Return results
        return {
            'recording_file': self.env.save_file,
            'recording_duration': self.time_elapsed()
        }
    
    def info(self) -> Dict[str, Any]:
        info = super().info()
        info.update({
            'duration': self.duration
        })
        return info


class NeuralSweepPhaseV3(PhaseV3):
    """
    Sweep stimulation amplitude to find response thresholds.
    
    Requires:
        - stim_electrodes: List of stimulation electrodes (from config)
    
    Provides:
        - sweep_results: Results of amplitude sweep
        - sweep_file: Path to the saved sweep data
    """
    
    inputs = ['stim_electrodes']  # Gets stim_electrodes from config
    outputs = ['sweep_results', 'sweep_file']
    
    def __init__(self, neuron_list: List[int] = None, 
                 amp_bounds: tuple = (150, 400, 10),
                 stim_freq: float = 1.0,
                 replicates: int = 30,
                 phase_length: int = 100,
                 order: str = 'ran',
                 single_connect: bool = False,
                 tag: str = 'neural_sweep',
                 name: str = None,
                 suffix: str = "_sweep",
                 recording_tag: str = "sweep"):
        
        super().__init__(name or "NeuralSweepPhase", suffix, recording_tag=recording_tag)
        
        self.neuron_list = neuron_list
        self._auto_neuron_list = neuron_list is None
        self.amp_bounds = amp_bounds
        self.stim_freq = stim_freq
        self.replicates = replicates
        self.phase_length = phase_length
        self.order = order
        self.single_connect = single_connect
        self.tag = tag
        self.verbose = False
        
        # Process amplitude bounds
        if isinstance(amp_bounds, (int, float)):
            self.amplitude_start = self.amplitude_end = amp_bounds
            self.n_amplitudes = 1
        else:
            self.amplitude_start = amp_bounds[0]
            self.amplitude_end = amp_bounds[1]
            self.n_amplitudes = amp_bounds[2] if len(amp_bounds) > 2 else 10
    
    def configure_from_experiment(self, experiment):
        """Auto-configure from experiment."""
        super().configure_from_experiment(experiment)
        
        # Inputs may come from a preceding analysis and change between runs.
        if self._auto_neuron_list:
            self.neuron_list = list(range(len(self.stim_electrodes)))
            
        self.verbose = experiment.get_param('verbose', False)

    def customize_environment_params(self, env_params: dict) -> dict:
        """Customize environment parameters."""
        env_params['stim_electrodes'] = self.stim_electrodes
        return env_params
    
    def generate_stim_commands(self) -> List[tuple]:
        """Generate stimulation commands for the sweep."""
        amplitudes = np.linspace(self.amplitude_start, self.amplitude_end, self.n_amplitudes)
        stim_commands = []
        
        # Generate commands based on order
        if self.order == 'ran':
            for _ in range(self.replicates):
                for amp in amplitudes:
                    for neuron in self.neuron_list:
                        stim_commands.append(([neuron], amp, self.phase_length))
        
        elif self.order == 'arn':
            for amp in amplitudes:
                for _ in range(self.replicates):
                    for neuron in self.neuron_list:
                        stim_commands.append(([neuron], amp, self.phase_length))
        
        elif self.order == 'nar':
            for neuron in self.neuron_list:
                for amp in amplitudes:
                    for _ in range(self.replicates):
                        stim_commands.append(([neuron], amp, self.phase_length))
        
        elif self.order == 'random':
            # Generate all combinations
            for _ in range(self.replicates):
                for amp in amplitudes:
                    for neuron in self.neuron_list:
                        stim_commands.append(([neuron], amp, self.phase_length))
            np.random.shuffle(stim_commands)
        
        return stim_commands
    
    def run(self, experiment) -> Dict[str, Any]:
        """Execute amplitude sweep."""
        if self.verbose:
            print(f"   Sweeping {len(self.neuron_list)} neurons, "
                  f"{self.n_amplitudes} amplitudes, {self.replicates} replicates")
        
        stim_commands = self.generate_stim_commands()
        time_between_stims = 1 / self.stim_freq
        stim_count = 0
        
        # Track results
        results = []
        
        # Single connect mode setup
        if self.single_connect and stim_commands:
            self.last_neuron = stim_commands[0][0][0]
            self.env.disconnect_all()
            self.env.connect_units([self.env.stim_units[self.last_neuron]])
        
        done = False
        self.start_time = time.perf_counter()
        
        while not done and stim_commands:
            if self.time_elapsed() >= time_between_stims * stim_count:
                # Get next command
                neurons, amplitude, phase_length = stim_commands.pop(0)
                
                # Handle single connect mode
                if self.single_connect and neurons[0] != self.last_neuron:
                    self.last_neuron = neurons[0]
                    self.env.disconnect_all()
                    self.env.connect_units([self.env.stim_units[self.last_neuron]])
                
                # Stimulate
                self.env.step(action=(neurons, amplitude, phase_length), tag=self.tag)
                
                # Record result
                results.append({
                    'time': self.time_elapsed(),
                    'neuron': neurons[0],
                    'amplitude': amplitude,
                    'stim_count': stim_count
                })
                
                stim_count += 1
                
                if self.verbose and stim_count % 100 == 0:
                    print(f"      Completed {stim_count}/{len(stim_commands) + stim_count} stims")
            
            else:
                buffer_size = None
                if getattr(self.env, "is_replay", False):
                    elapsed = self.time_elapsed()
                    next_deadline = time_between_stims * stim_count
                    frames_to_deadline = int(np.floor(
                        (next_deadline - elapsed)
                        * self.env.replay_source.sampling_hz
                        + 1e-9
                    ))
                    buffer_size = max(
                        1,
                        min(self.env.replay_source.chunk_frames, frames_to_deadline),
                    )
                obs, done = self.env.step(buffer_size=buffer_size)
        
        return {
            'sweep_results': results,
            'sweep_file': self.env.save_file
        }
    
    def info(self) -> Dict[str, Any]:
        info = super().info()
        info.update({
            'neuron_count': len(self.neuron_list) if self.neuron_list else 0,
            'amplitude_range': (self.amplitude_start, self.amplitude_end),
            'n_amplitudes': self.n_amplitudes,
            'stim_freq': self.stim_freq,
            'replicates': self.replicates,
            'order': self.order
        })
        return info 


class FrequencyStimPhaseV3(PhaseV3):
    """
    Phase for stimulating at a specific frequency.
    
    Provides:
        - stim_file: Path to the saved stimulation data
        - stim_count: Number of stimulations delivered
    """
    inputs = ['stim_electrodes']
    outputs = ['stim_file', 'stim_count']
    
    def __init__(self, stim_command: Union[tuple, List[tuple]], 
                 stim_freq: float = 1.0,
                 duration: int = 60,
                 tag: Union[str, List[str]] = 'frequency_stim',
                 name: str = None,
                 suffix: str = "_freq_stim",
                 recording_tag: str = "freq_stim"):
        
        super().__init__(name or "FrequencyStimPhase", suffix, recording_tag=recording_tag)
        
        self.stim_command = stim_command
        self.stim_freq = stim_freq
        self.duration = duration
        self.tag = tag
        self.verbose = False
        
        # Check if single command or list
        if isinstance(stim_command, list) and len(stim_command) > 0:
            self.single_command = not isinstance(stim_command[0][0], list)
        else:
            self.single_command = True
            
        # Validate tag format
        self.single_tag = isinstance(tag, str)
        if not self.single_tag and self.single_command:
            raise ValueError("Tag must be a string if stim_command is single")
        if not self.single_tag and not self.single_command and len(tag) != len(stim_command):
            raise ValueError("Tag list must match stim_command length")
    
    def configure_from_experiment(self, experiment):
        """Auto-configure from experiment."""
        super().configure_from_experiment(experiment)
        

            
        self.verbose = experiment.get_param('verbose', False)

    def customize_environment_params(self, env_params: dict) -> dict:
        """Customize environment parameters."""
        env_params['stim_electrodes'] = self.stim_electrodes
        return env_params
    
    def run(self, experiment) -> Dict[str, Any]:
        """Execute frequency stimulation."""
        if self.verbose:
            print(f"   Stimulating at {self.stim_freq} Hz for {self.duration}s")

        # Prepare commands
        if not self.single_command:
            stim_commands = self.stim_command.copy()
            tags = self.tag.copy() if not self.single_tag else [self.tag] * len(stim_commands)
        
        # Run stimulation
        done = False
        stim_count = 0
        time_between_stims = 1 / self.stim_freq
        self.start_time = time.perf_counter()
        
        while not done:
            if self.time_elapsed() >= self.duration:
                break
            if self.time_elapsed() >= time_between_stims * stim_count:
                # Get command and tag
                if not self.single_command:
                    if not stim_commands:
                        done = True
                        break
                    stim_cmd = stim_commands.pop(0)
                    tag = tags.pop(0)
                else:
                    stim_cmd = self.stim_command
                    tag = self.tag
                
                # Stimulate
                _, done = self.env.step(action=stim_cmd, tag=tag)
                stim_count += 1
                
                if self.verbose and stim_count % 10 == 0:
                    print(f"      Delivered {stim_count} stimulations")
            
            else:
                buffer_size = None
                if getattr(self.env, "is_replay", False):
                    elapsed = self.time_elapsed()
                    next_deadline = min(
                        self.duration,
                        time_between_stims * stim_count,
                    )
                    frames_to_deadline = int(np.floor(
                        (next_deadline - elapsed)
                        * self.env.replay_source.sampling_hz
                        + 1e-9
                    ))
                    buffer_size = max(
                        1,
                        min(self.env.replay_source.chunk_frames, frames_to_deadline),
                    )
                obs, done = self.env.step(buffer_size=buffer_size)
            
            # Check duration
            if self.time_elapsed() >= self.duration:
                done = True
        
        return {
            'stim_file': self.env.save_file,
            'stim_count': stim_count
        }
    
    def info(self) -> Dict[str, Any]:
        info = super().info()
        info.update({
            'stim_freq': self.stim_freq,
            'duration': self.duration,
            'single_command': self.single_command
        })
        return info


class CartPolePhaseV3(PhaseV3):
    """
    Phase for CartPole game with neural control.
    
    Requires:
        - sensory_neurons: Indices of neurons for sensory input
        - motor_neurons: Channel numbers for motor readout
        - training_neurons: Indices of neurons for training stimulation
    
    Provides:
        - game_log_file: Path to game log CSV
        - reward_log_file: Path to reward log CSV
        - pattern_log_file: Path to pattern log CSV
        - total_episodes: Number of episodes completed
        - final_rewards: List of rewards from all episodes
    """
    
    inputs = ['sensory_neurons', 'motor_neurons', 'training_neurons']
    outputs = ['game_log_file', 'reward_log_file', 'pattern_log_file',
                'total_episodes', 'final_rewards']
    
    def __init__(self, n_episodes: int = 10,
                 amp_mv: int = 400,
                 phase_width: int = 100,
                 read_period_ms: int = 200,
                 train_period_ms: int = 200,
                 wait_period_ms: int = 400,
                 trainer = None,
                 artifact_removal: bool = False,
                 continuous: bool = True,
                 assistive: float = 0.0,
                 minibatch_size: int = 5,
                 spike_thresh: List[float] = None,
                 max_time: float = np.inf,
                 normalization: float = 1.0,
                 force_train: bool = False,
                 name: str = None,
                 suffix: str = "_cartpole",
                 recording_tag: str = "cartpole"):
        
        super().__init__(name or "CartPolePhase", suffix, recording_tag=recording_tag)
        
        self.n_episodes = n_episodes
        self.amp_mv = amp_mv
        self.phase_width = phase_width
        self.read_period_ms = read_period_ms
        self.train_period_ms = train_period_ms
        self.wait_period_ms = wait_period_ms
        self.trainer = trainer
        self.artifact_removal = artifact_removal
        self.continuous = continuous
        self.assistive = assistive
        self.minibatch_size = minibatch_size
        self.spike_thresh = spike_thresh or [-3.1, -20]
        self.max_time = max_time
        self.normalization = normalization
        self.force_train = force_train
        self.verbose = False
        
        # Will be set from requirements
        self.sensory_neurons = None
        self.motor_neurons = None
        self.training_neurons = None
    
    def configure_from_experiment(self, experiment):
        """Auto-configure from experiment data and config."""
        super().configure_from_experiment(experiment)
        
        # These should be auto-populated from requirements
        # But we can also get them from config as backup
        if self.sensory_neurons is None:
            self.sensory_neurons = experiment.get_param('sensory_neurons', [])
        if self.motor_neurons is None:
            self.motor_neurons = experiment.get_param('motor_channels', [])
        if self.training_neurons is None:
            self.training_neurons = experiment.get_param('training_neurons', [])
            
        self.verbose = experiment.get_param('verbose', False)
        
        # Convert to numpy arrays
        self.sensory_neurons = np.array(self.sensory_neurons)
        self.motor_neurons = np.array(self.motor_neurons)
        self.training_neurons = np.array(self.training_neurons)
    
    def needs_environment(self) -> bool:
        """CartPole needs Maxwell environment."""
        return True
    
    def close_environment_after(self) -> bool:
        """Keep environment open for potential follow-up phases."""
        return False 
