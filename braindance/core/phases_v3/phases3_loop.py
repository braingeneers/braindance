"""
Closed-Loop Phases for Experiment Framework V3

These phases implement closed-loop stimulation based on neural activity.
"""

import numpy as np
import time
from typing import Dict, List, Any
from collections import deque
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from .phase_base_v3 import PhaseV3
from braindance.core.phases_v3.experiment_v3 import Experiment
from braindance.analysis.data_loader import convert_uint16_maxwell
from braindance.core.trainer import ContextualTrainer

from spikelab import SpikeData


class ClosedLoopPhaseV3(PhaseV3):
    """
    Basic closed-loop phase that maps input neuron activity to output neuron stimulation.

    Requires:
        - selected_neuron_pair: Tuple of (input_neuron_id, output_neuron_id)
        - channels_per_neuron: Channel mapping for neurons
        - electrodes_per_neuron: Electrode mapping for neurons

    Provides:
        - closed_loop_file: Path to saved data
        - stim_count: Number of stimulations delivered
        - spike_count: Number of spikes detected
    """

    inputs = ["selected_pair", "rt_sort_object", "electrodes_per_neuron"]
    outputs = ["closed_loop_file", "stim_count", "spike_count"]

    def __init__(
        self,
        duration: int = 600,
        amp_mv: int = 400,
        phase_width: int = 100,
        refractory_ms: float = 2.0,
        delay_ms: float = 1.0,
        name: str = None,
        suffix: str = "_closed_loop",
        rt_sort_path: str = None,
        detection_model_path: str | None = None,
        verbose: int = 1,
        timing_debug: bool = False,
        timing_interval: int = 100,
        use_artifact_removal: bool = True,
        art_removal_N: int = 60,
        art_removal_min_val: float = -100,
        art_removal_max_val: float = 100,
        art_removal_spike_thresh: List[float] = None,
        use_numba: bool = True,
        recording_tag: str = "closed_loop",
    ):
        """
        Initialize closed-loop phase.

        Args:
            duration: Duration in seconds
            amp_mv: Stimulation amplitude in mV
            phase_width: Stimulation phase width
            detection_threshold: Spike detection threshold (negative)
            refractory_ms: Refractory period after spike detection
            delay_ms: Delay between detection and stimulation
            timing_debug: Enable timing diagnostics
            timing_interval: Print timing stats every N steps
            use_artifact_removal: Enable linear artifact removal
            art_removal_N: Half-window size for artifact removal
            art_removal_min_val: Minimum threshold for artifact detection
            art_removal_max_val: Maximum threshold for artifact detection
            art_removal_spike_thresh: Spike detection thresholds [min, max]
            recording_tag: Tag for the recording subdirectory
        """
        super().__init__(name or "ClosedLoopPhase", suffix, recording_tag=recording_tag)
        from braindance import get_rt_sort_path

        self.duration = duration
        self.amp_mv = amp_mv
        self.phase_width = phase_width
        self.refractory_ms = refractory_ms
        self.delay_ms = delay_ms
        self.rt_sort_path = rt_sort_path
        if detection_model_path is None:
            self.detection_model_path = get_rt_sort_path()
        else:
            self.detection_model_path = detection_model_path
        self.use_numba = use_numba
        # Timing diagnostics
        self.timing_debug = timing_debug
        self.timing_interval = timing_interval

        # Artifact removal parameters
        self.use_artifact_removal = use_artifact_removal
        self.art_removal_N = art_removal_N
        self.art_removal_min_val = art_removal_min_val
        self.art_removal_max_val = art_removal_max_val
        if art_removal_spike_thresh is None:
            self.art_removal_spike_thresh = [-3.5, 20]
        else:
            self.art_removal_spike_thresh = art_removal_spike_thresh

        # Will be auto-populated
        self.selected_pair = None
        self.channels_per_neuron = None
        self.electrodes_per_neuron = None
        self.verbose = verbose
        self.artifact_remover = None

    def configure_from_experiment(self, experiment: Experiment):
        """Configure from experiment data."""
        super().configure_from_experiment(experiment)

        # Extract neuron info
        self.input_neuron_id = self.selected_pair[0]
        self.output_neuron_id = self.selected_pair[1]

        # TODO: Fix the way we are getting these electrodes...

        # These are the indices from the sorter
        # We need to instantiate the sorter object
        self.rt_sort = experiment.data.rt_sort_object
        self.rt_sort.set_model(self.detection_model_path)

        # Subset by the number of neurons
        # self.rt_sort = self.rt_sort.select_seqs(list(np.arange(40))) # Selects first 20 for now...

        channels = [
            int(ch) for ch in self.rt_sort.get_seq_root_elecs()
        ]  # These are actually the channels
        # Ensure we have a mapping
        experiment.load_mapping()

        # Convert to electrodes
        self.electrodes = experiment.mapping.get_electrodes(channels=channels)

        # Our stim electrode is just the input neuron's electrode
        stim_electrode = self.electrodes[self.input_neuron_id]
        self.stim_electrodes = [stim_electrode]
        self.stim_inds = np.arange(len(self.stim_electrodes))

        print(
            f"   Configured closed-loop: neuron {self.input_neuron_id} "
            f"-> neuron {self.output_neuron_id} "
        )

    def customize_environment_params(self, env_params: dict) -> dict:
        """Customize environment parameters."""
        env_params["stim_electrodes"] = self.stim_electrodes
        return env_params

    def _add_to_buffer(self, new_data, n_channels):
        """Add new data to the sliding window buffer."""
        new_data = np.asarray(new_data)
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)
        if new_data.shape[1] != n_channels:
            new_data = new_data[:, :n_channels]

        n_new_samples = new_data.shape[0]
        window_buffer_size = self.data_buffer.shape[0]

        if n_new_samples >= window_buffer_size:
            # If new data is larger than buffer, just keep the last window_buffer_size samples
            self.data_buffer[:] = new_data[-window_buffer_size:]
            self.buffer_write_pos = 0
        else:
            # Add new data to buffer with circular wrapping
            end_pos = self.buffer_write_pos + n_new_samples
            if end_pos <= window_buffer_size:
                # Simple case: no wraparound needed
                self.data_buffer[self.buffer_write_pos : end_pos] = new_data
            else:
                # Wraparound case
                split_point = window_buffer_size - self.buffer_write_pos
                self.data_buffer[self.buffer_write_pos :] = new_data[:split_point]
                self.data_buffer[: end_pos - window_buffer_size] = new_data[
                    split_point:
                ]

            self.buffer_write_pos = (
                self.buffer_write_pos + n_new_samples
            ) % window_buffer_size

    def _get_window(self):
        """Get the current window from the buffer in chronological order."""
        # The buffer is circular, so we need to reorder it chronologically
        if self.buffer_write_pos == 0:
            return self.data_buffer.copy()
        else:
            return np.vstack(
                [
                    self.data_buffer[self.buffer_write_pos :],
                    self.data_buffer[: self.buffer_write_pos],
                ]
            )

    def run(self, experiment) -> Dict[str, Any]:
        """Run closed-loop stimulation."""
        print(f"   Running closed-loop for {self.duration}s")

        # Initialize counters
        spike_count = 0
        stim_count = 0

        # Windowing parameters for rt_sort
        # 10ms windows with 4.5ms overlap as requested
        window_size_ms = 20.0
        overlap_ms = 10
        sample_rate_khz = 20.0  # 20kHz sampling rate

        window_size_samples = int(
            window_size_ms * sample_rate_khz
        )  # 200 samples for 10ms
        overlap_samples = int(overlap_ms * sample_rate_khz)  # 90 samples for 4.5ms
        step_size_samples = (
            window_size_samples - overlap_samples
        )  # 110 samples for 5.5ms step

        # Environment buffer size (smaller chunks for low latency)
        buffer_size = int(2.0 * sample_rate_khz)  # 2ms chunks (40 samples)

        # Sliding window buffer for accumulating data
        self.data_buffer = None
        self.buffer_write_pos = 0

        # Timing diagnostics
        step_count = 0
        step_times = []
        last_step_time = None
        expected_step_time_ms = buffer_size / sample_rate_khz  # Convert samples to ms

        obs, _ = self.env.step(
            buffer_size=buffer_size
        )  # Obs is list of lists (time, channels)
        obs = np.array(obs)
        # Get shape of the obs
        n_channels = obs.shape[1]
        time_chunk = obs.shape[0]
        if self.verbose:
            print(f"   Obs shape: {obs.shape}")
            print(f"   Time chunk: {time_chunk}")
            print(f"   N channels: {n_channels}")
            print(f"   Window size: {window_size_samples} samples ({window_size_ms}ms)")
            print(f"   Overlap: {overlap_samples} samples ({overlap_ms}ms)")
            print(
                f"   Step size: {step_size_samples} samples ({step_size_samples / sample_rate_khz}ms)"
            )

        # Initialize sliding window buffer
        self.data_buffer = np.zeros((window_size_samples, n_channels), dtype=obs.dtype)
        self.buffer_write_pos = 0

        # Clear buffer
        # self.env.clear_buffer()
        # warmup
        if self.verbose:
            print("   Warming up RTSort...")
        for _ in range(200):
            obs, _ = self.env.step(buffer_size=buffer_size)
            if obs is not None:
                obs = convert_uint16_maxwell(obs)

                spikes = self.rt_sort.running_sort(obs, use_numba=self.use_numba)
                self._add_to_buffer(obs, n_channels)

        sorted_spikes = []
        done = False
        samples_since_last_process = 0
        if self.verbose:
            print("Completed!")

        stim_command = (self.stim_inds, self.amp_mv, self.phase_width)

        self.start_time = time.perf_counter()

        while not done:
            # Timing measurement
            if self.timing_debug:
                current_time = time.perf_counter()
                if last_step_time is not None:
                    step_time_ms = (current_time - last_step_time) * 1000
                    step_times.append(step_time_ms)
                last_step_time = current_time
                step_count += 1

            # Step environment
            obs, done = self.env.step(buffer_size=buffer_size)  # 2ms chunk

            # Process observation
            if obs is not None:
                # Scale
                obs = convert_uint16_maxwell(obs)

                # Add to sliding window buffer
                self._add_to_buffer(obs, n_channels)
                samples_since_last_process += obs.shape[0]

                # Process window when we have accumulated enough new data
                if samples_since_last_process >= step_size_samples:
                    # Get the current 10ms window
                    window_data = self._get_window()

                    # Process through rt_sort
                    spikes = self.rt_sort.running_sort(
                        window_data,
                        use_numba=self.use_numba,
                        latest_frame=self.env.latest_frame,
                    )
                    sorted_spikes.extend(spikes)

                    # Reset counter
                    samples_since_last_process = 0
                    # Check for output neuron spikes and stimulate if found
                    found_output_spike = False
                    for spike in spikes:
                        if spike[0] == self.output_neuron_id:
                            found_output_spike = True
                            spike_count += 1
                            break

                    if found_output_spike:
                        _, _ = self.env.step(
                            action=stim_command, buffer_size=buffer_size
                        )
                        stim_count += 1

            # Print timing stats periodically
            if (
                self.timing_debug
                and step_count % self.timing_interval == 0
                and len(step_times) > 0
            ):
                recent_times = step_times[-self.timing_interval :]
                avg_time = np.mean(recent_times)
                std_time = np.std(recent_times)
                if self.verbose:
                    print(
                        f"   Step {step_count}: avg={avg_time:.2f}ms (±{std_time:.2f}), "
                        f"expected={expected_step_time_ms:.2f}ms, "
                        f"ratio={avg_time / expected_step_time_ms:.2f}"
                    )

            # Check duration
            if self.time_elapsed() > self.duration:
                done = True
                continue

        # Final timing summary
        if self.timing_debug and len(step_times) > 0:
            avg_time = np.mean(step_times)
            std_time = np.std(step_times)
            print(
                f"   Timing summary: {len(step_times)} steps, "
                f"avg={avg_time:.2f}ms (±{std_time:.2f}), "
                f"expected={expected_step_time_ms:.2f}ms"
            )

        print(
            f"   Completed: {spike_count} spikes detected, {stim_count} stims delivered"
        )

        # Make spikedata
        sd = SpikeData.from_events(sorted_spikes)
        return {
            "closed_loop_file": self.env.save_file,
            "stim_count": stim_count,
            "spike_count": spike_count,
            "sorted_spikedata": sd,
        }


class PolicyNetwork(nn.Module):
    """Simple policy network for REINFORCE algorithm."""

    def __init__(self, input_size, hidden_size=32):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)  # Single output for continuous control
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))  # Output in [-1, 1]
        return x


class CartPolePhase(PhaseV3):
    """
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

    """

    inputs = ["rt_sort_object", "recording_file"]
    outputs = ["total_episodes", "sorted_spikedata", "final_policy_state"]

    # Neuron allocation constants
    MIN_SEQUENCES = 8       # Minimum sequences needed (2 sensory + 4 motor + 2 training)
    N_SENSORY = 2           # Always 2 sensory neurons
    MIN_MOTOR = 4           # Minimum motor neurons
    MIN_TRAINING = 2        # Minimum training neurons
    MAX_TRAINING = 6        # Maximum training neurons

    @staticmethod
    def allocate_neurons(n_sequences, n_sensory=2, min_motor=4, min_training=2, max_training=6):
        """
        Allocate detected sequences into sensory, motor, and training groups.

        Allocation priority:
            1. Always n_sensory for sensory (first indices)
            2. At least min_training for training (up to max_training)
            3. Everything else goes to motor (at least min_motor)

        Parameters
        ----------
        n_sequences : int
            Total number of detected sequences
        n_sensory : int
            Number of sensory neurons (always fixed)
        min_motor : int
            Minimum number of motor neurons
        min_training : int
            Minimum number of training neurons
        max_training : int
            Maximum number of training neurons

        Returns
        -------
        tuple
            (sensory_neurons, motor_neurons, training_neurons) as lists of indices

        Raises
        ------
        ValueError
            If n_sequences < n_sensory + min_motor + min_training
        """
        min_total = n_sensory + min_motor + min_training
        if n_sequences < min_total:
            raise ValueError(
                f"Need at least {min_total} sequences "
                f"({n_sensory} sensory + {min_motor} motor + {min_training} training), "
                f"but only {n_sequences} detected."
            )

        remaining = n_sequences - n_sensory
        n_training = max(min_training, min(max_training, remaining - min_motor))
        n_motor = remaining - n_training

        sensory = list(range(n_sensory))
        motor = list(range(n_sensory, n_sensory + n_motor))
        training = list(range(n_sensory + n_motor, n_sequences))

        return sensory, motor, training

    def __init__(
        self,
        sensory_neurons: list | None = None,
        motor_neurons: list | None = None,
        training_neurons: list | None = None,
        amp_mv: int = 400,
        phase_width: int = 100,
        read_period_ms: int = 200,
        train_period_ms: int = 200,
        wait_period_ms: int = 400,
        n_episodes: int = 10,
        trainer=None,
        trainer_type: str | None = None,
        artifact_removal=False,
        continuous=True,
        minibatch_size=5,
        verbose=False,
        max_time=np.inf,
        normalization=1,
        suffix: str = "_cartpole",
        rt_sort_path: str = None,
        detection_model_path: str | None = None,
        use_numba: bool = True,
        learning_rate: float = 0.01,
        gamma: float = 0.99,
        policy_hidden_size: int = 32,
        assistive: float = 0.0,
        use_contextual_trainer: bool = False,
        contextual_trainer_kwargs: dict | None = None,
        recording_tag: str = "cartpole",
        render_mode: str | None = "human",
    ):

        super().__init__("CartPolePhase", suffix, recording_tag=recording_tag)

        import gymnasium as gym
        from braindance.games import cartpole_continuous
        from braindance import get_rt_sort_path

        self.continuous = continuous

        if continuous:
            self.game_env = cartpole_continuous.CartPoleContinuousEnv(
                render_mode=render_mode
            )
            pass
        else:
            self.game_env = gym.make("CartPole-v1", render_mode=render_mode)

        self.read_period_ms = read_period_ms
        self.train_period_ms = train_period_ms
        self.wait_period_ms = wait_period_ms
        self.n_episodes = n_episodes
        self.episode = 0
        self.max_time = max_time
        self.minibatch_size = minibatch_size
        self.normalization = normalization
        self.assistive = assistive

        self.suffix = suffix

        # IO
        self.sensory_neurons = (
            np.array(sensory_neurons, dtype=int)
            if sensory_neurons is not None
            else None
        )
        self.motor_neurons = (
            np.array(motor_neurons, dtype=int) if motor_neurons is not None else None
        )
        self.training_neurons = (
            np.array(training_neurons, dtype=int)
            if training_neurons is not None
            else None
        )

        self.amp_mv, self.phase_width = amp_mv, phase_width

        self.sensory_stim_Hz = np.zeros(
            len(self.sensory_neurons) if self.sensory_neurons is not None else 0
        )
        self.trainer = trainer
        self.trainer_type = trainer_type  # For deferred trainer creation
        if self.trainer:
            self.trainer.phase = self

        self.last_action_ind = None
        self.last_action_inds = deque(maxlen=self.minibatch_size)
        self.game_obs = None
        self.episode_reward = None
        self.episode_reward_change = None

        self.verbose = verbose

        # RT-sort parameters
        self.rt_sort_path = rt_sort_path
        if detection_model_path is None:
            self.detection_model_path = get_rt_sort_path()
        else:
            self.detection_model_path = detection_model_path
        self.use_numba = use_numba

        # REINFORCE parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.policy_hidden_size = policy_hidden_size

        # Initialize policy network and optimizer (will be set up after knowing motor neuron count)
        self.policy_net = None
        self.optimizer = None
        self.saved_log_probs = []
        self.rewards = []

        # RT-sort related
        self.rt_sort = None
        self.electrodes = None

        self.motor_spike_count = np.zeros(
            len(self.motor_neurons) if self.motor_neurons is not None else 0
        )  # Stores the spike count for each motor neuron
        self.motor_spike_rate = np.zeros(
            len(self.motor_neurons) if self.motor_neurons is not None else 0
        )
        self.state = "read"  # 'read', 'train', 'game', or 'wait'

        self.predicted_time = (
            (read_period_ms + train_period_ms) * 20 * n_episodes / 1000
        )  # x20 due to 20 steps of the game avg

        # Contextual trainer settings
        self.use_contextual_trainer = use_contextual_trainer
        self.contextual_trainer_kwargs = contextual_trainer_kwargs or {}
        self.episode_length = 0  # Track episode length for contextual trainer
        self.last_policy_grad_magnitude = 0.0  # Track gradient magnitude

    def configure_from_experiment(self, experiment: Experiment):
        """Configure from experiment data."""
        super().configure_from_experiment(experiment)

        # Set up RT-sort
        self.rt_sort = experiment.data.rt_sort_object
        self.rt_sort.set_model(self.detection_model_path)

        channels = [
            int(ch) for ch in self.rt_sort.get_seq_root_elecs()
        ]  # These are actually the channels

        # Auto-allocate neurons if any are None
        if (
            self.sensory_neurons is None
            or self.motor_neurons is None
            or self.training_neurons is None
        ):
            n_sequences = len(channels)
            print(f"  Auto-allocating {n_sequences} detected sequences into neuron groups...")
            sensory, motor, training = self.allocate_neurons(n_sequences)
            if self.sensory_neurons is None:
                self.sensory_neurons = sensory
            if self.motor_neurons is None:
                self.motor_neurons = motor
            if self.training_neurons is None:
                self.training_neurons = training

        # Ensure they are np arrays of ints
        self.sensory_neurons = np.array(self.sensory_neurons, dtype=int)
        self.motor_neurons = np.array(self.motor_neurons, dtype=int)
        self.training_neurons = np.array(self.training_neurons, dtype=int)

        print("=" * 20)
        print("Sensory neurons: ", self.sensory_neurons)
        print("Motor neurons: ", self.motor_neurons)
        print("Training neurons: ", self.training_neurons)
        print("=" * 20)

        # Ensure we have a mapping
        experiment.load_mapping()

        # Convert to electrodes
        self.electrodes = experiment.mapping.get_electrodes(channels=channels)

        # Create deferred trainer now that we know training neuron indices
        if self.trainer is None and self.trainer_type is not None and self.trainer_type != 'none':
            self._create_deferred_trainer()

        # Set up stimulation electrodes for sensory neurons
        if self.sensory_neurons is not None:
            # Convert electrodes to numpy array if it isn't already
            electrodes_array = np.array(self.electrodes)
            stim_electrodes = electrodes_array[
                self.sensory_neurons
            ]  # This is indexed by rt-sort ID
            self.stim_electrodes = stim_electrodes.tolist()
            self.stim_inds = np.arange(len(self.stim_electrodes))
            # Re-initialize sensory stim Hz (may have been size 0 if neurons were None at init)
            self.sensory_stim_Hz = np.zeros(len(self.sensory_neurons))

        # Initialize motor neuron arrays with correct size
        if self.motor_neurons is not None:
            self.motor_spike_count = np.zeros(len(self.motor_neurons))
            self.motor_spike_rate = np.zeros(len(self.motor_neurons))

            # Initialize policy network now that we know the motor neuron count
            self.policy_net = PolicyNetwork(
                len(self.motor_neurons)
            )  # , self.policy_hidden_size)
            self.optimizer = optim.Adam(
                self.policy_net.parameters(), lr=self.learning_rate
            )

        print(
            f"   Configured cartpole: sensory neurons {self.sensory_neurons} "
            f"-> motor neurons {self.motor_neurons} "
            f"-> training neurons {self.training_neurons} "
        )
        print(
            f"   Policy network initialized with input size: {len(self.motor_neurons) if self.motor_neurons is not None else 0}"
        )

    def _create_deferred_trainer(self):
        """Create trainer after neuron allocation is known (deferred from __init__)."""
        from braindance.core.trainer import TetanusTrainer, ContextualTrainer, generate_tetanus_pattern

        training_neurons = self.training_neurons.tolist()
        n_neurons = len(training_neurons)
        amp_mv = self.amp_mv
        phase_width = self.phase_width

        # Generate stimulation patterns from training neurons
        patterns = []
        if n_neurons >= 3:
            patterns.append(generate_tetanus_pattern(
                training_neurons[:3], stim_count=3, delay_ms=5,
                amp_mv=amp_mv, pulse_width=phase_width
            ))
            patterns.append(generate_tetanus_pattern(
                training_neurons[-3:], stim_count=3, delay_ms=5,
                amp_mv=amp_mv, pulse_width=phase_width
            ))
            if n_neurons >= 5:
                patterns.append(generate_tetanus_pattern(
                    training_neurons[::2][:3], stim_count=3, delay_ms=5,
                    amp_mv=amp_mv, pulse_width=phase_width
                ))
        else:
            patterns.append(generate_tetanus_pattern(
                training_neurons, stim_count=min(3, n_neurons), delay_ms=5,
                amp_mv=amp_mv, pulse_width=phase_width
            ))

        freq = 30  # Stimulation frequency in Hz

        if self.trainer_type == 'tetanus':
            self.trainer = TetanusTrainer(
                choices=patterns, freqs=freq,
                start_value=10, learning_rate=0.1,
                floor=10, discount_factor=0.3
            )
            self.use_contextual_trainer = False
        elif self.trainer_type == 'contextual':
            self.trainer = ContextualTrainer(
                choices=patterns, freqs=freq,
                hidden_sizes=[64, 32], learning_rate=0.01,
                gamma=0.95, baseline_window=20,
                normalize_context=True, no_stim_bias=0.0
            )
            self.use_contextual_trainer = True
        else:
            raise ValueError(f"Unknown trainer type: {self.trainer_type}")

        self.trainer.phase = self
        print(f"  Created {self.trainer_type} trainer with {len(patterns)} patterns "
              f"using {n_neurons} training neurons")

    def customize_environment_params(self, env_params: dict) -> dict:
        """Customize environment parameters."""
        env_params["stim_electrodes"] = self.stim_electrodes
        return env_params

    def _add_to_buffer(self, new_data, n_channels):
        """Add new data to the sliding window buffer."""
        new_data = np.asarray(new_data)
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)
        if new_data.shape[1] != n_channels:
            new_data = new_data[:, :n_channels]

        n_new_samples = new_data.shape[0]
        window_buffer_size = self.data_buffer.shape[0]

        if n_new_samples >= window_buffer_size:
            # If new data is larger than buffer, just keep the last window_buffer_size samples
            self.data_buffer[:] = new_data[-window_buffer_size:]
            self.buffer_write_pos = 0
        else:
            # Add new data to buffer with circular wrapping
            end_pos = self.buffer_write_pos + n_new_samples
            if end_pos <= window_buffer_size:
                # Simple case: no wraparound needed
                self.data_buffer[self.buffer_write_pos : end_pos] = new_data
            else:
                # Wraparound case
                split_point = window_buffer_size - self.buffer_write_pos
                self.data_buffer[self.buffer_write_pos :] = new_data[:split_point]
                self.data_buffer[: end_pos - window_buffer_size] = new_data[
                    split_point:
                ]

            self.buffer_write_pos = (
                self.buffer_write_pos + n_new_samples
            ) % window_buffer_size

    def _get_window(self):
        """Get the current window from the buffer in chronological order."""
        # The buffer is circular, so we need to reorder it chronologically
        if self.buffer_write_pos == 0:
            return self.data_buffer.copy()
        else:
            return np.vstack(
                [
                    self.data_buffer[self.buffer_write_pos :],
                    self.data_buffer[: self.buffer_write_pos],
                ]
            )

    def set_sensory_signal(self, game_env_obs):
        """
        Maps the game observation to the sensory neurons.
        This observation is of the form:
        ndarray with shape (4,):
            - Cart Position, Cart Velocity, Pole Angle, Pole Velocity At Tip

        We will map the pole angle to the 0th and 1st sensory neurons

        """
        pole_angle = game_env_obs[2]
        stim_bias = 0.15  # .15#.2 # was .4, .1
        stim_scale = 7  # 9#15#8# was 7,  20

        self.sensory_stim_Hz = np.zeros(len(self.sensory_neurons))

        self.sensory_stim_Hz[0] = ((-np.sin(pole_angle) + stim_bias) * stim_scale) ** 2
        self.sensory_stim_Hz[1] = ((np.sin(pole_angle) + stim_bias) * stim_scale) ** 2
        if self.verbose:
            # print(f'Stim:Hz: \t{self.sensory_stim_Hz}')
            print(
                f"Stim:Hz: \t{self.sensory_stim_Hz[0]:.2f} || {self.sensory_stim_Hz[1]:.2f}"
            )

    def get_motor_signal(self, moving_avg=0.2):
        """
        Readout from the motor neurons using policy network
        Returns action and saves log probability for REINFORCE
        """
        # Update moving average of spike rates
        self.motor_spike_rate = (
            moving_avg * self.motor_spike_rate
            + (1 - moving_avg) * self.motor_spike_count
        )

        # Normalize spike rates if needed
        # if isinstance(self.normalization, np.ndarray) and len(self.normalization) >= len(self.motor_neurons):
        #     normalized_rates = self.motor_spike_rate / self.normalization[:len(self.motor_neurons)]
        # else:
        #     normalized_rates = self.motor_spike_rate / self.normalization

        # Convert to tensor and pass through policy network
        state_tensor = torch.FloatTensor(self.motor_spike_rate).unsqueeze(
            0
        )  # Add batch dimension

        # Get action from policy network
        with torch.no_grad():  # No gradients needed for action selection
            action_value = self.policy_net(state_tensor)

        # For training, we need to compute log probability
        if self.state == "read":  # Only save log probs during actual gameplay
            # Re-compute with gradients for training
            action_output = self.policy_net(state_tensor)

            # For continuous action, we can use a Gaussian policy
            # For now, we'll just save the direct output as log prob (simplified)
            self.saved_log_probs.append(action_output)

        action = action_value.item()  # Convert to scalar

        if self.verbose:
            rates_str = " || ".join([f"{rate:.2f}" for rate in self.motor_spike_rate])
            print(f"Spike rates: {rates_str} \t[Action]: {action:.3f}")

        if self.continuous:
            return np.clip(action, -1, 1)  # Ensure action is in valid range
        else:
            # For discrete action space
            return 0 if action < 0 else 1

    def get_training_signal(
        self, train_Hz=30, stim_count=3, delay_ms=5, use_change=True
    ):
        """
        The stimulation on the training neurons.

        For ContextualTrainer: computes context and gets action based on state.
        For TetanusTrainer: uses the original multi-armed bandit approach.

        Returns
        -------
        tuple
            (train_action, train_Hz) where train_action is None for no-stim
        """
        if self.trainer:
            # Update last reward
            if (
                self.episode_reward is not None
                and len(self.last_action_inds) == self.minibatch_size
            ):
                if use_change:
                    self.trainer.update_values(
                        self.episode_reward_change, self.last_action_inds[-1]
                    )
                else:
                    self.trainer.update_values(
                        self.episode_reward, self.last_action_inds[-1]
                    )

            # Handle contextual trainer differently
            if self.use_contextual_trainer and isinstance(
                self.trainer, ContextualTrainer
            ):
                # Compute context for contextual trainer
                context = self.trainer.compute_context(
                    motor_spike_rates=self.motor_spike_rate,
                    episode_reward=self.episode_reward if self.episode_reward else 0,
                    episode_length=self.episode_length,
                )
                train_action, train_Hz, self.last_action_ind = self.trainer.get_action(
                    context=context
                )

                # Record episode reward for baseline
                if self.episode_reward is not None:
                    self.trainer.record_episode_reward(self.episode_reward)
            else:
                # Original TetanusTrainer behavior
                train_action, train_Hz, self.last_action_ind = self.trainer.get_action()

            self.last_action_inds.append(self.last_action_ind)
            return train_action, train_Hz

        neuron_order = np.random.choice(
            self.training_neurons, size=stim_count, replace=False
        )
        train_action = []
        for n in neuron_order:
            train_action.append(("stim", [n], self.amp_mv, self.phase_width))
            if n != neuron_order[-1]:
                train_action.append(("delay", delay_ms))

        return train_action, train_Hz

    def run(self, experiment):
        done = False

        state, inf = self.game_env.reset()

        just_entered = True
        reward = 0
        total_reward = 0
        rewards = []

        # RT-sort setup (similar to ClosedLoopPhaseV3)
        window_size_ms = 20.0
        overlap_ms = 4.5
        sample_rate_khz = 20.0

        window_size_samples = int(window_size_ms * sample_rate_khz)
        overlap_samples = int(overlap_ms * sample_rate_khz)
        step_size_samples = window_size_samples - overlap_samples

        # Environment buffer size
        buffer_size = int(2.0 * sample_rate_khz)  # 2ms chunks

        # Get initial observation to determine channel count
        obs, _ = self.env.step(buffer_size=buffer_size)
        obs = np.array(obs)
        n_channels = obs.shape[1]

        # Initialize sliding window buffer
        self.data_buffer = np.zeros((window_size_samples, n_channels), dtype=obs.dtype)
        self.buffer_write_pos = 0

        # Warmup RT-sort
        if self.verbose:
            print("   Warming up RTSort...")
        # self.env.clear_buffer()
        for _ in range(100):
            obs, _ = self.env.step(buffer_size=buffer_size)
            if obs is not None:
                obs = convert_uint16_maxwell(obs)
                spikes = self.rt_sort.running_sort(obs, use_numba=self.use_numba)
                self._add_to_buffer(obs, n_channels)

        if self.verbose:
            print("   Warmup completed!")

        # Game log
        game_log = open(self.env.save_file + "_game_log.csv", "w")
        # Header for csv
        game_logger = csv.writer(game_log)
        game_logger.writerow(
            ["time", "pole_angle", "reward", "action", "spike_rates", "state"]
        )

        reward_log = open(self.env.save_file + "_reward_log.csv", "w")
        # Header for csv
        reward_logger = csv.writer(reward_log)
        reward_logger.writerow(["time", "episode", "reward"])

        # Pattern log
        pattern_log = open(self.env.save_file + "_pattern_log.csv", "w")
        # Header for csv
        pattern_logger = csv.writer(pattern_log)
        pattern_logger.writerow(["time", "pattern", "reward", "probs", "vals"])

        self.start_time = time.perf_counter()
        state_timer = self.time_elapsed()
        samples_since_last_process = 0

        while not done:
            # ~~~~~~~~~~~ Read phase Logic ~~~~~~~~~~~
            if self.state == "read":
                if just_entered:
                    state_timer = self.time_elapsed()
                    just_entered = False
                    self.motor_spike_count = np.zeros(len(self.motor_neurons))

                if self.time_elapsed() - state_timer > self.read_period_ms / 1000:
                    self.state = "game"
                    just_entered = True
                    continue

                # Neurons to stimulate
                active_sensory_neurons = np.where(
                    self.env.stim_dts[self.sensory_neurons] > 1 / self.sensory_stim_Hz
                )[0]
                # Stim the neurons at the correct frequency
                if len(active_sensory_neurons) > 0:
                    obs, done = self.env.step(
                        action=(
                            self.sensory_neurons[active_sensory_neurons],
                            self.amp_mv,
                            self.phase_width,
                        ),
                        tag="sensory",
                        buffer_size=buffer_size,
                    )
                else:
                    # No action
                    obs, done = self.env.step(buffer_size=buffer_size)

                # Process observation with RT-sort
                if obs is not None:
                    # Scale
                    obs = convert_uint16_maxwell(obs)

                    # Add to sliding window buffer
                    self._add_to_buffer(obs, n_channels)
                    samples_since_last_process += obs.shape[0]

                    # Process window when we have accumulated enough new data
                    if samples_since_last_process >= step_size_samples:
                        # Get the current window
                        window_data = self._get_window()

                        # Process through rt_sort
                        spikes = self.rt_sort.running_sort(
                            window_data,
                            use_numba=self.use_numba,
                            latest_frame=self.env.latest_frame,
                        )

                        # Count spikes from motor neurons
                        for spike in spikes:
                            neuron_id = spike[0]
                            if neuron_id in self.motor_neurons:
                                motor_idx = np.where(self.motor_neurons == neuron_id)[
                                    0
                                ][0]
                                self.motor_spike_count[motor_idx] += 1

                        # Reset counter
                        samples_since_last_process = 0

            # ~~~~~~~~~~~ Game phase Logic ~~~~~~~~~~~
            elif self.state == "game":
                if just_entered:
                    state_timer = self.time_elapsed()

                    game_action = self.get_motor_signal()

                    self.game_obs, reward, game_done, trunc, inf = self.game_env.step(
                        game_action
                    )
                    total_reward += reward

                    # Store reward for REINFORCE
                    self.rewards.append(reward)

                    game_logger.writerow(
                        [
                            self.time_elapsed(),
                            self.game_obs[2],
                            reward,
                            game_action,
                            self.motor_spike_rate.tolist(),
                            self.state,
                        ]
                    )

                    self.set_sensory_signal(
                        self.game_obs
                    )  # Set the stim frequency rates

                    if self.verbose:
                        print(
                            "Spike count: ",
                            self.motor_spike_count,
                            f"Rate: {' || '.join([f'{r:.2f}' for r in self.motor_spike_rate])}",
                        )

                    if game_done:
                        self.state = "train"
                        just_entered = True
                        continue
                    else:
                        self.state = "read"
                        just_entered = True
                        continue

            elif self.state == "train":
                if just_entered:
                    state_timer = self.time_elapsed()
                    train_timer = self.time_elapsed()
                    just_entered = False
                    rewards.append(total_reward)
                    self.episode_reward = np.mean(rewards[-self.minibatch_size :])
                    self.episode_reward_change = (
                        np.mean(rewards[-self.minibatch_size :])
                        - np.mean(rewards[-20:])
                        if len(rewards) >= 20
                        else 0
                    )

                    # Update policy using REINFORCE
                    self.update_policy()

                    # Determine if we should do traditional training (if trainer exists)
                    do_training = False
                    train_pulse = None
                    train_freq = 30  # Default frequency

                    if self.trainer is not None:
                        if self.use_contextual_trainer and isinstance(
                            self.trainer, ContextualTrainer
                        ):
                            # Contextual trainer decides whether to stimulate
                            train_pulse, train_freq, _ = self.get_training_signal()
                            # For contextual trainer: do_training is True if a pattern was selected
                            do_training = train_pulse is not None
                            if self.verbose:
                                if do_training:
                                    print(f"ContextualTrainer selected STIM pattern")
                                else:
                                    print(f"ContextualTrainer selected NO-STIM")
                        else:
                            # Original TetanusTrainer behavior
                            reward_increased = self.episode_reward_change > 0
                            do_training = not reward_increased or getattr(
                                self, "force_train", False
                            )

                    reward_logger.writerow(
                        [self.time_elapsed(), self.episode, total_reward]
                    )
                    self.episode += 1

                    # Store episode length for contextual trainer before reset
                    self.episode_length = (
                        len(self.rewards) if hasattr(self, "rewards") else 0
                    )

                    if (
                        self.episode >= self.n_episodes
                        or self.time_elapsed() > self.max_time
                    ):
                        done = True
                        return

                    if self.verbose:
                        print("Reward:", total_reward)
                        print("Reward change:", self.episode_reward_change)
                        print(
                            f"Last 20 : {np.mean(rewards[-20:]):.2f}, last 5: {np.mean(rewards[-5:]):.2f}"
                        )
                        print("-" * 20)
                        print("Rewards:", rewards)

                    if do_training:
                        # For non-contextual trainer, get pattern now
                        if not (
                            self.use_contextual_trainer
                            and isinstance(self.trainer, ContextualTrainer)
                        ):
                            train_pulse, train_freq = self.get_training_signal()

                        # Log pattern
                        pattern_logger.writerow(
                            [
                                self.time_elapsed(),
                                self.trainer.get_action(self.last_action_ind)
                                if self.trainer
                                else None,
                                total_reward,
                                self.trainer.get_probs() if self.trainer else None,
                                self.trainer.get_values() if self.trainer else None,
                            ]
                        )

                        if train_pulse is not None:
                            obs, done = self.env.step(action=train_pulse, tag="train")
                        else:
                            obs, done = self.env.step()
                        train_timer = self.time_elapsed()
                        continue
                    else:
                        # Update even if it isn't currently training
                        if self.trainer is not None:
                            self.trainer.update_values(
                                self.episode_reward_change,
                                self.last_action_inds[-1]
                                if self.last_action_inds
                                else None,
                            )
                        self.last_action_inds.append(None)

                if self.time_elapsed() - state_timer > self.train_period_ms / 1000:
                    self.state = "wait"
                    just_entered = True
                    self.game_obs, inf = self.game_env.reset()
                    self.set_sensory_signal(self.game_obs)
                    total_reward = 0

                    continue

                if not (do_training):
                    self.game_obs, inf = self.game_env.reset()
                    self.set_sensory_signal(self.game_obs)
                    total_reward = 0
                    self.state = "wait"
                    just_entered = True
                    continue

                # Training stimulation pulse -- If reward has not increased
                if (
                    do_training
                    and train_freq > 0
                    and (self.time_elapsed() - train_timer > 1 / train_freq)
                ):
                    # Log pattern
                    pattern_logger.writerow(
                        [
                            self.time_elapsed(),
                            self.trainer.get_action(self.last_action_ind)
                            if self.trainer
                            else None,
                            total_reward,
                            self.trainer.get_probs() if self.trainer else None,
                            self.trainer.get_values() if self.trainer else None,
                        ]
                    )

                    if train_pulse is not None:
                        obs, done = self.env.step(action=train_pulse, tag="train")
                    else:
                        obs, done = self.env.step()
                    train_timer = self.time_elapsed()

                else:
                    obs, done = self.env.step()

            elif self.state == "wait":
                if just_entered:
                    state_timer = self.time_elapsed()
                    just_entered = False

                if self.time_elapsed() - state_timer > self.wait_period_ms / 1000:
                    self.state = "read"
                    just_entered = True
                    continue

                obs, done = self.env.step()

        # Close the logs
        game_log.close()
        reward_log.close()
        pattern_log.close()

        # Save sorted spikes similar to ClosedLoopPhaseV3
        sorted_spikes = []  # We could collect these during the run if needed
        sd = SpikeData.from_events(sorted_spikes) if sorted_spikes else None

        return {
            # 'cartpole_file': self.env.save_file,
            "total_episodes": self.episode,
            "sorted_spikedata": sd,
            "final_policy_state": self.policy_net.state_dict()
            if self.policy_net
            else None,
        }

    def update_policy(self):
        """Update policy network using REINFORCE algorithm."""
        if len(self.saved_log_probs) == 0 or len(self.rewards) == 0:
            self.last_policy_grad_magnitude = 0.0
            return

        # Calculate discounted rewards
        R = 0
        policy_loss = []
        returns = []

        # Calculate returns (discounted cumulative rewards)
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        # Normalize returns
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate loss
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        # Backprop
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()

        # Compute gradient magnitude before optimizer step
        self.last_policy_grad_magnitude = self._compute_gradient_magnitude()

        self.optimizer.step()

        # Update contextual trainer with gradient magnitude if applicable
        if self.use_contextual_trainer and self.trainer is not None:
            self.trainer.set_policy_grad_magnitude(self.last_policy_grad_magnitude)

        # Clear episode data
        del self.rewards[:]
        del self.saved_log_probs[:]

    def _compute_gradient_magnitude(self):
        """Compute L2 norm of the policy network gradients."""
        if self.policy_net is None:
            return 0.0

        total_norm = 0.0
        for param in self.policy_net.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return np.sqrt(total_norm)

    def time_elapsed(self):
        return super().time_elapsed()

    def info(self):
        return {
            "sensory_neurons": self.sensory_neurons,
            "motor_neurons": self.motor_neurons,
            "training_neurons": self.training_neurons,
            "read_period_ms": self.read_period_ms,
            "train_period_ms": self.train_period_ms,
            "n_episodes": self.n_episodes,
        }

    def predicted_time(self):
        """Returns the predicted time for the phase to run in seconds"""
        return self.predicted_time

