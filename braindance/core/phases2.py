"""
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
"""

from __future__ import annotations

import braindance.core.base_env as base_env
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proj.cartpole_v2 import experiment

import numpy as np
import time
import csv
from braindance.core.artifact_removal import ArtifactRemoval
from collections import deque


class Phase:
    """
    Base class for all phases
    """

    def __init__(self, env: base_env.BaseEnv, experiment=None, suffix=""):
        self.env = env
        self.start_time = time.perf_counter()
        self.requires = []  # List of data requirements
        self.provides = []  # List of data this phase provides
        self.experiment = experiment
        self.suffix = suffix

    def set_env(self, env):
        self.env = env

    def set_experiment(self, experiment):
        self.experiment = experiment

    def run(self, experiment=None):
        raise NotImplementedError

    def validate(self, experiment=None):
        """Validates the phase to make sure it can run"""
        return True

    def time_elapsed(self):
        return time.perf_counter() - self.start_time

    def predicted_time(self):
        """Returns the predicted time for the phase to run in seconds"""
        raise NotImplementedError

    def info(self):
        """Returns a dictionary of information about the phase"""
        raise NotImplementedError

    def cleanup(self):
        """Cleans up the phase after it is done"""
        pass


class PhaseManager:
    """
    Manages phases of an experiment
    """

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
        """
        Adds a group of phases to the manager,
        each group will belong to the same save file
        """
        self.phases.append(phase_group)

    def log_summary(self):
        """Logs the summary of the experiment to a text file"""
        summary_str = self.summary()
        with open(self.save_dir + "/summary.txt", "a") as f:
            f.write(summary_str)
            f.write("\n\n\n")
            f.write("=" * 20)

    def log_phase(self, phase):
        """Appends the phase and filename to the log file"""
        with open(self.save_dir + "/phase_log.csv", "a") as f:
            writer = csv.writer(f)
            if isinstance(phase, list):
                i = 0
                for sub_phase in phase:
                    writer.writerow([sub_phase.__class__.__name__, self.filenames[-i]])
                    i += 1
            else:
                writer.writerow([phase.__class__.__name__, self.filenames[-1]])

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
                    print("~" * 20)
                    print("Save file:", self.env.save_file)
                    if isinstance(phase, list):
                        print("Running phase group of:", end=" ")
                        for sub_phase in phase:
                            print(sub_phase.__class__.__name__, end=" ")
                        print()
                    else:
                        print("Running phase:", phase.__class__.__name__)
                    print("~" * 20)

                # Run the phase
                if isinstance(phase, list):
                    for sub_phase in phase:
                        print("Running sub phase:", sub_phase.__class__.__name__)
                        print("=" * 20)
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
        """Returns a summary of the experiment as a string"""

        def summarize_phase(phase):
            summary_str = phase.__class__.__name__ + "\n"
            predicted_time = phase.predicted_time
            if predicted_time > 60:
                summary_str += "\tPredicted Time: {:.0f}m {:.0f}s\n".format(
                    predicted_time // 60, predicted_time % 60
                )
            else:
                summary_str += "\tPredicted Time: {:.1f} seconds\n".format(
                    predicted_time
                )

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
                summary_str += "Group Total Time: {:.0f}m {:.0f}s\n\n".format(
                    group_time // 60, group_time % 60
                )
                total_time += group_time
            else:
                summary_str += summarize_phase(phase)
                total_time += phase.predicted_time

        summary_str += "Total Experiment Time: {:.0f}m {:.0f}s\n".format(
            total_time // 60, total_time % 60
        )
        summary_str += "-------------"
        return summary_str


class RecordPhase(Phase):
    """
    Phase for recording spontaneous activity

    Parameters
    ----------
    env : base_env.BaseEnv
        The environment to run the phase in
    duration : int
        The duration of the recording in seconds, by default 10
    """

    def __init__(
        self,
        env: base_env.BaseEnv = None,
        duration: int = 10,
        name: str = "RecordPhase",
        suffix: str = "_recording",
        verbose=False,
    ):
        super().__init__(env, experiment=experiment)

        self.duration = duration
        self.predicted_time = duration
        self.name = name
        self.verbose = verbose
        self.suffix = suffix

    def set_env(self, env):
        self.env = env

    def run(self, experiment=None):
        self.start_time = time.perf_counter()

        done = False
        while not done:
            obs, done = self.env.step()
            if self.time_elapsed() > self.duration:
                done = True

        self.env.close()

        return {"filename": self.env.save_file, "time_elapsed": self.time_elapsed()}

    def info(self):
        return {"duration": self.duration, "name": self.name}


class NeuralSweepPhase(Phase):
    """
    Sweep the amplitude of a stimulation to find the minimum amplitude that
    elicits a spike

    Parameters
    ----------
    env : base_env.BaseEnv
        The environment to run the phase in
    neuron_list : list
        The neurons to stimulate
    experiment : experiment.Experiment
        The experiment to run the phase in
        If provided, the neuron_list will be ignored, and env will be ignored
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
    """

    def __init__(
        self,
        env: base_env.BaseEnv = None,
        neuron_list: list = [],
        experiment: experiment.Experiment = None,
        amp_bounds=[150, 150, 1],
        stim_freq: float = 1,
        replicates=30,
        phase_length: int = 100,
        order="ran",
        single_connect=False,
        verbose=False,
        tag="neural_sweep",
        suffix: str = "_stim",
    ):
        super().__init__(env, experiment=experiment, suffix=suffix)

        assert order[0] in ["r", "a", "n"], (
            "First character of type must be 'r', 'a', or 'n'"
        )
        assert order[1] in ["r", "a", "n"], (
            "Second character of type must be 'r', 'a', or 'n'"
        )
        assert order[2] in ["r", "a", "n"], (
            "Third character of type must be 'r', 'a', or 'n'"
        )
        self.neuron_list = neuron_list

        # If int, we only have one value
        if isinstance(amp_bounds, int):
            amp_bounds = [amp_bounds, amp_bounds, 1]
        assert len(amp_bounds) >= 2, (
            "Amplitude bounds must have at least a start and end value"
        )
        assert amp_bounds[0] <= amp_bounds[1], (
            "Amplitude bounds must be same or increasing"
        )

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
        if (
            self.single_connect
            and (order[2] != "n" or order[1] != "n")
            or order == "random"
        ):
            print(
                "Warning: single_connect should only be used with *n* or **n in the order"
            )

        self.order = order
        self.tag = tag
        self.verbose = verbose

        if len(neuron_list) > 0:
            self.predicted_time = (
                self.n_amplitudes * len(neuron_list) * 1 / stim_freq * replicates
            )
        else:
            self.predicted_time = 1 / stim_freq * replicates

    def set_from_experiment(self, experiment: experiment.Experiment):
        self.experiment = experiment
        stim_electrodes = experiment.params.get("stim_electrodes", [])
        self.neuron_list = np.arange(len(stim_electrodes))
        self.env = experiment.env
        # Update the predicted time
        self.predicted_time = (
            self.n_amplitudes
            * len(self.neuron_list)
            * 1
            / self.stim_freq
            * self.replicates
        )
        if self.verbose:
            print("Stimulating", len(self.neuron_list), "neurons")
            print("From electrodes:", stim_electrodes)

    def generate_stim_commands(self):
        """
        Generates the stimulation commands for the amplitude sweep
        """
        amplitudes = np.linspace(
            self.amplitude_start, self.amplitude_end, self.n_amplitudes
        )

        stim_commands = []
        if self.order[2] == "r":
            for r in range(self.replicates):
                if self.order[1] == "a":
                    for a in amplitudes:
                        for n in self.neuron_list:
                            stim_commands.append(([n], a, self.phase_length))

                elif self.order[1] == "n":
                    for n in self.neuron_list:
                        for a in amplitudes:
                            stim_commands.append(([n], a, self.phase_length))

        elif self.order[2] == "a":
            for a in amplitudes:
                if self.order[1] == "r":
                    for r in range(self.replicates):
                        for n in self.neuron_list:
                            stim_commands.append(([n], a, self.phase_length))

                elif self.order[1] == "n":
                    for n in self.neuron_list:
                        for r in range(self.replicates):
                            stim_commands.append(([n], a, self.phase_length))

        elif self.order[2] == "n":
            for n in self.neuron_list:
                if self.order[1] == "r":
                    for r in range(self.replicates):
                        for a in amplitudes:
                            stim_commands.append(([n], a, self.phase_length))

                elif self.order[1] == "a":
                    for a in amplitudes:
                        for r in range(self.replicates):
                            stim_commands.append(([n], a, self.phase_length))

        if self.order == "random":
            np.random.shuffle(stim_commands)

        return stim_commands

    def run(self, experiment=None):
        done = False

        time_between_stims = 1 / self.stim_freq
        stim_count = 0

        if self.verbose:
            print("Generating stimulation commands from neuron list:", self.neuron_list)

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
                    print(
                        "Stimulating neuron",
                        stim_commands[0][0],
                        "at amplitude",
                        stim_commands[0][1],
                        "at time: {:.3f}".format(self.time_elapsed()),
                    )

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

        return {"filename": self.env.save_file, "time_elapsed": self.time_elapsed()}

    def info(self):
        return {
            "neuron_list": self.neuron_list,
            "amplitude_start": self.amplitude_start,
            "amplitude_end": self.amplitude_end,
            "n_amplitudes": self.n_amplitudes,
            "stim_freq": self.stim_freq,
            "replicates": self.replicates,
            "phase_length": self.phase_length,
            "type": self.order,
            "tag": self.tag,
        }


class FrequencyStimPhase(Phase):
    """
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
    experiment: experiment.Experiment, optional
        The experiment to run the phase in
    """

    def __init__(
        self,
        env: base_env.BaseEnv,
        stim_command: list,
        stim_freq: float = 1,
        duration: int = 10,
        tag="frequency_stim",
        verbose=False,
        connect_units=None,
        suffix: str = "_freq-stim",
        experiment: experiment.Experiment = None,
    ):
        super().__init__(env, experiment=experiment)
        self.stim_freq = stim_freq
        self.stim_command = stim_command
        # Check if stim_command is a single stim command or a list of stim commands
        if type(stim_command[0][0]) == list:
            self.single_command = False
        else:
            self.single_command = True

        self.duration = duration
        self.suffix = suffix
        self.predicted_time = duration

        self.tag = tag
        self.single_tag = type(self.tag) == str
        if not self.single_tag and self.single_command:
            raise ValueError("Tag must be a string if stim_command is a single command")
        # Or if not single commands, tag must be string or list of strings the same length as stim_command
        if (
            not self.single_tag
            and not self.single_command
            and len(self.tag) != len(self.stim_command)
        ):
            raise ValueError(
                "Tag must be a string or list of strings the same length as stim_command"
            )

        self.start_time = time.perf_counter()
        self.verbose = verbose
        self.connect_units = connect_units

    def set_from_experiment(self, experiment: experiment.Experiment):
        self.experiment = experiment
        self.env = experiment.env
        stim_electrodes = experiment.params.get("stim_electrodes", [])
        self.connect_units = np.arange(len(stim_electrodes))

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
                    print(
                        "Stimulating neuron(s) {} at time: {:.3f}".format(
                            stim_command[0], self.time_elapsed()
                        )
                    )

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
            "stim_command": self.stim_command,
            "stim_freq": self.stim_freq,
            "duration": self.duration,
            "tag": self.tag,
        }


class CartPolePhase(Phase):
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
    training_type = type of training to use, "punishment" or "reward" or "always"

    """

    def __init__(
        self,
        env: base_env.BaseEnv,
        sensory_neurons: list,
        motor_neurons: list,
        training_neurons: list,
        experiment: experiment.Experiment = None,
        amp_mv: int = 400,
        phase_width: int = 100,
        read_period_ms: int = 200,
        train_period_ms: int = 200,
        wait_period_ms: int = 400,
        n_episodes: int = 10,
        trainer=None,
        artifact_removal=False,
        continuous=True,
        assistive=0.0,
        minibatch_size=5,
        spike_thresh=[-3.1, -20],
        verbose=False,
        max_time=np.inf,
        normalization=1,
        suffix: str = "_cartpole",
        training_type: str = "punishment",
    ):
        import gymnasium as gym
        from braindance.games import cartpole_continuous

        self.continuous = continuous

        if continuous:
            self.game_env = cartpole_continuous.CartPoleContinuousEnv(
                render_mode="human"
            )
            pass
        else:
            self.game_env = gym.make("CartPole-v1", render_mode="human")

        self.env = env
        self.read_period_ms = read_period_ms
        self.train_period_ms = train_period_ms
        self.wait_period_ms = wait_period_ms
        self.n_episodes = n_episodes
        self.episode = 0
        self.max_time = max_time
        self.minibatch_size = minibatch_size
        self.normalization = normalization
        self.training_type = training_type

        self.suffix = suffix

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
                self.artifact_remover.append(
                    ArtifactRemoval(
                        N=20,
                        nc_start=20,
                        min_val=-25,
                        max_val=25,  # was -25, 25
                        spike_thresh=spike_thresh,
                    )
                )
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

        self.motor_spike_count = np.zeros(
            len(self.motor_neurons)
        )  # Stores the spike count for each motor neuron
        self.motor_spike_rate = np.zeros(len(self.motor_neurons))
        self.state = "read"  # 'read', 'train', 'game', or 'wait'

        self.predicted_time = (
            (read_period_ms + train_period_ms) * 20 * n_episodes / 1000
        )  # x20 due to 20 steps of the game avg

        super().__init__(env, experiment=experiment)

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
        Readout from the read/motor neurons set
        The action returned should be an integer {0,1} corresponding to the action

        Recommended to use self.spike_count, which is a list of the spike counts
        for each motor neuron

        """
        # if moving_avg:
        self.motor_spike_rate = (
            moving_avg * self.motor_spike_rate
            + (1 - moving_avg) * self.motor_spike_count
        )

        # spike_diff = self.motor_spike_count[0] - self.motor_spike_count[1]

        if type(self.normalization) == np.ndarray:
            # m0 mean, m1 mean, m0 std, m1 std
            # spike_diff /= self.normalization
            spike_diff = (self.motor_spike_rate[0] / self.normalization[0]) - (
                self.motor_spike_rate[1] / self.normalization[1]
            )
            # spike_diff *= 10
        else:
            spike_diff = self.motor_spike_rate[0] - self.motor_spike_rate[1]
            spike_diff /= self.normalization

        if self.assistive:
            assistive_factor = self.assistive * (
                self.sensory_stim_Hz[0] - self.sensory_stim_Hz[1]
            )
            print(
                f"Assistive factor: {assistive_factor:.2f} \t Spike diff: {spike_diff:.2f}"
            )

            spike_diff += assistive_factor

        if self.continuous:
            print(
                f"Spike rate: {self.motor_spike_rate[0]:.2f} || {self.motor_spike_rate[1]:.2f} \t[Action]: {max(min(-spike_diff, 1), -1)} \t Spike diff: {spike_diff:.2f}"
            )
            return max(min(-spike_diff, 1), -1)
        if spike_diff > 0:
            action = 0
            print("Going left")
        else:
            action = 1
            print("Going right")

        return action

    def get_training_signal(
        self, train_Hz=30, stim_count=3, delay_ms=5, use_change=True
    ):
        """
        The stimulation on the training neurons

        """
        if self.trainer:
            # Update last reward
            if (
                self.episode_reward != None
                and len(self.last_action_inds) == self.minibatch_size
            ):
                # self.trainer.update_values(self.episode_reward, self.last_action_ind)
                if use_change:
                    self.trainer.update_values(
                        self.episode_reward_change, self.last_action_inds[-1]
                    )
                else:
                    self.trainer.update_values(
                        self.episode_reward, self.last_action_inds[-1]
                    )

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

    def run(self):
        done = False

        assert self.env.observation_type == "raw", "Observation type must be raw"

        state, inf = self.game_env.reset()

        just_entered = True
        action = None
        reward = 0
        total_reward = 0
        rewards = []

        # Game log
        game_log = open(self.env.save_file + "_game_log.csv", "w")
        # Header for csv
        game_logger = csv.writer(game_log)
        game_logger.writerow(
            [
                "time",
                "pole_angle",
                "reward",
                "action",
                "spike_count_l",
                "spike_count_r",
                "state",
            ]
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
                    )
                else:
                    # No action
                    obs, done = self.env.step()

                # Faster with artifact removal
                if self.artifact_remover:
                    # obs = np.array(obs)[self.motor_neurons]
                    if obs == None:
                        continue
                    _, _, spike_0 = self.artifact_remover[0].fit_step(
                        obs[self.motor_neurons[0]]
                    )
                    _, _, spike_1 = self.artifact_remover[1].fit_step(
                        obs[self.motor_neurons[1]]
                    )
                    if spike_0:
                        self.motor_spike_count[0] += 1
                    if spike_1:
                        self.motor_spike_count[1] += 1
                else:
                    # Process the observation
                    for frame, ch, amp in obs:
                        if ch in self.motor_neurons:
                            self.motor_spike_count[
                                np.where(self.motor_neurons == ch)
                            ] += 1

            # ~~~~~~~~~~~ Game phase Logic ~~~~~~~~~~~
            elif self.state == "game":
                if just_entered:
                    state_timer = self.time_elapsed()

                    game_action = self.get_motor_signal()

                    self.game_obs, reward, game_done, trunc, inf = self.game_env.step(
                        game_action
                    )
                    total_reward += reward

                    game_logger.writerow(
                        [
                            self.time_elapsed(),
                            self.game_obs[2],
                            reward,
                            game_action,
                            self.motor_spike_count,
                            self.game_obs,
                        ]
                    )

                    self.set_sensory_signal(
                        self.game_obs
                    )  # Set the stim frequency rates

                    if self.verbose:
                        print(
                            "Spike count: ",
                            self.motor_spike_count,
                            f"Rate {self.motor_spike_rate[0]:.2f} || {self.motor_spike_rate[1]:.2f}",
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
                    self.episode_reward_change = np.mean(
                        rewards[-self.minibatch_size :]
                    ) - np.mean(rewards[-20:])

                    # reward_increased = total_reward > max(np.mean(rewards[-5:]),15)
                    reward_increased = self.episode_reward_change > 0
                    # reward_increased = np.mean(rewards[-self.minibatch_size:]) >= np.mean(rewards[-20:])

                    if self.training_type == "punishment":
                        do_training = not reward_increased
                    elif self.training_type == "reward":
                        do_training = reward_increased
                    elif self.training_type == "always":
                        do_training = True

                    reward_logger.writerow(
                        [self.time_elapsed(), self.episode, total_reward]
                    )
                    self.episode += 1
                    if (
                        self.episode >= self.n_episodes
                        or self.time_elapsed() > self.max_time
                    ):
                        done = True
                        return

                    if self.verbose:
                        print("Reward:", total_reward, "increased:", reward_increased)
                        print("Reward change:", self.episode_reward_change)
                        print(
                            f"Last 20 : {np.mean(rewards[-20:]):.2f}, last 5: {np.mean(rewards[-5:]):.2f}"
                        )
                        print("-" * 20)
                        print("Rewards:", rewards)

                    if do_training:
                        # Get pattern/update
                        train_pulse, train_freq = self.get_training_signal()
                        # Log pattern
                        pattern_logger.writerow(
                            [
                                self.time_elapsed(),
                                self.trainer.get_action(self.last_action_ind),
                                total_reward,
                                self.trainer.get_probs(),
                                self.trainer.get_values(),
                            ]
                        )

                        obs, done = self.env.step(action=train_pulse, tag="train")
                        train_timer = self.time_elapsed()
                        continue
                    else:
                        # Update even if it isn't currently training
                        if len(self.last_action_inds) > 0:
                            self.trainer.update_values(
                                self.episode_reward_change, self.last_action_inds[-1]
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
                if do_training and (self.time_elapsed() - train_timer > 1 / train_freq):
                    # Get pattern/update
                    # train_pulse, train_freq = self.get_training_signal()
                    # Log pattern
                    pattern_logger.writerow(
                        [
                            self.time_elapsed(),
                            self.trainer.get_action(self.last_action_ind),
                            total_reward,
                            self.trainer.get_probs(),
                            self.trainer.get_values(),
                        ]
                    )

                    obs, done = self.env.step(action=train_pulse, tag="train")
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
        return True

    def time_elapsed(self):
        return time.perf_counter() - self.start_time

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


class AntPhase(Phase):
    """
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
    """

    def __init__(
        self,
        env: base_env.BaseEnv,
        motor_neurons: list,
        training_neurons: list,
        experiment: experiment.Experiment = None,
        amp_mv: int = 400,
        phase_width: int = 100,
        read_period_ms: int = 200,
        train_period_ms: int = 200,
        wait_period_ms: int = 400,
        n_episodes: int = 10,
        trainer=None,
        artifact_removal=False,
        verbose=False,
        max_time=np.inf,
        dummy_mode: str = None,
        suffix: str = "_ant",
    ):
        import gymnasium as gym
        from braindance.utils.rt_linear_art_removal import LinearArtifactRemoval

        self.game_env = gym.make("Ant-v5", render_mode="human")

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
                N=60,
                nc_start=60,
                min_val=-85,
                max_val=85,
                spike_thresh=[-3.1, -25],
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
        self.state = "read"  # 'read', 'train', 'game', or 'wait'

        self.predicted_time = (
            (read_period_ms + train_period_ms) * 20 * n_episodes / 1000
        )

        super().__init__(env, experiment=experiment)

    def get_motor_signal(self, moving_avg=0.3):
        """
        Readout from the read/motor neurons set
        The action returned should be a numpy array of shape (8,) with values between -1 and 1
        """
        # Handle dummy mode - generate synthetic spike rates instead of using real ones
        if self.dummy_mode:
            if self.dummy_mode == "zeros":
                self.motor_spike_count = np.zeros(len(self.motor_neurons))
            elif self.dummy_mode == "random":
                self.motor_spike_count = (
                    np.random.rand(len(self.motor_neurons)) * 0.5
                )  # Random values between 0 and 0.5

        self.motor_spike_rate = (
            moving_avg * self.motor_spike_rate
            + (1 - moving_avg) * self.motor_spike_count
        )

        # Simple mapping: use first 8 neurons to control the 8 action dimensions
        # Scale the spike rates to be between -1 and 1
        action = np.zeros(8)
        for i in range(min(8, len(self.motor_spike_rate))):
            # Scale between 0 and 1 to between -1 and 1
            action[i] = np.clip(self.motor_spike_rate[i] * 3 - 1, -1, 1)
            # action[i] = np.clip(self.motor_spike_rate[i]*5, -1, 1)  # Scale factor of 10 is arbitrary

        if self.verbose:
            print(f"Spike rates: {self.motor_spike_rate[:8]}")
            print(f"Action: {action}")

        return action.flatten()

    def get_training_signal(self, train_Hz=30, stim_count=3, delay_ms=5):
        """
        The stimulation on the training neurons
        """
        if self.trainer:
            # Update last reward
            if self.episode_reward != None:
                self.trainer.update_values(self.episode_reward, self.last_action_ind)

            train_action, train_Hz, self.last_action_ind = self.trainer.get_action()
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
        game_log = open(self.env.save_file + "_game_log.csv", "w")
        game_logger = csv.writer(game_log)
        game_logger.writerow(
            ["time", "reward", "action", "spike_count", "state", "z_pos"]
        )

        reward_log = open(self.env.save_file + "_reward_log.csv", "w")
        reward_logger = csv.writer(reward_log)
        reward_logger.writerow(["time", "episode", "reward", "episode_steps"])

        pattern_log = open(self.env.save_file + "_pattern_log.csv", "w")
        pattern_logger = csv.writer(pattern_log)
        pattern_logger.writerow(["time", "pattern", "reward", "probs", "vals"])

        self.start_time = time.perf_counter()
        state_timer = self.time_elapsed()

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

                obs, done = self.env.step()

                if self.artifact_remover:
                    if obs is None:
                        continue
                    # Process all channels at once
                    obs_arr = (
                        np.array([obs[ch] for ch in self.motor_neurons])
                        .reshape(1, -1)
                        .T
                    )
                    clean_values, artifacts, spikes = self.artifact_remover.fit_step(
                        obs_arr
                    )
                    self.motor_spike_count += spikes.flatten()
                else:
                    # Process the observation
                    for frame, ch, amp in obs:
                        if ch in self.motor_neurons:
                            self.motor_spike_count[
                                np.where(self.motor_neurons == ch)
                            ] += 1

            # ~~~~~~~~~~~ Game phase Logic ~~~~~~~~~~~
            elif self.state == "game":
                if just_entered:
                    state_timer = self.time_elapsed()
                    episode_steps += 1

                    game_action = self.get_motor_signal()

                    self.game_obs, reward, game_done, trunc, inf = self.game_env.step(
                        game_action
                    )
                    game_done = (
                        game_done or trunc
                    )  # Handle both termination and truncation

                    # Check if episode timeout or ant flipped over
                    z_pos = (
                        self.game_obs[0] if len(self.game_obs) > 0 else 0
                    )  # z-coordinate of torso

                    total_reward += reward

                    game_logger.writerow(
                        [
                            self.time_elapsed(),
                            reward,
                            game_action,
                            self.motor_spike_count,
                            self.game_obs,
                            z_pos,
                        ]
                    )

                    if self.verbose:
                        print(
                            "Spike count: ",
                            self.motor_spike_count,
                            f"\nRate {self.motor_spike_rate[0]:.2f} || {self.motor_spike_rate[1]:.2f}",
                            end="",
                        )
                        print(
                            f"|| {self.motor_spike_rate[2]:.2f} || {self.motor_spike_rate[3]:.2f} || {self.motor_spike_rate[4]:.2f} || {self.motor_spike_rate[5]:.2f} || {self.motor_spike_rate[6]:.2f} || {self.motor_spike_rate[7]:.2f}"
                        )

                    if game_done or episode_steps >= max_episode_steps:
                        if self.verbose and episode_steps >= max_episode_steps:
                            print(f"Episode truncated after {episode_steps} steps")
                        self.state = "train"
                        just_entered = True
                        # Update networks at end of episode
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
                    self.episode_reward = np.mean(rewards[-5:])  # Use last 5 rewards

                    reward_logger.writerow(
                        [self.time_elapsed(), self.episode, total_reward, episode_steps]
                    )
                    self.episode += 1
                    episode_steps = 0  # Reset episode steps

                    if (
                        self.episode >= self.n_episodes
                        or self.time_elapsed() > self.max_time
                    ):
                        done = True
                        return

                    if self.verbose:
                        print("Reward:", total_reward)
                        print("-" * 20)
                        print("Rewards:", rewards)

                    # Get pattern/update
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

                    obs, done = self.env.step(action=train_pulse, tag="train")
                    train_timer = self.time_elapsed()
                    continue

                if self.time_elapsed() - state_timer > self.train_period_ms / 1000:
                    self.state = "wait"
                    just_entered = True
                    self.game_obs, inf = self.game_env.reset()
                    total_reward = 0
                    continue

                # Training stimulation pulse
                if self.time_elapsed() - train_timer > 1 / train_freq:
                    # Get pattern/update
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

                    obs, done = self.env.step(action=train_pulse, tag="train")
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
        return True

    def time_elapsed(self):
        return time.perf_counter() - self.start_time

    def info(self):
        return {
            "motor_neurons": self.motor_neurons,
            "training_neurons": self.training_neurons,
            "read_period_ms": self.read_period_ms,
            "train_period_ms": self.train_period_ms,
            "n_episodes": self.n_episodes,
        }

    def predicted_time(self):
        """Returns the predicted time for the phase to run in seconds"""
        return self.predicted_time


class NeuralAntPhase(AntPhase):
    """
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
    """

    def __init__(
        self,
        env: base_env.BaseEnv,
        motor_neurons: list,
        training_neurons: list,
        network_arch: list = [16],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.1,
        train_iters: int = 10,
        batch_size: int = 64,
        dummy_mode: str = None,
        **kwargs,
    ):
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

        # Initialize neural networks
        self.actor = self._build_actor().to(self.device)
        self.critic = self._build_critic().to(self.device)
        self.sensory_actor = self._build_sensory_actor().to(
            self.device
        )  # New sensory network

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.sensory_optimizer = optim.Adam(
            self.sensory_actor.parameters(), lr=learning_rate
        )  # New optimizer

        # Initialize memory buffer
        self.memory = []
        self.max_memory_size = 10000

        # Initialize sensory memory for training
        self.sensory_memory = []
        self.max_sensory_memory_size = 10000

    def _build_sensory_actor(self):
        """Build a neural network for mapping game state and motor rates to sensory signals"""
        import torch.nn as nn
        import torch

        class SensoryActorNetwork(nn.Module):
            def __init__(self, game_state_dim, motor_dim, hidden_dims):
                super().__init__()

                # Input normalization
                self.game_norm = nn.LayerNorm(game_state_dim)
                self.motor_norm = nn.LayerNorm(motor_dim)

                # Combine dimensions
                input_dim = game_state_dim + motor_dim

                # Hidden layers
                layers = []
                prev_dim = input_dim

                for dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, dim))
                    layers.append(nn.Tanh())
                    layers.append(nn.LayerNorm(dim))
                    prev_dim = dim

                self.hidden = nn.Sequential(*layers)

                # Output layer - produces firing rates for sensory neurons
                self.output = nn.Sequential(
                    nn.Linear(prev_dim, 2),  # 2 sensory neurons
                    nn.Softplus(),  # Ensure positive firing rates
                )

                # Initialize weights
                self.apply(self._init_weights)

            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=0.1)
                    nn.init.constant_(module.bias, 0)

            def forward(self, game_state, motor_rates):
                # Normalize inputs
                game_state = self.game_norm(game_state)
                motor_rates = self.motor_norm(motor_rates)

                # Concatenate inputs
                x = torch.cat([game_state, motor_rates], dim=-1)

                # Process through network
                x = self.hidden(x)
                firing_rates = self.output(x)

                return firing_rates

        # Create network with appropriate input dimensions
        return SensoryActorNetwork(
            game_state_dim=111,  # Ant environment state dimension
            motor_dim=len(self.motor_neurons),
            hidden_dims=[16],  # Single hidden layer
        )

    def set_sensory_signal(self, game_env_obs):
        """
        Maps the game observation and motor rates to sensory neuron firing rates using a neural network.

        Parameters
        ----------
        game_env_obs : ndarray
            Game observation array containing state information
        """
        import torch
        import numpy as np

        # Convert inputs to tensors
        game_state = torch.FloatTensor(game_env_obs).to(self.device)
        motor_rates = torch.FloatTensor(self.motor_spike_rate).to(self.device)

        # Add batch dimension
        game_state = game_state.unsqueeze(0)
        motor_rates = motor_rates.unsqueeze(0)

        # Get firing rates from network
        with torch.no_grad():
            firing_rates = self.sensory_actor(game_state, motor_rates)

            # Store in memory for training
            self.sensory_memory.append(
                {
                    "game_state": game_state.detach().cpu().numpy(),
                    "motor_rates": motor_rates.detach().cpu().numpy(),
                    "firing_rates": firing_rates.detach().cpu().numpy(),
                    "reward": 0,  # Will be updated later
                }
            )

            # Keep memory size limited
            if len(self.sensory_memory) > self.max_sensory_memory_size:
                self.sensory_memory.pop(0)

        # Update sensory stimulation rates
        self.sensory_stim_Hz = firing_rates.squeeze().cpu().numpy()

        if self.verbose:
            print(
                f"Sensory firing rates: {self.sensory_stim_Hz[0]:.2f} || {self.sensory_stim_Hz[1]:.2f}"
            )

    def _update_sensory_network(self):
        """Update the sensory network using the collected experience"""
        import torch
        import numpy as np

        if len(self.sensory_memory) < self.batch_size:
            return

        try:
            # Convert memory to tensors
            game_states = torch.FloatTensor(
                np.array([m["game_state"] for m in self.sensory_memory])
            ).to(self.device)
            motor_rates = torch.FloatTensor(
                np.array([m["motor_rates"] for m in self.sensory_memory])
            ).to(self.device)
            firing_rates = torch.FloatTensor(
                np.array([m["firing_rates"] for m in self.sensory_memory])
            ).to(self.device)
            rewards = torch.FloatTensor(
                np.array([m["reward"] for m in self.sensory_memory])
            ).to(self.device)

            # Normalize rewards
            if len(rewards) > 1:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # Update network
            for _ in range(self.train_iters):
                # Forward pass
                pred_firing_rates = self.sensory_actor(game_states, motor_rates)

                # Compute loss (MSE + reward-weighted term)
                mse_loss = torch.nn.functional.mse_loss(pred_firing_rates, firing_rates)
                reward_loss = -torch.mean(rewards * torch.sum(pred_firing_rates, dim=1))

                total_loss = mse_loss + 0.1 * reward_loss  # Weight the reward term

                # Update network
                self.sensory_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.sensory_actor.parameters(), max_norm=0.1
                )
                self.sensory_optimizer.step()

        except Exception as e:
            print(f"Error in sensory network update: {e}")

        # Clear memory after update
        self.sensory_memory = []

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
                    layers.append(
                        nn.LayerNorm(dim)
                    )  # Add normalization after activation
                    prev_dim = dim

                self.shared = nn.Sequential(*layers)

                # Separate heads for mean and log_std
                self.mean_head = nn.Sequential(
                    nn.Linear(prev_dim, 8),
                    nn.Tanh(),  # Keep means in [-1, 1] range
                )

                self.log_std_head = nn.Sequential(
                    nn.Linear(prev_dim, 8),
                    nn.Hardtanh(-2, 0),  # Constrain log_std to keep std in [0.13, 1.0]
                )

                # Initialize weights with small values
                self.apply(self._init_weights)

            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(
                        module.weight, gain=0.1
                    )  # Small initial weights
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
                # x = self.hidden(x)
                return self.output(x)

        # Use same architecture as actor for consistency
        return StableCriticNetwork(len(self.motor_neurons), [16])

    def get_motor_signal(self, moving_avg=0.3):
        """
        Readout from the read/motor neurons set using neural network
        The action returned should be a numpy array of shape (8,) with values between -1 and 1
        """
        import torch
        from torch.distributions import Normal
        import numpy as np

        # Handle dummy mode - generate synthetic spike rates instead of using real ones
        if self.dummy_mode:
            if self.dummy_mode == "zeros":
                self.motor_spike_count = np.zeros(len(self.motor_neurons))
            elif self.dummy_mode == "random":
                self.motor_spike_count = (
                    np.random.rand(len(self.motor_neurons)) * 0.5
                )  # Random values between 0 and 0.5

        # Update spike rates
        self.motor_spike_rate = (
            moving_avg * self.motor_spike_rate
            + (1 - moving_avg) * self.motor_spike_count
        )

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
                    print(
                        "Warning: NaN or Inf in action_params, returning zero actions"
                    )
                    return np.zeros(8)

                action_params = action_params.view(-1, 8, 2)
                means = action_params[..., 0]
                log_stds = action_params[..., 1]

                # Tight clamping of log_stds - very important for stability
                log_stds = torch.clamp(
                    log_stds, -1.5, 0
                )  # Constrain std to (0.22, 1.0)

                # Sample action directly using the reparameterization trick
                # This is safer than using Normal distribution
                noise = torch.randn_like(means)
                stds = torch.exp(log_stds)
                action = means + noise * stds
                action = torch.tanh(action)  # Bound to [-1, 1]

                # Compute log_prob safely
                log_prob = -0.5 * (noise**2 + 2 * log_stds + np.log(2 * np.pi))
                log_prob = torch.sum(log_prob, dim=-1)

                # Account for the tanh squashing
                log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1)

                # Get value estimate
                value = self.critic(state)

                # Store in memory - detach tensors before converting to numpy
                self.memory.append(
                    {
                        "state": state.detach().cpu().numpy(),
                        "action": action.detach().cpu().numpy().flatten(),
                        "log_prob": log_prob.detach().cpu().numpy(),
                        "value": value.detach().cpu().numpy(),
                        "reward": 0,  # Will be updated later
                        "next_value": 0,  # Will be updated later
                        "done": False,  # Will be updated later
                    }
                )

                # Keep memory size limited
                if len(self.memory) > self.max_memory_size:
                    self.memory.pop(0)

                if self.verbose:
                    print(f"Action: {action.detach().cpu().numpy().flatten()}")

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
            states = torch.FloatTensor(np.array([m["state"] for m in self.memory])).to(
                self.device
            )
            actions = torch.FloatTensor(
                np.array([m["action"] for m in self.memory])
            ).to(self.device)
            old_log_probs = torch.FloatTensor(
                np.array([m["log_prob"] for m in self.memory])
            ).to(self.device)
            rewards = torch.FloatTensor(
                np.array([m["reward"] for m in self.memory])
            ).to(self.device)
            next_values = torch.FloatTensor(
                np.array([m["next_value"] for m in self.memory])
            ).to(self.device)
            dones = torch.FloatTensor(np.array([m["done"] for m in self.memory])).to(
                self.device
            )

            # Handle extreme values in rewards
            rewards = torch.clamp(rewards, -10, 10)  # Clip large reward values

            # Compute advantages
            with torch.no_grad():
                values = self.critic(states).squeeze()
                advantages, returns = self._compute_gae(
                    rewards.cpu().numpy(),
                    values.detach().cpu().numpy(),
                    next_values.cpu().numpy(),
                    dones.cpu().numpy(),
                )
                advantages = torch.FloatTensor(advantages).to(self.device)
                returns = torch.FloatTensor(returns).to(self.device)

                # Normalize advantages - important for stable updates
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

            # Use a smaller learning rate for the first few episodes
            if self.episode < 5:
                lr_scale = 0.1
                for param_group in self.actor_optimizer.param_groups:
                    param_group["lr"] = self.learning_rate * lr_scale
                for param_group in self.critic_optimizer.param_groups:
                    param_group["lr"] = self.learning_rate * lr_scale

            # PPO update
            for _ in range(self.train_iters):
                # Forward pass
                action_params = self.actor(states)

                # Check for NaN values in action_params and skip update if found
                if torch.isnan(action_params).any() or torch.isinf(action_params).any():
                    print(
                        "Warning: NaN values detected in action parameters, skipping update"
                    )
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
                    new_log_probs = -0.5 * (
                        (actions - means) ** 2 / (stds**2 + 1e-8)
                        + 2 * log_stds
                        + np.log(2 * np.pi)
                    )
                    new_log_probs = torch.sum(new_log_probs, dim=-1)

                    # Account for tanh squashing (optional in this safe version)
                    # new_log_probs -= torch.sum(torch.log(1 - actions.pow(2) + 1e-8), dim=-1)

                    # Compute PPO ratio safely
                    ratio = torch.exp(
                        torch.clamp(new_log_probs - old_log_probs, -20, 20)
                    )
                    ratio = torch.clamp(ratio, 0.0, 5.0)  # Strict clipping of ratio

                    surr1 = ratio * advantages
                    surr2 = (
                        torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                        * advantages
                    )
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Compute value loss with clipping
                    value_pred = self.critic(states).squeeze()
                    value_loss = torch.nn.functional.mse_loss(value_pred, returns)

                    # Update networks with careful gradient clipping
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), max_norm=0.1
                    )  # Strict clipping
                    self.actor_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), max_norm=0.1
                    )
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
        game_log = open(self.env.save_file + "_game_log.csv", "w")
        game_logger = csv.writer(game_log)
        game_logger.writerow(
            ["time", "reward", "action", "spike_count", "state", "z_pos"]
        )

        reward_log = open(self.env.save_file + "_reward_log.csv", "w")
        reward_logger = csv.writer(reward_log)
        reward_logger.writerow(["time", "episode", "reward", "episode_steps"])

        pattern_log = open(self.env.save_file + "_pattern_log.csv", "w")
        pattern_logger = csv.writer(pattern_log)
        pattern_logger.writerow(["time", "pattern", "reward", "probs", "vals"])

        self.start_time = time.perf_counter()
        state_timer = self.time_elapsed()

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

                obs, done = self.env.step()

                if self.artifact_remover:
                    if obs is None:
                        continue
                    # Process all channels at once
                    obs_arr = (
                        np.array([obs[ch] for ch in self.motor_neurons])
                        .reshape(1, -1)
                        .T
                    )
                    clean_values, artifacts, spikes = self.artifact_remover.fit_step(
                        obs_arr
                    )
                    self.motor_spike_count += spikes.flatten()
                else:
                    # Process the observation
                    for frame, ch, amp in obs:
                        if ch in self.motor_neurons:
                            self.motor_spike_count[
                                np.where(self.motor_neurons == ch)
                            ] += 1

            # ~~~~~~~~~~~ Game phase Logic ~~~~~~~~~~~
            elif self.state == "game":
                if just_entered:
                    state_timer = self.time_elapsed()
                    episode_steps += 1

                    game_action = self.get_motor_signal()

                    self.game_obs, reward, game_done, trunc, inf = self.game_env.step(
                        game_action
                    )
                    game_done = (
                        game_done or trunc
                    )  # Handle both termination and truncation
                    if episode_steps >= max_episode_steps:
                        game_done = True

                    # Check if episode timeout or ant flipped over
                    z_pos = (
                        self.game_obs[0] if len(self.game_obs) > 0 else 0
                    )  # z-coordinate of torso

                    total_reward += reward

                    # Update memory with reward
                    if len(self.memory) > 0:
                        self.memory[-1]["reward"] = reward
                        self.memory[-1]["done"] = game_done

                    game_logger.writerow(
                        [
                            self.time_elapsed(),
                            reward,
                            game_action,
                            self.motor_spike_count,
                            self.game_obs,
                            z_pos,
                        ]
                    )

                    # Set sensory signal
                    self.set_sensory_signal(self.game_obs)

                    if self.verbose:
                        print(
                            f"E:{self.episode} S:{episode_steps} R:{reward} Z:{z_pos} ",
                            self.motor_spike_count,
                            f"\nRate {self.motor_spike_rate[0]:.2f} || {self.motor_spike_rate[1]:.2f}",
                            end="",
                        )
                        print(
                            f"|| {self.motor_spike_rate[2]:.2f} || {self.motor_spike_rate[3]:.2f} || {self.motor_spike_rate[4]:.2f} || {self.motor_spike_rate[5]:.2f} || {self.motor_spike_rate[6]:.2f} || {self.motor_spike_rate[7]:.2f}"
                        )

                    if game_done or episode_steps >= max_episode_steps:
                        if self.verbose and episode_steps >= max_episode_steps:
                            print(f"Episode truncated after {episode_steps} steps")
                        self.state = "train"
                        just_entered = True
                        # Update networks at end of episode
                        self._update_networks()
                        self._update_sensory_network()  # Add sensory network update
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
                    self.episode_reward = np.mean(rewards[-5:])  # Use last 5 rewards

                    reward_logger.writerow(
                        [self.time_elapsed(), self.episode, total_reward, episode_steps]
                    )
                    self.episode += 1
                    episode_steps = 0  # Reset episode steps

                    if (
                        self.episode >= self.n_episodes
                        or self.time_elapsed() > self.max_time
                    ):
                        done = True
                        return

                    if self.verbose:
                        print("Reward:", total_reward)
                        print("-" * 20)
                        print("Rewards:", rewards)

                    # Get pattern/update
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

                    obs, done = self.env.step(action=train_pulse, tag="train")
                    train_timer = self.time_elapsed()
                    continue

                if self.time_elapsed() - state_timer > self.train_period_ms / 1000:
                    self.state = "wait"
                    just_entered = True
                    self.game_obs, inf = self.game_env.reset()
                    total_reward = 0
                    continue

                # Training stimulation pulse
                if self.time_elapsed() - train_timer > 1 / train_freq:
                    # Get pattern/update
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

                    obs, done = self.env.step(action=train_pulse, tag="train")
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
        return True

    def time_elapsed(self):
        return time.perf_counter() - self.start_time

    def info(self):
        return {
            "motor_neurons": self.motor_neurons,
            "training_neurons": self.training_neurons,
            "read_period_ms": self.read_period_ms,
            "train_period_ms": self.train_period_ms,
            "n_episodes": self.n_episodes,
        }

    def predicted_time(self):
        """Returns the predicted time for the phase to run in seconds"""
        return self.predicted_time
