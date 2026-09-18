"""
Ant-v5 closed-loop phases for Experiment Framework V3.

Adapted for the Ant's
8-dimensional continuous action space.
"""

from __future__ import annotations

import csv
import time
from collections import deque

import numpy as np
import torch
import torch.optim as optim

from braindance.analysis.data_loader import convert_uint16_maxwell
from braindance.core.phases_v3.codec import (
    ContinuousOutputPPO,
    InputPolicyActorCritic,
    PPORolloutBuffer,
    ppo_update,
    sigmoid,
)
from braindance.core.phases_v3.experiment_v3 import Experiment
from braindance.core.trainer import ContextualTrainer, generate_tetanus_pattern
from braindance.games.ant import AntFeatureEnv
from spikelab import SpikeData

from .phase_base_v3 import PhaseV3


class AntPhaseV3(PhaseV3):
    """Closed-loop Ant phase with flexible encode/decode modes.

    Encode modes
    ------------
    ``"fixed_sigmoid"``
        Projected features are mapped through a fixed sigmoid to stim rates.
    ``"policy_continuous"``
        A learned ``InputPolicyActorCritic`` maps features to stim rates,
        trained via PPO from game reward.

    Decode modes
    ------------
    ``"ppo"``
        A ``ContinuousOutputPPO`` (Gaussian policy) maps motor spike rates
        to 8-dim continuous actions in [-1, 1].
    ``"direct"``
        Spike rates are linearly mapped 1:1 to the 8 action dimensions
        (no trainable parameters).
    """

    inputs = ["rt_sort_object", "electrodes_per_neuron"]
    outputs = ["total_episodes", "sorted_spikedata", "final_policy_state", "episode_rewards"]

    N_SENSORY = 8
    MIN_MOTOR = 8
    MIN_TRAINING = 2
    MAX_TRAINING = 6
    N_ACTIONS = 8

    @staticmethod
    def allocate_neurons(
        n_sequences,
        n_sensory=8,
        min_motor=8,
        min_training=2,
        max_training=6,
    ):
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
        amp_mv: int = 200,
        phase_width: int = 100,
        read_period_ms: int = 50,
        train_period_ms: int = 200,
        wait_period_ms: int = 400,
        n_episodes: int = 100,
        trainer=None,
        trainer_type: str | None = None,
        use_contextual_trainer: bool = False,
        contextual_trainer_kwargs: dict | None = None,
        minibatch_size: int = 5,
        verbose: bool = False,
        max_time=np.inf,
        rt_sort_path: str | None = None,
        detection_model_path: str | None = None,
        use_numba: bool = True,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        ppo_epochs: int = 4,
        ppo_minibatch_size: int = 32,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_hidden_size: int = 64,
        n_features: int = 8,
        projection_seed: int = 0,
        projection_scale: float = 4.0,
        encode_mode: str = "fixed_sigmoid",
        decode_mode: str = "ppo",
        reward_mode: str = "forward_x",
        max_stim_hz: float = 10.0,
        input_gain: float = 6.0,
        input_center: float = 0.5,
        direct_scale: float = 3.0,
        direct_offset: float = 1.0,
        render_mode: str | None = "human",
        max_episode_steps: int = 1000,
        recording_tag: str = "ant",
        suffix: str = "_ant",
    ):
        super().__init__("AntPhaseV3", suffix, recording_tag=recording_tag)
        from braindance import get_rt_sort_path

        if encode_mode not in {"fixed_sigmoid", "policy_continuous"}:
            raise ValueError(f"Unknown encode_mode: {encode_mode}")
        if decode_mode not in {"ppo", "direct"}:
            raise ValueError(f"Unknown decode_mode: {decode_mode}")

        self.game_env = AntFeatureEnv(
            render_mode=render_mode,
            n_features=n_features,
            projection_seed=projection_seed,
            projection_scale=projection_scale,
            reward_mode=reward_mode,
        )

        self.read_period_ms = read_period_ms
        self.train_period_ms = train_period_ms
        self.wait_period_ms = wait_period_ms
        self.n_episodes = n_episodes
        self.episode = 0
        self.max_time = max_time
        self.max_episode_steps = max_episode_steps
        self.verbose = verbose
        self.minibatch_size = minibatch_size

        self.amp_mv = amp_mv
        self.phase_width = phase_width
        self.trainer = trainer
        self.trainer_type = trainer_type
        if self.trainer is not None:
            self.trainer.phase = self
        self.use_contextual_trainer = use_contextual_trainer
        self.contextual_trainer_kwargs = contextual_trainer_kwargs or {}

        self.rt_sort_path = rt_sort_path
        self.detection_model_path = (
            get_rt_sort_path() if detection_model_path is None else detection_model_path
        )
        self.use_numba = use_numba

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.ppo_minibatch_size = ppo_minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.policy_hidden_size = policy_hidden_size

        self.n_features = n_features
        self.encode_mode = encode_mode
        self.decode_mode = decode_mode
        self.reward_mode = reward_mode
        self.max_stim_hz = float(max_stim_hz)
        self.input_gain = float(input_gain)
        self.input_center = float(input_center)
        self.direct_scale = float(direct_scale)
        self.direct_offset = float(direct_offset)

        self.sensory_neurons = (
            np.asarray(sensory_neurons, dtype=int) if sensory_neurons is not None else None
        )
        self.motor_neurons = (
            np.asarray(motor_neurons, dtype=int) if motor_neurons is not None else None
        )
        self.training_neurons = (
            np.asarray(training_neurons, dtype=int)
            if training_neurons is not None
            else None
        )

        self.sensory_stim_Hz = np.zeros(
            len(self.sensory_neurons) if self.sensory_neurons is not None else 0,
            dtype=np.float32,
        )
        self.motor_spike_count = np.zeros(
            len(self.motor_neurons) if self.motor_neurons is not None else 0,
            dtype=np.float32,
        )
        self.motor_spike_rate = np.zeros_like(self.motor_spike_count)

        self.state = "read"
        self.game_features = None
        self.game_info = None
        self.episode_rewards = []
        self.episode_reward = None
        self.episode_reward_change = None
        self.episode_length = 0
        self.last_action_ind = None
        self.last_action_inds = deque(maxlen=self.minibatch_size)

        self.rt_sort = None
        self.electrodes = None
        self.motor_lookup = {}
        self.data_buffer = None
        self.buffer_write_pos = 0

        self.input_policy = None
        self.input_optimizer = None
        self.output_policy = None
        self.output_optimizer = None
        self.input_buffer = PPORolloutBuffer()
        self.output_buffer = PPORolloutBuffer()
        self.pending_input_transition = None
        self.pending_output_transition = None
        self.last_input_policy_grad_magnitude = 0.0
        self.last_output_policy_grad_magnitude = 0.0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_from_experiment(self, experiment: Experiment):
        super().configure_from_experiment(experiment)

        self.rt_sort = experiment.data.rt_sort_object
        self.rt_sort.set_model(self.detection_model_path)

        channels = [int(ch) for ch in self.rt_sort.get_seq_root_elecs()]
        if (
            self.sensory_neurons is None
            or self.motor_neurons is None
            or self.training_neurons is None
        ):
            sensory, motor, training = self.allocate_neurons(
                len(channels),
                n_sensory=self.n_features,
                min_motor=self.MIN_MOTOR,
                min_training=self.MIN_TRAINING,
                max_training=self.MAX_TRAINING,
            )
            if self.sensory_neurons is None:
                self.sensory_neurons = np.asarray(sensory, dtype=int)
            if self.motor_neurons is None:
                self.motor_neurons = np.asarray(motor, dtype=int)
            if self.training_neurons is None:
                self.training_neurons = np.asarray(training, dtype=int)

        self.sensory_neurons = np.asarray(self.sensory_neurons, dtype=int)
        self.motor_neurons = np.asarray(self.motor_neurons, dtype=int)
        self.training_neurons = np.asarray(self.training_neurons, dtype=int)
        self.motor_lookup = {
            int(neuron): idx for idx, neuron in enumerate(self.motor_neurons)
        }

        experiment.load_mapping()
        self.electrodes = experiment.mapping.get_electrodes(channels=channels)
        electrodes_array = np.asarray(self.electrodes, dtype=int)

        self.sensory_electrodes = electrodes_array[self.sensory_neurons]
        self.motor_electrodes = electrodes_array[self.motor_neurons]
        self.training_electrodes = electrodes_array[self.training_neurons]

        self.stim_electrodes = np.concatenate(
            [self.sensory_electrodes, self.training_electrodes]
        ).astype(int)
        self.sensory_stim_inds = np.arange(len(self.sensory_electrodes), dtype=int)
        self.training_stim_inds = np.arange(
            len(self.sensory_electrodes),
            len(self.stim_electrodes),
            dtype=int,
        )

        self.sensory_stim_Hz = np.zeros(len(self.sensory_neurons), dtype=np.float32)
        self.motor_spike_count = np.zeros(len(self.motor_neurons), dtype=np.float32)
        self.motor_spike_rate = np.zeros(len(self.motor_neurons), dtype=np.float32)

        if self.trainer is None and self.trainer_type is not None and self.trainer_type != "none":
            self._create_deferred_trainer()

        if self.encode_mode == "policy_continuous":
            self.input_policy = InputPolicyActorCritic(
                self.game_env.get_feature_dim(),
                len(self.sensory_neurons),
                hidden_size=self.policy_hidden_size,
            )
            self.input_optimizer = optim.Adam(
                self.input_policy.parameters(),
                lr=self.learning_rate,
            )

        if self.decode_mode == "ppo":
            self.output_policy = ContinuousOutputPPO(
                len(self.motor_neurons),
                self.N_ACTIONS,
                hidden_size=self.policy_hidden_size,
            )
            self.output_optimizer = optim.Adam(
                self.output_policy.parameters(),
                lr=self.learning_rate,
            )

        if self.verbose:
            print("=" * 20)
            print("Sensory neurons:", self.sensory_neurons)
            print("Motor neurons:", self.motor_neurons)
            print("Training neurons:", self.training_neurons)
            print("Encode mode:", self.encode_mode)
            print("Decode mode:", self.decode_mode)
            print("Reward mode:", self.reward_mode)
            print("=" * 20)

    def _create_deferred_trainer(self):
        from braindance.core.trainer import TetanusTrainer

        training_neuron_inds = self.training_stim_inds.tolist()
        n_neurons = len(training_neuron_inds)
        patterns = []
        if n_neurons >= 3:
            patterns.append(
                generate_tetanus_pattern(
                    training_neuron_inds[:3],
                    stim_count=3, delay_ms=5,
                    amp_mv=self.amp_mv, pulse_width=self.phase_width,
                )
            )
            patterns.append(
                generate_tetanus_pattern(
                    training_neuron_inds[-3:],
                    stim_count=3, delay_ms=5,
                    amp_mv=self.amp_mv, pulse_width=self.phase_width,
                )
            )
        elif n_neurons > 0:
            patterns.append(
                generate_tetanus_pattern(
                    training_neuron_inds,
                    stim_count=min(3, n_neurons), delay_ms=5,
                    amp_mv=self.amp_mv, pulse_width=self.phase_width,
                )
            )

        freq = 30
        if self.trainer_type == "tetanus":
            self.trainer = TetanusTrainer(
                choices=patterns, freqs=freq,
                start_value=10, learning_rate=0.1,
                floor=10, discount_factor=0.3,
            )
            self.use_contextual_trainer = False
        elif self.trainer_type == "contextual":
            self.trainer = ContextualTrainer(
                choices=patterns, freqs=freq,
                hidden_sizes=[64, 32], learning_rate=0.01,
                gamma=0.95, baseline_window=20,
                normalize_context=True, no_stim_bias=0.0,
            )
            self.use_contextual_trainer = True
        else:
            raise ValueError(f"Unknown trainer type: {self.trainer_type}")

        self.trainer.phase = self

    def customize_environment_params(self, env_params: dict) -> dict:
        env_params["stim_electrodes"] = self.stim_electrodes.tolist()
        return env_params

    # ------------------------------------------------------------------
    # Data buffer helpers
    # ------------------------------------------------------------------

    def _add_to_buffer(self, new_data, n_channels):
        new_data = np.asarray(new_data)
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(-1, 1)
        if new_data.shape[1] != n_channels:
            new_data = new_data[:, :n_channels]

        n_new_samples = new_data.shape[0]
        window_buffer_size = self.data_buffer.shape[0]

        if n_new_samples >= window_buffer_size:
            self.data_buffer[:] = new_data[-window_buffer_size:]
            self.buffer_write_pos = 0
            return

        end_pos = self.buffer_write_pos + n_new_samples
        if end_pos <= window_buffer_size:
            self.data_buffer[self.buffer_write_pos:end_pos] = new_data
        else:
            split_point = window_buffer_size - self.buffer_write_pos
            self.data_buffer[self.buffer_write_pos:] = new_data[:split_point]
            self.data_buffer[: end_pos - window_buffer_size] = new_data[split_point:]
        self.buffer_write_pos = (self.buffer_write_pos + n_new_samples) % window_buffer_size

    def _get_window(self):
        if self.buffer_write_pos == 0:
            return self.data_buffer.copy()
        return np.vstack(
            [
                self.data_buffer[self.buffer_write_pos:],
                self.data_buffer[: self.buffer_write_pos],
            ]
        )

    # ------------------------------------------------------------------
    # Encoding: game features -> sensory stimulation rates
    # ------------------------------------------------------------------

    def _fixed_sigmoid_rates(self, features):
        features = np.asarray(features, dtype=np.float32)
        logits = self.input_gain * (features - self.input_center)
        return (self.max_stim_hz * sigmoid(logits)).astype(np.float32)

    def set_sensory_signal(self, game_features):
        if self.encode_mode == "fixed_sigmoid":
            self.sensory_stim_Hz = self._fixed_sigmoid_rates(game_features)
            self.pending_input_transition = None
            return self.sensory_stim_Hz

        state = np.asarray(game_features, dtype=np.float32)
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        latent_action, rates, log_prob, value = self.input_policy.act(
            state_tensor,
            max_stim_hz=self.max_stim_hz,
        )
        self.sensory_stim_Hz = rates.astype(np.float32)
        self.pending_input_transition = {
            "state": state.copy(),
            "action": latent_action.copy(),
            "log_prob": log_prob,
            "value": value,
        }
        return self.sensory_stim_Hz

    # ------------------------------------------------------------------
    # Decoding: motor spike rates -> 8-dim continuous action
    # ------------------------------------------------------------------

    def _update_motor_rates(self, moving_avg=0.2):
        self.motor_spike_rate = (
            moving_avg * self.motor_spike_rate
            + (1.0 - moving_avg) * self.motor_spike_count
        )
        return self.motor_spike_rate.copy()

    def get_motor_signal(self, moving_avg=0.2):
        """PPO continuous decode: spike rates -> Normal -> tanh-squashed action."""
        state = self._update_motor_rates(moving_avg=moving_avg)
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        action, log_prob, value = self.output_policy.act(state_tensor)
        self.pending_output_transition = {
            "state": state.copy(),
            "action": action.copy(),
            "log_prob": log_prob,
            "value": value,
        }
        return action

    def get_motor_signal_direct(self, moving_avg=0.2):
        """Direct decode: linearly map first N_ACTIONS motor spike rates to [-1, 1]."""
        rates = self._update_motor_rates(moving_avg=moving_avg)
        action = np.zeros(self.N_ACTIONS, dtype=np.float32)
        n = min(self.N_ACTIONS, len(rates))
        action[:n] = np.clip(rates[:n] * self.direct_scale - self.direct_offset, -1.0, 1.0)
        return action

    # ------------------------------------------------------------------
    # Sensory stimulation & observation processing
    # ------------------------------------------------------------------

    def _step_sensory_environment(self, buffer_size):
        thresholds = np.full(len(self.sensory_stim_Hz), np.inf, dtype=np.float32)
        positive_mask = self.sensory_stim_Hz > 0
        thresholds[positive_mask] = 1.0 / self.sensory_stim_Hz[positive_mask]
        active_local_inds = np.where(
            self.env.stim_dts[self.sensory_stim_inds] > thresholds
        )[0]

        if len(active_local_inds) > 0:
            return self.env.step(
                action=(
                    self.sensory_stim_inds[active_local_inds],
                    self.amp_mv,
                    self.phase_width,
                ),
                tag="sensory",
                buffer_size=buffer_size,
            )
        return self.env.step(buffer_size=buffer_size)

    def _process_observation(
        self, obs, n_channels, step_size_samples, samples_since_last_process,
    ):
        sorted_spikes = []
        if obs is None:
            return sorted_spikes, samples_since_last_process

        obs = convert_uint16_maxwell(obs)
        self._add_to_buffer(obs, n_channels)
        samples_since_last_process += obs.shape[0]

        if samples_since_last_process < step_size_samples:
            return sorted_spikes, samples_since_last_process

        window_data = self._get_window()
        spikes = self.rt_sort.running_sort(
            window_data,
            use_numba=self.use_numba,
            latest_frame=self.env.latest_frame,
        )
        sorted_spikes.extend(spikes)
        for spike in spikes:
            neuron_id = int(spike[0])
            if neuron_id in self.motor_lookup:
                self.motor_spike_count[self.motor_lookup[neuron_id]] += 1

        return sorted_spikes, 0

    # ------------------------------------------------------------------
    # Policy bookkeeping
    # ------------------------------------------------------------------

    def _record_policy_transitions(self, reward, done):
        if self.encode_mode == "policy_continuous" and self.pending_input_transition is not None:
            self.input_buffer.add(
                state=self.pending_input_transition["state"],
                action=self.pending_input_transition["action"],
                log_prob=self.pending_input_transition["log_prob"],
                value=self.pending_input_transition["value"],
                reward=reward,
                done=done,
            )
            self.pending_input_transition = None

        if self.decode_mode == "ppo" and self.pending_output_transition is not None:
            self.output_buffer.add(
                state=self.pending_output_transition["state"],
                action=self.pending_output_transition["action"],
                log_prob=self.pending_output_transition["log_prob"],
                value=self.pending_output_transition["value"],
                reward=reward,
                done=done,
            )
            self.pending_output_transition = None

    def get_training_signal(self, train_Hz=30, stim_count=3, delay_ms=5, use_change=True):
        if self.trainer:
            if self.episode_reward is not None and len(self.last_action_inds) == self.minibatch_size:
                trainer_reward = self.episode_reward_change if use_change else self.episode_reward
                self.trainer.update_values(trainer_reward, self.last_action_inds[-1])

            if self.use_contextual_trainer and isinstance(self.trainer, ContextualTrainer):
                context = self.trainer.compute_context(
                    motor_spike_rates=self.motor_spike_rate,
                    episode_reward=self.episode_reward if self.episode_reward else 0,
                    episode_length=self.episode_length,
                )
                train_action, train_Hz, self.last_action_ind = self.trainer.get_action(
                    context=context,
                )
                if self.episode_reward is not None:
                    self.trainer.record_episode_reward(self.episode_reward)
            else:
                train_action, train_Hz, self.last_action_ind = self.trainer.get_action()

            self.last_action_inds.append(self.last_action_ind)
            return train_action, train_Hz

        if len(self.training_stim_inds) == 0:
            return None, 0

        stim_count = min(stim_count, len(self.training_stim_inds))
        neuron_order = np.random.choice(self.training_stim_inds, size=stim_count, replace=False)
        train_action = []
        for index, stim_ind in enumerate(neuron_order):
            train_action.append(("stim", [int(stim_ind)], self.amp_mv, self.phase_width))
            if index != len(neuron_order) - 1:
                train_action.append(("delay", delay_ms))
        return train_action, train_Hz

    def _update_policies(self):
        ppo_kwargs = dict(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_epsilon=self.clip_epsilon,
            ppo_epochs=self.ppo_epochs,
            ppo_minibatch_size=self.ppo_minibatch_size,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
            max_grad_norm=self.max_grad_norm,
        )

        if self.encode_mode == "policy_continuous" and len(self.input_buffer) > 0:
            self.last_input_policy_grad_magnitude = ppo_update(
                self.input_policy, self.input_optimizer, self.input_buffer,
                action_type="continuous", **ppo_kwargs,
            )
        else:
            self.last_input_policy_grad_magnitude = 0.0

        if self.decode_mode == "ppo" and len(self.output_buffer) > 0:
            self.last_output_policy_grad_magnitude = ppo_update(
                self.output_policy, self.output_optimizer, self.output_buffer,
                action_type="continuous", **ppo_kwargs,
            )
        else:
            self.last_output_policy_grad_magnitude = 0.0

        if self.use_contextual_trainer and self.trainer is not None:
            self.trainer.set_policy_grad_magnitude(
                self.last_output_policy_grad_magnitude
                + self.last_input_policy_grad_magnitude
            )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, experiment):
        done = False
        sorted_spikes = []

        self.game_features, self.game_info = self.game_env.reset()
        self.set_sensory_signal(self.game_features)

        window_size_ms = 20.0
        overlap_ms = 4.5
        sample_rate_khz = 20.0
        window_size_samples = int(window_size_ms * sample_rate_khz)
        overlap_samples = int(overlap_ms * sample_rate_khz)
        step_size_samples = window_size_samples - overlap_samples
        buffer_size = int(2.0 * sample_rate_khz)

        obs, _ = self.env.step(buffer_size=buffer_size)
        obs = np.asarray(obs)
        n_channels = obs.shape[1]
        self.data_buffer = np.zeros((window_size_samples, n_channels), dtype=obs.dtype)
        self.buffer_write_pos = 0

        if self.verbose:
            print("   Warming up RTSort...")
        for _ in range(100):
            warmup_obs, _ = self.env.step(buffer_size=buffer_size)
            if warmup_obs is not None:
                warmup_obs = convert_uint16_maxwell(warmup_obs)
                self.rt_sort.running_sort(warmup_obs, use_numba=self.use_numba)
                self._add_to_buffer(warmup_obs, n_channels)
        if self.verbose:
            print("   Warmup completed!")

        game_log = open(self.env.save_file + "_ant_game_log.csv", "w", newline="")
        reward_log = open(self.env.save_file + "_ant_reward_log.csv", "w", newline="")
        pattern_log = open(self.env.save_file + "_ant_pattern_log.csv", "w", newline="")

        game_logger = csv.writer(game_log)
        reward_logger = csv.writer(reward_log)
        pattern_logger = csv.writer(pattern_log)

        game_logger.writerow([
            "time", "episode", "reward", "total_reward", "action",
            "decode_mode", "encode_mode", "spike_rates", "sensory_stim_hz",
        ])
        reward_logger.writerow(["time", "episode", "reward", "episode_steps"])
        pattern_logger.writerow(["time", "pattern", "reward", "probs", "vals"])

        self.start_time = time.perf_counter()
        state_timer = self.time_elapsed()
        samples_since_last_process = 0
        just_entered = True
        total_reward = 0.0
        current_episode_rewards = []
        episode_steps = 0
        train_timer = self.time_elapsed()
        last_status_time = 0.0

        try:
            while not done:
                # ~~~~~~~~~~~ Read ~~~~~~~~~~~
                if self.state == "read":
                    if just_entered:
                        state_timer = self.time_elapsed()
                        just_entered = False
                        self.motor_spike_count = np.zeros(
                            len(self.motor_neurons), dtype=np.float32,
                        )

                    if self.time_elapsed() - state_timer > self.read_period_ms / 1000.0:
                        self.state = "game"
                        just_entered = True
                        continue

                    obs, done = self._step_sensory_environment(buffer_size=buffer_size)
                    spikes, samples_since_last_process = self._process_observation(
                        obs,
                        n_channels=n_channels,
                        step_size_samples=step_size_samples,
                        samples_since_last_process=samples_since_last_process,
                    )
                    sorted_spikes.extend(spikes)

                    if self.verbose and self.time_elapsed() - last_status_time > 1.0:
                        last_status_time = self.time_elapsed()
                        spk = self.motor_spike_count
                        rate = self.motor_spike_rate
                        stim = self.sensory_stim_Hz
                        print(
                            f"  [{self.time_elapsed():6.1f}s] "
                            f"E{self.episode} S{episode_steps} "
                            f"R={total_reward:+.1f} "
                            f"spk=[{' '.join(f'{v:.0f}' for v in spk[:8])}] "
                            f"rate=[{' '.join(f'{v:.1f}' for v in rate[:8])}] "
                            f"stim=[{' '.join(f'{v:.1f}' for v in stim[:8])}]"
                        )

                # ~~~~~~~~~~~ Game ~~~~~~~~~~~
                elif self.state == "game":
                    if just_entered:
                        just_entered = False
                        episode_steps += 1

                        if self.decode_mode == "ppo":
                            game_action = self.get_motor_signal()
                        else:
                            game_action = self.get_motor_signal_direct()

                        next_features, reward, terminated, truncated, info = (
                            self.game_env.step(game_action)
                        )
                        game_done = bool(
                            terminated or truncated
                            or episode_steps >= self.max_episode_steps
                        )

                        total_reward += reward
                        current_episode_rewards.append(float(reward))
                        self._record_policy_transitions(reward=reward, done=game_done)

                        game_logger.writerow([
                            self.time_elapsed(),
                            self.episode,
                            reward,
                            total_reward,
                            game_action.tolist() if hasattr(game_action, "tolist") else game_action,
                            self.decode_mode,
                            self.encode_mode,
                            self.motor_spike_rate.tolist(),
                            self.sensory_stim_Hz.tolist(),
                        ])

                        self.game_features = next_features
                        self.game_info = info
                        self.set_sensory_signal(self.game_features)

                        self.state = "train" if game_done else "read"
                        just_entered = True
                        continue

                # ~~~~~~~~~~~ Train ~~~~~~~~~~~
                elif self.state == "train":
                    if just_entered:
                        state_timer = self.time_elapsed()
                        train_timer = self.time_elapsed()
                        just_entered = False

                        self.episode_reward = float(total_reward)
                        self.episode_rewards.append(self.episode_reward)
                        self.episode_length = len(current_episode_rewards)
                        recent_rewards = self.episode_rewards[-self.minibatch_size:]
                        self.episode_reward_change = (
                            float(
                                np.mean(recent_rewards)
                                - np.mean(self.episode_rewards[-20:])
                            )
                            if len(self.episode_rewards) >= 20
                            else 0.0
                        )

                        self._update_policies()

                        do_training = False
                        train_pulse = None
                        train_freq = 30
                        if self.trainer is not None:
                            if self.use_contextual_trainer and isinstance(
                                self.trainer, ContextualTrainer,
                            ):
                                train_pulse, train_freq = self.get_training_signal()
                                do_training = train_pulse is not None
                            else:
                                reward_increased = self.episode_reward_change > 0
                                do_training = not reward_increased or getattr(
                                    self, "force_train", False,
                                )

                        reward_logger.writerow([
                            self.time_elapsed(), self.episode,
                            self.episode_reward, episode_steps,
                        ])
                        self.episode += 1
                        episode_steps = 0

                        if self.verbose:
                            print(
                                f"Episode {self.episode}: reward={self.episode_reward:.2f} "
                                f"input_grad={self.last_input_policy_grad_magnitude:.4f} "
                                f"output_grad={self.last_output_policy_grad_magnitude:.4f}"
                            )

                        if (
                            self.episode >= self.n_episodes
                            or self.time_elapsed() > self.max_time
                        ):
                            done = True
                            break

                        if do_training:
                            if not (
                                self.use_contextual_trainer
                                and isinstance(self.trainer, ContextualTrainer)
                            ):
                                train_pulse, train_freq = self.get_training_signal()
                            pattern_logger.writerow([
                                self.time_elapsed(),
                                self.trainer.get_action(self.last_action_ind)
                                if self.trainer else None,
                                self.episode_reward,
                                self.trainer.get_probs() if self.trainer else None,
                                self.trainer.get_values() if self.trainer else None,
                            ])

                    if self.time_elapsed() - state_timer > self.train_period_ms / 1000.0:
                        self.state = "wait"
                        just_entered = True
                        self.game_features, self.game_info = self.game_env.reset()
                        self.set_sensory_signal(self.game_features)
                        total_reward = 0.0
                        current_episode_rewards = []
                        continue

                    if not do_training:
                        self.state = "wait"
                        just_entered = True
                        self.game_features, self.game_info = self.game_env.reset()
                        self.set_sensory_signal(self.game_features)
                        total_reward = 0.0
                        current_episode_rewards = []
                        continue

                    if (
                        train_freq > 0
                        and self.time_elapsed() - train_timer > 1.0 / train_freq
                    ):
                        if train_pulse is not None:
                            self.env.step(action=train_pulse, tag="train")
                        else:
                            self.env.step()
                        train_timer = self.time_elapsed()
                    else:
                        self.env.step()

                # ~~~~~~~~~~~ Wait ~~~~~~~~~~~
                elif self.state == "wait":
                    if just_entered:
                        state_timer = self.time_elapsed()
                        just_entered = False

                    if self.time_elapsed() - state_timer > self.wait_period_ms / 1000.0:
                        self.state = "read"
                        just_entered = True
                        continue

                    self.env.step()

        finally:
            game_log.close()
            reward_log.close()
            pattern_log.close()

        sorted_spikedata = SpikeData.from_events(sorted_spikes) if sorted_spikes else None
        final_policy_state = {
            "input_policy": self.input_policy.state_dict() if self.input_policy else None,
            "output_policy": self.output_policy.state_dict() if self.output_policy else None,
            "encode_mode": self.encode_mode,
            "decode_mode": self.decode_mode,
            "reward_mode": self.reward_mode,
        }
        return {
            "total_episodes": self.episode,
            "sorted_spikedata": sorted_spikedata,
            "final_policy_state": final_policy_state,
            "episode_rewards": list(self.episode_rewards),
        }

    def cleanup(self):
        self.game_env.close()

    def info(self):
        return {
            "sensory_neurons": self.sensory_neurons,
            "motor_neurons": self.motor_neurons,
            "training_neurons": self.training_neurons,
            "read_period_ms": self.read_period_ms,
            "train_period_ms": self.train_period_ms,
            "wait_period_ms": self.wait_period_ms,
            "n_episodes": self.n_episodes,
            "encode_mode": self.encode_mode,
            "decode_mode": self.decode_mode,
            "reward_mode": self.reward_mode,
        }
