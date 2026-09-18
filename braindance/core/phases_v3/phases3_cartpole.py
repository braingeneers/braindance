"""Native V3 implementation of the paper CartPole experiment.

The game loop, neural encoding/decoding, tetanus scheduling, electrode routing,
and ranked-pair metric live here. Baseline and evoked-response analysis live in
phases3_cartpole_analysis. Physical electrodes become stimulation indices and
acquisition channels only when the game session is configured.
"""
import csv
import hashlib
import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .phase_base_v3 import AnalysisPhaseV3, PhaseV3
from .phases3 import NeuralSweepPhaseV3
from .phases3_cartpole_analysis import CartPoleFootprintPhaseV3, CartPoleCausalAnalysisPhaseV3


def prepare_session(config, run_index=0, n_episodes=200, max_time_sec=900):
    """Resolve physical routing and the historical trainer for one session."""
    from braindance.config import get_output_dir
    from braindance.core.params import maxwell_params
    from braindance.core.trainer import TetanusTrainer, generate_permutations
    from braindance.analysis.mapping import Mapping

    mode = config["type"]
    if mode not in ("C1", "C2", "C7", "punishment", "reward", "always"):
        raise ValueError("type must be C1, C2, C7, punishment, reward, or always")
    if run_index < 0:
        raise ValueError("run_index must be nonnegative")
    config = dict(config)
    sensory = config["sensory_electrodes"]
    motor = config["motor_electrodes"]
    if len(sensory) != 2 or len(motor) != 2 or len(set(sensory + motor)) != 4:
        raise ValueError("Expected two distinct sensory and two distinct motor electrodes")
    stim = [e for e in config["stim_electrodes"] if e not in motor]
    if len(stim) != len(set(stim)):
        raise ValueError("stim_electrodes must be unique")
    sensory_indices = [stim.index(e) for e in sensory]
    training_indices = [i for i in range(len(stim)) if i not in sensory_indices]
    if len(training_indices) < 2:
        raise ValueError("The historical trainer needs at least two training electrodes")
    valid = config["valid_stim_electrodes"]
    derived = Path(config["derived_dir"])
    means = np.load(derived / "causal_connectivity_multi_mean.npy", allow_pickle=False)
    stds = np.load(derived / "causal_connectivity_multi_std.npy", allow_pickle=False)
    if means.shape != (len(valid),) or stds.shape != means.shape:
        raise ValueError("Normalization arrays must match valid_stim_electrodes order")
    motor_indices = [valid.index(e) for e in motor]
    normalization = np.hstack([means[motor_indices], stds[motor_indices]])
    if not np.isfinite(normalization).all() or np.any(means[motor_indices] == 0):
        raise ValueError("Motor normalization requires finite values and nonzero means")
    # Input mapping is explicit; never infer it from an output directory/name.
    mapping = Mapping.from_csv(config["mapping_file_path"])
    channels = [int(c) for c in mapping.get_orig_channels(electrodes=motor)]
    if len(channels) != 2 or len(set(channels)) != 2:
        raise ValueError("Both motor electrodes must map to distinct acquisition channels")

    patterns = generate_permutations(training_indices, stim_count=2, delay_ms=10)
    learned = TetanusTrainer(patterns, 10, learning_rate=.3, floor=.1, start_value=1)
    random = TetanusTrainer(patterns.copy(), 10, learning_rate=0)
    none = TetanusTrainer([None] * 20, 10, learning_rate=0)
    schedules = {"C1": [none, learned, none], "C2": [none, random, none]}
    schedule = schedules.get(mode, [learned])
    trainer = schedule[run_index % len(schedule)]
    if mode in schedules:
        normalization = 1
    training_type = mode if mode in ("punishment", "reward", "always") else "punishment"
    params = maxwell_params.copy()
    output = Path(config.get("save_dir", get_output_dir() / "cartpole"))
    output.mkdir(parents=True, exist_ok=True)
    params.update(save_dir=str(output), name=config["name"] + "_cartpole_F",
                  config=config["config"], observation_type="raw",
                  max_time_sec=max_time_sec, stim_electrodes=stim)
    config.update(stim_electrodes=stim, sensory_neurons=sensory_indices,
                  motor_channels=channels, training_neurons=training_indices,
                  run_index=run_index, n_episodes=n_episodes, max_time_sec=max_time_sec)
    phase_params = dict(
        sensory_neurons=sensory_indices, motor_neurons=channels,
        training_neurons=training_indices, verbose=True, read_period_ms=200,
        train_period_ms=400, wait_period_ms=3000, phase_width=200,
        n_episodes=n_episodes, trainer=trainer, artifact_removal=True,
        continuous=True, assistive=0, spike_thresh=[-4.4, -25],
        normalization=normalization, training_type=training_type)
    return config, params, phase_params


def find_connectivity_patterns(conn_matrix, high_threshold=1.5, low_threshold=-0.5,
                               w_ab=1.0, w_cd=1.0, w_ad=-1.0, w_cb=-1.0,
                               w_ac=-0.3, w_ca=-0.3):
    """Rank distinct sensory/motor pairs with the original paper metric."""
    n = conn_matrix.shape[0]
    results = []
    used_combinations = set()
    reaction_means = np.mean(conn_matrix, axis=0)
    reaction_stds = np.std(conn_matrix, axis=0)
    zscore_matrix = (conn_matrix - reaction_means) / reaction_stds
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            for c in range(n):
                if c == a or c == b:
                    continue
                for d in range(n):
                    if d == a or d == b or d == c:
                        continue
                    combo = frozenset([(a, b), (c, d)])
                    if combo in used_combinations:
                        continue
                    used_combinations.add(combo)
                    b_mean = reaction_means[b]
                    b_std = reaction_stds[b]
                    d_mean = reaction_means[d]
                    d_std = reaction_stds[d]
                    motor_similarity = -abs(b_mean - d_mean)
                    z_ab = zscore_matrix[a, b] * b_std
                    z_cd = zscore_matrix[c, d] * d_std
                    z_ad = zscore_matrix[a, d] * d_std
                    z_cb = zscore_matrix[c, b] * b_std
                    z_ac = zscore_matrix[a, c]
                    z_ca = zscore_matrix[c, a]
                    score = w_ab * z_ab + w_cd * z_cd + w_ad * z_ad + w_cb * z_cb + w_ac * z_ac + w_ca * z_ca + 5 * motor_similarity
                    results.append((a, b, c, d, score, 5 * motor_similarity))
    results.sort(key=lambda x: x[4], reverse=True)
    return results


class CartPoleCausalSweepPhaseV3(NeuralSweepPhaseV3):
    """Historical sweep, including acquisition of the final stimulus response."""

    def __init__(self):
        # One amplitude makes ran equivalent to the historical rna ordering.
        super().__init__(amp_bounds=400, stim_freq=2, replicates=50,
                         order='ran', single_connect=True, phase_length=200,
                         tag='causal', recording_tag='causal')

    def customize_environment_params(self, env_params):
        env_params = super().customize_environment_params(env_params)
        env_params['max_time_sec'] = max(1800, len(self.stim_electrodes) * 25 + 10)
        return env_params

    def run(self, experiment):
        result = super().run(experiment)
        if len(result['sweep_results']) != len(self.neuron_list) * self.replicates:
            raise RuntimeError('Causal sweep ended before all stimuli were delivered')
        # Include the analyzer's padding around its 300 ms response window.
        end = self.time_elapsed() + .5
        while self.time_elapsed() < end:
            _, done = self.env.step()
            if done:
                raise RuntimeError('Acquisition ended before the final causal response')
        return result


class CartPoleRankPairsPhaseV3(AnalysisPhaseV3):
    inputs = ['valid_stim_electrodes', 'derived_dir']
    outputs = ['sensory_electrodes', 'motor_electrodes', 'pair_selection']

    def __init__(self, rank=1, order='multi'):
        super().__init__()
        if rank < 1 or order not in ('first', 'multi'):
            raise ValueError('rank must be positive and order must be first or multi')
        self.rank, self.order = rank, order

    def run(self, experiment):
        electrodes = list(experiment.data.valid_stim_electrodes)
        path = Path(experiment.data.derived_dir) / f'causal_connectivity_{self.order}.npy'
        matrix = np.load(path, allow_pickle=False)
        if (len(electrodes) < 4 or len(set(electrodes)) != len(electrodes)
                or matrix.shape != (len(electrodes), len(electrodes))
                or not np.isfinite(matrix).all()):
            raise ValueError('Need a finite square matrix matching at least four unique electrode IDs')
        if np.any(np.std(matrix, axis=0) == 0):
            raise ValueError('Ranking metric is undefined for zero-variance columns')
        patterns = find_connectivity_patterns(matrix)
        if self.rank > len(patterns):
            raise ValueError(f'rank must be between 1 and {len(patterns)}')
        a, b, c, d, score, _ = patterns[self.rank - 1]
        sensory, motor = [electrodes[a], electrodes[c]], [electrodes[b], electrodes[d]]
        print(f'   Rank {self.rank}: sensory={sensory}, motor={motor}, score={score:.6g}')
        return dict(sensory_electrodes=sensory, motor_electrodes=motor,
                    pair_selection=dict(
                        metric='ranked_pairs.find_connectivity_patterns', order=self.order,
                        rank=self.rank, score=float(score), matrix_path=str(path.resolve()),
                        matrix_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        source_commit='5f733a8821d63d502042acc66e59abd303e6dae3'))


class PaperCartPolePhaseV3(PhaseV3):
    """Direct neural control and tetanus training for one paper CartPole session.

    V3 owns Maxwell acquisition. This phase owns the continuous CartPole game,
    spike detectors, sensory encoding, motor decoding, trainer, and CSV logs.
    The first read window has no sensory stimulation, as in the paper controller.
    """
    inputs = ['sensory_electrodes', 'motor_electrodes', 'stim_electrodes',
              'valid_stim_electrodes', 'derived_dir', 'mapping_file_path']
    outputs = ['game_log_file', 'reward_log_file', 'pattern_log_file',
               'total_episodes', 'final_rewards', 'cartpole_config']

    def __init__(self, run_index=0, n_episodes=200, max_time_sec=900,
                 read_period_ms=200, train_period_ms=400, wait_period_ms=3000,
                 render_mode='human', name=None):
        super().__init__(name=name, recording_tag='cartpole')
        if run_index < 0 or n_episodes < 1 or max_time_sec <= 0:
            raise ValueError('run_index must be nonnegative; episodes and time must be positive')
        if read_period_ms <= 0 or train_period_ms <= 0 or wait_period_ms < 0:
            raise ValueError('Read/train periods must be positive and wait period nonnegative')
        self.run_index = run_index
        self.n_episodes = n_episodes
        self.max_time_sec = max_time_sec
        self.read_period_ms = read_period_ms
        self.train_period_ms = train_period_ms
        self.wait_period_ms = wait_period_ms
        self.render_mode = render_mode
        self.amp_mv, self.phase_width = 400, 200
        self.minibatch_size = 5
        self.game_env = None

    def configure_from_experiment(self, experiment):
        config = dict(experiment.params)
        config.update({key: getattr(experiment.data, key) for key in self.inputs})
        config.update(name=experiment.name, save_dir=str(experiment.save_dir))
        self.config, self.env_params, options = prepare_session(
            config, self.run_index, self.n_episodes, self.max_time_sec)
        self.sensory_neurons = np.asarray(options['sensory_neurons'], dtype=int)
        self.motor_neurons = np.asarray(options['motor_neurons'], dtype=int)
        self.training_neurons = np.asarray(options['training_neurons'], dtype=int)
        self.trainer = options['trainer']
        self.trainer.phase = self
        self.normalization = options['normalization']
        self.training_type = options['training_type']
        self.verbose = bool(config.get('verbose', True))
        self.config.update(read_period_ms=self.read_period_ms,
                           train_period_ms=self.train_period_ms,
                           wait_period_ms=self.wait_period_ms,
                           amp_mv=self.amp_mv, phase_width=self.phase_width)

    def customize_environment_params(self, env_params):
        for key in ('stim_electrodes', 'observation_type', 'max_time_sec'):
            env_params[key] = self.env_params[key]
        return env_params

    def set_sensory_signal(self, observation):
        """Encode pole angle as the two paper sensory stimulation rates."""
        pole_angle = observation[2]
        self.sensory_stim_Hz = np.square(
            (np.array([-np.sin(pole_angle), np.sin(pole_angle)]) + .15) * 7)
        if self.verbose:
            print(f'   Sensory Hz: {self.sensory_stim_Hz[0]:.2f}, {self.sensory_stim_Hz[1]:.2f}')

    def get_motor_signal(self, moving_avg=.2):
        """Smooth motor counts and decode their normalized difference as force."""
        self.motor_spike_rate = (moving_avg * self.motor_spike_rate
                                 + (1 - moving_avg) * self.motor_spike_count)
        if isinstance(self.normalization, np.ndarray):
            difference = (self.motor_spike_rate[0] / self.normalization[0]
                          - self.motor_spike_rate[1] / self.normalization[1])
        else:
            difference = (self.motor_spike_rate[0] - self.motor_spike_rate[1]) / self.normalization
        return max(min(-difference, 1), -1)

    def get_training_signal(self):
        """Apply the paper reward-change update, then choose a tetanus pattern."""
        if self.episode_reward is not None and len(self.last_action_inds) == self.minibatch_size:
            self.trainer.update_values(self.episode_reward_change, self.last_action_inds[-1])
        pulse, frequency, self.last_action_ind = self.trainer.get_action()
        self.last_action_inds.append(self.last_action_ind)
        return pulse, frequency

    def run(self, experiment):
        from braindance.core.artifact_removal import ArtifactRemoval
        from braindance.games.cartpole_continuous import CartPoleContinuousEnv

        if self.env.observation_type != 'raw':
            raise ValueError('Paper CartPole needs raw acquisition samples')
        self.game_env = CartPoleContinuousEnv(render_mode=self.render_mode)
        self.artifact_remover = [ArtifactRemoval(
            N=20, nc_start=20, min_val=-25, max_val=25, spike_thresh=[-4.4, -25],
        ) for _ in self.motor_neurons]
        self.game_obs, _ = self.game_env.reset()
        self.motor_spike_count = np.zeros(2)
        self.motor_spike_rate = np.zeros(2)
        self.sensory_stim_Hz = np.zeros(2)
        self.last_action_inds = deque(maxlen=self.minibatch_size)
        self.last_action_ind = None
        self.episode_reward = self.episode_reward_change = None
        self.episode = 0
        self.state = 'read'
        self.start_time = time.perf_counter()
        self.config['save_dir'] = str(experiment.current_recording_dir or experiment.save_dir)
        self.config['recording_file'] = str(self.env.save_file)
        stem = str(self.env.save_file)
        rewards = []
        total_reward = 0
        done, just_entered = False, True
        state_timer = self.time_elapsed()

        # All exit paths (including episode limits and errors) flush/close logs.
        with ExitStack() as files:
            game_logger = csv.writer(files.enter_context(open(stem + '_game_log.csv', 'w', newline='')))
            reward_logger = csv.writer(files.enter_context(open(stem + '_reward_log.csv', 'w', newline='')))
            pattern_logger = csv.writer(files.enter_context(open(stem + '_pattern_log.csv', 'w', newline='')))
            game_logger.writerow(['time', 'pole_angle', 'reward', 'action',
                                  'spike_count_l', 'spike_count_r', 'state'])
            reward_logger.writerow(['time', 'episode', 'reward'])
            pattern_logger.writerow(['time', 'pattern', 'reward', 'probs', 'vals'])

            while not done and self.time_elapsed() < self.max_time_sec:
                if self.state == 'read':
                    if just_entered:
                        state_timer = self.time_elapsed()
                        just_entered = False
                        self.motor_spike_count[:] = 0
                    if self.time_elapsed() - state_timer > self.read_period_ms / 1000:
                        self.state, just_entered = 'game', True
                        continue
                    intervals = np.divide(1., self.sensory_stim_Hz,
                                          out=np.full(2, np.inf), where=self.sensory_stim_Hz > 0)
                    active = self.env.stim_dts[self.sensory_neurons] > intervals
                    if np.any(active):
                        obs, done = self.env.step(
                            action=(self.sensory_neurons[active], self.amp_mv, self.phase_width),
                            tag='sensory')
                    else:
                        obs, done = self.env.step()
                    if obs is not None:
                        for i, channel in enumerate(self.motor_neurons):
                            _, _, spike = self.artifact_remover[i].fit_step(obs[channel])
                            self.motor_spike_count[i] += bool(spike)

                elif self.state == 'game':
                    action = self.get_motor_signal()
                    self.game_obs, reward, terminated, truncated, _ = self.game_env.step(action)
                    total_reward += reward
                    game_logger.writerow([self.time_elapsed(), self.game_obs[2], reward, action,
                                          *self.motor_spike_count, self.game_obs])
                    self.set_sensory_signal(self.game_obs)
                    self.state = 'train' if terminated or truncated else 'read'
                    just_entered = True

                elif self.state == 'train':
                    if just_entered:
                        state_timer = train_timer = self.time_elapsed()
                        just_entered = False
                        rewards.append(total_reward)
                        self.episode_reward = np.mean(rewards[-self.minibatch_size:])
                        self.episode_reward_change = self.episode_reward - np.mean(rewards[-20:])
                        increased = self.episode_reward_change > 0
                        do_training = (self.training_type == 'always'
                                       or self.training_type == 'reward' and increased
                                       or self.training_type == 'punishment' and not increased)
                        reward_logger.writerow([self.time_elapsed(), self.episode, total_reward])
                        self.episode += 1
                        if self.verbose:
                            print(f'   Episode {self.episode}: reward={total_reward}, '
                                  f'change={self.episode_reward_change:.3f}')
                        if self.episode >= self.n_episodes:
                            break
                        if do_training:
                            train_pulse, train_freq = self.get_training_signal()
                            pattern_logger.writerow([
                                self.time_elapsed(), self.trainer.get_action(self.last_action_ind),
                                total_reward, self.trainer.get_probs(), self.trainer.get_values()])
                            _, done = self.env.step(action=train_pulse, tag='train')
                            train_timer = self.time_elapsed()
                            continue
                        if self.last_action_inds:
                            self.trainer.update_values(self.episode_reward_change, self.last_action_inds[-1])
                        self.last_action_inds.append(None)

                    if (not do_training
                            or self.time_elapsed() - state_timer > self.train_period_ms / 1000):
                        self.game_obs, _ = self.game_env.reset()
                        self.set_sensory_signal(self.game_obs)
                        total_reward = 0
                        self.state, just_entered = 'wait', True
                        continue
                    if self.time_elapsed() - train_timer > 1 / train_freq:
                        pattern_logger.writerow([
                            self.time_elapsed(), self.trainer.get_action(self.last_action_ind),
                            total_reward, self.trainer.get_probs(), self.trainer.get_values()])
                        _, done = self.env.step(action=train_pulse, tag='train')
                        train_timer = self.time_elapsed()
                    else:
                        _, done = self.env.step()

                elif self.state == 'wait':
                    if just_entered:
                        state_timer, just_entered = self.time_elapsed(), False
                    if self.time_elapsed() - state_timer > self.wait_period_ms / 1000:
                        self.state, just_entered = 'read', True
                        continue
                    _, done = self.env.step()

        return dict(game_log_file=stem + '_game_log.csv',
                    reward_log_file=stem + '_reward_log.csv',
                    pattern_log_file=stem + '_pattern_log.csv',
                    total_episodes=len(rewards), final_rewards=rewards,
                    cartpole_config=self.config)

    def cleanup(self):
        if self.game_env is not None:
            self.game_env.close()
            self.game_env = None
