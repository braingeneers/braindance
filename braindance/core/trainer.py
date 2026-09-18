import numpy as np
from itertools import permutations
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


class ContextualTrainer:
    """
    Contextual RL trainer that uses an MLP to select stimulation patterns
    based on the current state of the system.

    Input features (state vector):
        - Motor spike rate statistics (mean and variance)
        - Policy network gradient magnitude (L2 norm)
        - Reward delta (current - running average)
        - Episode length (normalized)

    Output:
        - Distribution over N+1 actions: N stimulation patterns + 1 "no-stim" action

    Parameters
    ----------
    choices : list
        List of stimulation patterns (actions)
    freqs : list or float
        Stimulation frequencies for each pattern
    hidden_sizes : list
        Hidden layer sizes for the MLP (default: [64, 32])
    learning_rate : float
        Learning rate for policy gradient updates
    gamma : float
        Discount factor for eligibility traces
    baseline_window : int
        Window size for running average baseline (default: 20)
    normalize_context : bool
        Whether to normalize context features (default: True)
    """

    def __init__(self, choices, freqs, hidden_sizes=None, learning_rate=0.01,
                 gamma=0.95, baseline_window=20, normalize_context=True,
                 no_stim_bias=0.0):
        self.choices = choices
        self.n_patterns = len(choices)
        self.n_actions = self.n_patterns + 1  # +1 for no-stim action

        if isinstance(freqs, (list, np.ndarray)):
            self.freqs = list(freqs)
        else:
            self.freqs = [freqs] * self.n_patterns

        # Add a placeholder freq for no-stim action
        self.freqs.append(0)

        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        self.hidden_sizes = hidden_sizes

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.baseline_window = baseline_window
        self.normalize_context = normalize_context
        self.no_stim_bias = no_stim_bias

        # Context feature dimensions:
        # - motor_mean, motor_var (2)
        # - policy_grad_magnitude (1)
        # - reward_delta (1)
        # - episode_length_normalized (1)
        self.context_dim = 5

        # Build the policy network
        self._build_network()

        # Running statistics for normalization
        self.context_mean = np.zeros(self.context_dim)
        self.context_var = np.ones(self.context_dim)
        self.context_count = 0

        # Eligibility traces for credit assignment
        self.eligibility_traces = torch.zeros(self.n_actions)

        # Running reward baseline
        self.reward_history = deque(maxlen=baseline_window)
        self.episode_rewards = deque(maxlen=baseline_window)

        # Logging
        self.action_history = []
        self.reward_deltas = []

        # External state that can be set
        self.policy_grad_magnitude = 0.0

        # For compatibility with TetanusTrainer interface
        self.phase = None
        self.values = np.ones(self.n_actions) * 10  # Pseudo-values for logging
        self.probs = np.ones(self.n_actions) / self.n_actions

        # Track last action for update
        self.last_context = None
        self.last_action_log_prob = None
        self.last_action_idx = None

    def _build_network(self):
        """Build the MLP policy network."""
        layers = []
        input_dim = self.context_dim

        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size

        # Output layer produces logits for each action
        layers.append(nn.Linear(input_dim, self.n_actions))

        self.policy_net = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def set_policy_grad_magnitude(self, value):
        """
        Set the policy gradient magnitude from external source.
        Called by CartPolePhase after policy update.
        """
        self.policy_grad_magnitude = float(value)

    def compute_context(self, motor_spike_rates, episode_reward, episode_length,
                        max_episode_length=500):
        """
        Compute the context/state vector from current observations.

        Parameters
        ----------
        motor_spike_rates : np.ndarray
            Spike rates from motor neurons
        episode_reward : float
            Total reward from the current/last episode
        episode_length : int
            Number of steps in the current/last episode
        max_episode_length : int
            Maximum expected episode length for normalization

        Returns
        -------
        np.ndarray
            Context vector of shape (context_dim,)
        """
        # Motor spike statistics
        motor_mean = np.mean(motor_spike_rates) if len(motor_spike_rates) > 0 else 0.0
        motor_var = np.var(motor_spike_rates) if len(motor_spike_rates) > 0 else 0.0

        # Reward delta (current - running average)
        if len(self.episode_rewards) > 0:
            reward_delta = episode_reward - np.mean(self.episode_rewards)
        else:
            reward_delta = 0.0

        # Normalized episode length
        episode_length_norm = episode_length / max_episode_length

        context = np.array([
            motor_mean,
            motor_var,
            self.policy_grad_magnitude,
            reward_delta,
            episode_length_norm
        ], dtype=np.float32)

        # Update running statistics for normalization
        if self.normalize_context:
            self._update_normalization_stats(context)
            context = self._normalize(context)

        return context

    def _update_normalization_stats(self, context):
        """Update running mean and variance for context normalization."""
        self.context_count += 1
        delta = context - self.context_mean
        self.context_mean += delta / self.context_count
        delta2 = context - self.context_mean
        self.context_var += (delta * delta2 - self.context_var) / self.context_count

    def _normalize(self, context):
        """Normalize context using running statistics."""
        std = np.sqrt(self.context_var + 1e-8)
        return (context - self.context_mean) / std

    def get_action(self, context=None, ind=None):
        """
        Select an action based on the current context.

        Parameters
        ----------
        context : np.ndarray, optional
            Context vector. If None, uses a default zero context.
        ind : int, optional
            If provided, return the action at this index (for compatibility)

        Returns
        -------
        tuple
            (pattern_or_none, freq, action_index)
            pattern_or_none is None if no-stim action is selected
        """
        if ind is not None:
            # Compatibility mode: return specific action
            if ind == self.n_patterns:
                # No-stim action
                return None, 0, ind
            return self.choices[ind], self.freqs[ind], ind

        if context is None:
            context = np.zeros(self.context_dim, dtype=np.float32)

        # Store context for update
        self.last_context = context

        # Convert to tensor
        context_tensor = torch.FloatTensor(context).unsqueeze(0)

        # Get action probabilities
        logits = self.policy_net(context_tensor)

        # Apply no-stim bias if specified
        if self.no_stim_bias != 0:
            logits[0, -1] += self.no_stim_bias

        probs = torch.softmax(logits, dim=-1)

        # Sample action
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()

        # Store for update
        self.last_action_log_prob = dist.log_prob(action_idx)
        self.last_action_idx = action_idx.item()

        # Update eligibility traces
        self.eligibility_traces *= self.gamma
        self.eligibility_traces[self.last_action_idx] += 1.0

        # Update pseudo-values and probs for logging
        self.probs = probs.detach().numpy().flatten()

        # Log action
        self.action_history.append(self.last_action_idx)

        # Return appropriate action
        if self.last_action_idx == self.n_patterns:
            # No-stim action
            print(f"ContextualTrainer: Selected NO-STIM (action {self.last_action_idx})")
            return None, 0, self.last_action_idx
        else:
            print(f"ContextualTrainer: Selected pattern {self.last_action_idx}")
            self._print_probs()
            return self.choices[self.last_action_idx], self.freqs[self.last_action_idx], self.last_action_idx

    def _print_probs(self):
        """Print current action probabilities."""
        print("Action probabilities:")
        for i, p in enumerate(self.probs):
            if i == self.n_patterns:
                print(f"  NO-STIM: {p:.3f}")
            else:
                print(f"  Pattern {i}: {p:.3f}")

    def update(self, reward_delta, action_index=None):
        """
        Update the policy based on the reward signal.

        Parameters
        ----------
        reward_delta : float
            Reward improvement signal (current reward - baseline)
        action_index : int, optional
            The action to update. If None, uses the last action taken.
        """
        if action_index is None:
            action_index = self.last_action_idx

        if action_index is None or self.last_action_log_prob is None:
            return

        # Record reward delta
        self.reward_deltas.append(reward_delta)

        # Policy gradient update using eligibility traces
        # Loss = -log_prob * reward_delta * eligibility
        loss = -self.last_action_log_prob * reward_delta * self.eligibility_traces[action_index]

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update pseudo-values based on reward
        self.values[action_index] = 0.9 * self.values[action_index] + 0.1 * (reward_delta + 10)
        self.values = np.clip(self.values, 1, 100)

    def update_values(self, reward, ind, update_probs=True):
        """
        Compatibility method matching TetanusTrainer interface.

        Parameters
        ----------
        reward : float
            The reward signal
        ind : int or None
            The action index that was taken
        update_probs : bool
            Ignored, for compatibility only
        """
        self.update(reward, ind)

    def record_episode_reward(self, episode_reward):
        """Record episode reward for baseline calculation."""
        self.episode_rewards.append(episode_reward)

    def get_probs(self):
        """Return current action probabilities."""
        return self.probs

    def get_values(self):
        """Return pseudo-values for logging."""
        return self.values

    def get_state_dict(self):
        """Get the state dict of the policy network."""
        return self.policy_net.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict into the policy network."""
        self.policy_net.load_state_dict(state_dict)


class TetanusTrainer:
    def __init__(self, choices, freqs,start_value=10, learning_rate=.1, floor=10,
                 discount_factor=0.3):
        self.choices = choices
        self.n_choices = len(choices)
        if isinstance(freqs, (list, np.ndarray)):
            self.freqs = freqs
        elif isinstance(freqs, (int, float)):
            self.freqs = [freqs] * self.n_choices
            
        self.start_value = start_value
        self.floor = floor

        self.values = np.ones((self.n_choices,))*start_value
        self._update_count = 0
        self.print_interval = 20
        self.probs = self.update_probs(self.values)

        self.learning_rate = learning_rate
        self.traces = np.zeros((self.n_choices,))
        self.discount_factor = discount_factor

    def update_values(self, reward, ind, update_probs=True):
        # Update traces
        self.traces *= self.discount_factor

        if ind is not None:
            self.traces[ind] += 1

        # Update values
        self.values += self.learning_rate * (reward - self.values) * self.traces

        # Cap min value
        self.values[self.values < self.floor] = self.floor

        self._update_count += 1

        if update_probs:
            self.update_probs(self.values)


    def update_probs(self, values, bias=0):
        temp_vals = values + bias
        self.probs = temp_vals / temp_vals.sum()
        if self._update_count % self.print_interval == 0:
            print("New probabilities/rewards:")
            for p,v in zip(self.probs, values):
                print(f'{v:.2f}>::<{p:.2f}',end='\t')
            print('')
    
    def get_action(self, ind=None):
        '''Return a random action based on the current probabilities, and the 
        index of the action'''
        if ind is None:
            ind = np.random.choice(self.n_choices, p=self.probs)
        return self.choices[ind], self.freqs[ind], ind
    
    
    
    def get_probs(self):
        return self.probs
    
    def get_values(self):
        return self.values
    

class AdaptiveTetanusTrainer:
    """Bandit trainer over synthesized tetanus parameter combinations."""

    def __init__(
        self,
        neurons,
        amp_mv,
        pulse_width,
        stim_count_choices=(2, 3, 4),
        delay_ms_choices=(3, 5, 8),
        freq_choices=(20, 30, 40),
        start_value=10,
        learning_rate=.1,
        floor=10,
        discount_factor=0.3,
        max_random_orders=4,
        seed=0,
    ):
        self.neurons = [int(n) for n in neurons]
        self.amp_mv = int(amp_mv)
        self.pulse_width = int(pulse_width)
        self.stim_count_choices = sorted(
            {max(1, min(int(v), len(self.neurons))) for v in stim_count_choices}
        )
        self.delay_ms_choices = sorted({max(1, int(v)) for v in delay_ms_choices})
        self.freq_choices = sorted({max(1, int(v)) for v in freq_choices})
        self.max_random_orders = max(0, int(max_random_orders))
        self.rng = np.random.default_rng(seed)

        self.choices, self.freqs, self.choice_specs = self._build_choice_space()
        self.n_choices = len(self.choices)
        if self.n_choices == 0:
            raise ValueError("AdaptiveTetanusTrainer requires at least one choice.")

        self.start_value = start_value
        self.floor = floor
        self.values = np.ones((self.n_choices,)) * start_value
        self.probs = self.update_probs(self.values)

        self.learning_rate = learning_rate
        self.traces = np.zeros((self.n_choices,))
        self.discount_factor = discount_factor
        self.phase = None

    def _canonical_orderings(self):
        if not self.neurons:
            return []

        orderings = []

        def add_order(order):
            order = tuple(int(n) for n in order)
            if order and order not in orderings:
                orderings.append(order)

        add_order(tuple(self.neurons))
        add_order(tuple(reversed(self.neurons)))

        center = len(self.neurons) // 2
        interleaved = []
        for step in range(len(self.neurons)):
            left = center - step
            right = center + step
            if 0 <= left < len(self.neurons):
                interleaved.append(self.neurons[left])
            if 0 <= right < len(self.neurons) and right != left:
                interleaved.append(self.neurons[right])
        add_order(interleaved)

        for _ in range(self.max_random_orders):
            shuffled = list(self.neurons)
            self.rng.shuffle(shuffled)
            add_order(shuffled)

        return orderings

    def _build_choice_space(self):
        choices = []
        freqs = []
        specs = []
        seen = set()
        for order in self._canonical_orderings():
            for stim_count in self.stim_count_choices:
                effective_order = tuple(order[:stim_count])
                if len(effective_order) == 0:
                    continue
                for delay_ms in self.delay_ms_choices:
                    for freq in self.freq_choices:
                        key = (effective_order, int(delay_ms), int(freq))
                        if key in seen:
                            continue
                        seen.add(key)
                        spec = {
                            "neurons": list(effective_order),
                            "stim_count": len(effective_order),
                            "delay_ms": int(delay_ms),
                            "freq": int(freq),
                        }
                        specs.append(spec)
                        choices.append(self._pattern_from_spec(spec))
                        freqs.append(int(freq))
        return choices, freqs, specs

    def _pattern_from_spec(self, spec):
        tetanus_action = []
        neuron_order = spec["neurons"][: spec["stim_count"]]
        for i, neuron in enumerate(neuron_order):
            tetanus_action.append(
                ("stim", [int(neuron)], self.amp_mv, self.pulse_width)
            )
            if i != len(neuron_order) - 1:
                tetanus_action.append(("delay", int(spec["delay_ms"])))
        return tetanus_action

    def update_values(self, reward, ind, update_probs=True):
        self.traces *= self.discount_factor

        if ind is not None:
            self.traces[ind] += 1

        self.values += self.learning_rate * (reward - self.values) * self.traces
        self.values[self.values < self.floor] = self.floor

        if update_probs:
            self.update_probs(self.values)

    def update_probs(self, values, bias=0):
        print("New adaptive tetanus probabilities/rewards:")
        temp_vals = values + bias
        self.probs = temp_vals / temp_vals.sum()
        for p, v, spec in zip(self.probs, values, self.choice_specs):
            desc = (
                f"neurons={spec['neurons']} delay={spec['delay_ms']}ms "
                f"freq={spec['freq']}Hz"
            )
            print(f"{v:.2f}>::<{p:.2f} ({desc})", end='\t')
        print('')
        return self.probs

    def get_action(self, ind=None):
        """Return a tetanus action and action index from the learned search space."""
        if ind is None:
            ind = np.random.choice(self.n_choices, p=self.probs)
        spec = self.choice_specs[ind]
        return self._pattern_from_spec(spec), self.freqs[ind], ind

    def get_probs(self):
        return self.probs

    def get_values(self):
        return self.values

    def get_choice_specs(self):
        return list(self.choice_specs)


def generate_tetanus_pattern(neurons, stim_count=5, delay_ms=5, amp_mv=400, pulse_width=100, random=False, replace=False):
    """
    Generates a tetanus pattern for neuron stimulation.

    Args:
    neurons (list): List of neurons.
    stim_count (int): Number of stimulations at most to use.
    delay_ms (int): Delay between stimulations in milliseconds.
    amp_mv (int): Amplitude in millivolts.
    pulse_width (int): Pulse width.
    replace (bool): Whether to replace selections.

    Returns:
    list: A sequence of stimulation and delay actions.
    """
    if random:
        neuron_order = np.random.choice(neurons, size=stim_count, replace=replace)
    else:
        neuron_order = neurons[:stim_count]
    tetanus_action = []
    for i, n in enumerate(neuron_order):
        tetanus_action.append(('stim', [n], amp_mv, pulse_width))
        if i != stim_count - 1:
            tetanus_action.append(('delay', delay_ms))

    return tetanus_action


def generate_permutations(neurons, stim_count=5, delay_ms=5, amp_mv=400, pulse_width=100):
    """
    Generates a list of tetanus patterns for all permutations of the given neurons.

    Args:
    neurons (list): List of neurons.
    stim_count (int): Number of stimulations.
    delay_ms (int): Delay between stimulations in milliseconds.
    amp_mv (int): Amplitude in millivolts.
    pulse_width (int): Pulse width.

    Returns:
    list: A list of sequences, each a unique permutation of stimulation and delay actions.
    """
    tetanus_action_list = []

    # Generating all permutations of the given neurons
    for perm in permutations(neurons, stim_count):
        tetanus_action = []
        for i, n in enumerate(perm):
            tetanus_action.append(('stim', [n], amp_mv, pulse_width))
            if i != stim_count - 1:
                tetanus_action.append(('delay', delay_ms))
        tetanus_action_list.append(tetanus_action)

    return tetanus_action_list



class HebbianTrainer:
    def __init__(self, choices, freqs, phase=None):
        self.choices = choices
        self.n_choices = len(choices)
        if isinstance(freqs, (list, np.ndarray)):
            self.freqs = freqs
        elif isinstance(freqs, (int, float)):
            self.freqs = [freqs] * self.n_choices

        self.phase = phase
       

    def update_values(self, reward, ind, update_probs=True):
        pass
    
    def update_probs(self, values, bias=-5):
        pass
    
    def get_action(self, ind=None):
        '''Returns an action based on the pole angle'''
        pole_angle = self.phase.game_obs[2] # Second element in cartpole state

        if pole_angle < 0:
            print("Fell left")
            return self.choices[0], self.freqs[0], 0
        else:
            print("Fell right")
            return self.choices[1], self.freqs[1], 1




def get_random_reward(reward_min = -10, reward_max = 10):
    return np.random.randint(reward_min, reward_max)

def get_random_correlated_reward(previous_reward, reward_min = -10, reward_max = 10):
    rew = previous_reward + np.random.randint(reward_min, reward_max)
    if rew < reward_min:
        rew = reward_min
    elif rew > reward_max:
        rew = reward_max
    return rew


def update_values(values, reward, ind, learning_rate = 0.1):
    values[ind] += learning_rate * (reward - values[ind])
    # Cap min and max values
    values[values < 10] = 10

    return values

def update_probs(values):
    probs = values / values.sum()
    return probs
 
def plot_probs(probs):
    plt.figure()
    plt.plot(probs)
    plt.show()


if __name__ == '__main__':
    # Initial probability of sampling
    n_choices = 10
    start_value = 10

    tt = TetanusTrainer(['r']*n_choices, start_value=start_value, freqs = 10,
                        learning_rate=.05, floor=.1, discount_factor=.2)

    # Run a simulation
    n_trials = 100
    probs_log = []
    values_log = []
    reward = 0
    last_reward = 0
    for i in range(n_trials):
        # Get a random reward
        reward = get_random_correlated_reward(last_reward)
        # reward = get_random_reward()

        # Randomly choose an index to update
        # ind = np.random.randint(0, n_choices)
        # Choose an ind weighted by the probs
        probs = tt.get_probs()
        ind = np.random.choice(n_choices, p=probs)

        

        if ind == 0:
            reward += 3
        elif ind == 1:
            reward+= 4
        elif ind == 2:
            reward += 13
        elif ind == 3:
            reward += 10
        elif ind == 4:
            reward += -10

        # If we improve, set last reward and continue
        if reward > last_reward:
            last_reward = reward
            tt.update_values(reward, None)
            probs_log.append(tt.get_probs().copy())
            values_log.append(tt.get_values().copy())

            continue

        # Update values and probs

        tt.update_values(reward, ind)
        
        values = tt.get_values()
        probs = tt.get_probs()
        # tt.probs = tt.update_probs(tt.values)
        # Log probs
        probs_log.append(tt.get_probs().copy())
        # values_log.append(tt.values.copy())
        values_log.append(tt.get_values().copy())
        last_reward = reward

    probs_log = np.array(probs_log)
    values_log = np.array(values_log)

    plt.figure()
    # plt.plot(probs_log[:, :])
    plt.plot(values_log[:, :])
    plt.show()


    # Plot heatmap of probs evolving over time
    plt.imshow(probs_log.T, aspect='auto')
    plt.colorbar()
    plt.show()

    # values = np.ones((n_choices,))*start_value


    # probs = update_probs(values)
    # print(probs)

    # # Run a simulation
    # n_trials = 1000
    # probs_log = []
    # values_log = []
    # for i in range(n_trials):
        
    #     # Get a random reward
    #     reward = get_random_reward(probs, n_choices)

    #     # Randomly choose an index to update
    #     ind = np.random.randint(0, n_choices)
    #     if ind == 0:
    #         reward += 70
    #     elif ind == 1:
    #         reward+= 40
    #     elif ind == 2:
    #         reward += 130
    #     elif ind == 3:
    #         reward += 100

    #     # Update values and probsq
    #     values = update_values(values, reward, ind, .2)
    #     probs = update_probs(values)
    #     # Log probs
    #     probs_log.append(probs)
    #     values_log.append(values.copy())

    # # Plot the final probs
    # # plot_probs(probs)
    # probs_log = np.array(probs_log)
    # values_log = np.array(values_log)

    # plt.figure()
    # # plt.plot(probs_log[:, :])
    # plt.plot(values_log[:, :])
    # plt.show()

