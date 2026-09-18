"""Edit these functions in the Functions tab or in the saved Python profile.

All observations/actions are named in the UI. Counts cover dt_s seconds.
Encode/decode keep state dictionaries; train can update both explicitly.
State resets on reload, phase restart and episode reset.
"""
import numpy as np


def decode(spike_counts, dt_s, params, state):
    """Convert one bin of neural activity into a game action.

    Args:
        spike_counts (numpy.ndarray): Shape (n_channels,), spike counts in
            the latest bin, not rates. Entries represent acquisition channels
            or sorted units; channel-group indices refer to this array.
        dt_s (float): Bin duration in seconds (0.02 in the workshop).
        params (dict): Shared configuration. This decoder uses:
            baseline_hz: Length-n_channels baseline firing rates in Hz.
            left_channels, right_channels: Lists of indices into spike_counts.
            decoder_gain: Scale from the rate difference in Hz to action.
            action_size: Number of action components.
            environment: 'cartpole', 'foodland', or 'ant'.
            Encoder settings are also present; see encode. Treat params as
            configuration and keep learned values in state.
        state (dict): Mutable decoder memory, initially empty. This example
            stores smoothed Hz in state['rates']. Persists between bins and
            is also available to train as state['decode']; resets on episode
            reset, phase restart, or function reload.

    Returns:
        tuple: (action, diagnostics). action has shape (action_size,):
            CartPole uses one normalized force in [-1, 1]; FoodLand uses
            [turn in [-1, 1], forward speed in [0, 1]]; Ant uses normalized
            joint torques in [-1, 1]. diagnostics is a JSON-compatible dict.
    """
    rates = np.asarray(spike_counts) / dt_s
    smoothed = state.get('rates', np.zeros_like(rates))
    alpha = 1 - np.exp(-dt_s / .1)
    smoothed = smoothed + alpha * (rates - smoothed)
    state['rates'] = smoothed
    centered = smoothed - np.asarray(params['baseline_hz'])
    left = np.mean(centered[params['left_channels']])
    right = np.mean(centered[params['right_channels']])
    before_clip = (right - left) * params['decoder_gain']
    action = np.full(params['action_size'], np.clip(before_clip, -1, 1))
    if params['environment'] == 'ant':
        # Each joint reads a channel (cycling if there are fewer than 8).
        # This demonstrates separate outputs; it is not a trained walking policy.
        before_clip = centered[np.arange(params['action_size']) % len(centered)] * params['decoder_gain']
        action = np.clip(before_clip, -1, 1)
    if params['environment'] == 'foodland':
        action[1] = .6  # [turn, forward speed], adapter handles legacy order
    return action, {'rates_hz': smoothed.tolist(), 'before_clip': np.asarray(before_clip).tolist()}


def encode(observation, dt_s, params, state):
    """Map a game observation to two stimulation rates in Hz.

    Args:
        observation (numpy.ndarray): One-dimensional observation after the
            game step. CartPole: shape (4,), [position (m), velocity (m/s),
            pole angle (rad), angular velocity (rad/s)]. FoodLand: shape (9,),
            [food signal, hazard signal, x (px), y (px), direction (rad),
            food captured, hazard hits, wall contact, food gradient]. Ant:
            the environment's observation vector; its length depends on the
            environment. Component names are shown in the workshop UI.
        dt_s (float): Bin duration in seconds (0.02); unused by this example.
        params (dict): Shared configuration. This encoder uses:
            sensory_index: Index into observation (default 2 for CartPole,
                otherwise 0).
            encoder_gain: Stimulation Hz per unit of the selected observation.
            max_stim_hz: Maximum rate for each stimulation input (40 Hz).
            stim_electrodes: Configured stimulation electrodes, in output order.
            Decoder settings are also present; see decode. Treat params as
            configuration and keep learned values in state.
        state (dict): Mutable encoder memory, initially empty; unused here.
            Persists between bins and is also available to train as
            state['encode']; resets on episode reset, phase restart, or
            function reload.

    Returns:
        tuple: (rates, diagnostics). rates has shape (2,), with finite Hz
            values in [0, max_stim_hz] for the two configured stimulation
            inputs. This example maps negative sensory values to the first
            input and positive values to the second. diagnostics is a
            JSON-compatible dict.
    """
    value = float(observation[params['sensory_index']])
    signed_rate = value * params['encoder_gain']
    before_clip = np.array([max(0, -signed_rate), max(0, signed_rate)])
    rates = np.clip(before_clip, 0, params['max_stim_hz'])
    return rates, {'sensory_value': value, 'before_clip_hz': before_clip.tolist()}


def train(transition, dt_s, params, state):
    """Optional learning hook, called after game.step and before encode.

    Args:
        transition (dict): A copy of the current step, with these keys:
            observation: List of observation values before the game step.
            next_observation: List after the step; see encode for layouts.
            action: List of applied action values; see decode for ordering.
            spike_counts: List of counts per channel/unit in this bin.
            reward: Float reward returned by the game step.
            done: Bool indicating that the episode ended on this step.
        dt_s (float): Bin duration in seconds (0.02 in the workshop).
        params (dict): Copy of the shared configuration documented in encode
            and decode. Changes here do not update the runtime configuration.
        state (dict): Mutable training memory. state['decode'] and
            state['encode'] reference the dictionaries passed to those hooks;
            update their contents to share learned values. Other keys can
            hold training-only memory. State resets on episode reset, phase
            restart, or function reload, so learning here is episode-local.

    Returns:
        dict: JSON-compatible diagnostics (an empty dict disables reporting).

    Runs in the order decode -> game.step -> train -> encode, including the
    terminal step before state resets. Loading/reloading functions also calls
    the hooks with dummy inputs and fresh state to validate their outputs.
    Native V3 phases use their own trainers and do not call this hook.
    """
    return {}
