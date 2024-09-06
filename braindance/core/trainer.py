import numpy as np
from itertools import permutations


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



def generate_stimulations(electrode_inds, amp=400, phase_width=200):
    """Creates a list of stimulation commands for the given electrodes
    with the given amplitude and phase width"""
    stim_commands = []
    for electrode_ind_set in electrode_inds:
        if type(electrode_ind_set) == int:
            electrode_ind_set = [electrode_ind_set]
        stim_commands.append((electrode_ind_set, amp, phase_width))
    return stim_commands
