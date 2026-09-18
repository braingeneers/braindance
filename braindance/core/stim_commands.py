

def is_single_command(command):
    """
    Check if the given command is a single command or sequence of commands
    """
    if type(command[0][0]) == str:
        return False
    else:
        return True
    
    

def generate_stimulations(electrode_inds, amp=400, phase_width=200):
    """Creates a list of stimulation commands for the given electrodes
    with the given amplitude and phase width"""
    stim_commands = []
    for electrode_ind_set in electrode_inds:
        if type(electrode_ind_set) == int:
            electrode_ind_set = [electrode_ind_set]
        stim_commands.append((electrode_ind_set, amp, phase_width))
    return stim_commands
