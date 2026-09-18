"""
Test script for Phase V3 spike sorting and data saving functionality
"""
from braindance.core.phases_v3.experiment_v3 import Experiment
from braindance.core.phases_v3.phases3 import RecordPhaseV3
from braindance.core.phases_v3.phases_analysis_3 import RTSortPhaseV3, ConnectivityPhaseV3
from braindance.core.phases_v3.phase_base_v3 import phase
import argparse
import numpy as np




# argparse config
parser = argparse.ArgumentParser(description="Closed-Loop Plasticity Experiment")
parser.add_argument("--config", type=str, default=None, help="Maxwell configuration file")
parser.add_argument("--stim_electrodes", type=list, default=[], help="List of stimulation electrodes")
parser.add_argument("--verbose", type=bool, default=True, help="Verbose mode")
parser.add_argument("--record_duration", type=int, default=300, help="Recording duration in seconds")
args = parser.parse_args()

params = {
    "config": args.config,
    # "stim_electrodes": args.stim_electrodes, # can leave as empty list or will infer from phases
    "verbose": args.verbose,
    "record_duration": args.record_duration
}

# Create experiment
exp = Experiment(
    "test_spike_sort",
    params = params,
    save_dir="./test_experiments/spike_sort_test"
)

# This decorator allows us to quickly make a phase object
# that can be used in the experiment
# You can access all data using the exp.data.<data_name> 
# >>> where exp.data.connectivity_matrix here is set from a previous phase
# >>> This previous phase "provides" the data to this phase
# >>> This phase "requires" the key 'connectivity_matrix' to be present in exp.data
@phase("SelectRandomPair")
def select_random_pair(exp):
    lower_bound = .3
    upper_bound = .6
    n_attempts = 500
    connectivity_matrix = exp.data.connectivity_matrix
    n_neurons = connectivity_matrix.shape[0]
    print(f"   Selecting random pair from {n_neurons} neurons")
    for i in range(n_attempts):
        random_pair = np.random.randint(0, n_neurons, 2)
        if connectivity_matrix[random_pair[0], random_pair[1]] > lower_bound and connectivity_matrix[random_pair[0], random_pair[1]] < upper_bound:
            print(f"   Found random pair {random_pair}: conn={connectivity_matrix[random_pair[0], random_pair[1]]} within bounds")
            return {'selected_pair': random_pair}
        else:
            print(f"   Random pair {random_pair} not within bounds")
    raise ValueError("No random pair found within bounds")


# We can do it this way

# exp.add_phases(
#     RecordPhaseV3(duration=20),
#     RTSortPhaseV3(
#         sorter='rt_sort',  # Will fall back to mock if RT sort not available
#         min_spikes=50,
#         recording_window_ms=(0, 20000),
#         verbose=True
#     ),
#     ConnectivityPhaseV3(
#         window_ms=20.0
#     ),
#     select_random_pair # This was made using the @phase decorator
# )


# Or we can do it this way
exp.add_phase(RecordPhaseV3(duration=20))  # Record for 20 seconds
exp.add_phase(RTSortPhaseV3(
    sorter='rt_sort',  # Will fall back to mock if RT sort not available
    min_spikes=50,
    recording_window_ms=(0, 20000),
    verbose=True
))
exp.add_phase(ConnectivityPhaseV3(window_ms=20.0))
exp.add_phase(select_random_pair)  # Use the decorated function

# exp.add_phase(RecordPhaseV3(duration=60)).add_phase(RecordPhaseV3(duration=60)) 

# Run experiment
success = exp.run()
print("\n✨ All tests completed!") 