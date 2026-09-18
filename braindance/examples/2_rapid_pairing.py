"""
Closed-loop plasticity experiment using Phase V3.

Phases:
  0. Record baseline spontaneous activity
  1. RT-Sort to detect neurons
  2. Compute connectivity matrix
  3. Select a neuron pair with moderate connectivity
  4. Closed-loop stimulation
  5. Record post-stimulation activity

Use ``--resume`` to pick up from the last successful phase if a previous
run failed partway through.
"""
import argparse
import numpy as np


def select_random_pair(exp):
    """Pick a neuron pair whose connectivity falls in a target range."""
    lower_bound = .3
    upper_bound = .6
    n_attempts = 500
    total_neurons = 20
    connectivity_matrix = exp.data.connectivity_matrix
    n_neurons = min(total_neurons, connectivity_matrix.shape[0])

    print(f"   Selecting random pair from {n_neurons} neurons")
    for i in range(n_attempts):
        random_pair = np.random.randint(0, n_neurons, 2)
        if connectivity_matrix[random_pair[0], random_pair[1]] > lower_bound and connectivity_matrix[random_pair[0], random_pair[1]] < upper_bound:
            print(f"   Found random pair {random_pair}: conn={connectivity_matrix[random_pair[0], random_pair[1]]} within bounds")
            return {'selected_pair': random_pair}
    raise ValueError("No random pair found within bounds")


def main():
    parser = argparse.ArgumentParser(description="Closed-Loop Plasticity Experiment")
    parser.add_argument("--config", type=str, default=None,
                        help="Maxwell configuration file")
    parser.add_argument("--record_duration", type=int, default=300,
                        help="Recording duration in seconds")
    parser.add_argument("--project_id", type=str, default="closed_loop",
                        help="Project identifier (default: closed_loop)")
    parser.add_argument("--chip_id", type=str, default="default_chip",
                        help="Chip identifier (default: default_chip)")
    parser.add_argument("--experiment_name", type=str, default="closed_loop_plasticity",
                        help="Experiment name")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the last successful phase of a previous run")
    args = parser.parse_args()

    from braindance.core.phases_v3.experiment_v3 import Experiment
    from braindance.core.phases_v3.phases3 import RecordPhaseV3
    from braindance.core.phases_v3.phases_analysis_3 import RTSortPhaseV3, ConnectivityPhaseV3
    from braindance.core.phases_v3.phase_base_v3 import phase
    from braindance.core.phases_v3.phases3_loop import ClosedLoopPhaseV3


    params = {
        "config": args.config,
        "record_duration": args.record_duration
    }

    exp = Experiment(
        args.experiment_name,
        params=params,
        project_id=args.project_id,
        chip_id=args.chip_id,
    )

    # --- Build phase pipeline ---
    exp.add_phase(RecordPhaseV3(duration=args.record_duration))
    exp.add_phase(RTSortPhaseV3(
        sorter='rt_sort',
        min_spikes=50,
        recording_window_ms=(0, 30000),
        verbose=True,
    ))
    exp.add_phase(ConnectivityPhaseV3(window_ms=20.0))
    exp.add_phase(phase("SelectRandomPair")(select_random_pair))
    exp.add_phase(ClosedLoopPhaseV3(
        duration=args.record_duration,
        timing_debug=True,
        timing_interval=20,
    ))
    exp.add_phase(RecordPhaseV3(duration=args.record_duration))

    # --- Run (with optional resume) ---
    success = exp.run(resume=args.resume)

    if success:
        print("\n All phases completed!")
    else:
        print(f"\n Experiment stopped at phase {exp.current_phase_idx}. "
              f"Re-run with --resume to continue.")


if __name__ == "__main__":
    main()
