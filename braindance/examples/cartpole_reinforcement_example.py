"""
Example of using CartPolePhase with RT-sort and REINFORCE learning.

This example demonstrates:
1. Using RT-sort for real-time spike detection from motor neurons
2. Neural network policy for mapping motor neuron activity to control actions
3. REINFORCE algorithm for optimizing the policy based on game rewards
"""

import argparse

from braindance.core.phases_v3.phases3 import RecordPhaseV3
from braindance.core.phases_v3.phases_analysis_3 import RTSortPhaseV3
from braindance.core.phases_v3.phases3_loop import CartPolePhasWithViz
from braindance.core.phases_v3.experiment_v3 import Experiment


def run_cartpole_reinforcement_experiment(params):
    """Run a CartPole experiment with REINFORCE learning.

    Neuron allocation is automatic based on detected sequences:
        - 2 sensory, 4+ motor, 2-6 training (min 8 sequences needed)
    """

    # Create experiment
    exp = Experiment(name="cartpole_reinforcement", params=params)

    # Phase 1: Brief recording to establish baseline
    recording_phase = RecordPhaseV3(duration=30, suffix="_baseline")

    rtsort_phase = RTSortPhaseV3(min_spikes=100)

    # Phase 2: CartPole with REINFORCE
    # Neurons auto-allocated from detected sequences
    cartpole_phase = CartPolePhasWithViz(
        sensory_neurons=None,
        motor_neurons=None,
        training_neurons=None,
        amp_mv=200,
        phase_width=100,
        read_period_ms=50,
        n_episodes=1000,
        continuous=True,  # Continuous control
        verbose=True,
        # REINFORCE parameters
        learning_rate=0.01,
        gamma=0.99,
        policy_hidden_size=64,
        # RT-sort parameters
        use_numba=True,
        assistive=0.0,  # No assistance, pure learning
    )

    # Add phases to experiment
    exp.add_phase(recording_phase)
    exp.add_phase(rtsort_phase)
    exp.add_phase(cartpole_phase)

    # Run experiment
    exp.run()

    # Access results
    results = exp.data.total_episodes
    print("\nExperiment completed!")
    print(f"Total episodes played: {results.get('total_episodes', 0)}")

    # The policy network state is saved and can be loaded for future use
    if "final_policy_state" in results:
        print("Policy network trained and saved.")


if __name__ == "__main__":
    # Example of how the motor neuron vector is processed:
    #
    # 1. RT-sort detects spikes from 6 motor neurons in real-time
    # 2. Spike counts are converted to firing rates with moving average
    # 3. Firing rates are normalized and fed to the policy network
    # 4. Policy network (2-layer NN) outputs a continuous action in [-1, 1]
    # 5. Action controls the cart (left/right force)
    # 6. Game rewards are used to update the policy via REINFORCE

    parser = argparse.ArgumentParser(
        description="CartPole Reinforcement Learning Example"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Maxwell configuration file"
    )
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose mode")
    # parser.add_argument("--record_duration", type=int, default=300, help="Recording duration in seconds")
    args = parser.parse_args()

    params = {
        "config": args.config,
        "verbose": args.verbose,
    }

    print("CartPole REINFORCE Example")
    print("=========================")
    print("This example uses:")
    print("- RT-sort for real-time spike detection")
    print("- Neural network policy mapping motor neurons to actions")
    print("- REINFORCE algorithm for policy optimization")
    print()

    run_cartpole_reinforcement_experiment(params)
