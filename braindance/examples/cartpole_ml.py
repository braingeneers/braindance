"""
Example of using CartPolePhase with RT-sort and REINFORCE learning.

This example demonstrates:
1. Using RT-sort for real-time spike detection from motor neurons
2. Neural network policy for mapping motor neuron activity to control actions
3. REINFORCE algorithm for optimizing the policy based on game rewards
4. Optional ContextualTrainer for context-aware stimulation pattern selection

Neuron Allocation:
- Sequences detected by RT-sort are automatically allocated into sensory, motor,
  and training groups. Minimum 8 sequences required (2 sensory + 4 motor + 2 training).
  Extra sequences go to training (up to 6 max), then the rest to motor.

Trainer Options:
- None: Pure REINFORCE learning without stimulation training
- TetanusTrainer: Multi-armed bandit approach (stimulates when reward decreases)
- ContextualTrainer: MLP-based contextual RL that learns when/what to stimulate
"""

import argparse

from braindance.core.phases_v3.phases3 import RecordPhaseV3
from braindance.core.phases_v3.phases_analysis_3 import RTSortPhaseV3
from braindance.core.phases_v3.phases3_loop import CartPolePhasWithViz
from braindance.core.phases_v3.experiment_v3 import Experiment


def run_cartpole_reinforcement_experiment(params):
    """Run a CartPole experiment with REINFORCE learning.

    Neuron allocation is handled automatically by CartPolePhase based on
    the number of sequences detected by RT-sort:
        - 2 sensory neurons (always)
        - 2-6 training neurons (scales with available sequences, capped at 6)
        - 4+ motor neurons (everything else)
        - Minimum 8 sequences required
    """

    # Create experiment
    exp = Experiment(name="cartpole_reinforcement", params=params)

    # Phase 1: Brief recording to establish baseline
    recording_phase = RecordPhaseV3(duration=30, suffix="_baseline")

    rtsort_phase = RTSortPhaseV3(min_spikes=100)

    # Neuron allocation is automatic - pass None and CartPolePhase will
    # allocate based on the number of detected sequences after RT-sort runs.
    # Trainer is also created automatically when trainer_type is specified.
    trainer_type = params.get('trainer_type', 'none')

    if trainer_type != 'none':
        print(f"\nUsing {trainer_type} trainer (will be created after sequence detection)")
    else:
        print("\nNo trainer - using pure REINFORCE learning")

    # Phase 2: CartPole with REINFORCE
    # sensory/motor/training neurons are left as None for auto-allocation
    cartpole_phase = CartPolePhasWithViz(
        sensory_neurons=None,
        motor_neurons=None,
        training_neurons=None,
        amp_mv=200,
        phase_width=100,
        read_period_ms=50,
        n_episodes=params.get('n_episodes', 1000),
        continuous=True,  # Continuous control
        verbose=params.get('verbose', True),
        # REINFORCE parameters
        learning_rate=0.01,
        gamma=0.99,
        policy_hidden_size=64,
        # RT-sort parameters
        use_numba=True,
        assistive=0.0,  # No assistance, pure learning
        # Trainer settings - deferred creation after neuron allocation
        trainer_type=trainer_type if trainer_type != 'none' else None,
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

    # Print contextual trainer stats if used
    trainer = cartpole_phase.trainer
    if cartpole_phase.use_contextual_trainer and trainer is not None:
        print("\nContextual Trainer Statistics:")
        print(f"  Actions taken: {len(trainer.action_history)}")
        if trainer.action_history:
            no_stim_count = sum(1 for a in trainer.action_history if a == trainer.n_patterns)
            stim_count = len(trainer.action_history) - no_stim_count
            print(f"  Stimulation actions: {stim_count}")
            print(f"  No-stim actions: {no_stim_count}")
            print(f"  Final action probabilities: {trainer.get_probs()}")


if __name__ == "__main__":
    # Example of how the motor neuron vector is processed:
    #
    # 1. RT-sort detects spikes from 6 motor neurons in real-time
    # 2. Spike counts are converted to firing rates with moving average
    # 3. Firing rates are normalized and fed to the policy network
    # 4. Policy network (2-layer NN) outputs a continuous action in [-1, 1]
    # 5. Action controls the cart (left/right force)
    # 6. Game rewards are used to update the policy via REINFORCE
    #
    # With ContextualTrainer:
    # - The trainer observes: motor spike stats, policy gradient magnitude,
    #   reward delta, and episode length
    # - It learns when to apply stimulation and which pattern to use
    # - It can also choose "no-stim" if stimulation isn't beneficial

    parser = argparse.ArgumentParser(
        description="CartPole Reinforcement Learning Example"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Maxwell configuration file"
    )
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose mode")
    parser.add_argument(
        "--trainer", type=str, default="none",
        choices=["none", "tetanus", "contextual"],
        help="Trainer type: 'none' (pure REINFORCE), 'tetanus' (multi-armed bandit), 'contextual' (MLP-based)"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=1000,
        help="Number of episodes to run"
    )
    args = parser.parse_args()

    params = {
        "config": args.config,
        "verbose": args.verbose,
        "trainer_type": args.trainer,
        "n_episodes": args.n_episodes,
    }

    print("CartPole REINFORCE Example")
    print("=========================")
    print("This example uses:")
    print("- RT-sort for real-time spike detection")
    print("- Neural network policy mapping motor neurons to actions")
    print("- REINFORCE algorithm for policy optimization")
    print()
    print("Trainer options:")
    print("  --trainer none       Pure REINFORCE (no stimulation training)")
    print("  --trainer tetanus    TetanusTrainer (multi-armed bandit)")
    print("  --trainer contextual ContextualTrainer (MLP-based contextual RL)")
    print()

    run_cartpole_reinforcement_experiment(params)
