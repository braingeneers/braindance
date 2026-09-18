"""
Example of using FoodLandPhaseV3 with RT-sort and manifold-based learning.
"""

import argparse

from braindance.core.phases_v3.experiment_v3 import Experiment
from braindance.core.phases_v3.phases3 import RecordPhaseV3
from braindance.core.phases_v3.phases3_foodland import FoodLandPhaseV3
from braindance.core.phases_v3.phases_analysis_3 import RTSortPhaseV3


def run_foodland_reinforcement_experiment(
    params,
    resume=False,
    name="foodland_reinforcement",
):
    """Run a FoodLand experiment with neural reservoir control."""

    exp = Experiment(name=name, params=params)

    recording_phase = RecordPhaseV3(duration=30, suffix="_baseline")
    rtsort_phase = RTSortPhaseV3(min_spikes=100)

    foodland_phase = FoodLandPhaseV3(
        sensory_neurons=None,
        motor_neurons=None,
        training_neurons=None,
        amp_mv=200,
        phase_width=100,
        read_period_ms=params.get("read_period_ms", 50),
        train_period_ms=200,
        wait_period_ms=400,
        n_episodes=params["n_episodes"],
        verbose=params["verbose"],
        use_numba=True,
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        encode_mode=params["encode_mode"],
        decode_mode=params["decode_mode"],
        render_mode=params["render_mode"],
        n_features=params["n_features"],
        max_episode_steps=params.get("max_episode_steps", 100),
        trainer_type=params.get("trainer_type"),
        reward_type=params["reward_type"],
        food_count=params["food_count"],
        spike_count=params["spike_count"],
        hunger=params["hunger"],
        pca_calibration_s=params["pca_calibration_s"],
    )

    exp.add_phase(recording_phase)
    exp.add_phase(rtsort_phase)
    exp.add_phase(foodland_phase)
    exp.run(resume=resume)

    rewards = exp.data.get("episode_rewards", [])
    print("\nExperiment completed!")
    print(f"Total episodes played: {exp.data.get('total_episodes', 0)}")
    if rewards:
        print(f"Final reward: {rewards[-1]:.2f}")
        print(f"Best reward: {max(rewards):.2f}")
        import numpy as np

        print(f"Mean reward (last 10): {np.mean(rewards[-10:]):.2f}")

    final_policy_state = exp.data.get("final_policy_state")
    if final_policy_state:
        print(f"Encode mode: {final_policy_state.get('encode_mode')}")
        print(f"Decode mode: {final_policy_state.get('decode_mode')}")
        if final_policy_state.get("pca_explained_variance") is not None:
            print(
                "PCA explained variance:",
                final_policy_state.get("pca_explained_variance"),
            )
        print("Final policy state saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FoodLand Reservoir Reinforcement Learning Example"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="foodland_reinforcement",
        help="Experiment name (creates a separate data directory per name)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Maxwell configuration file",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of game episodes",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--encode-mode",
        type=str,
        default="policy_continuous",
        choices=["fixed_sigmoid", "policy_continuous"],
        help="Input encoding mode",
    )
    parser.add_argument(
        "--decode-mode",
        type=str,
        default="pca",
        choices=["pca", "ppo", "direct"],
        help="Output decoding mode",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "none"],
        help="Game render mode",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=2,
        help="Number of FoodLand observation features / sensory neurons",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=100,
        help="Maximum steps per episode before truncation",
    )
    parser.add_argument(
        "--read-period-ms",
        type=int,
        default=50,
        help="Read period in milliseconds",
    )
    parser.add_argument(
        "--food-count",
        type=int,
        default=3,
        help="Number of food targets in the arena",
    )
    parser.add_argument(
        "--spike-count",
        type=int,
        default=0,
        help="Number of spike hazards in the arena",
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default="dense",
        choices=["dense", "sparse"],
        help="FoodLand reward function",
    )
    parser.add_argument(
        "--hunger",
        type=float,
        default=0.5,
        help="Per-step hunger penalty in FoodLand",
    )
    parser.add_argument(
        "--pca-calibration-s",
        type=float,
        default=30.0,
        help="Seconds of spontaneous activity used to fit PCA before gameplay",
    )
    parser.add_argument(
        "--trainer-type",
        type=str,
        default=None,
        choices=["tetanus", "contextual"],
        help="Training neuron stimulation strategy (optional)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (skip recording/rt-sort if already done)",
    )
    args = parser.parse_args()

    params = {
        "config": args.config,
        "verbose": args.verbose,
        "n_episodes": args.n_episodes,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "encode_mode": args.encode_mode,
        "decode_mode": args.decode_mode,
        "render_mode": None if args.render_mode == "none" else args.render_mode,
        "n_features": args.n_features,
        "max_episode_steps": args.max_episode_steps,
        "read_period_ms": args.read_period_ms,
        "reward_type": args.reward_type,
        "food_count": args.food_count,
        "spike_count": args.spike_count,
        "hunger": args.hunger,
        "pca_calibration_s": args.pca_calibration_s,
        "trainer_type": args.trainer_type,
    }

    print("FoodLand Reservoir Reinforcement Example")
    print("========================================")
    print("This example uses:")
    print("- RT-sort for real-time spike detection")
    print("- FoodLand sensory features for stimulation")
    print(f"- Encode mode: {params['encode_mode']}")
    print(f"- Decode mode: {params['decode_mode']}")
    print(f"- Reward type: {params['reward_type']}")
    print(f"- PCA calibration: {params['pca_calibration_s']}s")
    print(f"- Episodes: {params['n_episodes']}")
    print(f"- Experiment: {args.name}")
    if args.resume:
        print("- RESUMING from last checkpoint")
    print()

    run_foodland_reinforcement_experiment(
        params,
        resume=args.resume,
        name=args.name,
    )
