"""
Load catalog, compute per-episode reward stats for cartpole experiments.
reward_log has one row per episode where reward = timesteps balanced (episode length).
"""

import pandas as pd
from braindance.utils.data_manager import load_catalog


def main(save_csv=True):
    catalog = load_catalog()

    # Filter to cartpole experiments
    catalog = catalog.filter(experiment__contains="cartpole")
    print(f"Found {len(catalog)} cartpole experiments")

    def get_episode_stats(rec):
        print(f"  Processing {rec.proj}/{rec.chip}/{rec.experiment}...")
        stats = {"max_reward": None, "min_reward": None, "mean_reward": None, "n_episodes": None}

        # reward_log: one row per episode, reward = timesteps balanced
        reward_log = rec.reward_log
        if reward_log is not None and "reward" in reward_log.columns and len(reward_log) > 0:
            rewards = reward_log["reward"]
            stats["max_reward"] = rewards.max()
            stats["min_reward"] = rewards.min()
            stats["mean_reward"] = rewards.mean()
            stats["n_episodes"] = len(rewards)
        else:
            # Fallback: try game_log grouped by episode-like boundaries
            game_log = rec.game_log
            if game_log is not None and "reward" in game_log.columns:
                # game_log has per-timestep reward=1.0, total sum is all we can get
                stats["max_reward"] = game_log["reward"].sum()
                stats["n_episodes"] = 1  # can't distinguish episodes without reward_log

        rec.clear_cache()
        return stats

    results = catalog.apply(get_episode_stats, on_error="warn")

    # Build summary DataFrame
    summary = pd.DataFrame(
        {
            "proj": catalog.proj,
            "chip": catalog.chip,
            "experiment": catalog.experiment,
            "max_reward": [r["max_reward"] if isinstance(r, dict) else None for r in results],
            "min_reward": [r["min_reward"] if isinstance(r, dict) else None for r in results],
            "mean_reward": [r["mean_reward"] if isinstance(r, dict) else None for r in results],
            "n_episodes": [r["n_episodes"] if isinstance(r, dict) else None for r in results],
        }
    )
    summary = summary.sort_values("max_reward", ascending=False).reset_index(drop=True)

    # Round for readability
    for col in ["max_reward", "min_reward", "mean_reward"]:
        summary[col] = summary[col].round(1)

    print("\n" + "=" * 80)
    print("Cartpole Episode Reward Summary")
    print("(reward = timesteps balanced per episode)")
    print("=" * 80)
    print(summary.to_string(index=False))
    print(f"\nTotal experiments: {len(summary)}")
    print(f"With reward data: {summary['max_reward'].notna().sum()}")

    if save_csv:
        out_path = "reward_summary.csv"
        summary.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")

    return summary


if __name__ == "__main__":
    main()
