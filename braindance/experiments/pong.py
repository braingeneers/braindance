"""Run a simple closed-loop Pong experiment on Maxwell hardware.

Two configured stimulation electrodes encode whether the ball is above or below
the paddle. Spikes on configured recording channels move the paddle up or down.
This is a spike-window controller intended as a minimal hardware integration
example; it does not train a policy.
"""

import argparse
from pathlib import Path

from braindance.config import get_output_dir
from braindance.core.maxwell_env import MaxwellEnv
from braindance.games.pong_env import PongEnv


def main(
    config=None, stim_electrodes=None, up_channels=None, down_channels=None,
    save_dir=None, name="pong", episodes=5, render=True, seed=0, max_time_sec=300,
    read_period_ms=50, sampling_hz=20_000, amplitude_mv=200, phase_width_us=100,
):
    """Connect neural activity to Pong and Pong state to stimulation.

    ``up_channels`` and ``down_channels`` are Maxwell recording channel IDs.
    ``stim_electrodes`` contains two physical electrode IDs; index 0 signals
    that the ball is above the paddle and index 1 signals that it is below.
    """
    if config is None:
        raise ValueError("config is required for Maxwell hardware")
    if not Path(config).is_file():
        raise ValueError(f"Maxwell config does not exist: {config}")
    if stim_electrodes is None or len(stim_electrodes) != 2:
        raise ValueError("stim_electrodes must contain above and below electrode IDs")
    if not up_channels or not down_channels:
        raise ValueError("up_channels and down_channels must both be non-empty")
    all_ids = list(stim_electrodes) + list(up_channels) + list(down_channels)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for value in all_ids):
        raise ValueError("electrode and channel IDs must be non-negative integers")
    if len(set(stim_electrodes)) != 2:
        raise ValueError("above and below stimulation electrodes must be distinct")
    if (len(set(up_channels)) != len(up_channels)
            or len(set(down_channels)) != len(down_channels)):
        raise ValueError("recording channel groups must not contain duplicates")
    if set(up_channels) & set(down_channels):
        raise ValueError("up_channels and down_channels must not overlap")
    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if max_time_sec <= 0:
        raise ValueError("max_time_sec must be positive")
    if read_period_ms <= 0 or sampling_hz <= 0:
        raise ValueError("read_period_ms and sampling_hz must be positive")

    period_frames = round(sampling_hz * read_period_ms / 1_000)
    if period_frames < 1:
        raise ValueError("read_period_ms is shorter than one acquisition frame")
    if save_dir is None:
        save_dir = get_output_dir() / "pong"
    neural_env = MaxwellEnv(
        config=config, name=name, save_dir=save_dir,
        stim_electrodes=stim_electrodes, observation_type="spikes",
        max_time_sec=max_time_sec,
    )
    game_env = None
    try:
        if any(channel >= neural_env.num_channels for channel in up_channels + down_channels):
            raise ValueError("recording channel ID outside the configured Maxwell channels")
        game_env = PongEnv(render_mode="human" if render else None)
        up_channels = set(up_channels)
        down_channels = set(down_channels)
        episode_rewards = []
        previous_action = 0
        observation, _ = game_env.reset(seed=seed)
        reward_total = 0.0
        while len(episode_rewards) < episodes:
            sensory_index = 0 if observation[1] < observation[0] else 1
            stimulus = [("stim", [sensory_index], amplitude_mv, phase_width_us)]
            # Check completion before stimulating: step(action=...) dispatches
            # before checking done. stimulate() itself consumes no observation.
            events, acquisition_done = neural_env.step()
            if acquisition_done:
                break
            events = list(events or [])
            first_frame = neural_env.latest_frame
            if first_frame is None:
                raise RuntimeError("Maxwell did not provide a frame number")
            neural_env.stimulate(stimulus)
            # Include the boundary packet and acquire through one frame period.
            # This is an inclusive packet window, not a causal-latency estimate.
            target_frame = first_frame + period_frames
            while neural_env.latest_frame < target_frame:
                more_events, acquisition_done = neural_env.step()
                events.extend(more_events or [])
                if acquisition_done:
                    break
            if acquisition_done:
                break

            up_spikes = sum(event.channel in up_channels for event in events)
            down_spikes = sum(event.channel in down_channels for event in events)
            if up_spikes > down_spikes:
                previous_action = 0
            elif down_spikes > up_spikes:
                previous_action = 1
            observation, reward, terminated, truncated, _ = game_env.step(previous_action)
            reward_total += reward
            if terminated or truncated:
                episode_rewards.append(reward_total)
                print(f"Episode {len(episode_rewards)}: {reward_total:.0f} hits")
                if len(episode_rewards) < episodes:
                    observation, _ = game_env.reset(seed=seed + len(episode_rewards))
                    reward_total = 0.0
    finally:
        try:
            if game_env is not None:
                game_env.close()
        finally:
            neural_env.close()
    return episode_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Maxwell array configuration file")
    parser.add_argument("--stim-electrodes", nargs=2, type=int, required=True,
                        metavar=("ABOVE", "BELOW"))
    parser.add_argument("--up-channels", nargs="+", type=int, required=True)
    parser.add_argument("--down-channels", nargs="+", type=int, required=True)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--name", default="pong")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-time-sec", type=float, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    main(config=args.config, stim_electrodes=args.stim_electrodes,
         up_channels=args.up_channels, down_channels=args.down_channels,
         save_dir=args.save_dir, name=args.name, episodes=args.episodes,
         render=not args.no_render, seed=args.seed, max_time_sec=args.max_time_sec)
