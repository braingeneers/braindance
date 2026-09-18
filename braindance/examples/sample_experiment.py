"""Record activity while alternating stimulation between two electrodes.

Run a 5-second synthetic replay (no Maxwell hardware or recording file needed)::

    python -m braindance.examples.sample_experiment

For live acquisition, call ``main(replay_source=None, config="/path/to/routing.cfg",
stim_electrodes=(10254, 14130))`` with your routing file and routed electrode IDs.
Replay exercises the loop; it does not simulate a biological stimulation response.
"""

from braindance.config import get_output_dir
from braindance.core.maxwell_env import MaxwellEnv
from braindance.core.params import maxwell_params


def main(
    name="sample_experiment",
    stim_electrodes=(0, 1),
    max_time_sec=5,
    config=None,
    replay_source="sine",
    observation_type="raw",
    stim_interval_sec=1.0,
    amplitude_mv=150,
    phase_length_us=100,
    buffer_size=20,
):
    """Collect observations and stimulate each configured electrode in turn."""
    if not stim_electrodes:
        raise ValueError("Provide at least one stimulation electrode.")
    if max_time_sec <= 0 or stim_interval_sec <= 0:
        raise ValueError("Duration and stimulation interval must be positive.")
    if replay_source is None and config is None:
        raise ValueError("Live acquisition requires a Maxwell routing config.")

    save_dir = get_output_dir() / "sample_experiment"
    save_dir.mkdir(parents=True, exist_ok=True)
    params = maxwell_params.copy()
    params.update(
        name=name,
        save_dir=str(save_dir),
        stim_electrodes=list(stim_electrodes),
        max_time_sec=max_time_sec,
        config=config,
        multiprocess=False,
        render=False,
        observation_type=observation_type,
        replay={"source": replay_source} if replay_source is not None else None,
    )

    env = MaxwellEnv(**params)
    electrode_index = 0
    done = False
    print(f"Running {name}; output directory: {save_dir}")
    try:
        while not done:
            action = None
            if env.stim_dt >= stim_interval_sec:
                # Actions use indices into stim_electrodes, not physical IDs.
                action = ([electrode_index], amplitude_mv, phase_length_us)
                print(
                    f"{env.time_elapsed():.3f}s: stimulating electrode "
                    f"{stim_electrodes[electrode_index]}"
                )
                electrode_index = (electrode_index + 1) % len(stim_electrodes)

            # Reading a frame advances replay time; do not gate step() on env.dt.
            obs, done = env.step(action=action, buffer_size=buffer_size)
            # Add observation processing here. With observation_type="raw",
            # obs contains voltage data; use "spikes" for spike events instead.
    finally:
        env.close()
    print("Experiment complete.")


if __name__ == "__main__":
    main()
