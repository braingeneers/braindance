"""Run a short V3 experiment using a Maxwell H5 replay source."""

from braindance.config import get_data_dir, get_output_dir
from braindance.core.phases_v3.experiment_v3 import Experiment
from braindance.core.phases_v3.phases3 import FrequencyStimPhaseV3, RecordPhaseV3


def main(
    source=None,
    record_seconds=0.5,
    stim_seconds=0.5,
    stim_electrode=0,
    speed=1.0,
):
    if source is None:
        source = get_data_dir() / "replay_fixtures" / "maxwell_replay_smoke.raw.h5"

    params = {
        "record_duration": record_seconds,
        "stim_electrodes": [stim_electrode],
        "verbose": True,
        "maxwell_env": {
            "replay": {
                "source": source,
                "speed": speed,
                "loop": False,
                "write_output": True,
            }
        },
    }
    experiment = Experiment(
        "replay_example",
        params=params,
        save_dir=str(get_output_dir() / "replay_example"),
        overwrite_existing=True,
    )
    experiment.add_phase(RecordPhaseV3(duration=record_seconds, name="baseline"))
    experiment.add_phase(FrequencyStimPhaseV3(
        stim_command=([0], 250, 100),
        stim_freq=5.0,
        duration=stim_seconds,
        tag="replay_stim",
    ))
    return experiment.run()


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
