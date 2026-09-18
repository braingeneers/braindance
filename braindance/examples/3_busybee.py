"""
BusyBee recording and frequency sweeps using Phase V3.

Alternate spontaneous recording with .5, 1, 2, 4, and 8 Hz stimulation.
Use --resume to continue from the last successful phase.
"""
import argparse
import json
from pathlib import Path


def main(
    json_file=None, cycles=40, record_seconds=600, dry_run=False,
    project_id="busybee", chip_id="default_chip", resume=False,
):
    if cycles < 1 or record_seconds <= 0:
        raise ValueError("cycles and record_seconds must be positive")
    if dry_run:
        return [
            {"record_seconds": record_seconds, "frequency_hz": frequency,
             "replicates": replicates, "amplitude_mv": 400, "phase_width_us": 200,
             "order": "ran", "single_connect": True}
            for _ in range(cycles)
            for frequency, replicates in zip((.5, 1, 2, 4, 8), (50, 100, 200, 400, 800))
        ]
    if json_file is None:
        raise ValueError("--json is required for acquisition")
    config = json.loads(Path(json_file).read_text())
    from braindance.config import get_output_dir
    from braindance.core.phases_v3.experiment_v3 import Experiment
    from braindance.core.phases_v3.phases3 import RecordPhaseV3, NeuralSweepPhaseV3

    params = {
        "config": config["config"],
        "stim_electrodes": config["stim_electrodes"],
        "verbose": True,
    }
    if not params["stim_electrodes"]:
        raise ValueError("stim_electrodes must contain physical electrode IDs")

    exp = Experiment(
        config["name"] + "_cont",
        params=params,
        save_dir=config.get("save_dir", str(get_output_dir() / "busybee")),
        project_id=project_id,
        chip_id=chip_id,
    )

    # --- Build phase pipeline ---
    for _ in range(cycles):
        for frequency, replicates in zip((.5, 1, 2, 4, 8), (50, 100, 200, 400, 800)):
            exp.add_phase(RecordPhaseV3(duration=record_seconds))
            exp.add_phase(NeuralSweepPhaseV3(
                amp_bounds=400,
                stim_freq=frequency,
                replicates=replicates,
                phase_length=200,
                # With one amplitude, ran preserves the original rna order.
                order="ran",
                single_connect=True,
                tag="causal",
            ))

    # --- Run (with optional resume) ---
    success = exp.run(resume=resume)

    if success:
        print("\n All phases completed!")
    else:
        print(f"\n Experiment stopped at phase {exp.current_phase_idx}. "
              f"Re-run with --resume to continue.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", "-j")
    parser.add_argument("--cycles", type=int, default=40)
    parser.add_argument("--record-seconds", type=float, default=600)
    parser.add_argument("--dry-run", action="store_true", help="Print schedule without hardware")
    parser.add_argument("--project_id", default="busybee", help="Project identifier")
    parser.add_argument("--chip_id", default="default_chip", help="Chip identifier")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the last successful phase of a previous run")
    args = parser.parse_args()
    result = main(
        args.json, args.cycles, args.record_seconds, args.dry_run,
        project_id=args.project_id, chip_id=args.chip_id, resume=args.resume,
    )
    if result is not None:
        print(json.dumps(result, indent=2))
