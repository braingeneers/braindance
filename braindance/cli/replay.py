"""Command-line entry point for Maxwell H5 replay."""

import argparse
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from braindance.config import get_data_dir, get_output_dir
from braindance.core.replay import (
    SpikeEvent,
    find_recordings,
    inspect_recording,
    pack_replay_packet,
    resolve_replay_source,
    unpack_replay_packet,
)


# TODO: Fill these after publishing replay_fixtures/maxwell_replay_smoke.raw.h5.
EXAMPLE_H5_URL = None
EXAMPLE_H5_SHA256 = None


def _print_recording(info, root=None):
    path = info["path"]
    try:
        display_path = path.relative_to(root) if root is not None else path
    except ValueError:
        display_path = path
    print(
        f"{display_path} | {info['size_bytes'] / 1024 ** 2:.1f} MB | "
        f"{info['num_channels']} ch | {info['sampling_hz']:.0f} Hz | "
        f"{info['duration_s']:.3f} s | v{info['version']}"
    )


def list_command(args):
    root = Path(args.data_dir).expanduser() if args.data_dir else get_data_dir()
    if not root.exists():
        print(f"Data root unavailable: {root}", file=sys.stderr)
        return 1
    recordings = find_recordings(root)
    print(f"Data root: {root}")
    print(f"Compatible recordings: {len(recordings)}")
    for info in recordings:
        _print_recording(info, root=root)
    return 0


def _zmq_loopback():
    import zmq

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    subscriber = context.socket(zmq.SUB)
    try:
        publisher.setsockopt(zmq.LINGER, 0)
        subscriber.setsockopt(zmq.LINGER, 0)
        port = publisher.bind_to_random_port("tcp://127.0.0.1")
        subscriber.connect(f"tcp://127.0.0.1:{port}")
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        time.sleep(0.1)
        expected = pack_replay_packet(7, np.asarray([1.0, 2.0], dtype=np.float32), [SpikeEvent(7, 1, -4.0)])
        received = None
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline and received is None:
            publisher.send_multipart(expected)
            if subscriber.poll(timeout=100):
                received = subscriber.recv_multipart()
        if received is None:
            raise TimeoutError("ZMQ loopback timed out")
        frame, raw, events = unpack_replay_packet(received)
        if frame != 7 or not np.array_equal(raw, [1.0, 2.0]) or len(events) != 1:
            raise RuntimeError("ZMQ loopback payload mismatch")
    finally:
        subscriber.close(linger=0)
        publisher.close(linger=0)
        context.term()


def doctor_command(args):
    failures = []
    env_name = os.getenv("CONDA_DEFAULT_ENV") or Path(sys.prefix).name
    print(f"Python: {sys.executable}")
    print(f"Environment: {env_name}")
    if args.require_env and env_name != args.require_env:
        failures.append(f"expected environment {args.require_env}, found {env_name}")

    for module_name in ("numpy", "h5py", "zmq", "spikeinterface", "braindance"):
        try:
            module = __import__(module_name)
            print(f"[ok] import {module_name} {getattr(module, '__version__', '')}".rstrip())
        except Exception as exc:
            failures.append(f"import {module_name}: {exc}")

    try:
        import maxlab  # noqa: F401
        print("[info] real maxlab available")
    except ImportError:
        print("[ok] maxlab unavailable; replay uses dummy_maxlab")

    source = None
    try:
        source = resolve_replay_source(
            h5_path=args.h5,
            recording=args.recording,
            sample=args.sample,
            data_dir=args.data_dir,
            doctor=True,
        )
        info = inspect_recording(source)
        print("[ok] replay input")
        _print_recording(info)
    except Exception as exc:
        failures.append(f"replay input: {exc}")

    if source is not None:
        try:
            from braindance.analysis.data_loader import load_data_maxwell
            from braindance.core.maxwell_env import MaxwellEnv
            from spikeinterface.extractors import MaxwellRecordingExtractor

            with tempfile.TemporaryDirectory(prefix="braindance-replay-") as temp_dir:
                env = MaxwellEnv(
                    config=None,
                    name="doctor",
                    save_dir=temp_dir,
                    observation_type="raw",
                    verbose=0,
                    replay={
                        "source": source,
                        "speed": 0,
                        "start_frame": 0,
                        "stop_frame": min(100, info["num_frames"]),
                        "write_output": True,
                    },
                )
                observations, done = env.step(buffer_size=min(100, info["num_frames"]))
                env.close()
                if not observations or not done:
                    raise RuntimeError("direct replay did not return expected bounded batch")
                output = Path(temp_dir) / "doctor.raw.h5"
                loaded = load_data_maxwell(output, start=0, length=min(10, info["num_frames"]))
                extracted = MaxwellRecordingExtractor(output)
                if loaded.shape[0] != info["num_channels"]:
                    raise RuntimeError("BrainDance output channel count mismatch")
                if extracted.get_num_channels() != info["num_channels"]:
                    raise RuntimeError("SpikeInterface output channel count mismatch")
                if hasattr(extracted, "neo_reader") and hasattr(extracted.neo_reader, "h5_file"):
                    extracted.neo_reader.h5_file.close()
            print("[ok] direct replay and compatible output")
        except Exception as exc:
            failures.append(f"direct replay: {exc}")

    try:
        _zmq_loopback()
        print("[ok] ZMQ loopback")
    except Exception as exc:
        failures.append(f"ZMQ loopback: {exc}")

    if failures:
        print("Doctor failures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("Replay doctor passed")
    return 0


def _parse_speed(value):
    if str(value).lower() in {"max", "unthrottled"}:
        return 0.0
    speed = float(value)
    if speed < 0:
        raise argparse.ArgumentTypeError("speed must be >= 0 or 'max'")
    return speed


def run_command(args):
    from braindance.core.phases_v3.experiment_v3 import Experiment
    from braindance.core.phases_v3.phases3 import FrequencyStimPhaseV3, RecordPhaseV3

    source = resolve_replay_source(
        h5_path=args.h5,
        recording=args.recording,
        sample=args.sample,
        data_dir=args.data_dir,
    )
    info = inspect_recording(source)
    max_source_seconds = max(args.record_seconds, args.stim_seconds or 0.0)
    stop_frame = min(
        info["num_frames"],
        max(1, int(math.ceil(max_source_seconds * info["sampling_hz"]))),
    )
    output_dir = Path(args.output_dir) if args.output_dir else get_output_dir() / "fake_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or time.strftime("replay_%Y%m%d_%H%M%S")
    stim_electrodes = [] if args.stim_electrode is None else [int(args.stim_electrode)]
    params = {
        "record_duration": float(args.record_seconds),
        "replay_source_duration": float(args.record_seconds),
        "stim_electrodes": stim_electrodes,
        "verbose": bool(args.verbose),
        "maxwell_env": {
            "max_time_sec": max_source_seconds + 1.0,
            "replay": {
                "source": source,
                "speed": float(args.speed),
                "start_frame": 0,
                "stop_frame": stop_frame,
                "loop": False,
                "chunk_frames": 20000 if args.speed == 0 else 2000,
                "write_output": not args.no_write_output,
                "transport": "direct",
            }
        },
    }
    experiment = Experiment(name, params=params, save_dir=str(output_dir / name))
    experiment.add_phase(RecordPhaseV3(duration=float(args.record_seconds), name="replay_record"))
    if args.stim_seconds and args.stim_electrode is not None:
        experiment.add_phase(FrequencyStimPhaseV3(
            stim_command=([0], float(args.amplitude_mv), int(args.phase_width_us)),
            stim_freq=float(args.stim_hz),
            duration=float(args.stim_seconds),
            tag="replay_stim",
        ))
    print(f"Replay source: {source}")
    print(f"Experiment output: {experiment.save_dir}")
    success = experiment.run(validate=True)
    return 0 if success else 1


def build_parser():
    parser = argparse.ArgumentParser(description="Replay Maxwell H5 data without hardware")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List compatible recordings")
    list_parser.add_argument("--data-dir")
    list_parser.set_defaults(func=list_command)

    doctor_parser = subparsers.add_parser("doctor", help="Verify replay environment")
    doctor_parser.add_argument("--require-env")
    doctor_parser.add_argument("--h5")
    doctor_parser.add_argument("--recording")
    doctor_parser.add_argument("--sample", action="store_true")
    doctor_parser.add_argument("--data-dir")
    doctor_parser.set_defaults(func=doctor_command)

    run_parser = subparsers.add_parser("run", help="Run V3 replay experiment")
    source_group = run_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--h5")
    source_group.add_argument("--recording")
    source_group.add_argument("--sample", action="store_true")
    run_parser.add_argument("--data-dir")
    run_parser.add_argument("--output-dir")
    run_parser.add_argument("--name")
    run_parser.add_argument("--record-seconds", type=float, default=0.5)
    run_parser.add_argument(
        "--speed",
        type=_parse_speed,
        default=1.0,
        metavar="RATE|max",
        help="Wall pacing multiplier; 0 or 'max' disables sleeping",
    )
    run_parser.add_argument("--stim-seconds", type=float, default=0.0)
    run_parser.add_argument("--stim-electrode", type=int)
    run_parser.add_argument("--stim-hz", type=float, default=5.0)
    run_parser.add_argument("--amplitude-mv", type=float, default=250.0)
    run_parser.add_argument("--phase-width-us", type=int, default=100)
    run_parser.add_argument("--verbose", action="store_true")
    run_parser.add_argument(
        "--no-write-output",
        action="store_true",
        help="Replay without creating output H5 files (fastest stream-only mode)",
    )
    run_parser.set_defaults(func=run_command)
    return parser


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
