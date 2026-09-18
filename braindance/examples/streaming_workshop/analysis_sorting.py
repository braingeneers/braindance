"""Optional RT-Sort adapter for the workshop's background analysis worker.

Input recordings are never modified. Every invocation gets its own intermediates,
model and portable spike tables; no pickle is needed to read the analysis output.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import uuid


PARAMETERS = {
    "device": {"default": "cuda", "choices": ["cuda", "cpu"]},
    "baseline_start_ms": {"default": 0.0, "min": 0},
    "baseline_duration_ms": {"default": 60000.0, "min": 1},
    "num_processes": {"default": 1, "min": 1, "max": 32},
    "stringent_thresh": {"default": 0.275, "min": 0.001, "max": 1},
    "loose_thresh": {"default": 0.1, "min": 0.001, "max": 1},
    "min_activity_hz": {"default": 0.05, "min": 0},
    "min_seq_spikes_n": {"default": 10, "min": 1},
    "min_seq_spikes_hz": {"default": 0.05, "min": 0},
}


def sorting_capabilities():
    """Cheap dependency probe; actual imports/device checks happen in the worker."""
    dependencies = ["numpy", "scipy", "sklearn", "torch", "spikeinterface",
                    "diptest", "pynvml", "h5py", "threadpoolctl", "tqdm"]
    missing = [name for name in dependencies if importlib.util.find_spec(name) is None]
    from braindance import get_rt_sort_path
    model_path = Path(get_rt_sort_path())
    model_available = all((model_path / name).is_file()
                          for name in ("init_dict.json", "state_dict.pt"))
    reasons = (["Missing dependencies: " + ", ".join(missing)] if missing else [])
    if not model_available:
        reasons.append("RT-Sort detection model is unavailable: " + str(model_path))
    return {"available": not reasons, "missing_dependencies": missing,
            "reason": "; ".join(reasons), "model_path": str(model_path),
            "formats": [".h5", ".nwb"], "parameters": PARAMETERS,
            "note": "Requires raw voltage recordings with matching channel routing and sampling rate. "
                    "GPU availability is checked at launch. Progress reports processing stages."}


def validate_params(params=None):
    params = dict(params or {})
    unknown = set(params) - set(PARAMETERS)
    if unknown:
        raise ValueError("Unknown sorting parameters: " + ", ".join(sorted(unknown)))
    result = {}
    for name, spec in PARAMETERS.items():
        value = params.get(name, spec["default"])
        if "choices" in spec:
            if value not in spec["choices"]:
                raise ValueError(f"{name} must be one of {spec['choices']}")
        else:
            if isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            value = float(value)
            if not math.isfinite(value) or value < spec["min"] or value > spec.get("max", math.inf):
                raise ValueError(f"Invalid {name}: {value}")
            if isinstance(spec["default"], int):
                if not value.is_integer():
                    raise ValueError(f"{name} must be an integer")
                value = int(value)
        result[name] = value
    if result["loose_thresh"] > result["stringent_thresh"]:
        raise ValueError("Loose threshold must not exceed stringent threshold")
    return result


def run_sorting(baseline_path, recording_paths, output_dir, params=None, progress=None):
    """Initialize on baseline, sort targets, and return JSON-safe artifact metadata.

    ``progress(fraction, message)`` reports real stage boundaries (not an ETA).
    Call from a worker, since RT-Sort's sequence discovery is a blocking operation.
    Times in saved NPZs are milliseconds relative to each complete target recording.
    """
    params = validate_params(params)
    baseline_path = Path(baseline_path).expanduser().resolve()
    recording_paths = list(dict.fromkeys(Path(p).expanduser().resolve() for p in recording_paths))
    if not recording_paths:
        raise ValueError("Select at least one recording to sort")
    for path in [baseline_path, *recording_paths]:
        if not path.is_file():
            raise ValueError(f"Recording does not exist: {path}")
        if path.suffix.lower() not in (".h5", ".nwb"):
            raise ValueError(f"RT-Sort requires a raw Maxwell .h5 or .nwb recording: {path.name}")
    capabilities = sorting_capabilities()
    if not capabilities["available"]:
        raise RuntimeError(capabilities["reason"])
    import numpy as np
    import torch
    from braindance.core.spikesorter.rt_sort import detect_sequences, load_recording, save_traces

    if params["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Select CPU or run on a machine with a CUDA GPU.")
    report = progress or (lambda fraction, message: None)
    report(0.02, "Opening baseline and checking target channel routing")
    baseline = load_recording(baseline_path)
    targets = [load_recording(path) for path in recording_paths]
    channel_ids = np.asarray(baseline.get_channel_ids())
    locations = np.asarray(baseline.get_channel_locations())
    frequency = float(baseline.get_sampling_frequency())
    for path, target in zip(recording_paths, targets):
        if (float(target.get_sampling_frequency()) != frequency
                or not np.array_equal(channel_ids, target.get_channel_ids())
                or not np.array_equal(locations, target.get_channel_locations(), equal_nan=True)):
            raise ValueError(f"{path.name}: channel routing or sampling rate differs from baseline; "
                             "choose a baseline recorded with the same electrode configuration")
    start = params["baseline_start_ms"]
    end = min(start + params["baseline_duration_ms"], baseline.get_total_duration() * 1000)
    if end <= start:
        raise ValueError("Baseline window starts after the recording ends")
    output = Path(output_dir).expanduser().resolve() / ("rt_sort_" + uuid.uuid4().hex[:12])
    output.mkdir(parents=True)
    detector_params = {key: value for key, value in params.items() if not key.startswith("baseline_")}
    report(0.08, "Discovering units in baseline (this stage may take several minutes)")
    sorter = detect_sequences(baseline, output / "baseline", capabilities["model_path"],
                              recording_window_ms=(start, end), return_spikes=False,
                              delete_inter=False, verbose=True, **detector_params)
    if sorter is None or sorter.num_seqs == 0:
        raise RuntimeError("RT-Sort found no units in the baseline. Try a longer baseline or adjust thresholds.")
    model_path = output / "sorter.pickle"
    sorter.save(model_path)
    result = {"baseline_path": str(baseline_path), "baseline_window_ms": [start, end],
              "params": params, "sorter_path": str(model_path), "unit_count": int(sorter.num_seqs),
              "recordings": [], "output_dir": str(output)}
    for index, (path, recording) in enumerate(zip(recording_paths, targets)):
        report(0.45 + 0.5 * index / len(targets), f"Sorting {index + 1}/{len(targets)}: {path.name}")
        traces_path = save_traces(recording, output / f"target_{index}",
                                  num_processes=params["num_processes"], verbose=True)
        trains = sorter.sort_offline(traces_path, reset=True, verbose=True)
        duration = float(recording.get_total_duration()) * 1000
        # Retain empty units as a separate list; flat tables otherwise lose them.
        times = np.concatenate([np.asarray(train, dtype=float) for train in trains])
        labels = np.repeat(np.arange(len(trains), dtype=np.int64), [len(train) for train in trains])
        if not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(times > duration):
            raise RuntimeError(f"RT-Sort returned invalid spike times for {path.name}")
        order = np.argsort(times, kind="stable")
        artifact = output / f"{index:03d}_{path.stem}_spikes.npz"
        np.savez_compressed(artifact, times_ms=times[order], unit_ids=labels[order],
                            all_unit_ids=np.arange(len(trains)), duration_ms=duration,
                            sampling_frequency_hz=frequency,
                            source_path=str(path), source="rt-sort")
        result["recordings"].append({"recording_path": str(path), "spikes_path": str(artifact),
                                      "unit_count": len(trains), "spike_count": int(len(times)),
                                      "duration_ms": duration})
        (output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report(1.0, f"Sorted {len(targets)} recording(s); {sorter.num_seqs} units available for analysis")
    return result
