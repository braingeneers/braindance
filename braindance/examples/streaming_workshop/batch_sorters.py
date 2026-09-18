"""Batch sorting adapters for uploaded recordings, with portable spike outputs."""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import uuid

from .analysis_sorting import run_sorting, sorting_capabilities


def sorter_choices():
    """Offer RT-Sort and locally installed SpikeInterface sorters."""
    capabilities = sorting_capabilities()
    choices = [dict(id="rt-sort", label="RT-Sort", available=capabilities["available"],
                    reason=capabilities.get("reason", ""))]
    try:
        from spikeinterface.sorters import installed_sorters
        names = installed_sorters()
    except (ImportError, RuntimeError):
        names = []
    for name in sorted(names):
        dependencies = {
            'simple': ('scipy', 'sklearn', 'numba', 'hdbscan'),
            'spykingcircus2': ('scipy', 'sklearn', 'numba', 'hdbscan', 'networkx', 'torch'),
            'tridesclous2': ('scipy', 'sklearn', 'numba', 'hdbscan', 'networkx'),
        }.get(name, ())
        missing = [module for module in dependencies if importlib.util.find_spec(module) is None]
        reason = ('Missing dependencies: ' + ', '.join(missing) +
                  '. Use Install sorter dependencies below.') if missing else ''
        choices.append(dict(id="spikeinterface:" + name, label=name + " (SpikeInterface)",
                            available=not missing, reason=reason))
    return choices


def run_batch_sorting(baseline_path, recording_paths, output_dir, sorter="rt-sort",
                      params=None, progress=None):
    """Fit RT-Sort on the baseline, or sort targets independently with SpikeInterface.

    SpikeInterface unit identities belong to each individual recording. Its sorters
    do not use the selected baseline and do not transfer units between recordings.
    """
    paths = list(dict.fromkeys(Path(path).expanduser().resolve() for path in recording_paths))
    if not paths:
        raise ValueError("Select at least one recording to sort")
    if sorter == "rt-sort":
        return run_sorting(baseline_path, paths, output_dir, params=params, progress=progress)
    choices = {item["id"]: item for item in sorter_choices()}
    if sorter not in choices or not choices[sorter]["available"]:
        raise ValueError("Unknown or unavailable sorter: " + str(sorter))

    import numpy as np
    from spikeinterface import extractors
    from spikeinterface.sorters import get_default_sorter_params, run_sorter

    name = sorter.removeprefix("spikeinterface:")
    options = dict(params or {})
    unknown = set(options) - set(get_default_sorter_params(name))
    if unknown:
        raise ValueError("Unknown sorting parameters: " + ", ".join(sorted(unknown)))
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in (".h5", ".hdf5", ".nwb"):
            raise ValueError("Choose a raw Maxwell H5 or NWB recording: " + path.name)
    output = Path(output_dir).expanduser().resolve() / (name + "_" + uuid.uuid4().hex[:12])
    output.mkdir(parents=True)
    report = progress or (lambda fraction, message: None)
    result = dict(sorter=sorter, baseline_path=str(baseline_path or ""), baseline_used=False,
                  unit_identity="independent_per_recording", params=options,
                  recordings=[], output_dir=str(output),
                  note="This sorter runs independently on each target. The selected baseline is not used; "
                       "unit IDs are not matched across recordings.")
    for index, path in enumerate(paths):
        report(index / len(paths), f"Sorting {index + 1}/{len(paths)}: {path.name}")
        recording = (extractors.read_nwb(str(path)) if path.suffix.lower() == ".nwb"
                     else extractors.read_maxwell(str(path)))
        if recording.get_num_segments() != 1:
            raise ValueError(f"{path.name}: choose a recording with one continuous segment")
        frequency = float(recording.get_sampling_frequency())
        samples = recording.get_num_samples()
        if not np.isfinite(frequency) or frequency <= 0 or samples <= 0:
            raise ValueError(f"{path.name}: recording has no samples or an invalid sampling rate")
        sorting = run_sorter(name, recording, folder=output / f"target_{index}",
                            verbose=True, raise_error=True, **options)
        original_ids = list(sorting.get_unit_ids())
        trains = [np.asarray(sorting.get_unit_spike_train(unit, segment_index=0))
                  for unit in original_ids]
        frames = np.concatenate(trains) if trains else np.array([], dtype=np.int64)
        if (not np.all(np.isfinite(frames)) or np.any(frames < 0)
                or np.any(frames >= samples)):
            raise RuntimeError(f"{name} returned invalid spike times for {path.name}")
        times = frames.astype(float) * (1000 / frequency)
        labels = np.repeat(np.arange(len(trains), dtype=np.int64), [len(train) for train in trains])
        order = np.argsort(times, kind="stable")
        artifact = output / f"{index:03d}_{path.stem}_spikes.npz"
        duration = samples * 1000 / frequency
        np.savez_compressed(artifact, times_ms=times[order], unit_ids=labels[order],
                            all_unit_ids=np.arange(len(trains)),
                            original_unit_ids=np.asarray([str(unit) for unit in original_ids], dtype=str),
                            duration_ms=duration, sampling_frequency_hz=frequency,
                            source_path=str(path), source=sorter)
        result["recordings"].append(dict(recording_path=str(path), spikes_path=str(artifact),
                                          unit_count=len(trains), spike_count=len(times), duration_ms=duration))
        (output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report(1.0, f"Sorted {len(paths)} recording(s)")
    return result
