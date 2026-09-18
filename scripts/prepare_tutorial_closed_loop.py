"""Prepare the small real-data closed-loop tutorial bundle.

The input files are frozen numeric exports from the closed-loop figure analysis.
This module is import-safe: filesystem writes happen only from :func:`main`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from spikelab import SpikeData


def _sha256_bytes(values: np.ndarray, dtype: str) -> str:
    canonical = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_bundle(
    spikes_source: Path,
    stim_source: Path,
    mapping_source: Path,
    output_dir: Path,
    project="2026-04-10-closedloop",
    chip="25245hs5",
    experiment="closed_loop_plasticity/002_closed_loop",
    duration_ms=60_350.0,
    neuron_count=99,
    bin_ms=100.0,
    phase="during",
    mapping_sample_channels=(134, 148),
) -> Path:
    """Build and return a load_catalog-compatible tutorial bundle directory."""
    spikes_source = Path(spikes_source)
    stim_source = Path(stim_source)
    mapping_source = Path(mapping_source)
    output_dir = Path(output_dir)
    required = {
        "spikes": spikes_source,
        "stim": stim_source,
        "mapping": mapping_source,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen source file(s): " + ", ".join(missing))
    with np.load(required["spikes"], allow_pickle=False) as archive:
        times = np.asarray(archive[f"{phase}_times_ms"], dtype="<f8")
        raw_unit_ids = np.asarray(archive[f"{phase}_unit_ids"])
    if raw_unit_ids.dtype.kind not in "iu":
        raise ValueError("Frozen unit IDs must be integers")
    unit_ids = raw_unit_ids.astype("<i8", copy=False)
    if times.ndim != 1 or unit_ids.ndim != 1 or len(times) != len(unit_ids):
        raise ValueError("Frozen spike arrays have inconsistent values")
    if not np.isfinite(times).all() or np.any(times < 0) or np.any(times >= duration_ms):
        raise ValueError("Spike times must be finite and within [0, duration_ms)")
    if np.any(unit_ids < 0) or np.any(unit_ids >= neuron_count):
        raise ValueError("Unit IDs fall outside the configured neuron range")
    trains = [times[unit_ids == unit].astype(float) for unit in range(neuron_count)]
    if any(np.any(np.diff(train) < 0) for train in trains):
        raise ValueError("Spike times must be sorted within each neuron")

    stim = pd.read_csv(required["stim"])
    mapping = pd.read_csv(required["mapping"])
    if "time" not in stim or not {"channel", "electrode", "x", "y"}.issubset(mapping.columns):
        raise ValueError("Source CSV files are missing required columns")
    stim_times = pd.to_numeric(stim["time"], errors="raise").to_numpy(dtype="<f8")
    channels = pd.to_numeric(mapping["channel"], errors="raise").to_numpy(dtype="<i8")
    electrodes = pd.to_numeric(mapping["electrode"], errors="raise").to_numpy(dtype="<i8")
    if not np.isfinite(stim_times).all():
        raise ValueError("Stimulus times must be finite")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    experiment_base, recording_name = experiment.split("/", maxsplit=1)
    rec_dir = output_dir / project / chip / experiment
    mapping_dir = output_dir / project / chip / experiment_base
    rec_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir.mkdir(parents=True, exist_ok=True)

    spike_data = SpikeData(
        trains,
        N=neuron_count,
        length=duration_ms,
        metadata={
            "dataset": f"{project}/{chip}/{experiment}",
            "source": "offline RT-Sort cache used by the closed-loop figure",
            "time_unit": "ms",
        },
    )
    spike_path = rec_dir / f"{recording_name}_spike_data.pkl"
    with spike_path.open("wb") as handle:
        pickle.dump(spike_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    stim_path = rec_dir / f"{recording_name}_log.csv"
    mapping_path = mapping_dir / "mapping.csv"
    shutil.copyfile(required["stim"], stim_path)
    shutil.copyfile(required["mapping"], mapping_path)

    catalog = pd.DataFrame([{
        "proj": project,
        "chip": chip,
        "experiment": experiment,
        "type": "closed_loop",
        "duration": duration_ms / 1000,
        "n_neurons": neuron_count,
        "data_path": spike_path.relative_to(output_dir).as_posix(),
        "log_path": stim_path.relative_to(output_dir).as_posix(),
        "mapping_path": mapping_path.relative_to(output_dir).as_posix(),
        "results_path": (rec_dir / "results").relative_to(output_dir).as_posix(),
    }])
    catalog.to_csv(output_dir / "catalog.csv", index=False)

    n_bins = int(np.ceil(duration_ms / bin_ms))
    binned = np.zeros((n_bins, neuron_count), dtype="<i8")
    for unit, train in enumerate(trains):
        np.add.at(binned[:, unit], (train // bin_ms).astype(int), 1)
    concatenated = np.concatenate(trains).astype("<f8", copy=False)
    reference = {
        "format_version": 1,
        "recordings": [{
            "proj": project,
            "chip": chip,
            "experiment": experiment,
            "duration_ms": duration_ms,
            "neuron_count": neuron_count,
            "spike_count": int(len(concatenated)),
            "per_neuron_spike_counts": [int(len(train)) for train in trains],
            "spikes_sha256": _sha256_bytes(concatenated, "<f8"),
            "sample_spikes_ms": {
                str(unit): trains[unit][:5].tolist()
                for unit in (9, 11) if unit < neuron_count
            },
            "bin_ms": bin_ms,
            "binned_shape": list(binned.shape),
            "binned_counts_sha256": _sha256_bytes(binned, "<i8"),
            "stim_row_count": int(len(stim)),
            "stim_first_time_s": float(stim_times[0]),
            "stim_last_time_s": float(stim_times[-1]),
            "stim_times_sha256": _sha256_bytes(stim_times, "<f8"),
            "mapping_rows": int(len(mapping)),
            "mapping_channels_sha256": _sha256_bytes(channels, "<i8"),
            "mapping_electrodes_sha256": _sha256_bytes(electrodes, "<i8"),
            "mapping_samples": {
                str(channel): mapping.loc[mapping["channel"] == channel].iloc[0].to_dict()
                for channel in mapping_sample_channels
                if np.any(channels == channel)
            },
        }],
        "provenance": {
            "description": "Frozen real data from the middle phase of the closed-loop figure experiment; no raw voltage or sorter model included.",
            "source_files": {
                path.name: {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
                for path in required.values()
            },
        },
    }
    (output_dir / "reference.json").write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def main(
    spikes_source=None,
    stim_source=None,
    mapping_source=None,
    output_dir=None,
    project="2026-04-10-closedloop",
    chip="25245hs5",
    experiment="closed_loop_plasticity/002_closed_loop",
    duration_ms=60_350.0,
    neuron_count=99,
    bin_ms=100.0,
) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    source_root = repo_root / "proj/braindance_figs/closed_loop_fig"
    spikes_source = Path(spikes_source) if spikes_source else source_root / "narrative_2026_09_05/source_data/offline_phase_spikes.npz"
    stim_source = Path(stim_source) if stim_source else source_root / "timing_reanalysis_2026_09_05/source_data/original_stim_log.csv"
    mapping_source = Path(mapping_source) if mapping_source else source_root / "timing_reanalysis_2026_09_05/source_data/raw_electrode_mapping.csv"
    if output_dir is None:
        from braindance.config import get_output_dir
        output_dir = get_output_dir() / "tutorial_bundles/closed-loop-small"
    return prepare_bundle(
        spikes_source, stim_source, mapping_source, output_dir,
        project=project, chip=chip, experiment=experiment,
        duration_ms=duration_ms, neuron_count=neuron_count, bin_ms=bin_ms,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spikes-source", type=Path)
    parser.add_argument("--stim-source", type=Path)
    parser.add_argument("--mapping-source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--project", default="2026-04-10-closedloop")
    parser.add_argument("--chip", default="25245hs5")
    parser.add_argument("--experiment", default="closed_loop_plasticity/002_closed_loop")
    parser.add_argument("--duration-ms", type=float, default=60_350.0)
    parser.add_argument("--neuron-count", type=int, default=99)
    parser.add_argument("--bin-ms", type=float, default=100.0)
    args = parser.parse_args()
    result = main(**vars(args))
    print(result)


if __name__ == "__main__":
    _cli()
