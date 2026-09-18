"""Download tutorial data and validate it through the normal Recording API.

Prefer ``python -m braindance.examples.get_tutorial_data --validate``; this older
module remains available for compatibility. Use a fresh
``--cache-dir`` to exercise the network; ``--offline`` verifies a cached bundle.
No plots, hardware, or private recording disk are required.
"""


def main(name="closed-loop-small", version=None, cache_dir=None, offline=False, update=False,
         manifest_path=None):
    import hashlib
    import json

    import numpy as np

    from braindance.tutorial import get_test_data
    from braindance.utils.data_manager import load_catalog, bin_spike_data_vectorized

    folder = get_test_data(name, version=version, cache_dir=cache_dir,
                           offline=offline, update=update, manifest_path=manifest_path)
    reference = json.loads((folder / "reference.json").read_text())
    catalog = load_catalog(folder / "catalog.csv", base_path=folder)
    expected_recordings = reference["recordings"]
    if len(catalog) != len(expected_recordings) or not expected_recordings:
        raise AssertionError("Catalog does not match the reference recording count")
    reports = []
    for expected in expected_recordings:
        selected = catalog.filter(**{key: expected[key] for key in ("proj", "chip", "experiment")})
        if len(selected) != 1:
            raise AssertionError("Reference recording does not uniquely resolve in catalog")
        rec = selected[0]
        try:
            spikes = rec.spikes
            trains = [np.asarray(train, dtype=np.float64) for train in spikes.train]
            duration_ms = float(spikes.length)
            if not np.isfinite(duration_ms) or duration_ms <= 0:
                raise AssertionError("Recording duration must be finite and positive")
            for train in trains:
                if (train.ndim != 1 or not np.isfinite(train).all()
                        or np.any(np.diff(train) < 0) or np.any(train < 0)
                        or np.any(train > duration_ms)):
                    raise AssertionError("Spike times must be sorted milliseconds within the recording")
            counts, centers = bin_spike_data_vectorized(
                trains, bin_size_ms=expected["bin_ms"],
                time_range=(0, duration_ms), verbose=False,
            )
            per_neuron = [len(train) for train in trains]
            within_window = [int(np.count_nonzero(train < duration_ms)) for train in trains]
            np.testing.assert_array_equal(counts.sum(axis=0), within_window)
            if counts.shape != (len(centers), len(trains)):
                raise AssertionError("Binning lost the time-by-neuron orientation")
            stim_log = rec.stim_log
            mapping = rec.mapping
            # Recording's Mapping wrapper exposes routed channels/electrodes.
            mapping_channels = mapping.channels
            mapping_electrodes = mapping.electrodes
            arrays = {
                "spikes_sha256": np.concatenate(trains).astype("<f8"),
                "binned_counts_sha256": counts.astype("<i8"),
                "stim_times_sha256": stim_log["time"].to_numpy(dtype="<f8"),
                "mapping_channels_sha256": np.asarray(mapping_channels, dtype="<i8"),
                "mapping_electrodes_sha256": np.asarray(mapping_electrodes, dtype="<i8"),
            }
            observed = {
                "duration_ms": duration_ms,
                "neuron_count": len(trains),
                "spike_count": sum(per_neuron),
                "per_neuron_spike_counts": per_neuron,
                "stim_row_count": len(stim_log),
                "mapping_rows": len(mapping_channels),
            }
            observed.update({key: hashlib.sha256(value.tobytes(order="C")).hexdigest()
                             for key, value in arrays.items()})
            for key, value in observed.items():
                if value != expected[key]:
                    raise AssertionError(f"{rec.identifier}: {key} differs from the frozen source reference")
            report = {"identifier": rec.identifier, "duration_ms": duration_ms,
                      "neurons": len(trains), "spikes": sum(per_neuron),
                      "stimulations": len(stim_log), "mapping_rows": len(mapping_channels),
                      "bin_shape": list(counts.shape), "status": "passed"}
            reports.append(report)
            print(json.dumps(report, indent=2))
        finally:
            rec.clear_cache()
    return {"folder": str(folder), "recordings": reports, "status": "passed"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="closed-loop-small")
    parser.add_argument("--version")
    parser.add_argument("--cache-dir")
    parser.add_argument("--manifest-path")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
