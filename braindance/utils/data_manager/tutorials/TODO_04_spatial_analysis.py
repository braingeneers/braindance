"""Spatial peak-latency analysis using catalog-managed recording data.

Requires a recording with spike data, stimulation log, electrode mapping and
spike locations. Configure input locations through the data manager. This example
uses statistically validated responses; historical project helpers are not needed.
"""

from braindance.config import get_output_dir
from braindance.utils.data_manager import calculate_latencies, load_catalog


def main(
    proj="25-02-25_busybees",
    chip="25123ic",
    experiment="exp1/exp1_cont_95",
    recompute=False,
    show=True,
):
    catalog = load_catalog().filter(proj=proj, chip=chip, experiment=experiment)
    if len(catalog) != 1:
        raise ValueError(f"Expected one recording for {proj}/{chip}/{experiment}; found {len(catalog)}")
    rec = catalog[0]
    try:
        print(f"Loading spatial data for {rec.identifier}")
        mapping = rec.mapping
        locations = rec.spike_locations
        if mapping is None or locations is None:
            print("Spatial analysis requires electrode mapping and spike locations.")
            return {}
        stim_log = rec.stim_log
        if stim_log is None or len(stim_log) == 0:
            print("No stimulation data available for latency analysis.")
            return {}

        # Cache this tutorial's fixed scientific parameters with the results.
        params = {
            "min_response_ratio": 1.5,
            "max_p_value": 0.0001,
            "baseline_window": (-100, 0),
            "response_window": (0, 100),
            "use_time_mod": True,
        }
        cache_key = "spatial_latency_tutorial"
        cached = rec.results.get(cache_key)
        if not recompute and isinstance(cached, dict) and cached.get("params") == params:
            pairs = cached["data"]
        else:
            print("Calculating stimulus-evoked peak latencies...")
            pairs = calculate_latencies(rec.spikes, stim_log, **params)
            rec.results[cache_key] = {"data": pairs, "params": params}
            rec.save_results(keys=[cache_key])

        # The plotting API expects neuron-indexed results for one electrode.
        results_by_electrode = {}
        for result in pairs.values():
            results_by_electrode.setdefault(result["electrode_id"], {})[result["neuron_idx"]] = result
        if not results_by_electrode:
            print("No responses passed statistical validation.")
            return results_by_electrode

        electrode_id = next(iter(results_by_electrode))
        output_dir = get_output_dir() / "spatial_latency_tutorial" / proj / chip / experiment
        output_dir.mkdir(parents=True, exist_ok=True)
        rec.pl.spatial_latency(
            electrode_id=electrode_id,
            latency_results=results_by_electrode[electrode_id],
            time_window=(0, 100),
            save_path=str(output_dir),
            filename=f"spatial_latency_elec_{electrode_id}",
            show=show,
        )
        print(f"Spatial peak-latency plot saved under {output_dir}")
        return results_by_electrode
    finally:
        rec.clear_cache()


if __name__ == "__main__":
    main()
