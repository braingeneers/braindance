"""Load an already-downloaded experiment, inspect its data, and make simple plots.

Run: python -m braindance.examples.load_test_data
Use --cache-dir if you passed one to get_tutorial_data; --no-show saves plots only.
Requires the analysis dependencies. This example never downloads data or runs phases.
"""


def main(cache_dir=None, window_seconds=10.0, bin_ms=100.0, show=True):
    import matplotlib.pyplot as plt
    import numpy as np

    from braindance.config import get_output_dir
    from braindance.core.phases_v3.experiment_v3 import Experiment
    from braindance.tutorial import get_test_data
    from braindance.utils.data_manager import (
        Styler, bin_spike_data_vectorized, load_catalog,
    )

    if window_seconds <= 0 or bin_ms <= 0:
        raise ValueError("window_seconds and bin_ms must be positive")

    # 1. Locate the verified local tutorial and load its catalog of recordings.
    # For your own data, use catalog = load_catalog() (configured data paths).
    folder = get_test_data(cache_dir=cache_dir, offline=True)
    catalog = load_catalog(folder / "catalog.csv", base_path=folder)
    recordings = catalog.filter(
        proj="2026-04-10-closedloop", chip="25245hs5",
        experiment="closed_loop_plasticity/002_closed_loop",
    )
    rec = recordings[0]
    print(f"Recording: {rec.identifier}")
    print("Catalog metadata:", rec.metadata)

    try:
        # 2. Recording properties load data lazily on first access.
        spikes = rec.spikes                 # SpikeData object
        trains = spikes.train               # one array per neuron; times in ms
        duration_ms = float(spikes.length)
        stim_log = rec.stim_log              # pandas DataFrame
        mapping = rec.mapping               # routed channel/electrode mapping
        results = rec.results               # lazy, persistent recording DataContext
        print(f"{len(trains)} neurons, {duration_ms / 1000:.2f} seconds")
        print("Neuron 0 spike times (ms):", trains[0][:10])
        print("Stimulation log:\n", stim_log.head())
        print("Channels:", mapping.channels[:10])
        print("Physical electrodes:", mapping.electrodes[:10])
        print("Recording result keys:", list(results.keys()))
        # Access saved results by key, e.g. results["sorted_spikedata"].
        # Assigning results["my_analysis"] = value persists it; here we only read.

        # 3. Reopen the experiment and load its saved shared phase DataContext.
        # The catalog above describes a recording *inside* this experiment.
        experiment_dir = folder / "2026-04-10-closedloop" / "25245hs5" / "closed_loop_plasticity"
        exp = Experiment(experiment_dir.name, save_dir=experiment_dir, auto_load_data=False)
        # Choose the phase explicitly: the final phase need not contain the
        # earlier analysis. Select keys to avoid loading the RT-Sort model.
        exp.load_data(experiment_dir / "001_rec" / "results", keys=[
            "connectivity_matrix", "selected_pair", "channels_per_neuron",
        ])
        context = exp.data
        connectivity = context.connectivity_matrix  # numpy array; attribute access
        selected_pair = context.selected_pair
        channels_per_neuron = context.channels_per_neuron
        print("Phase context keys:", list(context.keys()))
        print("Connectivity shape:", connectivity.shape)
        print("Selected neuron pair:", selected_pair)
        print("Channels for baseline neuron 0:", channels_per_neuron[0])
        # This restores saved data, not the original runnable phase schedule.
        # Baseline and closed-loop neuron indices need not identify the same units.

        # 4. Plot a short raster and population rate directly from spike arrays.
        output_dir = get_output_dir() / "load_test_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        end_ms = min(window_seconds * 1000, duration_ms)
        counts, centers_ms = bin_spike_data_vectorized(
            trains, bin_size_ms=bin_ms, time_range=(0, end_ms), verbose=False,
        )  # counts has shape (time bins, neurons)
        styler = Styler(journal="draft")
        fig, axes = styler.create_figure(nrows=2, sharex=True, width_pt=500, height_pt=320)
        for neuron, train in enumerate(trains):
            times = np.asarray(train)
            times = times[(times >= 0) & (times < end_ms)] / 1000
            axes[0].scatter(times, np.full(len(times), neuron), s=2, color="black")
        axes[0].set(ylabel="Neuron", title="Closed-loop recording")
        axes[1].plot(centers_ms / 1000, counts.sum(axis=1) / (bin_ms / 1000))
        axes[1].set(xlabel="Time (s)", ylabel="Population rate (spikes/s)")
        fig.tight_layout()
        fig.savefig(output_dir / "spikes.png", dpi=150)

        # The saved baseline analysis can be plotted just like any numpy array.
        fig_context, ax = styler.create_figure(width_pt=360, height_pt=300)
        im = ax.imshow(connectivity, aspect="auto", interpolation="nearest")
        ax.set(xlabel="Baseline neuron index", ylabel="Baseline neuron index",
               title="Saved connectivity matrix")
        fig_context.colorbar(im, ax=ax, label="Connectivity value")
        fig_context.tight_layout()
        fig_context.savefig(output_dir / "connectivity.png", dpi=150)
        print(f"Plots saved to: {output_dir}")
        if show:
            plt.show()
        plt.close(fig)
        plt.close(fig_context)
        # Useful when importing main() in a notebook to explore interactively.
        return rec, exp
    finally:
        rec.clear_cache()  # properties will reload on access if main() returned rec


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", help="Same cache directory used when downloading")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--bin-ms", type=float, default=100.0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    main(args.cache_dir, args.window_seconds, args.bin_ms, show=not args.no_show)
