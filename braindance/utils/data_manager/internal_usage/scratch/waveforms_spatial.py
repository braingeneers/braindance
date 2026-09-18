"""
Explore an RT-Sort pickle: load neuron locations, footprints, and inventory
everything else that lives on the RTSort object.

Loads via the new RecordingNew wrapper in
  proj/Proj_Ari/Closed_loop_data/project_rec_rework.py
which resolves   {base}/{proj}/{chip}/{exp}/RT_sort.pkl   on S3 and caches locally.

WHAT IS / ISN'T IN THE PKL
--------------------------
The RT_sort.pkl is a `braindance.core.spikesorter.rt_sort.RTSort` object. It does
NOT store literal voltage waveforms. Per neuron ("sequence") it stores a spatial
*footprint*:
  - seq_locs            (n_neurons, 2)   root-electrode XY location  <- the "locations"
  - seqs_amps           (n_neurons, n_comp_elecs)  per-elec amplitude median  <- template
  - seqs_latencies      (n_neurons, n_comp_elecs)  per-elec latency
  - comp_elecs          (n_comp_elecs,)  electrode index for each footprint column
  - seq_comp_elecs      list, per-neuron electrode ids that define its footprint
  - seq_spike_trains    list, per-neuron spike times
  - _seq_root_elecs     list, per-neuron root electrode index
True per-spike voltage waveforms must be cut from the {rec}.raw.h5 (not done here).
"""

import sys
import importlib
from pathlib import Path

import numpy as np

# Make RecordingNew importable
REWORK_DIR = (
    Path(__file__).resolve().parents[5]
    / "proj" / "Proj_Ari" / "Closed_loop_data"
)
sys.path.insert(0, str(REWORK_DIR))


# --- numpy 2.x -> 1.x pickle compat -------------------------------------------
# The RT_sort.pkl on S3 was written with numpy>=2.0, which renamed the internal
# `numpy.core` package to `numpy._core`. This env has numpy 1.26, so unpickling
# blows up with "No module named 'numpy._core...'". Redirect those imports.
class _NumpyCoreCompat:
    def find_module(self, name, path=None):
        return self if name.startswith("numpy._core") else None

    def load_module(self, name):
        if name in sys.modules:
            return sys.modules[name]
        mod = importlib.import_module(name.replace("numpy._core", "numpy.core"))
        sys.modules[name] = mod
        return mod


if not any(isinstance(f, _NumpyCoreCompat) for f in sys.meta_path):
    sys.meta_path.append(_NumpyCoreCompat())


def robust_pickle_load(path):
    """Unpickle an RTSort obj saved on GPU/numpy2 from a CPU/numpy1 box."""
    import io
    import pickle
    import torch

    # remap any CUDA tensor storages to CPU during unpickle
    orig = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(
        io.BytesIO(b), map_location="cpu", weights_only=False
    )
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    finally:
        torch.storage._load_from_bytes = orig


def describe(name, obj, indent="  "):
    """Print a compact type/shape summary for one attribute."""
    import torch

    if isinstance(obj, torch.Tensor):
        print(f"{indent}{name:28s} torch.Tensor {tuple(obj.shape)} {obj.dtype} dev={obj.device}")
    elif isinstance(obj, np.ndarray):
        print(f"{indent}{name:28s} np.ndarray  {obj.shape} {obj.dtype}")
    elif isinstance(obj, (list, tuple)):
        n = len(obj)
        inner = type(obj[0]).__name__ if n else ""
        ln = ""
        if n and hasattr(obj[0], "__len__"):
            try:
                ln = f" inner_len[0]={len(obj[0])}"
            except TypeError:
                ln = ""
        print(f"{indent}{name:28s} {type(obj).__name__}(n={n}) of {inner}{ln}")
    elif isinstance(obj, (int, float, str, bool, np.integer, np.floating)):
        print(f"{indent}{name:28s} {type(obj).__name__} = {obj}")
    else:
        print(f"{indent}{name:28s} {type(obj).__name__}")


def main(
    base_path="s3://braingeneers/braindance/",
    proj="2026-04-10-closedloop",
    chip="25245hs5",
    experiment="closed_loop_plasticity",
    rec="002_closed_loop",
    save_footprints=False,
):
    from project_rec_rework import RecordingNew

    rec_obj = RecordingNew.from_identifiers(
        proj=proj,
        chip=chip,
        experiment=experiment,
        rec=rec,
        base_path=base_path,
        auto_upload=False,
    )

    resolver = rec_obj._load_manager.path_resolver
    rts_path = Path(resolver.rt_sort_path)
    print(f"\n=== {proj}/{chip}/{experiment}/{rec} ===")
    print(f"resolved RT_sort path: {rts_path}")

    # ---- make sure the pkl is present locally (download from S3 if needed) ----
    # rec.rt_sort goes through _load_pickle_safe, which swallows the numpy2/cuda
    # unpickle error and returns None -- but it DOES download the file first. So
    # trigger the download, then load the local file ourselves with the shim.
    if not rts_path.exists():
        s3_rt = resolver.construct_s3_path("rt_sort_obj")
        print(f"downloading from S3: {s3_rt}")
        rec_obj._load_manager._download_from_s3(s3_rt, rts_path)

    rt = robust_pickle_load(rts_path)

    print(f"\nRTSort object: {type(rt).__module__}.{type(rt).__name__}")
    if isinstance(rt, dict):
        print("  (loaded as a raw dict, not an RTSort instance)")
        for k, v in rt.items():
            describe(k, v)
        return rt

    # ---- full attribute inventory ----
    print("\n--- all public attributes ---")
    for name in sorted(vars(rt).keys()):
        if name.startswith("__"):
            continue
        describe(name, getattr(rt, name))

    # ---- the bits you care about ----
    n_neurons = getattr(rt, "num_seqs", None) or len(getattr(rt, "seq_spike_trains", []))
    print(f"\nn_neurons (sequences) = {n_neurons}")

    locs = np.asarray(getattr(rt, "seq_locs"))
    print(f"\nLOCATIONS  rt.seq_locs -> {locs.shape}  (root-electrode XY, microns)")
    print(locs[:5])

    def to_np(x):
        import torch
        return x.detach().cpu().float().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    amps = to_np(rt.seqs_amps) if hasattr(rt, "seqs_amps") else None
    lats = to_np(rt.seqs_latencies) if hasattr(rt, "seqs_latencies") else None
    comp_elecs = to_np(rt.comp_elecs).ravel() if hasattr(rt, "comp_elecs") else None

    if amps is not None:
        print(f"\nFOOTPRINT (amplitude template)  rt.seqs_amps -> {amps.shape}")
        print("  rows = neurons, cols = comp electrodes (rt.comp_elecs gives elec ids)")
        print(f"  amp range: [{np.nanmin(amps):.3f}, {np.nanmax(amps):.3f}]")
    if lats is not None:
        print(f"\nLATENCY footprint  rt.seqs_latencies -> {lats.shape}")
    if comp_elecs is not None:
        print(f"\ncomp_elecs (footprint columns -> electrode ids) -> {comp_elecs.shape}")

    # per-neuron spike counts
    trains = getattr(rt, "seq_spike_trains", None)
    if trains is not None:
        counts = np.array([len(t) for t in trains])
        print(f"\nspike counts per neuron: min={counts.min()} med={int(np.median(counts))} max={counts.max()}")

    # ---- also load locations from the standardized spike_info.json path ----
    print("\n--- rec.spike_locations (spike_info.json, extracted from RT-Sort) ---")
    try:
        sl = rec_obj.spike_locations
        if sl is not None:
            sl = np.asarray(sl)
            print(f"  spike_locations -> {sl.shape}")
            print(sl[:5])
        else:
            print("  None (spike_info.json not present / not extracted)")
    except Exception as e:
        print(f"  failed: {e}")

    if save_footprints and amps is not None:
        out = rec_obj.output_dir / "rt_sort_footprints.npz"
        np.savez_compressed(
            out,
            seq_locs=locs,
            seqs_amps=amps,
            seqs_latencies=lats if lats is not None else np.array([]),
            comp_elecs=comp_elecs if comp_elecs is not None else np.array([]),
        )
        print(f"\nsaved footprints -> {out}")

    return rt


def extract_footprint_waveforms(
    base_path="s3://braingeneers/braindance/",
    proj="2026-04-10-closedloop",
    chip="25245hs5",
    experiment="closed_loop_plasticity",
    rec="001_rec",                 # scaled_traces.npy is the 001 baseline
    n_before=10,
    n_after=10,
    save=True,
    upload_s3=False,
):
    """
    Rebuild per-neuron footprint WAVEFORMS by cutting windows out of
    rt_sort_inter/scaled_traces.npy at each neuron's spike times.

    For neuron i with comp channels C_i and spike frames F (from the full
    001 sort, spike_data.pkl), the footprint is:
        mean_{spikes} scaled_traces[C_i, f-n_before : f+n_after+1]   -> (|C_i|, 21)

    Returns a dict keyed by neuron index:
        {i: {"channels", "xy", "waveforms" (|C_i|,21), "n_spikes",
             "trough_amp", "trough_latency_frame", "root_channel", "root_xy"}}
    Also validates measured trough amp vs the stored rt.seqs_amps.
    """
    import subprocess
    import pandas as pd
    from braindance import get_data_dir

    win = n_before + n_after + 1

    rec_obj = _build_rec(base_path, proj, chip, experiment, rec)
    resolver = rec_obj._load_manager.path_resolver
    samp_freq = 20  # samples/ms (from RT_sort params)

    # ---- 1. RTSort object: comp channels + stored amp/latency for validation ----
    rts_path = Path(resolver.rt_sort_path)
    if not rts_path.exists():
        rec_obj._load_manager._download_from_s3(
            resolver.construct_s3_path("rt_sort_obj"), rts_path
        )
    rt = robust_pickle_load(rts_path)
    seq_comp_elecs = rt.seq_comp_elecs            # per-neuron channel idx lists
    root_chans = list(rt._seq_root_elecs)
    seq_locs = np.asarray(rt.seq_locs)
    comp_union = np.asarray(rt.comp_elecs).ravel().tolist()
    stored_amps = rt.seqs_amps.float().cpu().numpy()  # (99, n_union)

    # ---- 2. mapping channel -> (x,y) ----
    map_path = Path(resolver.mapping_path)
    if not map_path.exists():
        rec_obj._load_manager._download_from_s3(
            resolver.construct_s3_path("mapping"), map_path
        )
    mp = pd.read_csv(map_path)
    chan_xy = {int(r.channel): (float(r.x), float(r.y)) for r in mp.itertuples()}

    # ---- 3. spike times: full 001 sort (more spikes -> cleaner average) ----
    sd_path = Path(resolver.spikes_path)          # 001_rec/<rec>_spike_data.pkl
    if not sd_path.exists():
        rec_obj._load_manager._download_from_s3(
            resolver.construct_s3_path("spikes"), sd_path
        )
    spike_data = robust_pickle_load(sd_path)
    trains = spike_data.train                     # list of (ms) arrays, index-aligned to neurons
    if len(trains) != len(seq_comp_elecs):
        print(f"  [warn] {len(trains)} spike-trains vs {len(seq_comp_elecs)} neurons; "
              "falling back to rt.seq_spike_trains")
        trains = rt.seq_spike_trains

    # ---- 4. download scaled_traces.npy (2.4 GB) and mmap it ----
    st_local = get_data_dir() / proj / chip / experiment / "rt_sort_inter" / "scaled_traces.npy"
    st_local.parent.mkdir(parents=True, exist_ok=True)
    if not st_local.exists():
        s3_st = f"{base_path}{proj}/{chip}/{experiment}/rt_sort_inter/scaled_traces.npy"
        print(f"  downloading scaled_traces (2.4 GB): {s3_st}")
        subprocess.run(
            ["aws", "--endpoint", "https://s3-west.nrp-nautilus.io",
             "s3", "cp", s3_st, str(st_local)],
            check=True,
        )
    traces = np.load(st_local, mmap_mode="r")     # (n_elecs, n_frames) float16
    n_elecs, n_frames = traces.shape
    offsets = np.arange(-n_before, n_after + 1)
    print(f"  scaled_traces {traces.shape} {traces.dtype}; window={win} frames (~{win/samp_freq:.2f} ms)")

    # ---- 5. cut + average per neuron ----
    footprints = {}
    measured, stored = [], []   # for validation scatter
    for i, chans in enumerate(seq_comp_elecs):
        chans = [int(c) for c in chans]
        t_ms = np.asarray(trains[i], dtype=float)
        F = np.round(t_ms * samp_freq).astype(np.int64)
        F = F[(F - n_before >= 0) & (F + n_after < n_frames)]
        if len(F) == 0 or len(chans) == 0:
            continue
        sub = np.asarray(traces[chans], dtype=np.float32)        # (|C|, n_frames)
        baseline = np.median(sub, axis=1)                        # per-channel DC level
        widx = F[:, None] + offsets                              # (n_spk, 21)
        cut = sub[:, widx]                                       # (|C|, n_spk, 21)
        wf = cut.mean(axis=1) - baseline[:, None]                # (|C|, 21), baseline-subtracted

        trough_idx = wf.argmin(axis=1)                           # per-channel trough frame
        trough_amp = -wf.min(axis=1)                             # depth below baseline (positive)
        xy = np.array([chan_xy.get(c, (np.nan, np.nan)) for c in chans])

        footprints[i] = {
            "channels": np.array(chans),
            "xy": xy,
            "waveforms": wf,
            "n_spikes": int(len(F)),
            "trough_amp": trough_amp,
            "trough_latency_frame": trough_idx,
            "root_channel": int(root_chans[i]) if i < len(root_chans) else -1,
            "root_xy": tuple(seq_locs[i]),
        }

        # validation: measured trough amp vs stored seqs_amps for the same channels
        for c, a in zip(chans, trough_amp):
            if c in comp_union:
                s = stored_amps[i, comp_union.index(c)]
                if s != 1.0:        # 1.0 == off-footprint sentinel
                    measured.append(a)
                    stored.append(s)

    print(f"\n  built footprint waveforms for {len(footprints)} neurons")
    if measured:
        r = np.corrcoef(measured, stored)[0, 1]
        print(f"  validation: measured trough amp vs stored seqs_amps  Pearson r = {r:.3f}  (n={len(measured)})")

    # quick look at the strongest neuron
    if footprints:
        bi = max(footprints, key=lambda k: footprints[k]["waveforms"].min() * -1
                 if footprints[k]["n_spikes"] else 0)
        fp = footprints[bi]
        print(f"\n  example neuron {bi}: {len(fp['channels'])} chans, {fp['n_spikes']} spikes, "
              f"root chan {fp['root_channel']} @ {fp['root_xy']}")
        ri = list(fp["channels"]).index(fp["root_channel"]) if fp["root_channel"] in fp["channels"] else int(fp["trough_amp"].argmax())
        print(f"  root-elec mean waveform (21 samples): "
              f"{np.array2string(fp['waveforms'][ri], precision=1, max_line_width=200)}")

    if save:
        out = resolver.output_dir / "rt_sort_footprint_waveforms.npz"
        # object arrays since per-neuron channel counts differ
        np.savez_compressed(
            out,
            neuron_idx=np.array(list(footprints.keys())),
            channels=np.array([footprints[i]["channels"] for i in footprints], dtype=object),
            xy=np.array([footprints[i]["xy"] for i in footprints], dtype=object),
            waveforms=np.array([footprints[i]["waveforms"] for i in footprints], dtype=object),
            n_spikes=np.array([footprints[i]["n_spikes"] for i in footprints]),
            root_channel=np.array([footprints[i]["root_channel"] for i in footprints]),
            root_xy=np.array([footprints[i]["root_xy"] for i in footprints]),
            n_before=n_before, n_after=n_after, samp_freq=samp_freq,
        )
        print(f"\n  saved -> {out}")
        if upload_s3:
            s3_out = f"s3://braingeneersdev/hschweig/{proj}/{chip}/{experiment}/rt_sort_footprint_waveforms.npz"
            subprocess.run(["aws", "--endpoint", "https://s3-west.nrp-nautilus.io",
                            "s3", "cp", str(out), s3_out], check=True)
            print(f"  uploaded -> {s3_out}")

    return footprints


def _build_rec(base_path, proj, chip, experiment, rec):
    from project_rec_rework import RecordingNew
    return RecordingNew.from_identifiers(
        proj=proj, chip=chip, experiment=experiment, rec=rec,
        base_path=base_path, auto_upload=False,
    )


def plot_footprint_waveforms(
    footprints=None,
    npz_path=None,
    n_examples=6,
    sort_by="amp",          # "amp" -> strongest neurons first, else by neuron index
    elec_pitch=17.5,        # MaxWell microns between electrodes; sets waveform glyph scale
    save_path=None,
):
    """
    Plot example neuron footprints as SPATIAL waveforms: each comp-channel's
    mean waveform is drawn as a little trace centered on that electrode's (x, y).
    The root electrode is highlighted. One subplot per example neuron.

    Pass either `footprints` (dict from extract_footprint_waveforms) or
    `npz_path` to the saved rt_sort_footprint_waveforms.npz.
    """
    import matplotlib.pyplot as plt

    # ---- load from npz if a dict wasn't handed in ----
    if footprints is None:
        if npz_path is None:
            raise ValueError("pass either footprints=... or npz_path=...")
        z = np.load(npz_path, allow_pickle=True)
        footprints = {}
        for k, idx in enumerate(z["neuron_idx"]):
            footprints[int(idx)] = {
                "channels": z["channels"][k],
                "xy": z["xy"][k],
                "waveforms": z["waveforms"][k],
                "n_spikes": int(z["n_spikes"][k]),
                "root_channel": int(z["root_channel"][k]),
                "root_xy": tuple(z["root_xy"][k]),
            }

    if not footprints:
        print("  no footprints to plot")
        return

    # ---- pick the examples ----
    keys = list(footprints.keys())
    if sort_by == "amp":
        keys.sort(key=lambda k: -float(np.nanmax(-footprints[k]["waveforms"])))  # deepest trough first
    keys = keys[:n_examples]

    ncol = min(3, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, k in zip(axes, keys):
        fp = footprints[k]
        xy = np.asarray(fp["xy"], dtype=float)         # (|C|, 2)
        wf = np.asarray(fp["waveforms"], dtype=float)  # (|C|, win)
        win = wf.shape[1]
        # x-extent of each glyph ~ 0.8 of electrode pitch; y-scale so the biggest
        # trough on this neuron spans ~1.2 pitch (readable but not overlapping)
        xspan = 0.8 * elec_pitch
        peak = np.nanmax(np.abs(wf)) or 1.0
        yscale = 1.2 * elec_pitch / peak
        t = np.linspace(-xspan / 2, xspan / 2, win)

        root_c = fp["root_channel"]
        for c, (x, y), w in zip(fp["channels"], xy, wf):
            if not np.isfinite(x):
                continue
            is_root = (c == root_c)
            ax.plot(
                x + t, y + w * yscale,
                color="crimson" if is_root else "0.35",
                lw=1.6 if is_root else 0.8,
                zorder=3 if is_root else 2,
            )
        # mark electrode positions
        good = np.isfinite(xy[:, 0])
        ax.scatter(xy[good, 0], xy[good, 1], s=6, color="0.7", zorder=1)
        rx, ry = fp["root_xy"]
        ax.scatter([rx], [ry], s=40, facecolor="none", edgecolor="crimson", lw=1.5, zorder=4)

        ax.set_title(f"neuron {k}  ({len(fp['channels'])} ch, {fp['n_spikes']} spk)\n"
                     f"trough {peak:.1f}  root ch {root_c}", fontsize=9)
        ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
        ax.set_aspect("equal", adjustable="datalim")

    for ax in axes[len(keys):]:
        ax.set_visible(False)

    fig.suptitle("RT-Sort footprint waveforms (red = root electrode)", y=1.0)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved figure -> {save_path}")
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    import sys as _sys
    if "--plot" in _sys.argv:
        # build (or rebuild) waveforms, then show a few examples on screen
        fps = extract_footprint_waveforms(save=True)
        plot_footprint_waveforms(fps, n_examples=6, save_path=None)
    elif "--waveforms" in _sys.argv:
        extract_footprint_waveforms()
    else:
        main()
