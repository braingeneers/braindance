"""
Per-unit mean spike waveforms — the read side.

Produced by `proj/predictor/waveform_extractor/derive_data/step1_extract_waveforms.py`
and cached one `.npz` per recording under the recording's `results/` prefix. Reach
them through `rec.wf`:

    rec = catalog.filter(chip="23138", experiment="exp1/exp1_cartpole_long_14")[0]
    wf = rec.wf                       # None if this recording has no npz
    wf.waveforms                      # (n_units, K, n_samples) uV
    wf.fs                             # 20000.0  -- a float, not a 0-d array
    wf.t_ms                           # time axis, t=0 at the spike
    wf.summary()                      # DataFrame, one row per unit

`Waveforms` is dict-compatible (`wf["waveforms"]`, `"amp_map" in wf`, `wf.keys()`)
so anything written against the raw npz dict keeps working.

WHY THIS MODULE EXISTS AT ALL: the cache key is easy to get wrong in two ways
that both fail silently, and every consumer that rebuilt it by hand was a fresh
chance to hit them.

1. `ResultsCache._param_key()` sorts by KEY NAME alphabetically -- guard, k, msa,
   msb -- so `{msb: 1.0, msa: 2.0, k: 64, guard: 10.0}` becomes
   `waveforms_10.0_64_2.0_1.0.npz`. That tail reads like before-then-after and is
   after-then-before. Hand-building it with the two swapped gives a path that
   never existed, and a coverage audit then reports a confident `0 / 2852`.
2. `guard` is PART OF THE KEY. A guarded and an unguarded npz are different
   products. Omit `guard` and you miss every npz the fan-out produced, silently
   resolving a stale pre-guard file instead.

Use `WAVEFORM_PARAMS` / `waveform_params()`; do not retype either.
"""

from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

#: `rec.cache` result name.
WAVEFORM_NAME = "waveforms"

#: The parameters of the run of record (`wf_v1`, 2026-08-10) and step1's own
#: argparse defaults. `rec.wf` uses these.
WAVEFORM_PARAMS: Dict[str, float] = {"msb": 1.0, "msa": 2.0, "k": 64, "guard": 10.0}


def waveform_params(
    ms_before: float = 1.0,
    ms_after: float = 2.0,
    n_footprint: int = 64,
    artifact_guard_ms: float = 10.0,
) -> Dict[str, float]:
    """Build the cache params dict from step1's CLI arguments, named as step1
    names them. Mirrors `step1_extract_waveforms.py:451`."""
    return {
        "msb": float(ms_before),
        "msa": float(ms_after),
        "k": int(n_footprint),
        "guard": float(artifact_guard_ms),
    }


def waveform_s3_path(rec, **kwargs) -> Optional[str]:
    """Where this recording's npz lives on S3. Handy for `aws s3 ls` when
    something looks wrong. Takes the same keywords as `Recording.get_waveforms`.

    Worth knowing: the results prefix is built from the experiment LEAF, so
    `a/x` and `b/x` on one chip would collide and one npz would silently
    overwrite the other. Checked once across the `wf_v1` corpus — all 2,852 rows
    resolve to 2,852 distinct paths — but re-check if the catalog grows.
    """
    return rec.cache._s3_file_path(WAVEFORM_NAME, waveform_params(**kwargs))


class Waveforms:
    """One recording's extracted waveforms.

    Wraps the npz dict with scalar unwrapping, a time axis, and a per-unit
    summary. Every underlying array stays reachable by name, either as an
    attribute (`wf.amp_map`) or by key (`wf["amp_map"]`).
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        params: Optional[Dict[str, Any]] = None,
        identifier: Optional[str] = None,
    ):
        self._d = data
        self._params = dict(params) if params else dict(WAVEFORM_PARAMS)
        self._identifier = identifier

    # ---- dict compatibility -------------------------------------------
    def __getitem__(self, key: str) -> np.ndarray:
        return self._d[key]

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def __iter__(self) -> Iterator[str]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def keys(self):
        return self._d.keys()

    def items(self):
        return self._d.items()

    def get(self, key: str, default=None):
        return self._d.get(key, default)

    def to_dict(self) -> Dict[str, np.ndarray]:
        """The raw npz dict, unmodified."""
        return self._d

    # ---- attribute access ---------------------------------------------
    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails, so `_d` etc. resolve first.
        if name.startswith("_"):
            raise AttributeError(name)
        d = self.__dict__.get("_d")
        if d is not None and name in d:
            v = d[name]
            # step1 stores scalars as 0-d arrays and `load_local` only unwraps
            # 0-d OBJECT arrays, so `wf["fs"]` is `array(20000.)`. Unwrap here
            # -- `float(wf["fs"])` on every use is a papercut worth removing.
            return v.item() if isinstance(v, np.ndarray) and v.ndim == 0 else v
        raise AttributeError(
            f"Waveforms has no attribute {name!r}; available: {sorted(self._d)}"
        )

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._d))

    def __repr__(self) -> str:
        who = f"{self._identifier} " if self._identifier else ""
        n_live = int((self._d["n_spikes"] > 0).sum())
        return (
            f"Waveforms({who}{self.n_units} units [{n_live} live] x "
            f"{self.n_footprint_channels} ch x {self.n_samples} samples, "
            f"guard={self._params.get('guard')} ms)"
        )

    # ---- shape ---------------------------------------------------------
    @property
    def params(self) -> Dict[str, Any]:
        """The cache params this was loaded with."""
        return dict(self._params)

    @property
    def n_units(self) -> int:
        return int(self._d["waveforms"].shape[0])

    @property
    def n_footprint_channels(self) -> int:
        """K — how many channels per unit were kept. NOT a fixed radius: routed
        electrodes are sparse, so mask with `footprint_dist_um` if you need one."""
        return int(self._d["waveforms"].shape[1])

    @property
    def n_samples(self) -> int:
        return int(self._d["waveforms"].shape[2])

    @property
    def t_ms(self) -> np.ndarray:
        """Time axis in ms, t=0 at the spike.

        The window is `n_before + n_after + 1` samples — the spike sample itself
        is the +1, so at the defaults that is 20 + 40 + 1 = 61. Taken from the
        array's own shape so it cannot drift out of sync with the data.
        """
        return (np.arange(self.n_samples) - int(self._d["n_before"])) / float(
            self._d["fs"]
        ) * 1000.0

    # ---- convenience ----------------------------------------------------
    @property
    def peak(self) -> np.ndarray:
        """(n_units, n_samples) — the peak-channel waveform of every unit.

        Footprint columns are ordered by DISTANCE from the peak channel, so
        column 0 is always the peak channel. They are deliberately not ordered
        by amplitude: ranking by amplitude pulls in other neurons firing
        synchronously during network bursts.
        """
        return self._d["waveforms"][:, 0, :]

    @property
    def peak_p2p_uv(self) -> np.ndarray:
        """(n_units,) — peak-to-peak amplitude on the peak channel, uV."""
        p = self.peak
        return p.max(axis=1) - p.min(axis=1)

    @property
    def live(self) -> np.ndarray:
        """Boolean mask of units with at least one averaged spike.

        ALWAYS gate on this. Zero-spike units are normal and common — RT-Sort
        fits its sequences on the base `exp1` recording and applies them to every
        `_cont_*`, so many sequences are simply silent in any one recording. The
        catalog's `num_units` counts the sequence set, not the active units.
        """
        return self._d["n_spikes"] > 0

    def footprint(self, unit: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One unit's footprint as `(waveforms, channel_xy, dist_um)`,
        shapes `(K, n_samples)`, `(K, 2)`, `(K,)`."""
        u = int(unit)
        return (
            self._d["waveforms"][u],
            self._d["channel_xy"][u],
            self._d["footprint_dist_um"][u],
        )

    def summary(self, live_only: bool = False) -> "pd.DataFrame":  # noqa: F821
        """One row per unit: spike counts, guard counts, peak channel, p2p.

        `n_spikes` is capped by step1's `max_spikes` (2000 by default);
        `n_spikes_total` is the uncapped train.
        """
        import pandas as pd

        df = pd.DataFrame(
            {
                "unit": np.arange(self.n_units),
                "n_spikes": self._d["n_spikes"],
                "n_spikes_total": self._d["n_spikes_total"],
                "n_spikes_guarded": self._d["n_spikes_guarded"],
                "peak_channel": self._d["peak_channel"],
                "peak_p2p_uv": self.peak_p2p_uv,
            }
        )
        return df[self.live].reset_index(drop=True) if live_only else df
