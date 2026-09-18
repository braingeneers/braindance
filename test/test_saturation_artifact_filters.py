"""Verify rail-only validity, polynomial accuracy, and legacy compatibility."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numba
import numpy as np

from braindance.utils.rt_linear_art_removal import LinearArtifactRemoval2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'proj/braindance_figs/artifact_comparison/derive_data'))
from historical_filters import HistoricalCubicRemover, HistoricalLinearRemover


class SaturationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numba.set_num_threads(2)

    def test_validity_and_direct_fits(self):
        x = np.random.default_rng(29).normal(size=(4, 1600))
        x[0, :8] = 1000
        x[0, 400:405] = -1000
        x[0, 420:450] = 1000
        x[1, 300:700] = 1000
        x[2, 700:800] = 900  # High, but not clipped: never blank.
        x[3, 900] = np.nan
        for n in (20, 60):
            masks = []
            for cls, degree in ((HistoricalLinearRemover, 1), (HistoricalCubicRemover, 3)):
                options = dict(n_channels=4, half_window=n, blanking_mode='saturation',
                               rail_min=-1000, rail_max=1000)
                clean, art, spikes = cls(**options).process(x)
                weights = np.linalg.pinv(np.vander(np.arange(-n, n + 1, dtype=float), degree + 1, increasing=True))[0]
                valid = np.isfinite(clean)
                expected = np.zeros_like(valid)
                direct = np.full_like(art, np.nan)
                for t in range(2*n + 1, x.shape[1]):
                    window = x[:, t-2*n:t+1]
                    expected[:, t] = np.all(np.isfinite(window) & (window > -1000) & (window < 1000), axis=1)
                    direct[:, t] = window @ weights
                np.testing.assert_array_equal(valid, expected)
                np.testing.assert_array_equal(np.isfinite(art), expected)
                np.testing.assert_allclose(art[valid], direct[valid], atol=2e-7, rtol=1e-9)
                np.testing.assert_allclose((clean+art)[:, 2*n+1:][valid[:, 2*n+1:]],
                                           x[:, n+1:-n][valid[:, 2*n+1:]], atol=1e-12)
                self.assertFalse(np.any(spikes[~valid]))
                f = cls(**options)
                chunks = []
                for a, b in ((0, 3), (3, 404), (404, 430), (430, 950), (950, 1600)):
                    chunks.append(f.process(x[:, a:b]))
                    f.warmup()
                for k, whole in enumerate((clean, art, spikes)):
                    np.testing.assert_array_equal(whole, np.concatenate([p[k] for p in chunks], axis=1))
                f.reset()
                np.testing.assert_array_equal(clean, f.process(x)[0])
                masks.append(valid)
            np.testing.assert_array_equal(*masks)

    def test_default_regression(self):
        x = np.random.default_rng(91).normal(size=(3, 900))
        x[0, 300:315] = 1000
        scratch = ROOT / 'proj/braindance_figs/artifact_comparison/scratch'
        for kind, cls in (('linear', LinearArtifactRemoval2),):
            spec = importlib.util.spec_from_file_location('before_saturation_' + kind, scratch / (kind + '_before_saturation.py'))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            old_cls = getattr(mod, cls.__name__)
            for n in (20, 60):
                args = dict(n_channels=3, N=n, nc_start=n)
                for old, new in zip(old_cls(**args).fit_step(x), cls(**args).fit_step(x)):
                    np.testing.assert_array_equal(old, new)

    def test_configuration_validation(self):
        for cls in (HistoricalLinearRemover, HistoricalCubicRemover):
            for kwargs in (dict(blanking_mode='oops'), dict(blanking_mode='saturation'),
                           dict(blanking_mode='saturation', rail_min=1, rail_max=1)):
                with self.assertRaises(ValueError):
                    cls(2, **kwargs)


if __name__ == '__main__':
    unittest.main()
