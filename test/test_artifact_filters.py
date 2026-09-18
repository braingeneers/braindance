"""Numerical and streaming contract checks; run with unittest."""
import unittest
import numpy as np
import numba
from braindance.utils.artifact_filters import SalpaArtifactRemover, LinearFitArtifactRemover


class ArtifactFilterTests(unittest.TestCase):
    def test_polynomial_reference_and_alignment(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(2, 300))
        for cls in (SalpaArtifactRemover, LinearFitArtifactRemover):
            f = cls(2, half_window=8, excursion_threshold=np.inf)
            r = f.process(x)
            reference = np.stack([np.convolve(row, f.weights, 'valid') for row in x])
            np.testing.assert_allclose(r.artifact[:, 16:], reference, atol=1e-12)
            np.testing.assert_allclose((r.clean+r.artifact)[:, 16:], x[:, 8:-8])
            self.assertFalse(r.valid[:, :16].any())

    def test_polynomial_annihilation(self):
        t = np.linspace(-1, 1, 300)
        for cls, baseline in ((SalpaArtifactRemover, 3+t+2*t*t+t**3),
                              (LinearFitArtifactRemover, 3+t)):
            r = cls(1, half_window=8, excursion_threshold=np.inf).process(baseline[None])
            np.testing.assert_allclose(r.clean[:, 16:], 0, atol=1e-12)

    def test_chunk_reset_warmup_and_exclusions(self):
        x = np.random.default_rng(3).normal(size=(2, 300))
        x[0, 100:103] = 200
        for cls in (SalpaArtifactRemover, LinearFitArtifactRemover):
            f = cls(2, half_window=8, excursion_threshold=100, recovery_samples=10)
            whole = f.process(x)
            f.reset()
            parts = []
            for a, b in ((0, 5), (5, 99), (99, 110), (110, 300)):
                parts.append(f.process(x[:, a:b]))
                f.warmup()
            for attr in ('clean', 'artifact', 'valid'):
                np.testing.assert_equal(getattr(whole, attr),
                                        np.concatenate([getattr(p, attr) for p in parts], axis=1))
            self.assertFalse(whole.valid[0, 100:126].any())
            self.assertTrue(whole.valid[1, 100:].all())
            self.assertTrue(np.isnan(whole.clean[~whole.valid]).all())

    def test_input_validation_and_rails(self):
        f = LinearFitArtifactRemover(1, half_window=2, excursion_threshold=np.inf,
                                    recovery_samples=0, rail_min=0, rail_max=10)
        x = np.ones((1, 20)); x[0, 10] = 10
        r = f.process(x)
        self.assertFalse(r.valid[0, 10:15].any())
        self.assertTrue(r.valid[0, 15:].all())
        with self.assertRaises(ValueError):
            f.process([[np.nan]])
        with self.assertRaises(ValueError):
            f.process(np.ones((2, 10)))

    def test_long_stream_large_offset_reference(self):
        # Many rebases, a large DC level, and a slow drift stress cancellation.
        t=np.arange(100_000)
        x=(1e6+300*np.sin(t*.003)+np.random.default_rng(12).normal(size=len(t)))[None]
        for cls in (SalpaArtifactRemover,LinearFitArtifactRemover):
            for half in (2,8,60):
                f=cls(1,half_window=half,excursion_threshold=np.inf)
                r=f.process(x)
                expected=np.convolve(x[0],f.weights,'valid')
                np.testing.assert_allclose(r.artifact[0,2*half:],expected,rtol=0,atol=2e-8)
                np.testing.assert_allclose(r.clean[0,2*half:],x[0,half:-half]-expected,rtol=0,atol=2e-8)

    def test_parallel_serial_and_chunk_equivalence(self):
        previous=numba.get_num_threads()
        numba.set_num_threads(min(4,previous))
        try:
            x=np.random.default_rng(18).normal(size=(40,2200))
            x[3,251:254]=200
            for cls in (SalpaArtifactRemover,LinearFitArtifactRemover):
                serial=cls(40,parallel=False).process(x)
                f=cls(40,parallel=True)
                parts=[]
                for a,b in ((0,120),(120,256),(256,257),(257,780),(780,2200)):
                    parts.append(f.fit_step(x[:,a:b]))
                    f.warmup()
                for attr in ('clean','artifact','valid'):
                    np.testing.assert_equal(getattr(serial,attr),
                        np.concatenate([getattr(p,attr) for p in parts],axis=1))
        finally:
            numba.set_num_threads(previous)

    def test_empty_and_rejected_input_preserve_state(self):
        f=SalpaArtifactRemover(1)
        g=SalpaArtifactRemover(1)
        x=np.random.default_rng(11).normal(size=(1,1000))
        f.process(x[:,:250]); g.process(x[:,:250])
        self.assertEqual(f.process(np.empty((1,0))).clean.shape,(1,0))
        with self.assertRaises(ValueError): f.process([[np.inf]])
        np.testing.assert_equal(f.process(x[:,250:]).clean,g.process(x[:,250:]).clean)


if __name__ == '__main__':
    unittest.main()
