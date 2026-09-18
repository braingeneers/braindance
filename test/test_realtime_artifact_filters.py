"""Check existing realtime recurrences against direct fits and across chunks."""
import unittest
import numpy as np
import numba
from braindance.utils.rt_linear_art_removal import LinearArtifactRemoval, LinearArtifactRemoval2


class RealtimeFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numba.set_num_threads(2)

    def test_reference_and_delay(self):
        n=8
        t=np.linspace(-1,1,2000)
        x=np.vstack((3+2*t, np.random.default_rng(4).normal(size=len(t))))
        for cls,degree in ((LinearArtifactRemoval,1),(LinearArtifactRemoval2,1)):
            f=cls(2,N=n,nc_start=n,min_val=-1e10,max_val=1e10)
            clean,artifact,_=f.fit_step(x)
            weights=np.linalg.pinv(np.vander(np.arange(-n,n+1,dtype=float),degree+1,increasing=True))[0]
            baseline=np.array([np.convolve(row,weights,'valid') for row in x])
            np.testing.assert_allclose(artifact[:,2*n+1:],baseline[:,1:],atol=2e-7)
            np.testing.assert_allclose(clean[:,2*n+1:]+artifact[:,2*n+1:],x[:,n+1:-n],atol=1e-12)
            np.testing.assert_allclose(clean[0,2*n+1:],0,atol=2e-7)

    def test_chunk_invariance_and_linear_equivalence(self):
        x=np.random.default_rng(2).normal(size=(3,800)); x[0,300:305]=1000
        results=[]
        for cls in (LinearArtifactRemoval,LinearArtifactRemoval2):
            kwargs=dict(n_channels=3,N=8,nc_start=8,min_val=-100,max_val=100)
            whole=cls(**kwargs).fit_step(x)
            f=cls(**kwargs)
            chunks=[f.fit_step(x[:,a:b]) for a,b in ((0,5),(5,290),(290,304),(304,470),(470,800))]
            for k in range(3):
                np.testing.assert_allclose(whole[k],np.concatenate([p[k] for p in chunks],axis=1),atol=1e-10)
            results.append(whole)
        for k in range(2):
            np.testing.assert_allclose(results[0][k],results[1][k],atol=1e-10)


if __name__=='__main__': unittest.main()
