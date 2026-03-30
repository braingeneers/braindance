"""Tests for braindance.core.artifact_removal module."""

import numpy as np
import pytest

from braindance.core.artifact_removal import (
    ArtifactRemoval,
    LOOKUP_TABLE,
    fast_factorial,
    fast_median,
    fast_mmean,
)


# ---------------------------------------------------------------------------
# Warmup numba-compiled functions (JIT compilation on first call)
# ---------------------------------------------------------------------------

def _warmup():
    fast_factorial(0)
    fast_median(np.array([1.0, 2.0]))
    fast_mmean(0.0, 1.0)
    ArtifactRemoval.shift_ind(np.array([0.0, 0.0, 0.0]), 1.0)
    ar = ArtifactRemoval(N=5)
    ar.fit_step(0.0)


_warmup()


# ===== LOOKUP_TABLE ========================================================

class TestLookupTable:
    """Tests for the module-level LOOKUP_TABLE constant."""

    def test_length(self):
        """Tests:
        - LOOKUP_TABLE has exactly 21 elements (0! through 20!).
        """
        assert len(LOOKUP_TABLE) == 21

    def test_known_values(self):
        """Tests:
        - Selected entries match known factorial values.
        """
        expected = {
            0: 1,
            1: 1,
            2: 2,
            3: 6,
            4: 24,
            5: 120,
            10: 3628800,
            20: 2432902008176640000,
        }
        for n, val in expected.items():
            assert LOOKUP_TABLE[n] == val, f"LOOKUP_TABLE[{n}] expected {val}, got {LOOKUP_TABLE[n]}"


# ===== fast_factorial ======================================================

class TestFastFactorial:
    """Tests for the fast_factorial numba function."""

    def test_factorial_zero(self):
        """Tests:
        - factorial(0) returns 1.
        """
        assert fast_factorial(0) == 1

    def test_factorial_five(self):
        """Tests:
        - factorial(5) returns 120.
        """
        assert fast_factorial(5) == 120

    def test_factorial_twenty(self):
        """Tests:
        - factorial(20) returns 2432902008176640000.
        """
        assert fast_factorial(20) == 2432902008176640000


# ===== fast_mmean ==========================================================

class TestFastMmean:
    """Tests for the fast_mmean numba function."""

    def test_known_inputs(self):
        """Tests:
        - Returns 0.8*q + 0.2*frame for known inputs.
        """
        q, frame = 10.0, 5.0
        result = fast_mmean(q, frame)
        expected = 0.8 * q + 0.2 * frame
        assert result == pytest.approx(expected)

    def test_zero_inputs(self):
        """Tests:
        - Returns 0.0 when both inputs are zero.
        """
        assert fast_mmean(0.0, 0.0) == pytest.approx(0.0)

    def test_negative_inputs(self):
        """Tests:
        - Handles negative values correctly.
        """
        q, frame = -5.0, -10.0
        result = fast_mmean(q, frame)
        expected = 0.8 * (-5.0) + 0.2 * (-10.0)
        assert result == pytest.approx(expected)


# ===== ArtifactRemoval.__init__ ============================================

class TestArtifactRemovalInit:
    """Tests for ArtifactRemoval.__init__."""

    def test_default_parameters(self):
        """Tests:
        - Default parameter values are stored correctly.
        """
        ar = ArtifactRemoval(N=10)
        assert ar.N == 10
        assert ar.nc_start == 0
        assert ar.min_val == -100
        assert ar.max_val == 100
        assert ar.spike_thresh_min == -3.5
        assert ar.spike_thresh_max == -20

    def test_custom_parameters(self):
        """Tests:
        - Custom parameter values are stored correctly.
        """
        ar = ArtifactRemoval(N=20, nc_start=5, min_val=-50, max_val=50, spike_thresh=[-5, -30])
        assert ar.N == 20
        assert ar.nc_start == 5
        assert ar.min_val == -50
        assert ar.max_val == 50
        assert ar.spike_thresh_min == -5
        assert ar.spike_thresh_max == -30

    def test_state_is_init(self):
        """Tests:
        - Initial state is 'init'.
        """
        ar = ArtifactRemoval(N=10)
        assert ar.state == "init"

    def test_v_array_size(self):
        """Tests:
        - v array has size 2*N + 2.
        """
        for n in [5, 10, 20]:
            ar = ArtifactRemoval(N=n)
            assert ar.v.shape == (2 * n + 2,)

    def test_T_length(self):
        """Tests:
        - T vector has 7 elements.
        """
        ar = ArtifactRemoval(N=10)
        assert ar.T.shape == (7,)

    def test_S_shape(self):
        """Tests:
        - S matrix is 4x4.
        """
        ar = ArtifactRemoval(N=10)
        assert ar.S.shape == (4, 4)


# ===== ArtifactRemoval._compute_T =========================================

class TestArtifactRemovalComputeT:
    """Tests for ArtifactRemoval._compute_T."""

    def test_returns_length_7(self):
        """Tests:
        - Returns an array of length 7.
        """
        ar = ArtifactRemoval(N=10)
        T = ar._compute_T(0)
        assert T.shape == (7,)

    def test_T_first_element(self):
        """Tests:
        - T[0] equals 2*N + 1 (number of points in the window).
        """
        N = 10
        ar = ArtifactRemoval(N=N)
        T = ar._compute_T(0)
        assert T[0] == pytest.approx(2 * N + 1)


# ===== ArtifactRemoval._compute_S =========================================

class TestArtifactRemovalComputeS:
    """Tests for ArtifactRemoval._compute_S."""

    def test_returns_4x4(self):
        """Tests:
        - Returns a 4x4 matrix.
        """
        ar = ArtifactRemoval(N=10)
        T = ar._compute_T(0)
        S = ar._compute_S(T)
        assert S.shape == (4, 4)

    def test_S_is_inverse_of_T_matrix(self):
        """Tests:
        - S is the inverse of the matrix constructed from T.
        """
        ar = ArtifactRemoval(N=10)
        T = ar._compute_T(0)
        # Reconstruct the matrix that was inverted
        M = np.zeros((4, 4))
        for k in range(4):
            for l in range(4):
                M[k, l] = T[k + l]
        S = ar._compute_S(T)
        product = M @ S
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)


# ===== ArtifactRemoval.shift_ind ==========================================

class TestArtifactRemovalShiftInd:
    """Tests for ArtifactRemoval.shift_ind."""

    def test_shifts_left_and_appends(self):
        """Tests:
        - Array is shifted left by one and the new value is appended at the end.
        """
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ArtifactRemoval.shift_ind(arr, 99.0)
        expected = np.array([2.0, 3.0, 4.0, 5.0, 99.0])
        np.testing.assert_array_equal(result, expected)

    def test_single_element(self):
        """Tests:
        - Works with a single-element array.
        """
        arr = np.array([1.0])
        result = ArtifactRemoval.shift_ind(arr, 42.0)
        np.testing.assert_array_equal(result, np.array([42.0]))


# ===== ArtifactRemoval.fit_step ============================================

class TestArtifactRemovalFitStep:
    """Tests for ArtifactRemoval.fit_step."""

    def test_returns_zeros_during_init(self):
        """Tests:
        - Returns (0, 0, False) while in 'init' state.
        """
        ar = ArtifactRemoval(N=10)
        cleaned, artifact, new_spike = ar.fit_step(1.0)
        assert cleaned == 0
        assert artifact == 0
        assert new_spike is False

    def test_transitions_to_fit_state(self):
        """Tests:
        - After receiving 2*N + 2 frames the state transitions to 'fit'.
        """
        N = 5
        ar = ArtifactRemoval(N=N)
        # Need to feed 2*N + 2 frames to fill v and trigger transition
        for i in range(2 * N + 2):
            ar.fit_step(float(i))
        assert ar.state == "fit"

    def test_returns_tuple_of_three(self):
        """Tests:
        - Always returns a 3-tuple (cleaned, artifact, new_spike).
        """
        ar = ArtifactRemoval(N=5)
        result = ar.fit_step(1.0)
        assert isinstance(result, tuple)
        assert len(result) == 3


# ===== ArtifactRemoval.run =================================================


class TestArtifactRemovalRun:
    """Tests for ArtifactRemoval.run."""

    def test_1d_returns_same_shape(self):
        """Tests:
        - 1D input returns a cleaned array of the same shape.
        """
        np.random.seed(42)
        data = np.random.randn(500)
        N = 10
        ar = ArtifactRemoval(N=N)
        result = ar.run(data)
        assert result.shape == data.shape

    def test_2d_returns_same_shape(self):
        """Tests:
        - 2D input returns a cleaned array of the same shape.
        """
        np.random.seed(42)
        data = np.random.randn(3, 500)
        N = 10
        ar = ArtifactRemoval(N=N)
        result = ar.run(data)
        assert result.shape == data.shape

    def test_return_artifacts_1d(self):
        """Tests:
        - return_artifacts=True returns a tuple of (cleaned, artifacts) for 1D input.
        """
        np.random.seed(42)
        data = np.random.randn(500)
        N = 10
        ar = ArtifactRemoval(N=N)
        result = ar.run(data, return_artifacts=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        cleaned, artifacts = result
        assert cleaned.shape == data.shape
        assert artifacts.shape == data.shape

    def test_return_spikes_1d(self):
        """Tests:
        - return_spikes=True returns a tuple of (cleaned, spike_times) for 1D input.
        """
        np.random.seed(42)
        data = np.random.randn(500)
        N = 10
        ar = ArtifactRemoval(N=N)
        result = ar.run(data, return_spikes=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        cleaned, spike_times = result
        assert cleaned.shape == data.shape
        assert isinstance(spike_times, list)

    def test_return_artifacts_2d(self):
        """Tests:
        - return_artifacts=True returns a tuple of (cleaned, artifacts) for 2D input.
        """
        np.random.seed(42)
        data = np.random.randn(2, 500)
        N = 10
        ar = ArtifactRemoval(N=N)
        result = ar.run(data, return_artifacts=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        cleaned, artifacts = result
        assert cleaned.shape == data.shape
        assert artifacts.shape == data.shape

    def test_return_both_1d(self):
        """Tests:
        - return_artifacts=True and return_spikes=True returns a 3-tuple for 1D input.
        """
        np.random.seed(42)
        data = np.random.randn(500)
        N = 10
        ar = ArtifactRemoval(N=N)
        result = ar.run(data, return_artifacts=True, return_spikes=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        cleaned, artifacts, spike_times = result
        assert cleaned.shape == data.shape
        assert artifacts.shape == data.shape
        assert isinstance(spike_times, list)


# ===== fast_mmean edge cases =================================================

class TestFastMmeanEdgeCases:
    """Edge case tests for the fast_mmean numba function.

    Tests:
    - Stability with very large values.
    - Asymmetric convergence toward a repeated frame value.
    """

    def test_large_values(self):
        """Tests:
        - q=1e10, frame=1e10 produces a stable result equal to 1e10.
        """
        result = fast_mmean(1e10, 1e10)
        assert result == pytest.approx(1e10)

    def test_asymmetric_convergence(self):
        """Tests:
        - Starting from q=100 and repeatedly applying frame=0, the result
          converges toward 0.
        """
        q = 100.0
        for _ in range(200):
            q = fast_mmean(q, 0.0)
        assert abs(q) < 1e-6


# ===== ArtifactRemoval edge N values ========================================

class TestArtifactRemovalEdgeN:
    """Edge case tests for ArtifactRemoval with extreme N values.

    Tests:
    - Minimum viable N=1 produces correct array shapes and does not crash.
    - Large N=100 produces correct array shapes.
    """

    def test_n_equals_one_singular_matrix(self):
        """Tests:
        - N=1 causes a singular matrix in _compute_S because the T matrix
          is rank-deficient for such a small window.

        Notes:
            - This is a source limitation: N must be > 1 for the cubic fit
              to be well-conditioned.
        """
        with pytest.raises(np.linalg.LinAlgError):
            ArtifactRemoval(N=1)

    def test_n_equals_two(self):
        """Tests:
        - N=2 is the minimum viable value: v has 6 elements, T has 7, S is 4x4.
        """
        ar = ArtifactRemoval(N=2)
        assert ar.v.shape == (6,)
        assert ar.T.shape == (7,)
        assert ar.S.shape == (4, 4)

    def test_large_n(self):
        """Tests:
        - N=100 creates v with 202 elements and correct T/S shapes.
        """
        ar = ArtifactRemoval(N=100)
        assert ar.v.shape == (202,)
        assert ar.T.shape == (7,)
        assert ar.S.shape == (4, 4)


# ===== ArtifactRemoval.fit_step depeg edge cases ============================

class TestArtifactRemovalFitStepDepeg:
    """Edge case tests for depeg transitions in ArtifactRemoval.fit_step.

    Tests:
    - An extreme frame triggers the depeg state from fit.
    - Normal frames during depeg cause recovery back to init.
    - Depeg auto-recovers after 20+ frames even with extreme values.
    """

    def _advance_to_fit(self, ar):
        """Feed 2*N+2 normal frames to transition from 'init' to 'fit'."""
        for i in range(2 * ar.N + 2):
            ar.fit_step(float(i % 10))
        assert ar.state == "fit"

    def test_extreme_frame_triggers_depeg(self):
        """Tests:
        - After reaching 'fit' state, a frame exceeding max_val triggers 'depeg'.
        """
        ar = ArtifactRemoval(N=5, min_val=-100, max_val=100)
        self._advance_to_fit(ar)
        # Feed a frame far beyond max_val relative to moving_mean
        ar.fit_step(ar.moving_mean + 200.0)
        assert ar.state == "depeg"

    def test_depeg_recovers_after_normal_frame(self):
        """Tests:
        - In depeg state, feeding normal frames eventually transitions back
          to 'init'.
        """
        ar = ArtifactRemoval(N=5, min_val=-100, max_val=100)
        self._advance_to_fit(ar)
        ar.fit_step(ar.moving_mean + 200.0)
        assert ar.state == "depeg"

        # Feed normal frames (close to moving_mean) until recovery
        for _ in range(25):
            ar.fit_step(ar.moving_mean)
            if ar.state == "init":
                break
        assert ar.state == "init"

    def test_depeg_recovers_after_20_frames(self):
        """Tests:
        - Depeg auto-recovers after depeg_count > 20, even if every frame
          is extreme.
        """
        ar = ArtifactRemoval(N=5, min_val=-100, max_val=100)
        self._advance_to_fit(ar)
        ar.fit_step(ar.moving_mean + 200.0)
        assert ar.state == "depeg"

        # Feed 21 more extreme frames; depeg_count should exceed 20
        for _ in range(21):
            ar.fit_step(ar.moving_mean + 500.0)
        assert ar.state == "init"


# ===== ArtifactRemoval.run edge cases =======================================

class TestArtifactRemovalRunEdgeCases:
    """Edge case tests for ArtifactRemoval.run with unusual data.

    Tests:
    - All-zeros 1D data produces all-zeros output.
    - Constant-valued 1D data runs without error and preserves shape.
    - Very short data (just above 2*N+1) runs without error.
    """

    def test_all_zeros_1d(self):
        """Tests:
        - data=np.zeros(100) produces an all-zeros cleaned output.
        """
        data = np.zeros(100)
        ar = ArtifactRemoval(N=5)
        result = ar.run(data)
        assert result.shape == data.shape
        np.testing.assert_array_equal(result, np.zeros(100))

    def test_constant_data_1d(self):
        """Tests:
        - data=np.ones(100)*5.0 runs without crash and preserves shape.
        """
        data = np.ones(100) * 5.0
        ar = ArtifactRemoval(N=5)
        result = ar.run(data)
        assert result.shape == data.shape

    def test_very_short_data(self):
        """Tests:
        - Data with length 2*N+2 (minimum viable) runs without error.
        """
        N = 5
        length = 2 * N + 2
        data = np.random.RandomState(0).randn(length)
        ar = ArtifactRemoval(N=N)
        result = ar.run(data)
        assert result.shape == (length,)


# ===== ArtifactRemoval.shift_ind edge cases ==================================

class TestShiftIndEdgeCases:
    """Edge case tests for ArtifactRemoval.shift_ind.

    Tests:
    - Two-element array shifts correctly.
    - Float values are handled correctly.
    """

    def test_two_elements(self):
        """Tests:
        - shift_ind([1, 2], 3) produces [2, 3].
        """
        arr = np.array([1.0, 2.0])
        result = ArtifactRemoval.shift_ind(arr, 3.0)
        np.testing.assert_array_equal(result, np.array([2.0, 3.0]))

    def test_float_values(self):
        """Tests:
        - shift_ind with float values produces correct result.
        """
        arr = np.array([1.5, 2.7, 3.9])
        result = ArtifactRemoval.shift_ind(arr, 4.1)
        np.testing.assert_array_almost_equal(result, np.array([2.7, 3.9, 4.1]))
