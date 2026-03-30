"""Tests for braindance.core.spikedetector.data module (Waveform and Unit classes)."""

import numpy as np
import pytest

data_module = pytest.importorskip(
    "braindance.core.spikedetector.data",
    reason="braindance.core.spikedetector.data not importable (torch or transitive deps missing)",
    exc_type=ImportError,
)

Waveform = data_module.Waveform
Unit = data_module.Unit


# ===== Waveform =============================================================


class TestWaveformInit:
    """Tests for Waveform.__init__."""

    def test_stores_waveform_array(self):
        """Tests:
        - The waveform attribute holds the numpy array passed to __init__.
        """
        arr = np.array([1.0, 2.0, 3.0])
        wf = Waveform(waveform=arr, peak_idx=1, alpha=0.5, curated=True)
        assert wf.waveform is arr

    def test_stores_peak_idx(self):
        """Tests:
        - The peak_idx attribute matches the value passed to __init__.
        """
        wf = Waveform(waveform=np.zeros(5), peak_idx=3, alpha=0.1, curated=False)
        assert wf.peak_idx == 3

    def test_stores_alpha(self):
        """Tests:
        - The alpha attribute matches the value passed to __init__.
        """
        wf = Waveform(waveform=np.zeros(4), peak_idx=0, alpha=0.99, curated=True)
        assert wf.alpha == 0.99

    def test_stores_curated(self):
        """Tests:
        - The curated attribute matches the boolean passed to __init__.
        """
        wf = Waveform(waveform=np.zeros(4), peak_idx=0, alpha=0.5, curated=False)
        assert wf.curated is False

    def test_curated_true(self):
        """Tests:
        - The curated attribute is True when True is passed.
        """
        wf = Waveform(waveform=np.zeros(4), peak_idx=0, alpha=0.5, curated=True)
        assert wf.curated is True


class TestWaveformLen:
    """Tests for Waveform.len attribute."""

    def test_len_matches_array_size(self):
        """Tests:
        - The len attribute equals waveform.size for a 1-D array.
        """
        arr = np.array([10.0, 20.0, 30.0, 40.0])
        wf = Waveform(waveform=arr, peak_idx=0, alpha=0.0, curated=False)
        assert wf.len == 4

    def test_len_single_element(self):
        """Tests:
        - The len attribute is 1 for a single-element array.
        """
        arr = np.array([7.0])
        wf = Waveform(waveform=arr, peak_idx=0, alpha=0.0, curated=False)
        assert wf.len == 1

    def test_len_2d_array(self):
        """Tests:
        - The len attribute equals total element count for a 2-D array
          (numpy .size returns the product of all dimensions).
        """
        arr = np.zeros((3, 4))
        wf = Waveform(waveform=arr, peak_idx=0, alpha=0.0, curated=False)
        assert wf.len == 12


class TestWaveformUnravel:
    """Tests for Waveform.unravel."""

    def test_returns_tuple_of_five(self):
        """Tests:
        - unravel() returns a tuple with exactly 5 elements.
        """
        arr = np.array([1.0, 2.0])
        wf = Waveform(waveform=arr, peak_idx=0, alpha=0.5, curated=True)
        result = wf.unravel()
        assert len(result) == 5

    def test_field_order(self):
        """Tests:
        - unravel() returns (waveform, peak_idx, len, alpha, curated) in that order.
        """
        arr = np.array([5.0, 6.0, 7.0])
        wf = Waveform(waveform=arr, peak_idx=2, alpha=0.75, curated=False)
        w, p, l, a, c = wf.unravel()
        assert w is arr
        assert p == 2
        assert l == 3
        assert a == 0.75
        assert c is False

    def test_unravel_reflects_stored_values(self):
        """Tests:
        - Each field returned by unravel() matches the corresponding attribute.
        """
        arr = np.arange(10, dtype=np.float32)
        wf = Waveform(waveform=arr, peak_idx=5, alpha=1.0, curated=True)
        w, p, l, a, c = wf.unravel()
        assert w is wf.waveform
        assert p == wf.peak_idx
        assert l == wf.len
        assert a == wf.alpha
        assert c == wf.curated


class TestWaveformShapes:
    """Tests for Waveform with various numpy array shapes."""

    def test_empty_array(self):
        """Tests:
        - Waveform accepts an empty array; len is 0.
        """
        arr = np.array([])
        wf = Waveform(waveform=arr, peak_idx=0, alpha=0.0, curated=False)
        assert wf.len == 0

    def test_large_1d_array(self):
        """Tests:
        - Waveform works correctly with a large 1-D array.
        """
        arr = np.random.randn(10000)
        wf = Waveform(waveform=arr, peak_idx=500, alpha=0.3, curated=True)
        assert wf.len == 10000
        assert wf.peak_idx == 500

    def test_3d_array_size(self):
        """Tests:
        - The len attribute equals total element count for a 3-D array.
        """
        arr = np.zeros((2, 3, 4))
        wf = Waveform(waveform=arr, peak_idx=0, alpha=0.0, curated=False)
        assert wf.len == 24


# ===== Unit =================================================================


class TestUnitInit:
    """Tests for Unit.__init__."""

    def test_stores_waveforms_list(self):
        """Tests:
        - The wfs attribute holds the list of Waveform objects passed to __init__.
        """
        wf1 = Waveform(waveform=np.zeros(3), peak_idx=0, alpha=0.1, curated=False)
        wf2 = Waveform(waveform=np.ones(5), peak_idx=2, alpha=0.9, curated=True)
        unit = Unit(waveforms=[wf1, wf2])
        assert unit.wfs is not None
        assert len(unit.wfs) == 2

    def test_wfs_identity(self):
        """Tests:
        - The wfs attribute is the exact same list object passed to __init__.
        """
        wf_list = [
            Waveform(waveform=np.zeros(4), peak_idx=0, alpha=0.0, curated=False),
        ]
        unit = Unit(waveforms=wf_list)
        assert unit.wfs is wf_list


class TestUnitAccess:
    """Tests for accessing individual waveforms via Unit.wfs."""

    def test_index_access(self):
        """Tests:
        - Individual Waveform objects are accessible by index through .wfs.
        """
        wf0 = Waveform(waveform=np.array([1.0]), peak_idx=0, alpha=0.5, curated=True)
        wf1 = Waveform(waveform=np.array([2.0]), peak_idx=0, alpha=0.6, curated=False)
        unit = Unit(waveforms=[wf0, wf1])
        assert unit.wfs[0] is wf0
        assert unit.wfs[1] is wf1

    def test_empty_waveforms(self):
        """Tests:
        - Unit accepts an empty list of waveforms.
        """
        unit = Unit(waveforms=[])
        assert unit.wfs == []
        assert len(unit.wfs) == 0


# ===== Edge Cases ===========================================================


class TestWaveformEdgeCases:
    """Edge case tests for Waveform."""

    def test_negative_peak_idx(self):
        """Tests:
        - A negative peak_idx (-1) is stored without raising an error.
        """
        arr = np.array([1.0, 2.0, 3.0])
        wf = Waveform(waveform=arr, peak_idx=-1, alpha=0.5, curated=True)
        assert wf.peak_idx == -1

    def test_zero_alpha(self):
        """Tests:
        - An alpha of 0.0 is stored correctly and returned by unravel().
        """
        arr = np.array([1.0, 2.0])
        wf = Waveform(waveform=arr, peak_idx=0, alpha=0.0, curated=False)
        assert wf.alpha == 0.0
        _, _, _, a, _ = wf.unravel()
        assert a == 0.0

    def test_large_waveform(self):
        """Tests:
        - A waveform with 10000 elements has len == 10000.
        """
        arr = np.zeros(10000)
        wf = Waveform(waveform=arr, peak_idx=0, alpha=0.5, curated=False)
        assert wf.len == 10000

    def test_boolean_curated_values(self):
        """Tests:
        - curated=True and curated=False are stored correctly.
        """
        wf_true = Waveform(waveform=np.zeros(3), peak_idx=0, alpha=0.5, curated=True)
        wf_false = Waveform(waveform=np.zeros(3), peak_idx=0, alpha=0.5, curated=False)
        assert wf_true.curated is True
        assert wf_false.curated is False

    def test_float32_waveform(self):
        """Tests:
        - A np.float32 array has its len computed correctly.
        """
        arr = np.arange(50, dtype=np.float32)
        wf = Waveform(waveform=arr, peak_idx=0, alpha=0.1, curated=False)
        assert wf.len == 50


class TestUnitEdgeCases:
    """Edge case tests for Unit."""

    def test_empty_waveforms_list(self):
        """Tests:
        - Unit([]) stores an empty list with len 0.
        """
        unit = Unit(waveforms=[])
        assert unit.wfs == []
        assert len(unit.wfs) == 0

    def test_single_waveform(self):
        """Tests:
        - A Unit with one Waveform stores it accessible via wfs[0].
        """
        wf = Waveform(waveform=np.array([1.0, 2.0]), peak_idx=0, alpha=0.5, curated=True)
        unit = Unit(waveforms=[wf])
        assert len(unit.wfs) == 1
        assert unit.wfs[0] is wf

    def test_many_waveforms(self):
        """Tests:
        - A Unit with 100 waveforms stores all of them.
        """
        wfs = [
            Waveform(waveform=np.zeros(5), peak_idx=0, alpha=0.1, curated=False)
            for _ in range(100)
        ]
        unit = Unit(waveforms=wfs)
        assert len(unit.wfs) == 100
