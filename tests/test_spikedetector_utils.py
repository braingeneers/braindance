"""Tests for braindance.core.spikedetector.utils."""

import datetime
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
utils = pytest.importorskip("braindance.core.spikedetector.utils")


class TestConfusionMatrix:
    """Tests for the confusion_matrix function."""

    def test_all_correct_predictions(self):
        """Tests:
        All correct predictions yield FN=0, TN=2, FP=0, TP=2.
        """
        result = utils.confusion_matrix([0, 0, 1, 1], [0, 0, 1, 1])
        np.testing.assert_array_equal(result, [0, 2, 0, 2])

    def test_all_wrong_predictions(self):
        """Tests:
        All wrong predictions yield FN=2, TN=0, FP=2, TP=0.
        """
        result = utils.confusion_matrix([1, 1, 0, 0], [0, 0, 1, 1])
        np.testing.assert_array_equal(result, [2, 0, 2, 0])

    def test_mixed_predictions(self):
        """Tests:
        Mixed predictions produce the expected confusion counts.
        """
        # preds=[1,0,1,0], labels=[1,0,0,1]
        # (1,1)->TP, (0,0)->TN, (1,0)->FP, (0,1)->FN
        result = utils.confusion_matrix([1, 0, 1, 0], [1, 0, 0, 1])
        np.testing.assert_array_equal(result, [1, 1, 1, 1])

    def test_mismatched_lengths_raises(self):
        """Tests:
        Mismatched lengths between preds and labels raises ValueError.
        """
        with pytest.raises(ValueError, match="len"):
            utils.confusion_matrix([0, 1], [0, 1, 1])


class TestConfusionStats:
    """Tests for the confusion_stats function."""

    def test_perfect_predictions(self):
        """Tests:
        Perfect confusion matrix [0, 2, 0, 2] gives 100% accuracy, recall, precision.
        """
        stats = utils.confusion_stats(np.array([0, 2, 0, 2]))
        assert stats[0] == pytest.approx(100.0)  # accuracy
        assert stats[1] == pytest.approx(100.0)  # recall
        assert stats[2] == pytest.approx(100.0)  # precision

    def test_known_confusion_matrix(self):
        """Tests:
        Known confusion matrix [1, 3, 2, 4] produces correct accuracy, recall, precision.
        """
        # FN=1, TN=3, FP=2, TP=4
        # accuracy = (4+3)/(1+3+2+4) = 7/10 = 70%
        # recall = 4/(4+1) = 80%
        # precision = 4/(4+2) = 66.666...%
        stats = utils.confusion_stats(np.array([1, 3, 2, 4]))
        assert stats[0] == pytest.approx(70.0)
        assert stats[1] == pytest.approx(80.0)
        assert stats[2] == pytest.approx(100 * 4 / 6)


class TestRandomSeed:
    """Tests for the random_seed function."""

    def test_reproducible_numpy_random(self):
        """Tests:
        Setting the same seed twice produces identical numpy random numbers.
        """
        utils.random_seed(42, silent=True)
        a = np.random.rand(5)
        utils.random_seed(42, silent=True)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_silent_suppresses_print(self, capsys):
        """Tests:
        silent=True suppresses the seed print message.
        """
        utils.random_seed(0, silent=True)
        captured = capsys.readouterr()
        assert captured.out == ""

        utils.random_seed(0, silent=False)
        captured = capsys.readouterr()
        assert "random seed" in captured.out.lower()


class TestGetTime:
    """Tests for the get_time function."""

    def test_returns_string(self):
        """Tests:
        get_time returns a string.
        """
        result = utils.get_time()
        assert isinstance(result, str)

    def test_format_length(self):
        """Tests:
        Returned string is 20 characters long (yymmdd_HHMMSS_ffffff).
        """
        result = utils.get_time()
        assert len(result) == 20

    def test_format_matches_pattern(self):
        """Tests:
        Returned string matches the expected datetime format pattern.
        """
        result = utils.get_time()
        assert re.match(r"\d{6}_\d{6}_\d{6}", result)


class TestCopyFile:
    """Tests for the copy_file function."""

    def test_copies_file_to_destination(self, tmp_path):
        """Tests:
        File is copied to the destination folder with the same name.
        """
        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"
        dest.mkdir()

        src_file = src / "data.txt"
        src_file.write_text("hello")

        utils.copy_file(str(src_file), str(dest))

        copied = dest / "data.txt"
        assert copied.exists()
        assert copied.read_text() == "hello"


class TestRound:
    """Tests for the custom round function."""

    def test_half_rounds_up_0_5(self):
        """Tests:
        round(0.5) returns 1 (rounds up, unlike Python banker's rounding).
        """
        assert utils.round(0.5) == 1

    def test_half_rounds_up_1_5(self):
        """Tests:
        round(1.5) returns 2.
        """
        assert utils.round(1.5) == 2

    def test_truncates_below_half(self):
        """Tests:
        round(2.3) returns 2 (truncates fractional part below 0.5).
        """
        assert utils.round(2.3) == 2

    def test_numpy_array(self):
        """Tests:
        round on a numpy array returns element-wise truncation.

        Notes:
            The numpy path uses .astype(int) which truncates the fractional
            part, so 0.5 -> 0 and 1.5 -> 1 (unlike the scalar path which
            rounds 0.5 up). This is a known behavioral difference in the
            source implementation.
        """
        result = utils.round(np.array([0.5, 1.5, 2.3]))
        np.testing.assert_array_equal(result, np.array([0, 1, 2]))


class TestTorchToNp:
    """Tests for the torch_to_np function."""

    def test_tensor_returns_numpy(self):
        """Tests:
        A torch.Tensor input is converted to a numpy array.
        """
        t = torch.tensor([1.0, 2.0, 3.0])
        result = utils.torch_to_np(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_numpy_returned_as_is(self):
        """Tests:
        A numpy array input is returned as-is (same object).
        """
        arr = np.array([1.0, 2.0])
        result = utils.torch_to_np(arr)
        assert result is arr


class TestConfusionMatrixEdgeCases:
    """Edge case tests for the confusion_matrix function."""

    def test_empty_inputs(self):
        """Tests:
        Empty preds and labels produce all-zero confusion array.
        """
        result = utils.confusion_matrix([], [])
        np.testing.assert_array_equal(result, [0, 0, 0, 0])

    def test_single_element_correct(self):
        """Tests:
        Single correct positive prediction yields TP=1.
        """
        result = utils.confusion_matrix([1], [1])
        np.testing.assert_array_equal(result, [0, 0, 0, 1])

    def test_single_element_wrong(self):
        """Tests:
        Single wrong positive prediction yields FP=1.
        """
        result = utils.confusion_matrix([1], [0])
        np.testing.assert_array_equal(result, [0, 0, 1, 0])


class TestConfusionStatsEdgeCases:
    """Edge case tests for the confusion_stats function."""

    def test_all_zeros(self):
        """Tests:
        All-zero confusion matrix produces NaN values due to division by zero.
        """
        stats = utils.confusion_stats(np.array([0, 0, 0, 0]))
        assert np.isnan(stats[0])  # accuracy
        assert np.isnan(stats[1])  # recall
        assert np.isnan(stats[2])  # precision

    def test_no_positives(self):
        """Tests:
        No positive predictions or labels: accuracy=100, recall=NaN, precision=NaN.
        """
        # FN=0, TN=5, FP=0, TP=0
        stats = utils.confusion_stats(np.array([0, 5, 0, 0]))
        assert stats[0] == pytest.approx(100.0)  # accuracy
        assert np.isnan(stats[1])  # recall (0/0)
        assert np.isnan(stats[2])  # precision (0/0)

    def test_no_negatives(self):
        """Tests:
        All true positives: accuracy=100, recall=100, precision=100.
        """
        # FN=0, TN=0, FP=0, TP=5
        stats = utils.confusion_stats(np.array([0, 0, 0, 5]))
        assert stats[0] == pytest.approx(100.0)  # accuracy
        assert stats[1] == pytest.approx(100.0)  # recall
        assert stats[2] == pytest.approx(100.0)  # precision


class TestRoundEdgeCases:
    """Edge case tests for the custom round function."""

    def test_negative_number(self):
        """Tests:
        round(-0.5) returns -1 because int(-0.5)=0 and int((-0.5-0)*2)=-1.
        """
        assert utils.round(-0.5) == -1

    def test_integer_input(self):
        """Tests:
        Integer input is returned unchanged.
        """
        assert utils.round(3) == 3

    def test_zero(self):
        """Tests:
        Zero input returns zero.
        """
        assert utils.round(0) == 0

    def test_large_float(self):
        """Tests:
        Large float with 0.5 fractional part rounds up correctly.
        """
        assert utils.round(1000000.5) == 1000001


class TestTorchToNpEdgeCases:
    """Edge case tests for the torch_to_np function."""

    def test_multidimensional_tensor(self):
        """Tests:
        A 3D torch.Tensor is converted to a 3D numpy array with correct shape.
        """
        t = torch.zeros(2, 3, 4)
        result = utils.torch_to_np(t)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3, 4)

    def test_empty_tensor(self):
        """Tests:
        An empty torch.Tensor is converted to an empty numpy array.
        """
        t = torch.tensor([])
        result = utils.torch_to_np(t)
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)

    def test_scalar_tensor(self):
        """Tests:
        A scalar torch.Tensor is converted to a numpy scalar.
        """
        t = torch.tensor(5.0)
        result = utils.torch_to_np(t)
        assert isinstance(result, np.ndarray)
        assert result.item() == 5.0

    def test_plain_python_int(self):
        """Tests:
        A plain Python int is returned as-is (not a Tensor).
        """
        result = utils.torch_to_np(42)
        assert result == 42
        assert isinstance(result, int)


class TestCopyFileEdgeCases:
    """Edge case tests for the copy_file function."""

    def test_copy_preserves_content(self, tmp_path):
        """Tests:
        Copied file content is byte-for-byte identical to the source.
        """
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        content = "line1\nline2\nspecial chars: àéîõü\n"
        src_file = src_dir / "test_data.txt"
        src_file.write_text(content, encoding="utf-8")

        utils.copy_file(str(src_file), str(dest_dir))

        copied = dest_dir / "test_data.txt"
        assert copied.read_text(encoding="utf-8") == content
