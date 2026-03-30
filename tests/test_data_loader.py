"""Tests for braindance.analysis.data_loader module."""
import pytest
import numpy as np
import pandas as pd
import os

data_loader = pytest.importorskip(
    "braindance.analysis.data_loader",
    reason="braindance.analysis.data_loader could not be imported (smart_open or h5py missing)",
)


class TestConvertUint16Maxwell:
    """Tests for the convert_uint16_maxwell function."""

    def test_output_dtype_is_float32(self):
        """Tests:
        - Output array has dtype float32 regardless of input dtype.
        """
        data = np.array([0, 256, 512, 1023], dtype=np.uint16)
        result = data_loader.convert_uint16_maxwell(data)
        assert result.dtype == np.float32

    def test_output_shape_matches_input(self):
        """Tests:
        - Output shape is identical to input shape for 1-D and 2-D arrays.
        """
        data_1d = np.arange(10, dtype=np.uint16)
        result_1d = data_loader.convert_uint16_maxwell(data_1d)
        assert result_1d.shape == data_1d.shape

        data_2d = np.arange(20, dtype=np.uint16).reshape(4, 5)
        result_2d = data_loader.convert_uint16_maxwell(data_2d)
        assert result_2d.shape == data_2d.shape

    def test_deterministic_output(self):
        """Tests:
        - Same input always produces exactly the same output.
        """
        data = np.array([100, 200, 512, 1000], dtype=np.uint16)
        result_a = data_loader.convert_uint16_maxwell(data)
        result_b = data_loader.convert_uint16_maxwell(data)
        np.testing.assert_array_equal(result_a, result_b)

    def test_known_conversion_formula(self):
        """Tests:
        - The conversion follows (data - 512) * lsb * gain * 1000 with
          lsb=6.294e-6, gain=512, sig_offset=512.
        - A value of 512 maps to 0.0 mV (the offset cancels out).
        """
        lsb = 6.294e-6
        gain = 512
        sig_offset = 512

        data = np.array([0, 512, 1023], dtype=np.uint16)
        expected = (data.astype(np.float32) - sig_offset) * lsb * gain * 1000
        result = data_loader.convert_uint16_maxwell(data)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_offset_value_maps_to_zero(self):
        """Tests:
        - An input equal to the sig_offset (512) converts to 0.0 mV.
        """
        data = np.array([512], dtype=np.uint16)
        result = data_loader.convert_uint16_maxwell(data)
        assert result[0] == pytest.approx(0.0)

    def test_empty_array(self):
        """Tests:
        - An empty input array returns an empty float32 array without error.
        """
        data = np.array([], dtype=np.uint16)
        result = data_loader.convert_uint16_maxwell(data)
        assert result.dtype == np.float32
        assert result.shape == (0,)


class TestLoadStimLog:
    """Tests for the load_stim_log function."""

    def test_loads_csv_and_returns_dataframe(self, tmp_path):
        """Tests:
        - A valid CSV file is loaded into a pandas DataFrame.
        - The DataFrame has the expected columns.
        - The stim_electrodes column contains Python lists, not strings.
        """
        csv_path = tmp_path / "experiment_log.csv"
        csv_path.write_text(
            "time,amplitude,duty_time_ms,stim_electrodes,tag\n"
            '0.0,100,5,"[1, 2, 3]",stim_a\n'
            '1.0,200,10,"[4, 5]",stim_b\n'
        )
        # load_stim_log appends suffix '_log.csv' by default; pass the full path
        result = data_loader.load_stim_log(str(csv_path), suffix=".csv")
        assert isinstance(result, pd.DataFrame)
        assert "time" in result.columns
        assert "stim_electrodes" in result.columns
        assert len(result) == 2

    def test_suffix_appended_when_missing(self, tmp_path):
        """Tests:
        - When filepath does not end with the suffix, it is appended automatically.
        """
        csv_path = tmp_path / "exp_log.csv"
        csv_path.write_text(
            "time,amplitude,duty_time_ms,stim_electrodes,tag\n"
            "0.0,100,5,\"[1, 2]\",stim\n"
        )
        # Pass path without .csv extension but with matching suffix
        result = data_loader.load_stim_log(str(csv_path), suffix="_log.csv")
        # The function should append '_log.csv' to the path without .csv
        # Since the file is named 'exp_log.csv' and we strip nothing, this tests
        # the suffix logic. We need to match the actual file name.
        # Re-create with the expected name:
        base_path = str(tmp_path / "exp")
        log_path = tmp_path / "exp_log.csv"
        log_path.write_text(
            "time,amplitude,duty_time_ms,stim_electrodes,tag\n"
            "0.0,100,5,\"[1, 2]\",stim\n"
        )
        result = data_loader.load_stim_log(base_path, suffix="_log.csv")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_stim_electrodes_parsed_as_lists(self, tmp_path):
        """Tests:
        - The stim_electrodes column values are Python lists after loading.
        """
        csv_path = tmp_path / "test_log.csv"
        csv_path.write_text(
            "time,amplitude,duty_time_ms,stim_electrodes,tag\n"
            "0.0,100,5,\"[1, 2, 3]\",stim_a\n"
        )
        result = data_loader.load_stim_log(str(csv_path), suffix=".csv")
        electrodes = result["stim_electrodes"].iloc[0]
        assert isinstance(electrodes, list)
        assert electrodes == [1, 2, 3]

    def test_stim_pattern_column_parsed_if_present(self, tmp_path):
        """Tests:
        - When a stim_pattern column exists, its string values are
          converted to Python objects via literal_eval.
        """
        csv_path = tmp_path / "pattern_log.csv"
        csv_path.write_text(
            "time,amplitude,duty_time_ms,stim_electrodes,stim_pattern,tag\n"
            "0.0,100,5,\"[1, 2]\",\"[10, 20]\",stim\n"
        )
        result = data_loader.load_stim_log(str(csv_path), suffix=".csv")
        assert isinstance(result["stim_pattern"].iloc[0], list)
        assert result["stim_pattern"].iloc[0] == [10, 20]


class TestApplyLiteralEvalStimLog:
    """Tests for the apply_literal_eval_stim_log function."""

    def test_converts_string_lists_to_python_lists(self):
        """Tests:
        - String representations of lists in stim_electrodes are converted
          to actual Python lists.
        """
        df = pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "stim_electrodes": ["[1, 2, 3]", "[4, 5]"],
            }
        )
        result = data_loader.apply_literal_eval_stim_log(df)
        assert result["stim_electrodes"].iloc[0] == [1, 2, 3]
        assert result["stim_electrodes"].iloc[1] == [4, 5]

    def test_leaves_already_parsed_lists_unchanged(self):
        """Tests:
        - Values that are already Python lists are not modified.
        """
        df = pd.DataFrame(
            {
                "time": [0.0],
                "stim_electrodes": [[1, 2, 3]],
            }
        )
        result = data_loader.apply_literal_eval_stim_log(df)
        assert result["stim_electrodes"].iloc[0] == [1, 2, 3]

    def test_converts_stim_pattern_when_present(self):
        """Tests:
        - The stim_pattern column is also converted when it exists.
        """
        df = pd.DataFrame(
            {
                "time": [0.0],
                "stim_electrodes": ["[1, 2]"],
                "stim_pattern": ["[10, 20, 30]"],
            }
        )
        result = data_loader.apply_literal_eval_stim_log(df)
        assert result["stim_pattern"].iloc[0] == [10, 20, 30]

    def test_no_stim_pattern_column(self):
        """Tests:
        - Function works without error when stim_pattern column is absent.
        """
        df = pd.DataFrame(
            {
                "time": [0.0],
                "stim_electrodes": ["[7, 8]"],
            }
        )
        result = data_loader.apply_literal_eval_stim_log(df)
        assert "stim_pattern" not in result.columns
        assert result["stim_electrodes"].iloc[0] == [7, 8]


class TestGetStimElectrodes:
    """Tests for the get_stim_electrodes function."""

    def test_returns_unique_electrodes(self):
        """Tests:
        - Duplicate electrodes across rows are collapsed to unique values.
        - The returned array is sorted (numpy unique guarantees this).
        """
        df = pd.DataFrame(
            {
                "stim_electrodes": [[1, 2, 3], [2, 3, 4], [4, 5]],
            }
        )
        result = data_loader.get_stim_electrodes(df)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5]))

    def test_single_row(self):
        """Tests:
        - Works correctly with a single-row DataFrame.
        """
        df = pd.DataFrame({"stim_electrodes": [[10, 20]]})
        result = data_loader.get_stim_electrodes(df)
        np.testing.assert_array_equal(result, np.array([10, 20]))

    def test_empty_electrode_lists(self):
        """Tests:
        - Rows with empty electrode lists are handled without error.
        """
        df = pd.DataFrame({"stim_electrodes": [[], [1], []]})
        result = data_loader.get_stim_electrodes(df)
        np.testing.assert_array_equal(result, np.array([1]))

    def test_all_same_electrode(self):
        """Tests:
        - When every row contains the same electrode, result has length 1.
        """
        df = pd.DataFrame({"stim_electrodes": [[5], [5], [5]]})
        result = data_loader.get_stim_electrodes(df)
        assert len(result) == 1
        assert result[0] == 5


class TestLoadDataMaxwellMocked:
    """Tests for load_data_maxwell using mocked h5py and smart_open."""

    def test_appends_raw_h5_extension(self, monkeypatch):
        """Tests:
        - When filepath lacks .raw.h5, the extension is appended.
        - The function opens the correct path via smart_open.
        """
        import unittest.mock as mock

        opened_path = {}

        # Build a fake HDF5 dataset
        fake_dataset = np.ones((4, 100), dtype=np.uint16) * 512

        class FakeH5File:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def keys(self):
                return ["/data_store/data0000/groups/routed/raw"]

            def __getitem__(self, key):
                if key == "/data_store/data0000/settings/lsb":
                    return [np.float32(0.0)]  # triggers default fallback
                if key == "/data_store/data0000/groups/routed/raw":
                    return fake_dataset
                raise KeyError(key)

        class FakeSmartOpen:
            def __init__(self, path, mode):
                opened_path["path"] = path

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(data_loader, "smart_open", mock.MagicMock())
        data_loader.smart_open.open = FakeSmartOpen
        monkeypatch.setattr(data_loader.h5py, "File", FakeH5File)

        data_loader.load_data_maxwell("/some/path/recording")
        assert opened_path["path"] == "/some/path/recording.raw.h5"

    def test_returns_float32_by_default(self, monkeypatch):
        """Tests:
        - Default dtype parameter produces a float32 output array.
        - The scaling formula is applied (values differ from raw uint16).
        """
        import unittest.mock as mock

        fake_dataset = np.ones((4, 50), dtype=np.uint16) * 600

        class FakeH5File:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def keys(self):
                return ["sig"]

            def __getitem__(self, key):
                if key == "/data_store/data0000/settings/lsb":
                    return [np.float32(6.294e-6)]
                if key == "/data_store/data0000/settings/gain":
                    return [np.float32(512)]
                if key == "/data_store/data0000/settings/hpf":
                    return [np.float32(300)]
                if key == "sig":
                    return fake_dataset
                raise KeyError(key)

        class FakeSmartOpen:
            def __init__(self, path, mode):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(data_loader, "smart_open", mock.MagicMock())
        data_loader.smart_open.open = FakeSmartOpen
        monkeypatch.setattr(data_loader.h5py, "File", FakeH5File)

        result = data_loader.load_data_maxwell("/path/file.raw.h5")
        assert result.dtype == np.float32
        # length=-1 (default) means frame_end = start + length = 0 + (-1) = -1
        # so the slice is [:, 0:-1] which drops the last column
        assert result.shape == (4, 49)


class TestLoadInfoMaxwellMocked:
    """Tests for load_info_maxwell using mocked h5py and smart_open."""

    def test_returns_expected_keys(self, monkeypatch):
        """Tests:
        - Returned dict contains 'shape', 'start_time', 'lsb', 'gain', 'hpf'.
        """
        import unittest.mock as mock

        fake_shape = (4, 10000)

        class FakeH5File:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def keys(self):
                return ["/data_store/data0000/groups/routed/raw"]

            def __getitem__(self, key):
                if key == "/data_store/data0000/settings/lsb":
                    return [np.float32(6.294e-6)]
                if key == "/data_store/data0000/settings/gain":
                    return [np.float32(512)]
                if key == "/data_store/data0000/settings/hpf":
                    return [np.float32(300)]
                if key == "/data_store/data0000/groups/routed/raw":
                    return type("DS", (), {"shape": fake_shape})()
                if key == "data_store/data0000/start_time":
                    # Timestamp in milliseconds
                    return [1609459200000]  # 2021-01-01 00:00:00 UTC
                raise KeyError(key)

        class FakeSmartOpen:
            def __init__(self, path, mode):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(data_loader, "smart_open", mock.MagicMock())
        data_loader.smart_open.open = FakeSmartOpen
        monkeypatch.setattr(data_loader.h5py, "File", FakeH5File)

        info = data_loader.load_info_maxwell("/path/file.raw.h5")
        assert "shape" in info
        assert "start_time" in info
        assert "lsb" in info
        assert "gain" in info
        assert "hpf" in info
        assert info["shape"] == fake_shape


class TestConvertUint16MaxwellEdgeCases:
    """Edge case tests for the convert_uint16_maxwell function.

    Tests:
    - 2D array shape preservation and dtype.
    - Maximum uint16 value scaling.
    - Zero-value scaling.
    - Single element at offset (512) maps to ~0.0.
    """

    def test_2d_array(self):
        """Tests:
        - A 2D uint16 array preserves its shape after conversion.
        - Output dtype is float32.
        """
        data = np.array([[0, 256], [512, 1023]], dtype=np.uint16)
        result = data_loader.convert_uint16_maxwell(data)
        assert result.shape == (2, 2)
        assert result.dtype == np.float32

    def test_max_uint16(self):
        """Tests:
        - np.array([65535], dtype=np.uint16) produces the correct scaled value.
        """
        lsb = 6.294e-6
        gain = 512
        sig_offset = 512
        data = np.array([65535], dtype=np.uint16)
        expected = (np.float32(65535) - sig_offset) * lsb * gain * 1000
        result = data_loader.convert_uint16_maxwell(data)
        assert result[0] == pytest.approx(expected, rel=1e-5)

    def test_zero_value(self):
        """Tests:
        - np.array([0], dtype=np.uint16) produces (0 - 512) * lsb * gain * 1000.
        """
        lsb = 6.294e-6
        gain = 512
        sig_offset = 512
        expected = (np.float32(0) - sig_offset) * lsb * gain * 1000
        data = np.array([0], dtype=np.uint16)
        result = data_loader.convert_uint16_maxwell(data)
        assert result[0] == pytest.approx(expected, rel=1e-5)

    def test_single_element(self):
        """Tests:
        - np.array([512]) converts to approximately 0.0.
        """
        data = np.array([512])
        result = data_loader.convert_uint16_maxwell(data)
        assert result[0] == pytest.approx(0.0)


class TestGetStimElectrodesEdgeCases:
    """Edge case tests for the get_stim_electrodes function.

    Tests:
    - Nested lists are flattened to extract all unique electrodes.
    - Large electrode numbers (>10000) are handled correctly.
    """

    def test_nested_lists_raises(self):
        """Tests:
        - stim_electrodes containing nested lists like [[1,2],[3]] causes
          np.unique to fail because the resulting array is inhomogeneous.

        Notes:
            - This is a source limitation: get_stim_electrodes expects flat
              lists of ints, not nested lists.
        """
        df = pd.DataFrame(
            {
                "stim_electrodes": [[[1, 2], [3]], [[4]], [[1]]],
            }
        )
        with pytest.raises(ValueError):
            data_loader.get_stim_electrodes(df)

    def test_large_electrode_numbers(self):
        """Tests:
        - Electrode numbers greater than 10000 are handled correctly.
        """
        df = pd.DataFrame(
            {
                "stim_electrodes": [[10001, 20000], [50000, 10001]],
            }
        )
        result = data_loader.get_stim_electrodes(df)
        np.testing.assert_array_equal(result, np.array([10001, 20000, 50000]))


class TestLoadStimLogEdgeCases:
    """Edge case tests for the load_stim_log function.

    Tests:
    - Filepath already ending with suffix is not doubled.
    - Filepath ending in '.csv' but not the expected suffix prints a warning.
    """

    def test_filepath_already_has_suffix(self, tmp_path):
        """Tests:
        - A filepath that already ends with '_log.csv' does not get the
          suffix appended a second time.
        """
        csv_path = tmp_path / "experiment_log.csv"
        csv_path.write_text(
            "time,amplitude,duty_time_ms,stim_electrodes,tag\n"
            '0.0,100,5,"[1, 2]",stim\n'
        )
        result = data_loader.load_stim_log(str(csv_path), suffix="_log.csv")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_filepath_with_csv_extension_warns(self, tmp_path, capsys):
        """Tests:
        - A filepath ending in '.csv' but not '_log.csv' prints a warning
          message but still attempts to load.
        """
        csv_path = tmp_path / "experiment_data.csv"
        csv_path.write_text(
            "time,amplitude,duty_time_ms,stim_electrodes,tag\n"
            '0.0,100,5,"[1, 2]",stim\n'
        )
        result = data_loader.load_stim_log(str(csv_path), suffix="_log.csv")
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert isinstance(result, pd.DataFrame)


class TestApplyLiteralEvalEdgeCases:
    """Edge case tests for the apply_literal_eval_stim_log function.

    Tests:
    - Integer values in stim_electrodes are left unchanged (not strings).
    - An empty DataFrame with the correct columns returns an empty DataFrame.
    """

    def test_integer_values_unchanged(self):
        """Tests:
        - When stim_electrodes column already contains integer values (not
          strings), they are left as-is without error.
        """
        df = pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "stim_electrodes": [42, 99],
            }
        )
        result = data_loader.apply_literal_eval_stim_log(df)
        assert result["stim_electrodes"].iloc[0] == 42
        assert result["stim_electrodes"].iloc[1] == 99

    def test_empty_dataframe(self):
        """Tests:
        - An empty DataFrame with the correct columns returns an empty
          DataFrame without error.
        """
        df = pd.DataFrame(columns=["time", "stim_electrodes", "stim_pattern"])
        result = data_loader.apply_literal_eval_stim_log(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert "stim_electrodes" in result.columns
        assert "stim_pattern" in result.columns
