"""Tests for Config class and module-level parsing functions in maxwell_env."""

import array
import struct

import pytest

from braindance.core.maxwell_env import (
    Config,
    SpikeEvent,
    _spike_struct,
    _spike_struct_size,
    parse_events_list,
    parse_frame,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfigParsing:
    """Tests for Config.__init__ parsing logic."""

    def test_parses_channels_electrodes_and_coordinates(self, sample_config_file):
        """Tests:
        - Config file is parsed into the expected (channel, electrode, x, y)
          tuples stored in config attribute.
        """
        cfg = Config(sample_config_file)
        assert cfg.config == [
            (0, 100, 1.0, 2.0),
            (1, 101, 3.0, 4.0),
            (2, 102, 5.0, 6.0),
        ]

    def test_mappings_created_for_each_entry(self, sample_config_file):
        """Tests:
        - A Mapping object is created for every semicolon-delimited entry.
        """
        cfg = Config(sample_config_file)
        assert len(cfg.mappings) == 3


class TestConfigGetChannels:
    """Tests for Config.get_channels."""

    def test_returns_channel_list(self, sample_config_file):
        """Tests:
        - get_channels returns the channel value from each mapping in order.
        """
        cfg = Config(sample_config_file)
        assert cfg.get_channels() == [0, 1, 2]


class TestConfigGetElectrodes:
    """Tests for Config.get_electrodes."""

    def test_returns_electrode_list(self, sample_config_file):
        """Tests:
        - get_electrodes returns the electrode value from each mapping in order.
        """
        cfg = Config(sample_config_file)
        assert cfg.get_electrodes() == [100, 101, 102]


class TestConfigGetChannelsForElectrodes:
    """Tests for Config.get_channels_for_electrodes."""

    def test_filters_to_requested_electrodes(self, sample_config_file):
        """Tests:
        - Only channels whose electrode is in the given subset are returned.
        """
        cfg = Config(sample_config_file)
        assert cfg.get_channels_for_electrodes([100, 102]) == [0, 2]


class TestConfigGetNumChannels:
    """Tests for Config.get_num_channels."""

    def test_returns_correct_count(self, sample_config_file):
        """Tests:
        - get_num_channels returns the total number of channel/electrode pairs.
        """
        cfg = Config(sample_config_file)
        assert cfg.get_num_channels() == 3


class TestConfigNoneFilename:
    """Tests for Config initialised with filename=None."""

    def test_none_filename_does_not_crash(self):
        """Tests:
        - Passing None as the filename produces an empty Config without raising.
        """
        cfg = Config(None)
        assert cfg.config == []
        assert cfg.mappings == []
        assert cfg.get_channels() == []
        assert cfg.get_num_channels() == 0


class TestConfigMapping:
    """Tests for the Config.Mapping inner class."""

    def test_stores_correct_types(self):
        """Tests:
        - Mapping converts its arguments to int (channel, electrode) and
          float (x, y) regardless of input string types.
        """
        m = Config.Mapping("3", "200", "5.5", "6.6")
        assert m.channel == 3 and isinstance(m.channel, int)
        assert m.electrode == 200 and isinstance(m.electrode, int)
        assert m.x == 5.5 and isinstance(m.x, float)
        assert m.y == 6.6 and isinstance(m.y, float)


# ---------------------------------------------------------------------------
# parse_events_list
# ---------------------------------------------------------------------------


class TestParseEventsListNone:
    """Tests for parse_events_list with None input."""

    def test_none_returns_empty_list(self):
        """Tests:
        - None input produces an empty list with no errors.
        """
        assert parse_events_list(None) == []


class TestParseEventsListEmpty:
    """Tests for parse_events_list with empty bytes."""

    def test_empty_bytes_returns_empty_list(self):
        """Tests:
        - An empty bytes object produces an empty list.
        """
        assert parse_events_list(b"") == []


class TestParseEventsListValid:
    """Tests for parse_events_list with valid packed spike events."""

    @pytest.mark.xfail(
        reason=(
            "Struct format '@Lfhc0L' unpacks 4 values (frame, channel, "
            "amplitude, padding-char) but SpikeEvent namedtuple only has "
            "3 fields — causes TypeError. Tracked for fix."
        ),
        strict=True,
    )
    def test_valid_events_parsed_correctly(self):
        """Tests:
        - Correctly packed binary data is decoded into SpikeEvent objects
          with the expected frame, channel, and amplitude values.
        """
        frame, channel_f, amplitude = 42, 1.5, 7
        raw = struct.pack(_spike_struct, frame, channel_f, amplitude, b"\x00")
        events = parse_events_list(raw)
        assert len(events) == 1
        ev = events[0]
        assert ev.frame == frame
        assert abs(ev.channel - channel_f) < 1e-5
        assert ev.amplitude == amplitude


# ---------------------------------------------------------------------------
# parse_frame
# ---------------------------------------------------------------------------


class TestParseFrameNone:
    """Tests for parse_frame with None input."""

    def test_none_returns_none(self):
        """Tests:
        - None input returns None.
        """
        assert parse_frame(None) is None


class TestParseFrameValid:
    """Tests for parse_frame with valid binary float data."""

    def test_valid_float_data_parsed(self):
        """Tests:
        - Binary-packed float data is converted to an array.array of floats
          with correct values.
        """
        values = [1.0, 2.0, 3.0]
        raw = struct.pack(f"{len(values)}f", *values)
        result = parse_frame(raw)
        assert isinstance(result, array.array)
        assert result.typecode == "f"
        assert list(result) == pytest.approx(values)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


class TestConfigEdgeCases:
    """Edge-case tests for Config parsing and query methods.

    Tests:
    - Single-entry config file produces one mapping.
    - Querying channels for a non-existent electrode returns empty list.
    - Querying channels for an empty electrode list returns empty list.
    - Config(None) leaves config and mappings as empty lists.
    - Negative coordinate values are parsed correctly.
    """

    def test_single_entry_config(self, tmp_path):
        """Tests:
        - A config file with exactly one mapping entry produces one channel
          and one electrode.
        """
        cfg_path = tmp_path / "single.cfg"
        cfg_path.write_text("0/100/1.5/2.5;")
        cfg = Config(str(cfg_path))
        assert cfg.get_channels() == [0]
        assert cfg.get_electrodes() == [100]
        assert cfg.config == [(0, 100, 1.5, 2.5)]
        assert len(cfg.mappings) == 1

    def test_get_channels_for_nonexistent_electrode(self, sample_config_file):
        """Tests:
        - Requesting channels for an electrode ID that does not exist in the
          config returns an empty list.
        """
        cfg = Config(sample_config_file)
        assert cfg.get_channels_for_electrodes([999]) == []

    def test_get_channels_for_empty_list(self, sample_config_file):
        """Tests:
        - Requesting channels for an empty electrode list returns an empty list.
        """
        cfg = Config(sample_config_file)
        assert cfg.get_channels_for_electrodes([]) == []

    def test_none_config_has_empty_lists(self):
        """Tests:
        - Config(None) sets config and mappings to empty lists without raising.
        """
        cfg = Config(None)
        assert cfg.config == []
        assert cfg.mappings == []

    def test_negative_coordinates(self, tmp_path):
        """Tests:
        - Negative x/y coordinate values are parsed correctly into Mapping
          objects and config tuples.
        """
        cfg_path = tmp_path / "negative.cfg"
        cfg_path.write_text("5/200/-1.0/-2.5;")
        cfg = Config(str(cfg_path))
        assert cfg.config == [(5, 200, -1.0, -2.5)]
        m = cfg.mappings[0]
        assert m.x == -1.0
        assert m.y == -2.5


class TestParseFrameEdgeCases:
    """Edge-case tests for parse_frame.

    Tests:
    - Empty bytes input produces an empty array.
    - A single 4-byte float is parsed into a one-element array.
    """

    def test_empty_bytes(self):
        """Tests:
        - parse_frame(b'') returns an empty float array rather than None.
        """
        result = parse_frame(b"")
        assert isinstance(result, array.array)
        assert len(result) == 0

    def test_single_float(self):
        """Tests:
        - Four bytes encoding a single float are correctly decoded.
        """
        raw = struct.pack("f", 3.14)
        result = parse_frame(raw)
        assert isinstance(result, array.array)
        assert len(result) == 1
        assert abs(result[0] - 3.14) < 1e-5


class TestParseEventsListEdgeCases:
    """Edge-case tests for parse_events_list.

    Tests:
    - Partial (misaligned) event data prints a warning but does not raise.
    """

    @pytest.mark.xfail(
        reason="Source bug: struct format '@Lfhc0L' unpacks 4 values but SpikeEvent has 3 fields",
        strict=True,
    )
    def test_partial_event_data(self):
        """Tests:
        - When events_data length is not divisible by _spike_struct_size, the
          function prints a warning and does not raise an exception.

        Notes:
            - This test is xfail because parse_events_list crashes on ANY valid
              data due to a mismatch between the struct format (4 values) and
              SpikeEvent namedtuple (3 fields).
        """
        partial = b"\x00" * (_spike_struct_size + 1)
        result = parse_events_list(partial)
        assert isinstance(result, list)
