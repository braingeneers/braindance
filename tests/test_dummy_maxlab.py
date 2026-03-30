"""Tests for braindance.core.dummy_maxlab mock MaxLab SDK."""

import braindance.core.dummy_maxlab as maxlab
from braindance.core.dummy_maxlab import (
    Sequence,
    Core,
    Amplifier,
    unit,
    chip,
    saving,
    system,
    util,
)


class TestSequence:
    """Tests for the Sequence class.

    Tests:
    - Creating a Sequence, appending items, and calling send succeeds.
    - append does not return a value (returns None).
    - send returns the internal list.
    - Supports len, indexing, iteration, and str.
    """

    def test_create_append_send(self):
        """Tests: Sequence can be created, have items appended, and send without errors."""
        seq = Sequence()
        seq.append("item1")
        seq.append("item2")
        result = seq.send()
        assert result == ["item1", "item2"]

    def test_append_returns_none(self):
        """Tests: append returns None."""
        seq = Sequence()
        result = seq.append("x")
        assert result is None

    def test_send_returns_list(self):
        """Tests: send returns the accumulated sequence list."""
        seq = Sequence()
        seq.append("a")
        assert seq.send() == ["a"]

    def test_len_and_indexing(self):
        """Tests: Sequence supports __len__ and __getitem__."""
        seq = Sequence()
        seq.append("first")
        seq.append("second")
        assert len(seq) == 2
        assert seq[0] == "first"
        assert seq[1] == "second"

    def test_iteration(self):
        """Tests: Sequence supports iteration."""
        seq = Sequence()
        seq.append(1)
        seq.append(2)
        assert list(seq) == [1, 2]

    def test_str(self):
        """Tests: Sequence supports str conversion."""
        seq = Sequence()
        seq.append("val")
        assert str(seq) == "['val']"

    def test_empty_send(self):
        """Tests: Sending an empty sequence returns an empty list."""
        seq = Sequence()
        assert seq.send() == []


class TestCore:
    """Tests for the top-level Core class.

    Tests:
    - enable_stimulation_power is callable and returns a string.
    """

    def test_enable_stimulation_power(self):
        """Tests: enable_stimulation_power returns a confirmation string."""
        core = Core()
        result = core.enable_stimulation_power(True)
        assert result == "Stimulation Power Set"


class TestAmplifier:
    """Tests for the top-level Amplifier class.

    Tests:
    - set_gain is callable and returns a string.
    """

    def test_set_gain(self):
        """Tests: set_gain returns a confirmation string."""
        amp = Amplifier()
        result = amp.set_gain(512)
        assert result == "Gain Set"


class TestUnit:
    """Tests for the unit class.

    Tests:
    - All methods return self for chaining.
    - power_up, connect, set_voltage_mode, dac_source are all callable.
    """

    def test_power_up_returns_self(self):
        """Tests: power_up returns self."""
        u = unit()
        assert u.power_up(True) is u

    def test_connect_returns_self(self):
        """Tests: connect returns self."""
        u = unit()
        assert u.connect(True) is u

    def test_set_voltage_mode_returns_self(self):
        """Tests: set_voltage_mode returns self."""
        u = unit()
        assert u.set_voltage_mode() is u

    def test_dac_source_returns_self(self):
        """Tests: dac_source returns self."""
        u = unit()
        assert u.dac_source(0) is u

    def test_method_chaining(self):
        """Tests: All unit methods can be chained together."""
        u = unit()
        result = u.power_up(True).connect(True).set_voltage_mode().dac_source(0)
        assert result is u


class TestChip:
    """Tests for the chip class and its nested classes.

    Tests:
    - DAC, DelaySamples, StimulationUnit return values.
    - chip.Array methods are all callable without errors.
    - query_stimulation_at_electrode returns a truthy value (1).
    - chip.Core and chip.Amplifier static methods work.
    - chip.send and chip.send_raw are callable.
    """

    def test_dac_returns_string(self):
        """Tests: DAC returns a formatted string."""
        result = chip.DAC(0, 512)
        assert result == "DAC,0,512"

    def test_delay_samples_returns_string(self):
        """Tests: DelaySamples returns a formatted string."""
        result = chip.DelaySamples(100)
        assert result == "DelaySamples,100"

    def test_stimulation_unit_returns_unit(self):
        """Tests: StimulationUnit returns a unit instance."""
        result = chip.StimulationUnit(0)
        assert isinstance(result, unit)

    def test_array_select_stimulation_electrodes(self):
        """Tests: Array.select_stimulation_electrodes is callable."""
        arr = chip.Array("test")
        result = arr.select_stimulation_electrodes([1, 2, 3])
        assert result == "Electrodes Selected"

    def test_array_connect_electrode_to_stimulation(self):
        """Tests: Array.connect_electrode_to_stimulation is callable."""
        arr = chip.Array("test")
        result = arr.connect_electrode_to_stimulation(5)
        assert result == "Electrode Connected"

    def test_array_query_stimulation_at_electrode_returns_truthy(self):
        """Tests: query_stimulation_at_electrode returns 1 (truthy)."""
        arr = chip.Array("test")
        result = arr.query_stimulation_at_electrode(5)
        assert result == 1
        assert result  # truthy

    def test_array_load_config(self):
        """Tests: Array.load_config is callable."""
        arr = chip.Array("test")
        result = arr.load_config("some_config")
        assert result == "Config Loaded"

    def test_array_reset(self):
        """Tests: Array.reset is callable."""
        arr = chip.Array("test")
        assert arr.reset() == "Reset Done"

    def test_array_download(self):
        """Tests: Array.download is callable."""
        arr = chip.Array("test")
        assert arr.download() == "Download Done"

    def test_chip_core_enable_stimulation_power(self):
        """Tests: chip.Core.enable_stimulation_power is callable."""
        result = chip.Core.enable_stimulation_power(True)
        assert result == "Stimulation Power Set"

    def test_chip_amplifier_set_gain(self):
        """Tests: chip.Amplifier.set_gain is callable."""
        result = chip.Amplifier.set_gain(1024)
        assert result == "Gain Set"

    def test_chip_send(self):
        """Tests: chip.send is callable and returns the value."""
        assert chip.send("test_val") == "test_val"

    def test_chip_send_raw(self):
        """Tests: chip.send_raw is callable and returns the value."""
        assert chip.send_raw("raw_msg") == "raw_msg"


class TestSaving:
    """Tests for the saving.Saving class.

    Tests:
    - Full lifecycle: open_directory -> set_legacy_format -> group_define ->
      start_file -> start_recording -> stop_recording -> stop_file.
    - group_delete_all is callable.
    """

    def test_full_lifecycle(self):
        """Tests: Complete recording lifecycle runs without errors."""
        s = saving.Saving()
        assert s.open_directory("/tmp/data") == "Directory Opened"
        assert s.set_legacy_format(False) == "Format Set"
        assert s.group_delete_all() == "Groups Deleted"
        assert s.group_define(0, "group0") == "Group Defined"
        assert s.start_file("recording001") == "File Started"
        assert s.start_recording([0]) == "Recording Started"
        assert s.stop_recording() == "Recording Stopped"
        assert s.stop_file() == "File Stopped"

    def test_group_delete_all(self):
        """Tests: group_delete_all is callable independently."""
        s = saving.Saving()
        result = s.group_delete_all()
        assert result == "Groups Deleted"


class TestSystem:
    """Tests for the system class.

    Tests:
    - DelaySamples is callable and returns the value.
    - Event is callable and returns args as a tuple.
    """

    def test_delay_samples(self):
        """Tests: system.DelaySamples returns the input value."""
        result = system.DelaySamples(200)
        assert result == 200

    def test_event(self):
        """Tests: system.Event returns args as a tuple."""
        result = system.Event("a", "b", "c", "d")
        assert result == ("a", "b", "c", "d")


class TestUtil:
    """Tests for the util class.

    Tests:
    - offset, initialize, set_gain, hpf are all callable without errors.
    """

    def test_offset(self):
        """Tests: util.offset is callable without errors."""
        result = util.offset()
        assert result is None

    def test_initialize(self):
        """Tests: util.initialize is callable without errors."""
        result = util.initialize()
        assert result is None

    def test_set_gain(self):
        """Tests: util.set_gain is callable without errors."""
        result = util.set_gain(512)
        assert result is None

    def test_hpf(self):
        """Tests: util.hpf is callable without errors."""
        result = util.hpf()
        assert result is None


class TestModuleLevelFunctions:
    """Tests for module-level send and send_raw functions.

    Tests:
    - maxlab.send returns the value passed to it.
    - maxlab.send_raw returns the value passed to it.
    """

    def test_send(self):
        """Tests: Module-level send returns the input value."""
        result = maxlab.send("hello")
        assert result == "hello"

    def test_send_raw(self):
        """Tests: Module-level send_raw returns the input value."""
        result = maxlab.send_raw("raw_hello")
        assert result == "raw_hello"
