"""Tests for braindance.core.phases module."""
import time

import pytest
import numpy as np

from braindance.core.phases import (
    Phase,
    PhaseManager,
    RecordPhase,
    NeuralSweepPhase,
    FrequencyStimPhase,
)


# ---------------------------------------------------------------------------
# Phase (base class)
# ---------------------------------------------------------------------------


class TestPhaseInit:
    """Tests for Phase.__init__."""

    def test_sets_env(self, mock_env):
        """Tests:
        - env attribute is set to the provided environment.
        """
        phase = Phase(mock_env)
        assert phase.env is mock_env

    def test_sets_start_time(self, mock_env):
        """Tests:
        - start_time is set to a recent perf_counter value.
        """
        before = time.perf_counter()
        phase = Phase(mock_env)
        after = time.perf_counter()
        assert before <= phase.start_time <= after


class TestPhaseAbstractMethods:
    """Tests for Phase abstract method stubs."""

    def test_run_raises_not_implemented(self, mock_env):
        """Tests:
        - run() raises NotImplementedError on the base class.
        """
        phase = Phase(mock_env)
        with pytest.raises(NotImplementedError):
            phase.run()

    def test_predicted_time_raises_not_implemented(self, mock_env):
        """Tests:
        - predicted_time() raises NotImplementedError on the base class.
        """
        phase = Phase(mock_env)
        with pytest.raises(NotImplementedError):
            phase.predicted_time()

    def test_info_raises_not_implemented(self, mock_env):
        """Tests:
        - info() raises NotImplementedError on the base class.
        """
        phase = Phase(mock_env)
        with pytest.raises(NotImplementedError):
            phase.info()


class TestPhaseTimeElapsed:
    """Tests for Phase.time_elapsed."""

    def test_returns_positive_elapsed(self, mock_env):
        """Tests:
        - time_elapsed() returns a non-negative value after construction.
        """
        phase = Phase(mock_env)
        elapsed = phase.time_elapsed()
        assert elapsed >= 0


# ---------------------------------------------------------------------------
# PhaseManager
# ---------------------------------------------------------------------------


class TestPhaseManagerInit:
    """Tests for PhaseManager.__init__."""

    def test_initializes_empty_phases(self, mock_env):
        """Tests:
        - phases list is initially empty.
        - filenames list is initially empty.
        - save_dir is taken from env.
        """
        pm = PhaseManager(mock_env)
        assert pm.phases == []
        assert pm.filenames == []
        assert pm.save_dir == mock_env.save_dir

    def test_verbose_flag(self, mock_env):
        """Tests:
        - verbose flag is stored correctly.
        """
        pm = PhaseManager(mock_env, verbose=True)
        assert pm.verbose is True


class TestPhaseManagerAddPhase:
    """Tests for PhaseManager.add_phase and add_phase_group."""

    def test_add_phase_appends(self, mock_env):
        """Tests:
        - add_phase appends a single phase to the list.
        """
        pm = PhaseManager(mock_env)
        rp = RecordPhase(mock_env, duration=5)
        pm.add_phase(rp)
        assert len(pm.phases) == 1
        assert pm.phases[0] is rp

    def test_add_multiple_phases(self, mock_env):
        """Tests:
        - Multiple add_phase calls append in order.
        """
        pm = PhaseManager(mock_env)
        rp1 = RecordPhase(mock_env, duration=5)
        rp2 = RecordPhase(mock_env, duration=10)
        pm.add_phase(rp1)
        pm.add_phase(rp2)
        assert len(pm.phases) == 2
        assert pm.phases[0] is rp1
        assert pm.phases[1] is rp2

    def test_add_phase_group(self, mock_env):
        """Tests:
        - add_phase_group appends a list as a single entry.
        """
        pm = PhaseManager(mock_env)
        rp1 = RecordPhase(mock_env, duration=5)
        rp2 = RecordPhase(mock_env, duration=10)
        pm.add_phase_group([rp1, rp2])
        assert len(pm.phases) == 1
        assert isinstance(pm.phases[0], list)
        assert len(pm.phases[0]) == 2


class TestPhaseManagerSummary:
    """Tests for PhaseManager.summary."""

    def test_summary_contains_header(self, mock_env):
        """Tests:
        - summary() output contains the 'Phase Summary' header.
        """
        pm = PhaseManager(mock_env)
        rp = RecordPhase(mock_env, duration=30)
        pm.add_phase(rp)
        s = pm.summary()
        assert "Phase Summary" in s
        assert "-------------" in s

    def test_summary_contains_phase_name(self, mock_env):
        """Tests:
        - summary() includes the class name of each added phase.
        """
        pm = PhaseManager(mock_env)
        rp = RecordPhase(mock_env, duration=10)
        pm.add_phase(rp)
        s = pm.summary()
        assert "RecordPhase" in s

    def test_summary_contains_predicted_time(self, mock_env):
        """Tests:
        - summary() includes predicted time information.
        """
        pm = PhaseManager(mock_env)
        rp = RecordPhase(mock_env, duration=10)
        pm.add_phase(rp)
        s = pm.summary()
        assert "Predicted Time" in s

    def test_summary_contains_total_time(self, mock_env):
        """Tests:
        - summary() includes total experiment time line.
        """
        pm = PhaseManager(mock_env)
        rp = RecordPhase(mock_env, duration=120)
        pm.add_phase(rp)
        s = pm.summary()
        assert "Total Experiment Time" in s

    def test_summary_phase_group(self, mock_env):
        """Tests:
        - summary() includes 'Phase Group' label and group total time for groups.
        """
        pm = PhaseManager(mock_env)
        rp1 = RecordPhase(mock_env, duration=60)
        rp2 = RecordPhase(mock_env, duration=60)
        pm.add_phase_group([rp1, rp2])
        s = pm.summary()
        assert "Phase Group" in s
        assert "Group Total Time" in s


class TestPhaseManagerRun:
    """Tests for PhaseManager.run."""

    def test_single_phase_runs(self, mock_env):
        """Tests:
        - A single RecordPhase is run and env.step is called.
        - env.close is called after the phase.
        """
        pm = PhaseManager(mock_env)
        rp = RecordPhase(mock_env, duration=0.01)
        pm.add_phase(rp)
        pm.run()
        assert mock_env.step_count > 0
        assert mock_env.close_count >= 1

    def test_no_reset_for_first_phase(self, mock_env):
        """Tests:
        - env.reset() is NOT called before the first phase.
        """
        pm = PhaseManager(mock_env)
        rp = RecordPhase(mock_env, duration=0.01)
        pm.add_phase(rp)
        pm.run()
        assert mock_env.reset_count == 0

    def test_reset_between_phases(self, mock_env):
        """Tests:
        - env.reset() is called between consecutive phases (but not before the first).
        """
        pm = PhaseManager(mock_env)
        rp1 = RecordPhase(mock_env, duration=0.01)
        rp2 = RecordPhase(mock_env, duration=0.01)
        pm.add_phase(rp1)
        pm.add_phase(rp2)
        pm.run()
        assert mock_env.reset_count == 1

    def test_close_called_for_each_phase(self, mock_env):
        """Tests:
        - env.close() is called after each phase (plus final close in finally block).
        """
        pm = PhaseManager(mock_env)
        rp1 = RecordPhase(mock_env, duration=0.01)
        rp2 = RecordPhase(mock_env, duration=0.01)
        pm.add_phase(rp1)
        pm.add_phase(rp2)
        pm.run()
        # close called once per phase in the loop + once in finally
        assert mock_env.close_count >= 2

    def test_phase_group_runs_sub_phases(self, mock_env):
        """Tests:
        - A phase group runs each sub-phase sequentially.
        """
        pm = PhaseManager(mock_env)
        rp1 = RecordPhase(mock_env, duration=0.01)
        rp2 = RecordPhase(mock_env, duration=0.01)
        pm.add_phase_group([rp1, rp2])
        pm.run()
        assert mock_env.step_count > 0


# ---------------------------------------------------------------------------
# RecordPhase
# ---------------------------------------------------------------------------


class TestRecordPhaseInit:
    """Tests for RecordPhase.__init__."""

    def test_default_duration(self, mock_env):
        """Tests:
        - Default duration is 10.
        - predicted_time equals duration.
        """
        rp = RecordPhase(mock_env)
        assert rp.duration == 10
        assert rp.predicted_time == 10

    def test_custom_duration(self, mock_env):
        """Tests:
        - Custom duration is stored correctly.
        - predicted_time matches custom duration.
        """
        rp = RecordPhase(mock_env, duration=60)
        assert rp.duration == 60
        assert rp.predicted_time == 60


class TestRecordPhaseInfo:
    """Tests for RecordPhase.info."""

    def test_info_returns_duration(self, mock_env):
        """Tests:
        - info() returns dict with 'duration' key matching self.duration.
        """
        rp = RecordPhase(mock_env, duration=42)
        info = rp.info()
        assert info == {"duration": 42}


class TestRecordPhaseRun:
    """Tests for RecordPhase.run."""

    def test_run_calls_env_step(self, mock_env):
        """Tests:
        - run() calls env.step() at least once.
        - run() completes within a short duration.
        """
        rp = RecordPhase(mock_env, duration=0.01)
        rp.run()
        assert mock_env.step_count > 0

    def test_run_respects_duration(self, mock_env):
        """Tests:
        - run() completes in approximately the specified duration.
        """
        rp = RecordPhase(mock_env, duration=0.01)
        start = time.perf_counter()
        rp.run()
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0  # Should finish well under 1 second


# ---------------------------------------------------------------------------
# NeuralSweepPhase
# ---------------------------------------------------------------------------


class TestNeuralSweepPhaseInit:
    """Tests for NeuralSweepPhase.__init__."""

    def test_basic_init(self, mock_env):
        """Tests:
        - Attributes are set from constructor arguments.
        """
        nsp = NeuralSweepPhase(
            mock_env,
            neuron_list=[0, 1],
            amp_bounds=[100, 200, 3],
            stim_freq=2,
            replicates=10,
            phase_length=50,
            order="ran",
        )
        assert nsp.neuron_list == [0, 1]
        assert nsp.amplitude_start == 100
        assert nsp.amplitude_end == 200
        assert nsp.n_amplitudes == 3
        assert nsp.stim_freq == 2
        assert nsp.replicates == 10
        assert nsp.phase_length == 50

    def test_int_amp_bounds(self, mock_env):
        """Tests:
        - Integer amp_bounds is converted to [val, val, 1].
        """
        nsp = NeuralSweepPhase(
            mock_env,
            neuron_list=[0],
            amp_bounds=150,
        )
        assert nsp.amplitude_start == 150
        assert nsp.amplitude_end == 150
        assert nsp.n_amplitudes == 1

    def test_predicted_time_calculation(self, mock_env):
        """Tests:
        - predicted_time = n_amplitudes * len(neuron_list) * (1/stim_freq) * replicates.
        """
        nsp = NeuralSweepPhase(
            mock_env,
            neuron_list=[0, 1],
            amp_bounds=[100, 200, 5],
            stim_freq=2,
            replicates=10,
        )
        expected = 5 * 2 * (1 / 2) * 10  # 50
        assert nsp.predicted_time == expected

    def test_empty_neuron_list_raises(self, mock_env):
        """Tests:
        - Empty neuron_list triggers an AssertionError.
        """
        with pytest.raises(AssertionError, match="at least one neuron"):
            NeuralSweepPhase(mock_env, neuron_list=[], amp_bounds=[100, 200, 3])

    def test_invalid_order_raises(self, mock_env):
        """Tests:
        - Invalid order string triggers an AssertionError.
        """
        with pytest.raises(AssertionError):
            NeuralSweepPhase(
                mock_env,
                neuron_list=[0],
                amp_bounds=[100, 200, 3],
                order="xyz",
            )


class TestNeuralSweepPhaseGenerateStimCommands:
    """Tests for NeuralSweepPhase.generate_stim_commands."""

    def test_ran_order_count(self, mock_env):
        """Tests:
        - Total number of stim commands equals n_amplitudes * n_neurons * replicates.
        """
        nsp = NeuralSweepPhase(
            mock_env,
            neuron_list=[0, 1],
            amp_bounds=[100, 200, 3],
            replicates=5,
            order="ran",
        )
        cmds = nsp.generate_stim_commands()
        expected_count = 3 * 2 * 5  # n_amps * n_neurons * replicates
        assert len(cmds) == expected_count

    def test_command_structure(self, mock_env):
        """Tests:
        - Each command is a tuple of ([neuron], amplitude, phase_length).
        """
        nsp = NeuralSweepPhase(
            mock_env,
            neuron_list=[0],
            amp_bounds=150,
            replicates=2,
            phase_length=100,
            order="ran",
        )
        cmds = nsp.generate_stim_commands()
        for cmd in cmds:
            assert isinstance(cmd, tuple)
            assert len(cmd) == 3
            assert isinstance(cmd[0], list)
            assert cmd[2] == 100

    def test_arn_order_changes_ordering(self, mock_env):
        """Tests:
        - 'arn' order iterates amplitudes in the outermost loop, producing a
          different sequence than 'ran'.
        """
        neurons = [0, 1]
        kwargs = dict(
            neuron_list=neurons,
            amp_bounds=[100, 200, 3],
            replicates=5,
            phase_length=100,
        )
        nsp_ran = NeuralSweepPhase(mock_env, order="ran", **kwargs)
        nsp_arn = NeuralSweepPhase(mock_env, order="arn", **kwargs)
        cmds_ran = nsp_ran.generate_stim_commands()
        cmds_arn = nsp_arn.generate_stim_commands()
        # Same count but different ordering
        assert len(cmds_ran) == len(cmds_arn)
        assert cmds_ran != cmds_arn

    def test_rna_order_count(self, mock_env):
        """Tests:
        - 'rna' order produces the correct total number of commands.
        """
        nsp = NeuralSweepPhase(
            mock_env,
            neuron_list=[0, 1, 2],
            amp_bounds=[100, 300, 3],
            replicates=4,
            order="rna",
        )
        cmds = nsp.generate_stim_commands()
        assert len(cmds) == 3 * 3 * 4

    def test_single_amplitude(self, mock_env):
        """Tests:
        - When amp_bounds start == end with 1 step, all commands use same amplitude.
        """
        nsp = NeuralSweepPhase(
            mock_env,
            neuron_list=[0],
            amp_bounds=[150, 150, 1],
            replicates=3,
            order="ran",
        )
        cmds = nsp.generate_stim_commands()
        assert len(cmds) == 3
        for cmd in cmds:
            assert cmd[1] == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# FrequencyStimPhase
# ---------------------------------------------------------------------------


class TestFrequencyStimPhaseInit:
    """Tests for FrequencyStimPhase.__init__."""

    def test_single_command_mode(self, mock_env):
        """Tests:
        - A single stim command is detected as single_command = True.
        - predicted_time = min(duration, len(stim_command) / stim_freq).
        """
        cmd = ([0], 150, 100)
        fsp = FrequencyStimPhase(
            mock_env,
            stim_command=cmd,
            stim_freq=1,
            duration=10,
        )
        assert fsp.single_command is True
        # predicted_time = min(duration, len(stim_command) / stim_freq)
        # For a single command tuple ([0], 150, 100), len() == 3
        assert fsp.predicted_time == min(10, 3 / 1)

    def test_multi_command_mode(self, mock_env):
        """Tests:
        - A list of stim commands is detected as single_command = False.
        """
        cmds = [([0], 150, 100), ([1], 200, 100)]
        fsp = FrequencyStimPhase(
            mock_env,
            stim_command=cmds,
            stim_freq=1,
            duration=10,
        )
        assert fsp.single_command is False
        assert fsp.predicted_time == min(10, 2 / 1)

    def test_predicted_time_respects_duration(self, mock_env):
        """Tests:
        - When duration is shorter than commands/freq, predicted_time equals duration.
        """
        cmds = [([0], 150, 100)] * 100
        fsp = FrequencyStimPhase(
            mock_env,
            stim_command=cmds,
            stim_freq=1,
            duration=5,
        )
        assert fsp.predicted_time == 5


class TestFrequencyStimPhaseTagValidation:
    """Tests for FrequencyStimPhase tag validation."""

    def test_list_tag_with_single_command_raises(self, mock_env):
        """Tests:
        - A list tag with a single stim command raises ValueError.
        """
        cmd = ([0], 150, 100)
        with pytest.raises(ValueError, match="Tag must be a string"):
            FrequencyStimPhase(
                mock_env,
                stim_command=cmd,
                tag=["tag1", "tag2"],
            )

    def test_tag_length_mismatch_raises(self, mock_env):
        """Tests:
        - Tag list length != stim_command list length raises ValueError.
        """
        cmds = [([0], 150, 100), ([1], 200, 100)]
        with pytest.raises(ValueError, match="same length"):
            FrequencyStimPhase(
                mock_env,
                stim_command=cmds,
                tag=["t1", "t2", "t3"],
            )

    def test_string_tag_always_valid(self, mock_env):
        """Tests:
        - A string tag is accepted for both single and multi command modes.
        """
        cmds = [([0], 150, 100), ([1], 200, 100)]
        fsp = FrequencyStimPhase(
            mock_env,
            stim_command=cmds,
            tag="my_tag",
        )
        assert fsp.single_tag is True

    def test_matching_tag_list_accepted(self, mock_env):
        """Tests:
        - Tag list of the same length as multi stim_command is accepted.
        """
        cmds = [([0], 150, 100), ([1], 200, 100)]
        fsp = FrequencyStimPhase(
            mock_env,
            stim_command=cmds,
            tag=["t1", "t2"],
        )
        assert fsp.single_tag is False


class TestFrequencyStimPhaseInfo:
    """Tests for FrequencyStimPhase.info."""

    def test_info_returns_correct_dict(self, mock_env):
        """Tests:
        - info() returns dict with stim_command, stim_freq, duration, and tag.
        """
        cmd = ([0], 150, 100)
        fsp = FrequencyStimPhase(
            mock_env,
            stim_command=cmd,
            stim_freq=2,
            duration=15,
            tag="my_stim",
        )
        info = fsp.info()
        assert info["stim_freq"] == 2
        assert info["duration"] == 15
        assert info["tag"] == "my_stim"


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


class TestNeuralSweepPhaseEdgeCases:
    """Edge-case tests for NeuralSweepPhase."""

    def test_random_order_shuffles(self, mock_env):
        """Tests:
        - order='random' produces the same set of commands but in a shuffled order.
        - Uses np.random.seed for reproducibility.
        """
        np.random.seed(42)
        nsp_random = NeuralSweepPhase(
            mock_env,
            neuron_list=[0, 1],
            amp_bounds=[100, 200, 3],
            replicates=2,
            order="random",
        )
        random_cmds = nsp_random.generate_stim_commands()

        nsp_ordered = NeuralSweepPhase(
            mock_env,
            neuron_list=[0, 1],
            amp_bounds=[100, 200, 3],
            replicates=2,
            order="ran",
        )
        ordered_cmds = nsp_ordered.generate_stim_commands()

        # Same elements (as sets of tuples), but order may differ
        assert len(random_cmds) == len(ordered_cmds)
        random_tuples = sorted(
            (tuple(c[0]), c[1], c[2]) for c in random_cmds
        )
        ordered_tuples = sorted(
            (tuple(c[0]), c[1], c[2]) for c in ordered_cmds
        )
        assert random_tuples == ordered_tuples

    def test_single_neuron_single_amplitude(self, mock_env):
        """Tests:
        - neuron_list=[0], amp_bounds=150 (int), replicates=3 yields 3 commands.
        """
        nsp = NeuralSweepPhase(
            mock_env,
            neuron_list=[0],
            amp_bounds=150,
            replicates=3,
        )
        cmds = nsp.generate_stim_commands()
        assert len(cmds) == 3
        for cmd in cmds:
            assert cmd[0] == [0]
            assert cmd[1] == 150

    def test_amp_bounds_two_elements(self, mock_env):
        """Tests:
        - amp_bounds with only two elements raises IndexError because
          amp_bounds[2] is accessed during __init__.
        """
        with pytest.raises(IndexError):
            NeuralSweepPhase(
                mock_env,
                neuron_list=[0],
                amp_bounds=[100, 200],
            )

    def test_multiple_amplitudes(self, mock_env):
        """Tests:
        - amp_bounds=[100, 200, 3] produces 3 amplitude steps matching
          np.linspace(100, 200, 3).
        """
        nsp = NeuralSweepPhase(
            mock_env,
            neuron_list=[0],
            amp_bounds=[100, 200, 3],
            replicates=1,
        )
        cmds = nsp.generate_stim_commands()
        expected_amps = np.linspace(100, 200, 3)
        assert len(cmds) == 3
        actual_amps = [cmd[1] for cmd in cmds]
        np.testing.assert_allclose(actual_amps, expected_amps)

    def test_equal_amp_bounds(self, mock_env):
        """Tests:
        - amp_bounds=[150, 150, 1] produces a single amplitude; all commands
          have amplitude 150.
        """
        nsp = NeuralSweepPhase(
            mock_env,
            neuron_list=[0, 1],
            amp_bounds=[150, 150, 1],
            replicates=2,
        )
        cmds = nsp.generate_stim_commands()
        assert len(cmds) == 1 * 2 * 2  # n_amps * neurons * replicates
        for cmd in cmds:
            assert cmd[1] == 150


class TestRecordPhaseEdgeCases:
    """Edge-case tests for RecordPhase."""

    def test_zero_duration(self, mock_env):
        """Tests:
        - duration=0 causes run() to return after the first step, because the
          elapsed time exceeds the zero-second duration immediately.
        """
        rp = RecordPhase(mock_env, duration=0)
        rp.run()
        # The while loop runs at least once; the first step triggers the
        # time_elapsed > 0 check, which sets done = True.
        assert mock_env.step_count == 1


class TestPhaseManagerEdgeCases:
    """Edge-case tests for PhaseManager."""

    def test_run_with_no_phases(self, mock_env):
        """Tests:
        - PhaseManager.run() with no phases does not crash and calls
          env.close() in the finally block.
        """
        pm = PhaseManager(mock_env)
        pm.run()
        # close() is called at least once (in the finally block)
        assert mock_env.close_count >= 1

    def test_run_propagates_exception(self, mock_env):
        """Tests:
        - If a phase raises RuntimeError, PhaseManager.run() calls env.close()
          and re-raises the exception.
        """

        class _FailingPhase(Phase):
            def __init__(self, env):
                super().__init__(env)
                self.predicted_time = 0

            def run(self):
                raise RuntimeError("phase failed")

            def info(self):
                return {}

        pm = PhaseManager(mock_env)
        pm.add_phase(_FailingPhase(mock_env))
        with pytest.raises(RuntimeError, match="phase failed"):
            pm.run()
        # close() called in the except block and the finally block
        assert mock_env.close_count >= 2

    def test_summary_with_no_phases(self, mock_env):
        """Tests:
        - summary() with an empty phase list returns the header and a total
          time of 0m 0s.
        """
        pm = PhaseManager(mock_env)
        s = pm.summary()
        assert "Phase Summary" in s
        assert "Total Experiment Time: 0m 0s" in s


class TestFrequencyStimPhaseEdgeCases:
    """Edge-case tests for FrequencyStimPhase."""

    def test_very_high_frequency(self, mock_env):
        """Tests:
        - stim_freq=1000 does not crash and predicted_time is calculated
          correctly as min(duration, len/freq).
        """
        cmd = ([0], 150, 100)
        fsp = FrequencyStimPhase(
            mock_env,
            stim_command=cmd,
            stim_freq=1000,
            duration=10,
        )
        # single command: len(cmd) == 3, predicted = min(10, 3/1000) = 0.003
        assert fsp.predicted_time == pytest.approx(3 / 1000)

    def test_empty_connect_units(self, mock_env):
        """Tests:
        - connect_units=[] is falsy, so run() skips the disconnect/connect
          logic entirely.
        """
        cmd = ([0], 150, 100)
        fsp = FrequencyStimPhase(
            mock_env,
            stim_command=cmd,
            stim_freq=1,
            duration=0,
            connect_units=[],
        )
        fsp.run()
        # Should have run at least one step without crashing
        assert mock_env.step_count >= 1
