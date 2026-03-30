"""Tests for braindance.core.trainer module."""

import math
import numpy as np
import pytest

from braindance.core.trainer import (
    generate_tetanus_pattern,
    generate_permutations,
    generate_stimulations,
)


class TestGenerateTetanusPattern:
    """Tests for generate_tetanus_pattern."""

    def test_sequential_mode_structure(self):
        """Tests:
        Sequential mode returns correct stim/delay structure with correct neurons.
        """
        neurons = [10, 20, 30, 40, 50]
        result = generate_tetanus_pattern(neurons, stim_count=3)

        # 3 stim tuples + 2 delay tuples = 5 entries
        assert len(result) == 5
        # Stim entries at even indices
        assert result[0] == ("stim", [10], 400, 100)
        assert result[2] == ("stim", [20], 400, 100)
        assert result[4] == ("stim", [30], 400, 100)
        # Delay entries at odd indices
        assert result[1] == ("delay", 5)
        assert result[3] == ("delay", 5)

    def test_random_mode_with_replace(self):
        """Tests:
        Random mode with replace=True returns correct length and stim/delay structure.
        """
        neurons = [1, 2, 3]
        np.random.seed(42)
        result = generate_tetanus_pattern(
            neurons, stim_count=5, random=True, replace=True
        )

        # 5 stim + 4 delay = 9 entries
        assert len(result) == 9
        for i, entry in enumerate(result):
            if i % 2 == 0:
                assert entry[0] == "stim"
                assert entry[2] == 400
                assert entry[3] == 100
            else:
                assert entry[0] == "delay"

    def test_fewer_neurons_than_stim_count(self):
        """Tests:
        When fewer neurons are available than stim_count, only available neurons are used.
        """
        neurons = [10, 20]
        result = generate_tetanus_pattern(neurons, stim_count=5, random=False)

        # Sequential slicing gives only 2 neurons
        stim_entries = [e for e in result if e[0] == "stim"]
        assert len(stim_entries) == 2

    def test_custom_amp_and_pulse_width(self):
        """Tests:
        Custom amp_mv and pulse_width are propagated to all stim tuples.
        """
        neurons = [1, 2, 3]
        result = generate_tetanus_pattern(
            neurons, stim_count=3, amp_mv=800, pulse_width=250
        )

        for entry in result:
            if entry[0] == "stim":
                assert entry[2] == 800
                assert entry[3] == 250

    def test_no_delay_after_last_stim(self):
        """Tests:
        Delay tuples are placed between stims but not after the last stim.
        """
        neurons = [1, 2, 3, 4]
        result = generate_tetanus_pattern(neurons, stim_count=4)

        assert result[-1][0] == "stim"
        delay_count = sum(1 for e in result if e[0] == "delay")
        stim_count = sum(1 for e in result if e[0] == "stim")
        assert delay_count == stim_count - 1


class TestGeneratePermutations:
    """Tests for generate_permutations."""

    def test_correct_number_of_permutations(self):
        """Tests:
        Number of permutations equals n! / (n - stim_count)!.
        """
        neurons = [1, 2, 3, 4]
        stim_count = 3
        result = generate_permutations(neurons, stim_count=stim_count)

        expected_count = math.factorial(len(neurons)) // math.factorial(
            len(neurons) - stim_count
        )
        assert len(result) == expected_count

    def test_each_permutation_has_correct_structure(self):
        """Tests:
        Each permutation contains alternating stim and delay tuples with correct counts.
        """
        neurons = [1, 2, 3]
        stim_count = 2
        result = generate_permutations(neurons, stim_count=stim_count)

        for perm in result:
            stims = [e for e in perm if e[0] == "stim"]
            delays = [e for e in perm if e[0] == "delay"]
            assert len(stims) == stim_count
            assert len(delays) == stim_count - 1
            # Last entry should be a stim
            assert perm[-1][0] == "stim"

    def test_all_permutations_are_unique(self):
        """Tests:
        All generated permutations are unique sequences of neurons.
        """
        neurons = [10, 20, 30]
        stim_count = 3
        result = generate_permutations(neurons, stim_count=stim_count)

        neuron_sequences = []
        for perm in result:
            seq = tuple(e[1][0] for e in perm if e[0] == "stim")
            neuron_sequences.append(seq)

        assert len(neuron_sequences) == len(set(neuron_sequences))


class TestGenerateStimulations:
    """Tests for generate_stimulations."""

    def test_ints_wrapped_in_lists(self):
        """Tests:
        Single integer electrode indices are wrapped in a list.
        """
        result = generate_stimulations([1, 2, 3])

        assert result[0] == ([1], 400, 200)
        assert result[1] == ([2], 400, 200)
        assert result[2] == ([3], 400, 200)

    def test_lists_passed_through(self):
        """Tests:
        Electrode indices already in list form are passed through unchanged.
        """
        result = generate_stimulations([[1, 2], [3, 4]])

        assert result[0] == ([1, 2], 400, 200)
        assert result[1] == ([3, 4], 400, 200)

    def test_custom_amp_and_phase_width(self):
        """Tests:
        Custom amp and phase_width values are propagated to all stim commands.
        """
        result = generate_stimulations([5], amp=600, phase_width=300)

        assert result[0] == ([5], 600, 300)


class TestGenerateTetanusPatternEdgeCases:
    """Edge case tests for generate_tetanus_pattern.

    Tests:
    Boundary conditions including single neuron, zero stim count, zero delay,
    random mode without replacement over all neurons, and empty neuron list.
    """

    def test_single_neuron_single_stim(self):
        """Tests:
        Single neuron with stim_count=1 produces one stim tuple and no delays.
        """
        result = generate_tetanus_pattern([0], stim_count=1)

        assert len(result) == 1
        assert result[0] == ("stim", [0], 400, 100)

    def test_stim_count_zero(self):
        """Tests:
        stim_count=0 produces an empty list (neurons[:0] yields no neurons).
        """
        result = generate_tetanus_pattern([1, 2, 3], stim_count=0)

        assert result == []

    def test_delay_ms_zero(self):
        """Tests:
        delay_ms=0 produces delay tuples with value 0 between stims.
        """
        result = generate_tetanus_pattern([1, 2, 3], stim_count=3, delay_ms=0)

        delays = [e for e in result if e[0] == "delay"]
        assert len(delays) == 2
        for d in delays:
            assert d == ("delay", 0)

    def test_random_no_replace_all_neurons(self):
        """Tests:
        Random mode with replace=False and stim_count=len(neurons) uses all neurons.
        """
        neurons = [10, 20, 30, 40]
        np.random.seed(0)
        result = generate_tetanus_pattern(
            neurons, stim_count=len(neurons), random=True, replace=False
        )

        stim_neurons = [e[1][0] for e in result if e[0] == "stim"]
        assert sorted(stim_neurons) == sorted(neurons)
        assert len(stim_neurons) == len(neurons)

    def test_empty_neurons_sequential(self):
        """Tests:
        Empty neuron list with stim_count=0 produces an empty list.
        """
        result = generate_tetanus_pattern([], stim_count=0)

        assert result == []


class TestGeneratePermutationsEdgeCases:
    """Edge case tests for generate_permutations.

    Tests:
    Boundary conditions including single neuron, stim_count equal to neuron count,
    and stim_count of zero.
    """

    def test_single_neuron(self):
        """Tests:
        Single neuron with stim_count=1 produces exactly one permutation.
        """
        result = generate_permutations([0], stim_count=1)

        assert len(result) == 1
        assert result[0] == [("stim", [0], 400, 100)]

    def test_stim_count_equals_neurons(self):
        """Tests:
        stim_count equal to neuron count produces factorial(n) permutations.
        """
        neurons = [0, 1]
        result = generate_permutations(neurons, stim_count=2)

        assert len(result) == math.factorial(2)

    def test_stim_count_zero(self):
        """Tests:
        stim_count=0 produces one permutation (the empty permutation) with an
        empty action list.
        """
        result = generate_permutations([0, 1, 2], stim_count=0)

        assert len(result) == 1
        assert result[0] == []


class TestGenerateStimulationsEdgeCases:
    """Edge case tests for generate_stimulations.

    Tests:
    Boundary conditions including empty input, single integer, mixed integers
    and lists, and a nested list with a single element.
    """

    def test_empty_list(self):
        """Tests:
        Empty electrode_inds produces an empty list.
        """
        result = generate_stimulations([])

        assert result == []

    def test_single_int(self):
        """Tests:
        Single integer electrode index is wrapped in a list with default params.
        """
        result = generate_stimulations([5])

        assert result == [([5], 400, 200)]

    def test_mixed_int_and_list(self):
        """Tests:
        Mixed integers and lists are handled correctly — ints wrapped, lists passed through.
        """
        result = generate_stimulations([1, [2, 3]])

        assert result == [([1], 400, 200), ([2, 3], 400, 200)]

    def test_nested_list_single_element(self):
        """Tests:
        A nested list with a single element is passed through unchanged.
        """
        result = generate_stimulations([[7]])

        assert result == [([7], 400, 200)]
