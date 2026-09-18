#!/usr/bin/env python
"""
Test Baseline Inheritance Logic
================================

Tests the resolve_baseline_inheritance function to ensure correct
behavior for multi-experiment series.

Usage:
    python test_baseline_inheritance.py
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.catalogging.validators import parse_exp_number, compare_exp_names
from utils.catalogging.postprocessing import resolve_baseline_inheritance

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_test_header(test_name):
    """Print a formatted test header."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{test_name}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")


def print_test_result(test_name, passed, details=""):
    """Print a formatted test result."""
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"{status} - {test_name}")
    if details:
        print(f"  {details}")


def test_parse_exp_number():
    """Test numeric parsing from experiment names."""
    print_test_header("Test 1: Parse Exp Number")

    tests = [
        ('exp1', 1),
        ('exp10', 10),
        ('exp2', 2),
        ('BL2', 2),
        ('david', None),
        ('RL_k252a', 252),  # Has numeric component, so extracts 252
        ('exp1_repeat', 1),
        ('freqs', None),
    ]

    all_passed = True
    for exp_name, expected in tests:
        result = parse_exp_number(exp_name)
        passed = result == expected
        all_passed = all_passed and passed
        print_test_result(
            f"parse_exp_number('{exp_name}')",
            passed,
            f"Expected {expected}, got {result}"
        )

    return all_passed


def test_compare_exp_names():
    """Test comparison logic for experiment names."""
    print_test_header("Test 2: Compare Exp Names")

    tests = [
        ('exp1', 'exp2', True),   # 1 < 2
        ('exp2', 'exp1', False),  # 2 > 1
        ('exp1', 'exp10', True),  # 1 < 10 (numeric, not alphabetical)
        ('exp10', 'exp2', False), # 10 > 2
        ('exp1', 'exp1_repeat', True),  # Same number, alphabetical
        ('david', 'freqs', True),  # Both non-numeric, alphabetical
        ('exp1', 'david', True),   # Numeric comes before non-numeric
        ('david', 'exp1', False),  # Non-numeric comes after numeric
    ]

    all_passed = True
    for exp_a, exp_b, expected in tests:
        result = compare_exp_names(exp_a, exp_b)
        passed = result == expected
        all_passed = all_passed and passed
        print_test_result(
            f"compare_exp_names('{exp_a}', '{exp_b}')",
            passed,
            f"Expected {expected}, got {result}"
        )

    return all_passed


def test_basic_inheritance():
    """Test Case 1: Basic numeric inheritance."""
    print_test_header("Test 3: Basic Numeric Inheritance")

    # exp1 has baseline, exp2 doesn't -> exp2 inherits from exp1
    df = pd.DataFrame([
        {'proj': 'test', 'chip': 'chip1', 'exp': 'exp1', 'experiment': 'exp1/exp1', 'baseline': True},
        {'proj': 'test', 'chip': 'chip1', 'exp': 'exp1', 'experiment': 'exp1/exp1_cont_1', 'baseline': False},
        {'proj': 'test', 'chip': 'chip1', 'exp': 'exp2', 'experiment': 'exp2/exp2_cont_1', 'baseline': False},
        {'proj': 'test', 'chip': 'chip1', 'exp': 'exp2', 'experiment': 'exp2/exp2_cont_2', 'baseline': False},
    ])

    df = resolve_baseline_inheritance(df)

    # Check exp1 recordings: should have no inheritance
    exp1_inheritance = df[df['exp'] == 'exp1']['inherited_baseline_from'].unique()
    test1_passed = len(exp1_inheritance) == 1 and pd.isna(exp1_inheritance[0])

    # Check exp2 recordings: should inherit from exp1
    exp2_inheritance = df[df['exp'] == 'exp2']['inherited_baseline_from'].unique()
    test2_passed = len(exp2_inheritance) == 1 and exp2_inheritance[0] == 'exp1'

    print_test_result(
        "exp1 recordings have no inheritance",
        test1_passed,
        f"Got: {exp1_inheritance}"
    )
    print_test_result(
        "exp2 recordings inherit from exp1",
        test2_passed,
        f"Got: {exp2_inheritance}"
    )

    return test1_passed and test2_passed


def test_multiple_baselines():
    """Test Case 2: Multiple baselines (most recent inheritance)."""
    print_test_header("Test 4: Multiple Baselines - Most Recent Inheritance")

    # exp1 has baseline, exp2 has baseline, exp3 doesn't -> exp3 inherits from exp2
    df = pd.DataFrame([
        {'proj': 'test', 'chip': 'chip1', 'exp': 'exp1', 'experiment': 'exp1/exp1', 'baseline': True},
        {'proj': 'test', 'chip': 'chip1', 'exp': 'exp2', 'experiment': 'exp2/exp2', 'baseline': True},
        {'proj': 'test', 'chip': 'chip1', 'exp': 'exp3', 'experiment': 'exp3/exp3_cont_1', 'baseline': False},
    ])

    df = resolve_baseline_inheritance(df)

    # exp3 should inherit from exp2 (not exp1)
    exp3_inheritance = df[df['exp'] == 'exp3']['inherited_baseline_from'].unique()
    passed = len(exp3_inheritance) == 1 and exp3_inheritance[0] == 'exp2'

    print_test_result(
        "exp3 inherits from exp2 (most recent)",
        passed,
        f"Got: {exp3_inheritance}"
    )

    return passed


def test_gaps_in_sequence():
    """Test Case 3: Gaps in sequence."""
    print_test_header("Test 5: Gaps in Sequence")

    # exp1 has baseline, exp3 doesn't (no exp2) -> exp3 inherits from exp1
    df = pd.DataFrame([
        {'proj': 'test', 'chip': 'chip1', 'exp': 'exp1', 'experiment': 'exp1/exp1', 'baseline': True},
        {'proj': 'test', 'chip': 'chip1', 'exp': 'exp3', 'experiment': 'exp3/exp3_cont_1', 'baseline': False},
    ])

    df = resolve_baseline_inheritance(df)

    exp3_inheritance = df[df['exp'] == 'exp3']['inherited_baseline_from'].unique()
    passed = len(exp3_inheritance) == 1 and exp3_inheritance[0] == 'exp1'

    print_test_result(
        "exp3 inherits from exp1 (gap in sequence)",
        passed,
        f"Got: {exp3_inheritance}"
    )

    return passed


def test_no_earlier_baseline():
    """Test Case 4: No earlier baseline."""
    print_test_header("Test 6: No Earlier Baseline")

    # exp1 has no baseline -> exp1 still fails validation (no inheritance)
    df = pd.DataFrame([
        {'proj': 'test', 'chip': 'chip1', 'exp': 'exp1', 'experiment': 'exp1/exp1_cont_1', 'baseline': False},
        {'proj': 'test', 'chip': 'chip1', 'exp': 'exp1', 'experiment': 'exp1/exp1_cont_2', 'baseline': False},
    ])

    df = resolve_baseline_inheritance(df)

    # All should have no inheritance
    exp1_inheritance = df[df['exp'] == 'exp1']['inherited_baseline_from'].unique()
    passed = len(exp1_inheritance) == 1 and pd.isna(exp1_inheritance[0])

    print_test_result(
        "exp1 has no inheritance (no earlier baseline)",
        passed,
        f"Got: {exp1_inheritance}"
    )

    return passed


def test_non_numeric_experiments():
    """Test Case 5: Non-numeric experiments."""
    print_test_header("Test 7: Non-Numeric Experiments")

    # david has baseline, freqs doesn't -> freqs inherits alphabetically
    df = pd.DataFrame([
        {'proj': 'test', 'chip': 'chip1', 'exp': 'david', 'experiment': 'david/david', 'baseline': True},
        {'proj': 'test', 'chip': 'chip1', 'exp': 'freqs', 'experiment': 'freqs/freqs_cont_1', 'baseline': False},
    ])

    df = resolve_baseline_inheritance(df)

    freqs_inheritance = df[df['exp'] == 'freqs']['inherited_baseline_from'].unique()
    passed = len(freqs_inheritance) == 1 and freqs_inheritance[0] == 'david'

    print_test_result(
        "freqs inherits from david (alphabetical)",
        passed,
        f"Got: {freqs_inheritance}"
    )

    return passed


def test_mixed_numeric_text():
    """Test Case 6: Mixed numeric/text (tiebreaking)."""
    print_test_header("Test 8: Mixed Numeric/Text Tiebreaking")

    # exp1 has baseline, exp1_repeat doesn't -> exp1_repeat inherits from exp1
    df = pd.DataFrame([
        {'proj': 'test', 'chip': 'chip1', 'exp': 'exp1', 'experiment': 'exp1/exp1', 'baseline': True},
        {'proj': 'test', 'chip': 'chip1', 'exp': 'exp1_repeat', 'experiment': 'exp1_repeat/exp1_repeat_cont_1', 'baseline': False},
    ])

    df = resolve_baseline_inheritance(df)

    exp1_repeat_inheritance = df[df['exp'] == 'exp1_repeat']['inherited_baseline_from'].unique()
    passed = len(exp1_repeat_inheritance) == 1 and exp1_repeat_inheritance[0] == 'exp1'

    print_test_result(
        "exp1_repeat inherits from exp1",
        passed,
        f"Got: {exp1_repeat_inheritance}"
    )

    return passed


def test_project_isolation():
    """Test Case 7: Project isolation."""
    print_test_header("Test 9: Project Isolation")

    # proj1/chip1/exp1 has baseline, proj2/chip1/exp2 doesn't
    # -> proj2/chip1/exp2 CANNOT inherit (different project)
    df = pd.DataFrame([
        {'proj': 'proj1', 'chip': 'chip1', 'exp': 'exp1', 'experiment': 'exp1/exp1', 'baseline': True},
        {'proj': 'proj2', 'chip': 'chip1', 'exp': 'exp2', 'experiment': 'exp2/exp2_cont_1', 'baseline': False},
    ])

    df = resolve_baseline_inheritance(df)

    # proj2 exp2 should have no inheritance (different project)
    proj2_inheritance = df[(df['proj'] == 'proj2') & (df['exp'] == 'exp2')]['inherited_baseline_from'].unique()
    passed = len(proj2_inheritance) == 1 and pd.isna(proj2_inheritance[0])

    print_test_result(
        "proj2/exp2 has no inheritance (different project)",
        passed,
        f"Got: {proj2_inheritance}"
    )

    return passed


def test_numeric_ordering():
    """Test Case 8: Numeric ordering correctness."""
    print_test_header("Test 10: Numeric Ordering Correctness")

    # exp1, exp10, exp2 -> should order as exp1(1) < exp2(2) < exp10(10)
    df = pd.DataFrame([
        {'proj': 'test', 'chip': 'chip1', 'exp': 'exp1', 'experiment': 'exp1/exp1', 'baseline': True},
        {'proj': 'test', 'chip': 'chip1', 'exp': 'exp2', 'experiment': 'exp2/exp2', 'baseline': True},
        {'proj': 'test', 'chip': 'chip1', 'exp': 'exp10', 'experiment': 'exp10/exp10_cont_1', 'baseline': False},
    ])

    df = resolve_baseline_inheritance(df)

    # exp10 should inherit from exp2 (not exp1)
    exp10_inheritance = df[df['exp'] == 'exp10']['inherited_baseline_from'].unique()
    passed = len(exp10_inheritance) == 1 and exp10_inheritance[0] == 'exp2'

    print_test_result(
        "exp10 inherits from exp2 (numeric ordering)",
        passed,
        f"Got: {exp10_inheritance}"
    )

    return passed


def test_real_data_p001237():
    """Test Case 9: Real data case (P001237)."""
    print_test_header("Test 11: Real Data - P001237")

    # Simulating P001237: exp1/exp1 (baseline), exp2 (no baseline)
    df = pd.DataFrame([
        {'proj': '24-04-18_butterfly', 'chip': 'p001237', 'exp': 'exp1', 'experiment': 'exp1/exp1', 'baseline': True},
        {'proj': '24-04-18_butterfly', 'chip': 'p001237', 'exp': 'exp1', 'experiment': 'exp1/exp1_cont_1', 'baseline': False},
        {'proj': '24-04-18_butterfly', 'chip': 'p001237', 'exp': 'exp2', 'experiment': 'exp2/exp2_cont_1', 'baseline': False},
        {'proj': '24-04-18_butterfly', 'chip': 'p001237', 'exp': 'exp2', 'experiment': 'exp2/exp2_cont_2', 'baseline': False},
    ])

    df = resolve_baseline_inheritance(df)

    # exp2 recordings should inherit from exp1
    exp2_inheritance = df[df['exp'] == 'exp2']['inherited_baseline_from'].unique()
    passed = len(exp2_inheritance) == 1 and exp2_inheritance[0] == 'exp1'

    print_test_result(
        "P001237 exp2 inherits from exp1",
        passed,
        f"Got: {exp2_inheritance}"
    )

    return passed


def test_bee_ctrl_pattern():
    """Test Case 10: bee_ctrl pattern."""
    print_test_header("Test 12: Bee Control Pattern")

    # bee_ctrl/bee_ctrl_recordphase_recording (baseline)
    # bee_ctrl/bee_ctrl_recordphase_recording_1 (not baseline)
    df = pd.DataFrame([
        {'proj': 'test', 'chip': 'chip1', 'exp': 'bee_ctrl',
         'experiment': 'bee_ctrl/bee_ctrl_recordphase_recording', 'baseline': True},
        {'proj': 'test', 'chip': 'chip1', 'exp': 'bee_ctrl',
         'experiment': 'bee_ctrl/bee_ctrl_recordphase_recording_1', 'baseline': False},
    ])

    df = resolve_baseline_inheritance(df)

    # Both should be in same exp, so no inheritance needed across exp values
    # (inheritance is for different exp values, not within same exp)
    all_inheritance = df['inherited_baseline_from'].unique()
    passed = len(all_inheritance) == 1 and pd.isna(all_inheritance[0])

    print_test_result(
        "bee_ctrl recordings in same exp (no cross-exp inheritance)",
        passed,
        f"Got: {all_inheritance}"
    )

    return passed


def main():
    """Run all tests."""
    print(f"\n{YELLOW}{'='*60}{RESET}")
    print(f"{YELLOW}BASELINE INHERITANCE TEST SUITE{RESET}")
    print(f"{YELLOW}{'='*60}{RESET}")

    all_tests = [
        test_parse_exp_number,
        test_compare_exp_names,
        test_basic_inheritance,
        test_multiple_baselines,
        test_gaps_in_sequence,
        test_no_earlier_baseline,
        test_non_numeric_experiments,
        test_mixed_numeric_text,
        test_project_isolation,
        test_numeric_ordering,
        test_real_data_p001237,
        test_bee_ctrl_pattern,
    ]

    results = []
    for test in all_tests:
        try:
            passed = test()
            results.append(passed)
        except Exception as e:
            print(f"{RED}✗ EXCEPTION: {e}{RESET}")
            results.append(False)

    # Summary
    print(f"\n{YELLOW}{'='*60}{RESET}")
    print(f"{YELLOW}TEST SUMMARY{RESET}")
    print(f"{YELLOW}{'='*60}{RESET}")

    passed_count = sum(results)
    total_count = len(results)

    if passed_count == total_count:
        print(f"{GREEN}✓ All tests passed ({passed_count}/{total_count}){RESET}")
        sys.exit(0)
    else:
        print(f"{RED}✗ Some tests failed ({passed_count}/{total_count} passed){RESET}")
        sys.exit(1)


if __name__ == '__main__':
    main()
