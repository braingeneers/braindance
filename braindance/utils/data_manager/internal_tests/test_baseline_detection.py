#!/usr/bin/env python3
"""
Test Suite for Baseline Detection Logic

This test script verifies that the is_baseline() function correctly identifies
baseline recordings across all supported patterns.

Usage:
    cd braindance/utils/data_manager/internal_tests
    python test_baseline_detection.py
    
The script will exit with code 0 if all tests pass, or 1 if any test fails.
"""

from braindance.utils.data_manager.utils.catalogging.validators import is_baseline, is_baseline_with_reason


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def test_baseline_detection():
    """
    Test baseline detection across all known patterns.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    
    # Define test cases: (experiment_path, expected_result, description)
    test_cases = [
        # =====================================================================
        # EXACT FOLDER MATCH PATTERN (folder/folder)
        # =====================================================================
        ('exp1/exp1', True, 'Exact folder match: exp1/exp1'),
        ('exp2/exp2', True, 'Exact folder match: exp2/exp2'),
        ('david/david', True, 'Exact folder match: david/david'),
        ('exp4/exp4', True, 'Exact folder match: exp4/exp4'),
        ('data/test', False, 'Non-matching: data/test'),
        
        # =====================================================================
        # FIRST IN SERIES PATTERN (cartpole baselines)
        # =====================================================================
        ('exp2/exp2_cartpole_long', True, 'First in series: exp2_cartpole_long (no suffix)'),
        ('exp2/exp2_cartpole_long_1', False, 'Numbered continuation: exp2_cartpole_long_1'),
        ('exp2/exp2_cartpole_long_2', False, 'Numbered continuation: exp2_cartpole_long_2'),
        ('exp2/exp2_cartpole_long_3', False, 'Numbered continuation: exp2_cartpole_long_3'),
        ('exp2/exp2_cartpole_long_4', False, 'Numbered continuation: exp2_cartpole_long_4'),
        ('exp1/exp1_cartpole_F', True, 'First in series: exp1_cartpole_F (no suffix)'),
        ('exp1/exp1_cartpole_F_1', False, 'Numbered continuation: exp1_cartpole_F_1'),
        
        # =====================================================================
        # CONTINUATION PATTERNS (_cont)
        # =====================================================================
        ('exp1/exp1_cont_5', False, 'Continuation: exp1_cont_5'),
        ('exp1/exp1_cont_12', False, 'Continuation: exp1_cont_12'),
        ('freqs/freqs_cont_1', False, 'Continuation: freqs_cont_1'),
        
        # =====================================================================
        # BL VARIANTS
        # =====================================================================
        ('BL1/BL1', True, 'BL1 exact match'),
        ('bl_1/bl_1', True, 'BL1 variant: bl_1'),
        ('bl-1/bl-1', True, 'BL1 variant: bl-1'),
        ('BL2/BL2', False, 'BL2 not baseline'),
        ('bl-2/bl-2', False, 'BL2 variant not baseline'),
        ('bl_3/bl_3', False, 'BL3 variant not baseline'),
        ('bl-4/bl-4', False, 'BL4 variant not baseline'),
        
        # NEW: Lowercase 'bl' exact matches (fixed in priority update)
        ('bl/bl', True, 'Exact match: bl/bl (lowercase)'),
        ('bl2/bl2', True, 'Exact match: bl2/bl2 (folder name match takes priority)'),
        
        # =====================================================================
        # CAUSAL EXPERIMENTS
        # =====================================================================
        ('exp1/exp1_causal', False, 'Causal experiment with _causal suffix'),
        ('david/david_causal', False, 'Causal experiment with _causal suffix'),
        ('exp2/exp2_causal_test', False, 'Causal experiment with _causal in middle'),
        
        # NEW: Causal experiments that match folder name exactly (fixed in priority update)
        ('causal_BL1/causal_BL1', True, 'Exact match: causal_BL1/causal_BL1 (folder name match priority)'),
        ('causal_BL2/causal_BL2', True, 'Exact match: causal_BL2/causal_BL2 (folder name match priority)'),
        ('causal_thc/causal_thc', True, 'Exact match: causal_thc/causal_thc (folder name match priority)'),
        
        # But _causal suffix is still rejected
        ('causal_BL1/causal_BL1_causal', False, 'Causal suffix: causal_BL1_causal'),
        ('causal_thc/causal_thc_causal', False, 'Causal suffix: causal_thc_causal'),
        
        # =====================================================================
        # RECORDING PHASE PATTERNS
        # =====================================================================
        ('bee_ctrl/bee_ctrl_recordphase_recording', True, 'Base recording phase'),
        ('bee_ctrl/bee_ctrl_recordphase_recording_1', False, 'Numbered recording phase'),
        ('bee_ctrl/bee_ctrl_recordphase_recording_2', False, 'Numbered recording phase'),
        
        # =====================================================================
        # FREQUENCY STIMULATION PHASE (never baselines)
        # =====================================================================
        ('exp1/exp1_frequencystimphase', False, 'Frequency stim phase'),
        ('david/david_frequencystimphase_test', False, 'Frequency stim phase with suffix'),
        
        # =====================================================================
        # EDGE CASES
        # =====================================================================
        ('', False, 'Empty string'),
        ('single', False, 'No folder separator'),
        ('RL_k252a/RL_k252a', True, 'Exact match with special chars'),
        ('RL_k252a/RL_k252a_1', False, 'Numbered continuation with special chars'),
        
        # =====================================================================
        # REAL CATALOG EXAMPLES (from user's data)
        # =====================================================================
        ('25178ic/exp1/exp1', True, 'Real example: 25178ic/exp1/exp1'),
        ('23186/RL_k252a/RL_k252a', True, 'Real example: RL_k252a baseline'),
        ('p001237/exp2/exp2', True, 'Real example: p001237/exp2/exp2'),
        ('25126hs1/david/david', True, 'Real example: david/david'),
        ('20217/exp1/exp1', True, 'Real example: 20217/exp1/exp1'),
    ]
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}Baseline Detection Test Suite{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")
    
    passed = 0
    failed = 0
    
    for experiment, expected, description in test_cases:
        result = is_baseline(experiment)
        success = (result == expected)
        
        if success:
            passed += 1
            status = f"{Colors.GREEN}✓ PASS{Colors.RESET}"
        else:
            failed += 1
            status = f"{Colors.RED}✗ FAIL{Colors.RESET}"
            
        # Get detailed reason for debugging
        _, reason = is_baseline_with_reason(experiment)
        
        print(f"{status} {description}")
        if not success:
            print(f"     {Colors.YELLOW}Expected: {expected}, Got: {result}{Colors.RESET}")
            print(f"     {Colors.YELLOW}Reason: {reason}{Colors.RESET}")
        print(f"     Path: '{experiment}'")
        print()
    
    # Summary
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}Test Summary{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"Total tests: {len(test_cases)}")
    print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
    if failed > 0:
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
    else:
        print(f"Failed: {failed}")
    print()
    
    if failed == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All tests passed!{Colors.RESET}")
        return True
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some tests failed!{Colors.RESET}")
        return False


if __name__ == '__main__':
    import sys
    success = test_baseline_detection()
    sys.exit(0 if success else 1)
