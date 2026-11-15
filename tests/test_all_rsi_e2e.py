"""
Master End-to-End Test Runner for Recursive Self-Improvement (RSI) Methods Suite.
Executes all test tiers (Tier 1 Features, Tier 2 Boundaries, Tier 3 Interactions, Tier 4 Scenarios)
and asserts a 100% pass rate across the comprehensive test matrix.
"""

import sys
import os
import pytest
from pathlib import Path


def run_full_suite() -> int:
    """Executes all RSI test tiers programmatically and returns the pytest exit code."""
    tests_dir = Path(__file__).resolve().parent
    base_tmp = tests_dir.parent / ".pytest_tmp"
    args = [
        str(tests_dir),
        f"--basetemp={base_tmp}",
        "-v",
        "--tb=short",
        "-k", "not test_master_full_suite_execution"  # Avoid recursive self-invocation
    ]
    return pytest.main(args)


def test_master_full_suite_execution():
    """Master E2E assertion: executes all tier suites and verifies 100% pass rate."""
    tests_dir = Path(__file__).resolve().parent
    tier_files = [
        "test_tier1_features.py",
        "test_tier2_boundaries.py",
        "test_tier3_interactions.py",
        "test_tier4_scenarios.py"
    ]
    for tier_file in tier_files:
        assert (tests_dir / tier_file).exists(), f"Missing required tier test file: {tier_file}"

    # Execute all tiers and assert 0 failures
    exit_code = run_full_suite()
    assert exit_code == 0, f"Master E2E test run failed with pytest exit code {exit_code}"


if __name__ == "__main__":
    code = run_full_suite()
    sys.exit(code)
