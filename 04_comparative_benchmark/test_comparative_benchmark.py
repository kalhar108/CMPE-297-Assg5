"""Comprehensive Unit and Integration Test Suite for Codelab 4: Comparative RSI Benchmark.

Verifies:
1. BenchmarkRunner multi-paradigm execution contracts across Dream-RSI, AIDE2, and The AI Scientist 2.
2. ParetoAnalytics multi-objective non-dominated frontier computation and monotonicity.
3. Multi-objective trade-off analysis (Drawdown vs Turnover, Sharpe vs Compute).
4. ASCII generational convergence curves and 2D Pareto frontier rendering.
5. Standalone demo entrypoint and JSON telemetry export.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure assignment root is on sys.path
_current_dir = Path(__file__).resolve().parent
_assignment_root = _current_dir.parent
if str(_assignment_root) not in sys.path:
    sys.path.insert(0, str(_assignment_root))
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from common.domain.market_data import generate_multi_era_market_data
try:
    from benchmark_runner import BenchmarkRunner, BenchmarkResultDict
    from pareto_analytics import ParetoAnalytics
    from demo import run_benchmark_demo
except ImportError:
    import importlib
    _br = importlib.import_module("04_comparative_benchmark.benchmark_runner")
    _pa = importlib.import_module("04_comparative_benchmark.pareto_analytics")
    _dm = importlib.import_module("04_comparative_benchmark.demo")
    BenchmarkRunner = _br.BenchmarkRunner
    BenchmarkResultDict = _br.BenchmarkResultDict
    ParetoAnalytics = _pa.ParetoAnalytics
    run_benchmark_demo = _dm.run_benchmark_demo


@pytest.fixture(scope="module")
def standard_data():
    """Generates standard 756-day multi-era market data."""
    return generate_multi_era_market_data(seed=42, num_days=756)


# ==============================================================================
# 1. BenchmarkRunner Tests
# ==============================================================================

def test_benchmark_runner_initialization(standard_data):
    """Verifies clean initialization and parameter validation."""
    runner = BenchmarkRunner(market_data=standard_data, mode="mock", generations=3, iterations=2)
    assert runner.market_data.num_days == 756
    assert runner.generations == 3
    assert runner.iterations == 2


def test_benchmark_runner_executes_all_three_paradigms(standard_data):
    """Verifies head-to-head benchmarking produces valid results for all 3 paradigms."""
    runner = BenchmarkRunner(market_data=standard_data, mode="mock", generations=3, iterations=2)
    results = runner.run_comparative_benchmark(verbose=False)

    assert isinstance(results, BenchmarkResultDict)
    assert "Dream-RSI" in results
    assert "AIDE2" in results
    assert "AI-Scientist-2" in results

    # Verify alias support
    assert results["The AI Scientist 2"] == results["AI-Scientist-2"]

    for paradigm in ["Dream-RSI", "AIDE2", "AI-Scientist-2"]:
        data = results[paradigm]
        assert data["final_sharpe"] > 1.50, f"{paradigm} final Sharpe too low"
        assert data["max_drawdown"] < 0.20, f"{paradigm} max drawdown too high"
        assert data["compute_seconds"] > 0.0, f"{paradigm} compute time non-positive"
        assert "cagr" in data
        assert "sortino_ratio" in data
        assert "generations" in data
        assert len(data["generations"]) >= 3


def test_benchmark_runner_summary_table(standard_data):
    """Verifies formatted ASCII summary table generation."""
    runner = BenchmarkRunner(market_data=standard_data, mode="mock")
    results = runner.run_comparative_benchmark(verbose=False)
    table = runner.generate_summary_table(results)

    assert "RSI CROSS-PARADIGM COMPARATIVE BENCHMARK" in table
    assert "Dream-RSI" in table
    assert "AIDE2" in table
    assert "AI-Scientist-2" in table
    assert "Final Sharpe" in table


# ==============================================================================
# 2. ParetoAnalytics Tests
# ==============================================================================

def test_pareto_frontier_empty():
    """Verifies handling of empty inputs."""
    frontier = ParetoAnalytics.compute_pareto_frontier([])
    assert frontier == []


def test_pareto_frontier_single_point():
    """Verifies handling of single-point inputs."""
    frontier = ParetoAnalytics.compute_pareto_frontier([(1.5, 2.0)])
    assert frontier == [(1.5, 2.0)]


def test_pareto_frontier_monotonicity_and_dominance():
    """Verifies that non-dominated points satisfy strict Pareto dominance."""
    # (cost, performance) -> lower cost is better, higher performance is better
    points = [
        (1.0, 1.0),
        (2.0, 1.5),
        (3.0, 1.8),
        (2.5, 1.2),  # Dominated by (2.0, 1.5): higher cost, lower performance
        (4.0, 1.7),  # Dominated by (3.0, 1.8): higher cost, lower performance
    ]
    frontier = ParetoAnalytics.compute_pareto_frontier(points, maximize_x=False, maximize_y=True)

    expected = [(1.0, 1.0), (2.0, 1.5), (3.0, 1.8)]
    assert frontier == expected

    # Monotonicity check: as cost increases, performance must strictly increase
    for i in range(len(frontier) - 1):
        assert frontier[i][0] <= frontier[i + 1][0]
        assert frontier[i][1] <= frontier[i + 1][1]


def test_pareto_frontier_minimize_both():
    """Verifies non-dominated frontier when minimizing both metrics (e.g. Drawdown vs Turnover)."""
    # (turnover, drawdown) -> lower is better for both
    points = [
        (1.0, 0.20),
        (2.0, 0.15),
        (3.0, 0.10),
        (2.5, 0.18),  # Dominated by (2.0, 0.15)
    ]
    frontier = ParetoAnalytics.compute_pareto_frontier(points, maximize_x=False, maximize_y=False)
    expected = [(1.0, 0.20), (2.0, 0.15), (3.0, 0.10)]
    assert frontier == expected


def test_pareto_analyze_tradeoffs(standard_data):
    """Verifies structured multi-objective trade-off analysis across paradigms."""
    runner = BenchmarkRunner(market_data=standard_data, mode="mock")
    results = runner.run_comparative_benchmark(verbose=False)
    tradeoffs = ParetoAnalytics.analyze_tradeoffs(results)

    assert "sharpe_vs_compute_frontier" in tradeoffs
    assert "mdd_vs_turnover_frontier" in tradeoffs
    assert "rankings" in tradeoffs
    assert "dominance_matrix" in tradeoffs
    assert len(tradeoffs["sharpe_vs_compute_frontier"]) >= 1


def test_pareto_render_visualizations(standard_data):
    """Verifies ASCII convergence curves and 2D Pareto frontier rendering."""
    runner = BenchmarkRunner(market_data=standard_data, mode="mock")
    results = runner.run_comparative_benchmark(verbose=False)

    curves = ParetoAnalytics.render_ascii_convergence_curves(results)
    assert "MULTI-GENERATIONAL RSI SHARPE CONVERGENCE" in curves
    assert "Gen 0" in curves
    assert "Gen 2" in curves

    frontier_plot = ParetoAnalytics.render_ascii_pareto_frontier(results)
    assert "MULTI-OBJECTIVE PARETO FRONTIER" in frontier_plot
    assert "PARETO OPTIMAL" in frontier_plot

    full_report = ParetoAnalytics.generate_pareto_report(results)
    assert "KEY ARCHITECTURAL FINDINGS" in full_report


# ==============================================================================
# 3. Standalone Demo Entrypoint & JSON Export Tests
# ==============================================================================

def test_run_benchmark_demo_mock_execution():
    """Verifies demo.py run_benchmark_demo execution in mock mode."""
    telemetry = run_benchmark_demo(mode="mock", generations=3, iterations=2, verbose=False)
    assert telemetry["status"] == "success"
    assert telemetry["benchmark"] == "CrossParadigmRSI"
    assert "Dream-RSI" in telemetry["results"]
    assert "AIDE2" in telemetry["results"]
    assert "AI-Scientist-2" in telemetry["results"]


def test_run_benchmark_demo_json_export(tmp_path):
    """Verifies telemetry JSON serialization to file."""
    json_file = tmp_path / "benchmark_telemetry.json"
    telemetry = run_benchmark_demo(
        mode="mock",
        generations=3,
        iterations=2,
        verbose=False,
        output_json=str(json_file),
    )

    assert json_file.exists()
    with open(json_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["status"] == "success"
    assert loaded["results"]["AI-Scientist-2"]["final_sharpe"] > 1.5
