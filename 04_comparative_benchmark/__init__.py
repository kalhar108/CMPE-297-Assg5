"""Codelab 4: Comparative RSI Benchmark & Multi-Generational Pareto Analytics.

Exports:
- BenchmarkRunner: Cross-paradigm benchmark harness executing Dream-RSI, AIDE2,
  and The AI Scientist 2 head-to-head on identical 756-day multi-era market data.
- ParetoAnalytics: Multi-generational Pareto efficiency frontier analysis and
  ASCII convergence curve rendering.
- BenchmarkRunResult: Telemetry dataclass.
"""

from __future__ import annotations

try:
    from .benchmark_runner import BenchmarkRunner, BenchmarkRunResult
    from .pareto_analytics import ParetoAnalytics
except (ImportError, ValueError):
    try:
        from benchmark_runner import BenchmarkRunner, BenchmarkRunResult  # type: ignore
        from pareto_analytics import ParetoAnalytics  # type: ignore
    except ImportError:
        import importlib
        _br = importlib.import_module("04_comparative_benchmark.benchmark_runner")
        _pa = importlib.import_module("04_comparative_benchmark.pareto_analytics")
        BenchmarkRunner = _br.BenchmarkRunner
        BenchmarkRunResult = _br.BenchmarkRunResult
        ParetoAnalytics = _pa.ParetoAnalytics

__all__ = [
    "BenchmarkRunner",
    "BenchmarkRunResult",
    "ParetoAnalytics",
]
