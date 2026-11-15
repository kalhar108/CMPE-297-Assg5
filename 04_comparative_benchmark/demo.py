"""Standalone Runnable CLI Entrypoint for Codelab 4: Comparative RSI Benchmark.

Executes a head-to-head empirical comparison across:
1. Dream-RSI (World Models & Off-Policy Dreaming)
2. AIDE2 (Code Tree MCTS & Bi-Level Meta-Optimization)
3. The AI Scientist 2 (Automated Discovery & MASS Peer Review)

Evaluates on identical 756-day multi-era market data under equal iteration budgets,
computes the multi-generational Pareto frontier, and renders ASCII convergence curves.

Execution Modes:
  python demo.py
  python -m 04_comparative_benchmark.demo
  python demo.py --mode mock --generations 3 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Reconfigure Windows console to UTF-8
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure assignment root and current package are in sys.path
_current_dir = Path(__file__).resolve().parent
_assignment_root = _current_dir.parent
if str(_assignment_root) not in sys.path:
    sys.path.insert(0, str(_assignment_root))
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from common.config import RSIExecutionConfig
from common.domain.market_data import generate_multi_era_market_data

try:
    from .benchmark_runner import BenchmarkRunner
    from .pareto_analytics import ParetoAnalytics
except (ImportError, ValueError):
    from benchmark_runner import BenchmarkRunner
    from pareto_analytics import ParetoAnalytics


def run_benchmark_demo(
    mode: str = "mock",
    generations: int = 3,
    iterations: int = 3,
    verbose: bool = True,
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes the complete cross-paradigm RSI comparative benchmark workflow.

    Args:
        mode: 'mock', 'live', or 'auto' execution mode.
        generations: Number of discovery generations per method.
        iterations: Number of search iterations per generation.
        verbose: If True, prints formatted console tables and ASCII plots.
        output_json: Optional path to save telemetry JSON.

    Returns:
        Dictionary containing comparative benchmark metrics and Pareto analytics.
    """
    t_start = time.perf_counter()

    config = RSIExecutionConfig(mode=mode)
    resolved_mode = config.resolved_mode()

    if verbose:
        print("=" * 80)
        print("  RECURSIVE SELF-IMPROVEMENT (RSI) SUITE - CODELAB 4: COMPARATIVE BENCHMARK")
        print("  Cross-Paradigm Evaluation: Dream-RSI vs AIDE2 vs The AI Scientist 2")
        print("=" * 80)
        print(f"  Execution Mode       : {resolved_mode.upper()}")
        print(f"  Multi-Era Market Span: 756 Trading Days (Bull, Choppy, Crisis Regimes)")
        print(f"  Iteration Budget     : {generations} Generations, {iterations} Iterations per Generation")
        print(f"  Transaction Friction : 10 basis points (0.10%) per turnover")
        print("-" * 80)

    # 1. Generate standardized 756-day multi-era market environment
    market_data = generate_multi_era_market_data(seed=42, num_days=756)

    # 2. Instantiate and execute BenchmarkRunner
    runner = BenchmarkRunner(
        market_data=market_data,
        mode=resolved_mode,
        seed=42,
        default_fee_bps=10.0,
        generations=generations,
        iterations=iterations,
    )

    if verbose:
        print("\n[*] Launching Head-to-Head Multi-Paradigm Benchmark Engine...")

    results = runner.run_comparative_benchmark(verbose=verbose)

    # 3. Compute Pareto Analytics & Trade-Offs
    tradeoffs = ParetoAnalytics.analyze_tradeoffs(results)

    # 4. Render Reports & Visualizations
    summary_table = runner.generate_summary_table(results)
    pareto_report = ParetoAnalytics.generate_pareto_report(results)

    if verbose:
        print("\n" + summary_table + "\n")
        print(pareto_report)

    t_total = time.perf_counter() - t_start

    telemetry: Dict[str, Any] = {
        "status": "success",
        "benchmark": "CrossParadigmRSI",
        "execution_mode": resolved_mode,
        "total_wall_time_seconds": round(t_total, 4),
        "results": {k: dict(v) for k, v in results.items()},
        "tradeoffs": tradeoffs,
    }

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2)
        if verbose:
            print(f"\n[+] Saved benchmark telemetry to: {out_path}")

    if verbose:
        print(f"\n[OK] Comparative Benchmark completed cleanly in {t_total:.2f} seconds.")

    return telemetry


def main() -> int:
    """CLI argument parser and entrypoint."""
    parser = argparse.ArgumentParser(
        description="Codelab 4: Comparative RSI Benchmark CLI Entrypoint"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live", "auto"],
        default="mock",
        help="Execution mode (default: mock, 100% offline deterministic)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of discovery generations per method (default: 3)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per generation (default: 3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable detailed formatted console logging",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to output execution telemetry JSON",
    )

    args = parser.parse_args()

    try:
        run_benchmark_demo(
            mode=args.mode,
            generations=args.generations,
            iterations=args.iterations,
            verbose=args.verbose,
            output_json=args.output_json,
        )
        return 0
    except Exception as exc:
        print(f"\n[ERROR] Benchmark demo failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
