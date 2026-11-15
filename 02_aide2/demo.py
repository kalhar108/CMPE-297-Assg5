"""Standalone Runnable CLI Entrypoint for Codelab 2: AIDE2.

Demonstrates:
(a) Drafting baseline alpha strategy from task description;
(b) Intentional traceback trigger (ZeroDivisionError / runtime crash);
(c) Debugging operator repairing the AST and code logic;
(d) UCT exploration branching, dynamic pruning, and adaptive backtracking;
(e) Bi-level meta-harness self-improvement across generations (Gen 0 -> Gen 1 -> Gen 2);
(f) Dual-mode execution support (deterministic offline mock vs live Gemini API).

Supports:
- python demo.py (run directly from inside 02_aide2/)
- python -m 02_aide2.demo (run as module from assignment root)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Windows console UTF-8 reconfiguration
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path for both direct and module invocation
current_file = Path(__file__).resolve()
aide2_dir = current_file.parent
assignment_root = aide2_dir.parent
if str(assignment_root) not in sys.path:
    sys.path.insert(0, str(assignment_root))
if str(aide2_dir) not in sys.path:
    sys.path.insert(0, str(aide2_dir))

from common.config import RSIExecutionConfig
from common.domain.backtester import BacktestEngine, BacktestResult
from common.domain.market_data import generate_multi_era_market_data
from common.domain.strategy_base import DynamicCodeStrategy
from common.llm_provider import get_llm_provider

try:
    from .mcts_search import AIDE2Search
    from .meta_harness import MetaHarness
    from .operators import DebuggingOperator, DraftingOperator, RefinementOperator
    from .tree_node import CodeTree, StrategyNode
except (ImportError, ValueError):
    from mcts_search import AIDE2Search
    from meta_harness import MetaHarness
    from operators import DebuggingOperator, DraftingOperator, RefinementOperator
    from tree_node import CodeTree, StrategyNode


def run_demo(
    mode: str = "mock",
    generations: int = 3,
    iterations: int = 4,
    verbose: bool = True,
) -> int:
    """Runs the complete AIDE2 Code Tree MCTS & Bi-Level Meta-Optimization walkthrough."""
    start_total_time = time.perf_counter()

    print("=" * 80)
    print("  CODELAB 2: AIDE2 (Code Tree MCTS & Bi-Level Meta-Optimization)")
    print("  Domain: Automated Quantitative Alpha Feature Discovery & Signal Engineering")
    print("=" * 80)

    # 1. Environment & Provider Configuration
    config = RSIExecutionConfig(mode=mode)
    resolved_mode = config.resolved_mode()
    provider = get_llm_provider(config)
    print(f"\n[1] Environment & Provider Setup:")
    print(f"    - Requested Mode : {mode}")
    print(f"    - Resolved Mode  : {resolved_mode}")
    print(f"    - Provider Class : {provider.__class__.__name__}")

    # 2. Market Data & Backtester Setup
    market_data = generate_multi_era_market_data(seed=42, num_days=756)
    engine = BacktestEngine(default_fee_bps=10.0)
    print(f"\n[2] Quantitative Alpha Domain Environment:")
    print(f"    - Market History : 756 Trading Days (3 full years)")
    print(f"    - Multi-Era Data : Era 1 (Bull) | Era 2 (Choppy) | Era 3 (Crisis Shock)")
    print(f"    - Assets Traded  : {market_data.symbols}")
    print(f"    - Fee Model      : 10 basis points (0.10%) transaction friction per turnover")

    # --------------------------------------------------------------------------
    # Step A: Drafting Baseline Strategy
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  STEP A: Drafting Operator — Synthesizing Baseline Alpha Code")
    print("-" * 80)
    drafter = DraftingOperator(provider)
    draft_prompt = "Synthesize an adaptive trend and momentum alpha strategy across SPY, TLT, and GLD."
    drafted_code = drafter.draft(draft_prompt)
    print("Drafting Operator successfully synthesized strategy code:")
    print("```python")
    code_lines = drafted_code.strip().splitlines()
    for line in code_lines[:15]:
        print(f"    {line}")
    if len(code_lines) > 15:
        print(f"    ... [{len(code_lines) - 15} lines omitted for brevity]")
    print("```")

    # Backtest drafted code
    strat_drafted = DynamicCodeStrategy(drafted_code)
    res_drafted = engine.run(strat_drafted, market_data)
    print(f"\nInitial Draft Backtest:")
    print(f"    - Sharpe Ratio : {res_drafted.sharpe_ratio:.4f}")
    print(f"    - Max Drawdown : {res_drafted.max_drawdown * 100:.2f}%")
    print(f"    - CAGR         : {res_drafted.cagr * 100:.2f}%")
    print(f"    - Wall Time    : {res_drafted.execution_time_ms:.2f}ms")

    # --------------------------------------------------------------------------
    # Step B: Intentional / Simulated Traceback Trigger
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  STEP B: Traceback Trigger — Simulating Runtime Division-by-Zero Exception")
    print("-" * 80)
    # Inject intentional zero division bug into strategy signal computation
    buggy_code = drafted_code.replace(
        "signals[t] = trend * np.clip(mom * 4.0, -1.0, 1.0)",
        "# Injected bug: unshielded division by zero\n"
        "            zero_scalar = 1.0 / 0.0\n"
        "            signals[t] = trend * np.clip(mom * 4.0, -1.0, 1.0)",
    )
    if "1.0 / 0.0" not in buggy_code:
        buggy_code += "\n# Intentional error: 1.0 / 0.0\n"

    print("Simulating execution of flawed candidate code...")
    try:
        strat_buggy = DynamicCodeStrategy(buggy_code)
        res_buggy = engine.run(strat_buggy, market_data)
    except Exception as exc:
        res_buggy = BacktestResult(
            sharpe_ratio=-999.0, max_drawdown=1.0, sortino_ratio=-999.0, calmar_ratio=-999.0,
            cagr=-1.0, turnover=0.0, total_cost=0.0, equity_curve=[100000.0],
            daily_returns=[], weights_history=[], execution_time_ms=0.5, success=False,
            error_message=f"{type(exc).__name__}: {str(exc)}",
            traceback_log=f"ZeroDivisionError: float division by zero in line 54 of DraftedStrategy.generate_signals",
        )

    print(f"--> Intercepted Failure: {res_buggy.error_message}")
    print(f"--> Traceback Log Snippet:")
    print(f"    {res_buggy.traceback_log}")

    # --------------------------------------------------------------------------
    # Step C: Debugging Operator Repairing the Code
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  STEP C: Debugging Operator — AST & Code Surgical Repair")
    print("-" * 80)
    debugger = DebuggingOperator(provider)
    fixed_code = debugger.debug_and_fix(
        buggy_code,
        res_buggy.traceback_log or "ZeroDivisionError: float division by zero"
    )

    print(f"Debugger Diagnosis: {debugger.last_diagnosis}")
    print(f"Verifying bug eradication: '1.0 / 0.0' in fixed code -> {'1.0 / 0.0' in fixed_code}")

    strat_repaired = DynamicCodeStrategy(fixed_code)
    res_repaired = engine.run(strat_repaired, market_data)
    print(f"Post-Repair Backtest Result:")
    print(f"    - Execution Success : {res_repaired.success}")
    print(f"    - Sharpe Ratio      : {res_repaired.sharpe_ratio:.4f}")
    print(f"    - Max Drawdown      : {res_repaired.max_drawdown * 100:.2f}%")
    print(f"    - Wall Time         : {res_repaired.execution_time_ms:.2f}ms")

    # --------------------------------------------------------------------------
    # Step D: UCT Exploration Branching, Dynamic Pruning & Backtracking
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  STEP D: MCTS Code Search — UCT Selection, Branching & Adaptive Backtracking")
    print("-" * 80)
    refiner = RefinementOperator(provider)
    search = AIDE2Search(
        c_param=1.414,
        drafter=drafter,
        debugger=debugger,
        refiner=refiner,
        backtest_engine=engine,
        market_data=market_data,
    )

    # 1. Root node
    root_node = StrategyNode(
        node_id="node_0_root",
        code=fixed_code,
        metrics=res_repaired,
        operator="root",
        visits=1,
        total_value=res_repaired.sharpe_ratio,
    )
    search.add_node(root_node)

    # 2. Branch 1: Refinement (parameter tuning & volatility filter)
    child_refined = search.expand(root_node, operator_type="refine")
    res_child1 = search.evaluate_node(child_refined)
    search.backpropagate(child_refined, reward=search.compute_reward(res_child1))

    # 3. Branch 2: Aggressive Overfitting (Simulated dead-end in Era 3 Crisis)
    flawed_metrics = BacktestResult(
        sharpe_ratio=-1.85, max_drawdown=0.62, sortino_ratio=-1.90, calmar_ratio=-2.0,
        cagr=-0.35, turnover=12.0, total_cost=0.05, equity_curve=[100000.0, 50000.0],
        daily_returns=[], weights_history=[], execution_time_ms=1.1, success=True,
    )
    dead_end_node = StrategyNode(
        node_id="node_dead_end_2",
        code=fixed_code + "\n# Aggressive unhedged high-turnover strategy\n",
        metrics=flawed_metrics,
        parent_id="node_0_root",
        operator="refine_aggressive",
        visits=1,
        total_value=-1.85,
    )
    search.add_node(dead_end_node)

    # Show UCT scores balancing exploration and exploitation
    uct_good = search.uct_score(child_refined, parent_visits=root_node.visits)
    uct_bad = search.uct_score(dead_end_node, parent_visits=root_node.visits)

    print(f"UCT Scores from Root (Parent visits = {root_node.visits}):")
    print(f"    - Branch 1 (Refined Node) : UCT = {uct_good:.3f} (Q={child_refined.value:.2f}, N={child_refined.visits})")
    print(f"    - Branch 2 (Flawed Node)  : UCT = {uct_bad:.3f} (Q={dead_end_node.value:.2f}, N={dead_end_node.visits})")

    # Prune dead-end branch and backtrack
    print(f"\nDetecting severe drawdown (-62%) on Branch 2 -> Triggering Dynamic Pruning & Backtracking...")
    pruned = search.prune_dead_ends(threshold=-0.5)
    print(f"Pruned dead-end node IDs : {pruned}")
    print(f"Backtracking selection   : returns to ancestor '{root_node.node_id}' to branch into high-conviction paths.")

    # Render MCTS Tree Topology
    print("\nCurrent Code Search Tree Structure:")
    print(search.tree.render_ascii())

    # --------------------------------------------------------------------------
    # Step E: Bi-Level Meta-Harness Self-Improvement
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  STEP E: Bi-Level Meta-Harness Self-Improvement Across Generations")
    print("-" * 80)
    print("Executing bi-level recursive meta-optimization:")
    print("  - Inner Loop: MCTS explores strategy variations in code space")
    print("  - Outer Loop: Meta-agent analyzes search tree and mutates operator prompt weights & heuristics")

    harness = MetaHarness(
        provider=provider,
        market_data=market_data,
        backtest_engine=engine,
    )

    gens = harness.run_meta_optimization(
        generations=generations,
        iterations_per_gen=iterations,
        verbose=verbose,
    )

    print("\n" + harness.summary_table())

    # Verification of recursive progression
    if len(gens) >= 2:
        assert gens[-1]["best_sharpe"] >= gens[0]["best_sharpe"], "Recursive self-improvement progression failed!"
        print(f"\n[Verification] Verified non-negative progression: Gen 0 Sharpe {gens[0]['best_sharpe']:.2f} -> Gen {len(gens)-1} Sharpe {gens[-1]['best_sharpe']:.2f}")
    elif len(gens) == 1:
        print(f"\n[Verification] Single generation executed: Gen 0 Sharpe {gens[0]['best_sharpe']:.2f}")
    else:
        print("\n[Verification] Zero generations requested; clean exit.")

    elapsed_total = time.perf_counter() - start_total_time
    print(f"\n[OK] AIDE2 Demo completed successfully in {elapsed_total:.2f} seconds.")
    print("=" * 80)
    return 0


def main() -> None:
    """CLI entrypoint parsing command line arguments."""
    parser = argparse.ArgumentParser(
        description="AIDE2: Code Tree MCTS & Bi-Level Meta-Optimization CLI Demo"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live", "auto"],
        default="mock",
        help="LLM execution mode (default: mock, zero network calls)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of bi-level meta-optimization generations (default: 3)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of MCTS iterations per generation (default: 3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable detailed execution logging",
    )

    args = parser.parse_args()
    exit_code = run_demo(
        mode=args.mode,
        generations=args.generations,
        iterations=args.iterations,
        verbose=args.verbose,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
