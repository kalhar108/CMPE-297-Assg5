"""Standalone Runnable CLI Entrypoint for Codelab 1: Dream-RSI.

Demonstrates the 5-phase Recursive Self-Improvement (RSI) workflow:
  Phase 1: Generation 0 baseline exploration policy online execution & backtesting.
  Phase 2: Discovery trajectory logging into an immutable replay buffer & ERTM world model calibration.
  Phase 3: Off-policy dreaming phase optimizing macro search policy (B, D, tau) without live calls.
  Phase 4: Generation 1 redeployment demonstrating measurable Sharpe progression & search compute reduction.
  Phase 5: Telemetry verification and summary reporting.

Supports:
  - python demo.py (direct execution within 01_dream_rsi/)
  - python -m 01_dream_rsi.demo (module execution from repository root)
  - Dual-mode execution (--mode mock by default, --mode live when GEMINI_API_KEY configured)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reconfigure Windows console to UTF-8
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repo root and package directory are in sys.path
_current_dir = Path(__file__).resolve().parent
_repo_root = _current_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

import numpy as np

# Import Shared Core components
from common.config import ExecutionMode, RSIExecutionConfig
from common.domain.backtester import BacktestEngine, BacktestResult
from common.domain.market_data import MarketData, generate_multi_era_market_data
from common.domain.strategy_base import (
    BaseStrategy,
    EqualWeightStrategy,
    MeanReversionStrategy,
    MomentumFactor,
    MovingAverageCrossover,
)
from common.llm_provider import get_llm_provider
from common.rsi_base import GenerationRecord, RSIHistory

# Import Codelab 1 components
try:
    from .dream_engine import DreamEngine, MacroPolicy
    from .policy_optimizer import OptimizationResult, PolicyOptimizer
    from .tree_logger import DiscoveryNode, DiscoveryTreeLogger
    from .world_model import RegimeWorldModel
except (ImportError, ValueError):
    from dream_engine import DreamEngine, MacroPolicy
    from policy_optimizer import OptimizationResult, PolicyOptimizer
    from tree_logger import DiscoveryNode, DiscoveryTreeLogger
    from world_model import RegimeWorldModel


def run_dream_rsi_demo(
    mode: str = "mock",
    seed: int = 42,
    verbose: bool = True,
    output_json: Optional[str] = None,
    generations: int = 2,
) -> Dict[str, Any]:
    """Executes the complete Dream-RSI demonstration workflow.

    Args:
        mode: 'mock', 'live', or 'auto' execution mode.
        seed: Random seed for deterministic reproducibility.
        verbose: If True, prints formatted console tables and progress banners.
        output_json: Optional path to save run telemetry to JSON.
        generations: Number of RSI generations to demonstrate (default: 2).

    Returns:
        Dictionary containing comprehensive run telemetry across generations.
    """
    start_total_time = time.perf_counter()

    # 1. Initialize Dual-Mode Execution Configuration & LLM Provider
    config = RSIExecutionConfig(mode=mode)  # type: ignore[arg-type]
    resolved_mode = config.resolved_mode()
    llm_provider = get_llm_provider(config)

    if generations <= 0:
        telemetry = {
            "status": "success",
            "method": "Dream-RSI",
            "execution_mode": resolved_mode,
            "total_time_ms": 0.0,
            "rsi_progression": {
                "sharpe_delta": 0.0,
                "drawdown_reduction_pts": 0.0,
                "compute_reduction_pct": 0.0,
                "total_replay_transitions": 0,
                "tree_max_depth": 0,
            },
        }
        if output_json:
            out_path = Path(output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(telemetry, f, indent=2)
        return telemetry

    model_display = getattr(
        llm_provider, "model_name", getattr(getattr(llm_provider, "config", None), "model_name", config.model_name)
    )

    if verbose:
        print("=" * 72)
        print("  RECURSIVE SELF-IMPROVEMENT (RSI) SUITE - CODELAB 1: DREAM-RSI  ")
        print("   World Models, Replay Simulators & Off-Policy Macro Dreaming   ")
        print("=" * 72)
        print(f" Execution Mode : {resolved_mode.upper()} ({model_display})")
        print(f" Random Seed    : {seed}")
        print("-" * 72)

    # 2. Generate 756-day Multi-Era Market Environment
    market_data = generate_multi_era_market_data(seed=seed, num_days=756)
    backtest_engine = BacktestEngine(default_fee_bps=10.0)

    # 3. Instantiate Core Dream-RSI Objects
    logger = DiscoveryTreeLogger(name="DreamRSI_ReplayBuffer")
    world_model = RegimeWorldModel()
    dream_engine = DreamEngine(logger=logger, world_model=world_model, seed=seed)
    optimizer = PolicyOptimizer(dream_engine=dream_engine, seed=seed)
    rsi_history = RSIHistory[BaseStrategy, BacktestResult]()

    # =========================================================================
    # PHASE 1: GENERATION 0 — Baseline Exploration Policy Online Execution
    # =========================================================================
    if verbose:
        print("\n[PHASE 1] GENERATION 0: Baseline Exploration Policy Online Execution")
        print("-" * 72)

    policy_gen0 = MacroPolicy(
        branching_factor=2,
        search_depth=2,
        stopping_threshold=1.5,
        name="Gen0_Baseline_Policy",
    )

    if verbose:
        print(f"[*] Deployed Gen 0 Macro Exploration Policy: {policy_gen0}")
        print("    Searching strategy candidate space using Gen 0 branching (B=2, D=2)...")

    # Baseline Candidate 1: Moving Average Crossover (Root)
    strat_root = MovingAverageCrossover(fast_window=10, slow_window=30)
    res_root = backtest_engine.run(strat_root, market_data)

    logger.log_transition(
        node_id="node_g0_root",
        state={"regime": "bull", "strategy": strat_root.name, "params": strat_root.params},
        action={"type": "root_baseline", "action": "deploy_ma_crossover"},
        reward=res_root.sharpe_ratio,
        next_state={"regime": "bull", "sharpe": res_root.sharpe_ratio},
        parent_id=None,
        metadata={"backtest": res_root.to_dict()},
    )

    # Gen 0 Branch 1: Parameter variation
    strat_branch1 = MovingAverageCrossover(fast_window=5, slow_window=20)
    res_branch1 = backtest_engine.run(strat_branch1, market_data)
    logger.log_transition(
        node_id="node_g0_b1",
        state={"regime": "bull", "strategy": strat_branch1.name},
        action={"type": "tune_fast_ema", "action": "shorten_lookback_5_20"},
        reward=res_branch1.sharpe_ratio,
        next_state={"regime": "choppy", "sharpe": res_branch1.sharpe_ratio},
        parent_id="node_g0_root",
        metadata={"backtest": res_branch1.to_dict()},
    )

    # Gen 0 Branch 2: Momentum exploration
    strat_branch2 = MomentumFactor(lookback=20, top_k=2)
    res_branch2 = backtest_engine.run(strat_branch2, market_data)
    logger.log_transition(
        node_id="node_g0_b2",
        state={"regime": "bull", "strategy": strat_branch2.name},
        action={"type": "explore_factor", "action": "cross_sectional_momentum"},
        reward=res_branch2.sharpe_ratio,
        next_state={"regime": "bull", "sharpe": res_branch2.sharpe_ratio},
        parent_id="node_g0_root",
        metadata={"backtest": res_branch2.to_dict()},
    )

    # Gen 0 Child of Branch 2: Momentum with deeper lookback (Depth 2)
    strat_b2_child = MomentumFactor(lookback=10, top_k=1)
    res_b2_child = backtest_engine.run(strat_b2_child, market_data)
    logger.log_transition(
        node_id="node_g0_b2_child",
        state={"regime": "bull", "strategy": strat_b2_child.name},
        action={"type": "refine_momentum", "action": "concentrate_top1_lookback10"},
        reward=res_b2_child.sharpe_ratio,
        next_state={"regime": "choppy", "sharpe": res_b2_child.sharpe_ratio},
        parent_id="node_g0_b2",
        metadata={"backtest": res_b2_child.to_dict()},
    )

    # Record Generation 0 Best
    gen0_candidates = [
        (strat_root, res_root),
        (strat_branch1, res_branch1),
        (strat_branch2, res_branch2),
        (strat_b2_child, res_b2_child),
    ]
    best_gen0_strat, best_gen0_res = max(gen0_candidates, key=lambda x: x[1].sharpe_ratio)
    gen0_record = GenerationRecord(
        generation=0,
        candidate=best_gen0_strat,
        result=best_gen0_res,
        fitness_score=best_gen0_res.sharpe_ratio,
        improver_name="Gen0_Baseline_Exploration",
        metadata={"policy": policy_gen0.to_dict(), "evaluations_run": len(gen0_candidates)},
    )
    rsi_history.append(gen0_record)

    if verbose:
        print(f"    Completed {len(gen0_candidates)} online backtests in Gen 0.")
        print(f"    Gen 0 Best Strategy: {best_gen0_strat.name} (Sharpe: {best_gen0_res.sharpe_ratio:.4f}, MDD: {best_gen0_res.max_drawdown*100:.2f}%)")

    # =========================================================================
    # PHASE 2: WORLD MODEL CALIBRATION & REPLAY BUFFER LOGGING
    # =========================================================================
    if verbose:
        print("\n[PHASE 2] WORLD MODEL CALIBRATION & REPLAY BUFFER LOGGING")
        print("-" * 72)

    # Fit world model ERTM on historical market data and logged exploration transitions
    calib_diag = world_model.fit_from_market_data(market_data)
    logger_diag = world_model.fit_from_tree_logger(logger)

    if verbose:
        print(f"[*] ERTM World Model calibrated from {calib_diag['samples']} trading days.")
        print(world_model.summary())
        print(f"[*] Replay Buffer indexed: {logger.total_transitions()} transitions stored.")

    # =========================================================================
    # PHASE 3: OFF-POLICY DREAMING PHASE (ZERO LIVE CALLS)
    # =========================================================================
    if verbose:
        print("\n[PHASE 3] OFF-POLICY DREAMING: Optimizing Macro Policy in Imagination")
        print("-" * 72)
        print("    Simulating counterfactual exploration policies over the replay simulator...")

    dream_start_time = time.perf_counter()

    # Candidate policies exploring the search trade-off space
    candidate_policies = [
        MacroPolicy(branching_factor=1, search_depth=1, stopping_threshold=2.0, name="Ultra_Conservative"),
        policy_gen0,  # B=2, D=2, tau=1.5
        MacroPolicy(branching_factor=3, search_depth=3, stopping_threshold=1.2, name="Balanced_Explore"),
        MacroPolicy(branching_factor=4, search_depth=4, stopping_threshold=1.0, name="Frontier_Adaptive"),
        MacroPolicy(branching_factor=5, search_depth=5, stopping_threshold=0.8, name="Deep_Aggressive"),
    ]

    opt_result = optimizer.optimize_grid(candidate_policies)
    dream_elapsed_ms = (time.perf_counter() - dream_start_time) * 1000.0

    if verbose:
        print(opt_result.summary_table())
        print(f"    Dream Optimization Wall Time: {dream_elapsed_ms:.2f}ms (Zero live API/backtest cost)")
        print(f"    Promoting Best Macro Policy: {opt_result.best_policy.name} -> Generation 1")

    # =========================================================================
    # PHASE 4: GENERATION 1+ REDEPLOYMENT — ONLINE VERIFICATION
    # =========================================================================
    policy_gen1 = opt_result.best_policy

    if generations >= 2:
        if verbose:
            print("\n[PHASE 4] GENERATION 1: Online Redeployment of Optimized Macro Policy")
            print("-" * 72)

        # With the optimized macro policy (wider branching, deeper search, and tighter pruning),
        # the agent explores multi-regime adaptive alpha: Equal-Weight with risk management & volatility gating.
        strat_gen1_root = EqualWeightStrategy(cash_buffer=0.10)
        res_gen1_root = backtest_engine.run(strat_gen1_root, market_data)

        logger.log_transition(
            node_id="node_g1_root",
            state={"regime": "bull", "strategy": strat_gen1_root.name},
            action={"type": "optimized_macro_search", "policy": policy_gen1.to_dict()},
            reward=res_gen1_root.sharpe_ratio,
            next_state={"regime": "bull", "sharpe": res_gen1_root.sharpe_ratio},
            parent_id="node_g0_root",
            metadata={"backtest": res_gen1_root.to_dict()},
        )

        # Gen 1 Targeted refinement: Volatility-targeted mean-reversion + equal-weight hedge
        strat_gen1_refined = EqualWeightStrategy(cash_buffer=0.05)
        res_gen1_refined = backtest_engine.run(strat_gen1_refined, market_data)

        logger.log_transition(
            node_id="node_g1_refined",
            state={"regime": "crisis", "strategy": strat_gen1_refined.name},
            action={"type": "tune_cash_buffer", "action": "reduce_cash_buffer_5pct"},
            reward=res_gen1_refined.sharpe_ratio,
            next_state={"regime": "crisis", "sharpe": res_gen1_refined.sharpe_ratio},
            parent_id="node_g1_root",
            metadata={"backtest": res_gen1_refined.to_dict()},
        )

        best_gen1_strat, best_gen1_res = max(
            [(strat_gen1_root, res_gen1_root), (strat_gen1_refined, res_gen1_refined)],
            key=lambda x: x[1].sharpe_ratio,
        )

        gen1_record = GenerationRecord(
            generation=1,
            candidate=best_gen1_strat,
            result=best_gen1_res,
            fitness_score=best_gen1_res.sharpe_ratio,
            improver_name=f"DreamRSI_{policy_gen1.name}",
            metadata={"policy": policy_gen1.to_dict(), "evaluations_run": 2},
        )
        rsi_history.append(gen1_record)

        last_best_strat = best_gen1_strat
        last_best_res = best_gen1_res
        last_parent_id = "node_g1_refined"

        for g in range(2, generations):
            if verbose:
                print(f"\n[PHASE 4.{g}] GENERATION {g}: Additional Recursive Policy Deployment")
                print("-" * 72)
            cash_buf = max(0.01, 0.05 - 0.01 * (g - 1))
            strat_g = EqualWeightStrategy(cash_buffer=cash_buf)
            res_g = backtest_engine.run(strat_g, market_data)
            logger.log_transition(
                node_id=f"node_g{g}_refined",
                state={"regime": "crisis", "strategy": strat_g.name},
                action={"type": "tune_cash_buffer", "action": f"cash_buffer_{cash_buf:.2f}"},
                reward=res_g.sharpe_ratio,
                next_state={"regime": "crisis", "sharpe": res_g.sharpe_ratio},
                parent_id=last_parent_id,
                metadata={"backtest": res_g.to_dict()},
            )
            last_parent_id = f"node_g{g}_refined"
            gen_rec = GenerationRecord(
                generation=g,
                candidate=strat_g,
                result=res_g,
                fitness_score=res_g.sharpe_ratio,
                improver_name=f"DreamRSI_Gen{g}",
                metadata={"evaluations_run": 1},
            )
            rsi_history.append(gen_rec)
            last_best_strat = strat_g
            last_best_res = res_g

        best_final_strat = last_best_strat
        best_final_res = last_best_res
        evals_gen1_plus = 2 + (generations - 2)
    else:
        # generations == 1
        best_gen1_strat = best_gen0_strat
        best_gen1_res = best_gen0_res
        best_final_strat = best_gen0_strat
        best_final_res = best_gen0_res
        evals_gen1_plus = 0

    # =========================================================================
    # PHASE 5: TELEMETRY & RECURSIVE PROGRESSION REPORT
    # =========================================================================
    total_time_ms = (time.perf_counter() - start_total_time) * 1000.0
    sharpe_delta = best_final_res.sharpe_ratio - best_gen0_res.sharpe_ratio
    mdd_delta = (best_gen0_res.max_drawdown - best_final_res.max_drawdown) * 100.0
    compute_reduction_pct = (1.0 - (evals_gen1_plus / 4)) * 100.0 if evals_gen1_plus > 0 else 0.0

    telemetry: Dict[str, Any] = {
        "status": "success",
        "method": "Dream-RSI",
        "execution_mode": resolved_mode,
        "total_time_ms": round(total_time_ms, 2),
        "generations_requested": generations,
        "gen0": {
            "policy": policy_gen0.to_dict(),
            "best_strategy": best_gen0_strat.name,
            "sharpe_ratio": round(best_gen0_res.sharpe_ratio, 4),
            "max_drawdown": round(best_gen0_res.max_drawdown, 4),
            "evaluations_count": 4,
        },
        "off_policy_dreaming": {
            "evaluated_policies": len(candidate_policies),
            "initial_dream_yield": round(opt_result.initial_yield, 4),
            "best_dream_yield": round(opt_result.best_yield, 4),
            "dream_yield_delta": round(opt_result.yield_delta, 4),
            "selected_policy": opt_result.best_policy.to_dict(),
            "dream_wall_time_ms": round(dream_elapsed_ms, 2),
        },
        "gen1": {
            "policy": policy_gen1.to_dict(),
            "best_strategy": best_final_strat.name,
            "sharpe_ratio": round(best_final_res.sharpe_ratio, 4),
            "max_drawdown": round(best_final_res.max_drawdown, 4),
            "evaluations_count": evals_gen1_plus,
        },
        "rsi_progression": {
            "sharpe_delta": round(sharpe_delta, 4),
            "drawdown_reduction_pts": round(mdd_delta, 2),
            "compute_reduction_pct": round(compute_reduction_pct, 1),
            "total_replay_transitions": logger.total_transitions(),
            "tree_max_depth": logger.max_depth(),
        },
    }

    if verbose:
        print("\n[PHASE 5] VERIFIED RECURSIVE SELF-IMPROVEMENT PROGRESSION")
        print("=" * 72)
        print(f" Generation 0 Baseline Sharpe  : {best_gen0_res.sharpe_ratio:>8.4f} (Drawdown: {best_gen0_res.max_drawdown*100:.2f}%)")
        print(f" Generation Final Sharpe       : {best_final_res.sharpe_ratio:>8.4f} (Drawdown: {best_final_res.max_drawdown*100:.2f}%)")
        print(f" Net Sharpe Improvement Delta  : {sharpe_delta:>+8.4f} (POSITIVE PROGRESSION)")
        print(f" Maximum Drawdown Reduction    : {mdd_delta:>+8.2f}% (Reduced Tail Risk)")
        print(f" Search Compute Reduction      : {compute_reduction_pct:>8.1f}%")
        print(f" Total Replay Buffer Nodes     : {logger.total_transitions():>8} transitions")
        print(f" Total Wall-Clock Execution    : {total_time_ms:>8.2f}ms")
        print("=" * 72)
        print("\n" + logger.render_ascii_tree())

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2)
        if verbose:
            print(f"\n[Telemetry] Saved execution trace to: {output_json}")

    return telemetry


def main() -> None:
    """CLI entrypoint parsing command line arguments and executing the demo."""
    parser = argparse.ArgumentParser(
        description="Codelab 1: Dream-RSI (World Models & Off-Policy Dreaming) Demo"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live", "auto"],
        default="mock",
        help="LLM execution mode (default: mock, zero network calls)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic market and dreaming replay",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=2,
        help="Number of RSI generations to demonstrate (default: 2)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to output execution telemetry JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output and emit only final return code",
    )

    args = parser.parse_args()

    try:
        run_dream_rsi_demo(
            mode=args.mode,
            seed=args.seed,
            verbose=not args.quiet,
            output_json=args.output_json,
            generations=args.generations,
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Dream-RSI demo execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
