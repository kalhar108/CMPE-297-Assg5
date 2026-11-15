"""Adversarial stress test harness authored by challenger_rsi_2.

Systematically challenges:
1. Generational progression authenticity (Dream-RSI, AIDE2, AI Scientist 2).
2. Code tree MCTS, AST repair, and MetaHarness progression in AIDE2.
3. Multi-era ablation, reviewer personas, and MASS loop prior integration in The AI Scientist 2.
4. Multi-CWD execution stability across /, assignment-5-rsi/, and individual codelabs.
5. Live-to-mock fallback resilience with invalid API keys.
6. Non-standard generation counts (generations 0, 1, 2, 4, 5).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# Setup paths
_script_dir = Path(__file__).resolve().parent
_assignment_root = _script_dir.parent
if str(_assignment_root) not in sys.path:
    sys.path.insert(0, str(_assignment_root))


def run_cmd(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Runs command with captured output and UTF-8 decoding."""
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_dream_rsi_progression_authenticity() -> Dict[str, Any]:
    """Test 1.1: Verify Dream-RSI Gen 0 vs Gen 1 progression."""
    from common.domain.backtester import BacktestEngine
    from common.domain.market_data import generate_multi_era_market_data
    from common.domain.strategy_base import (
        EqualWeightStrategy,
        MeanReversionStrategy,
        MomentumFactor,
        MovingAverageCrossover,
    )
    import importlib
    dream_demo = importlib.import_module("01_dream_rsi.demo")
    run_dream_rsi_demo = dream_demo.run_dream_rsi_demo
    dream_eng_mod = importlib.import_module("01_dream_rsi.dream_engine")
    DreamEngine, MacroPolicy = dream_eng_mod.DreamEngine, dream_eng_mod.MacroPolicy
    pol_opt_mod = importlib.import_module("01_dream_rsi.policy_optimizer")
    PolicyOptimizer = pol_opt_mod.PolicyOptimizer
    tree_log_mod = importlib.import_module("01_dream_rsi.tree_logger")
    DiscoveryTreeLogger = tree_log_mod.DiscoveryTreeLogger
    wm_mod = importlib.import_module("01_dream_rsi.world_model")
    RegimeWorldModel = wm_mod.RegimeWorldModel

    results: Dict[str, Any] = {}

    # Check backtest math across multiple seeds
    engine = BacktestEngine(default_fee_bps=10.0)
    seed_comparisons = []
    for s in [42, 100, 2026]:
        mdata = generate_multi_era_market_data(seed=s, num_days=756)
        res_ma = engine.run(MovingAverageCrossover(fast_window=10, slow_window=30), mdata)
        res_mom = engine.run(MomentumFactor(lookback=20, top_k=2), mdata)
        best_g0 = max(res_ma.sharpe_ratio, res_mom.sharpe_ratio)

        res_eq = engine.run(EqualWeightStrategy(cash_buffer=0.10), mdata)
        seed_comparisons.append({
            "seed": s,
            "best_g0_sharpe": best_g0,
            "g1_equal_weight_sharpe": res_eq.sharpe_ratio,
            "g1_outperforms_g0": res_eq.sharpe_ratio > best_g0,
        })
    results["seed_comparisons"] = seed_comparisons

    # Inspect code structure: Is EqualWeightStrategy statically hardcoded in demo.py?
    demo_path = _assignment_root / "01_dream_rsi" / "demo.py"
    demo_code = demo_path.read_text(encoding="utf-8")
    results["hardcoded_strategy_in_demo"] = (
        "EqualWeightStrategy(cash_buffer=0.10)" in demo_code
        and "strat_gen1_root = EqualWeightStrategy" in demo_code
    )

    # Check if MacroPolicy in Gen 1 is actually connected to code synthesis
    # Does MacroPolicy have an improver or synthesizer?
    policy = MacroPolicy(branching_factor=5, search_depth=5, stopping_threshold=0.8)
    results["policy_has_code_synthesis_method"] = hasattr(policy, "synthesize") or hasattr(policy, "improve")

    return results


def test_aide2_mcts_ast_and_metaharness() -> Dict[str, Any]:
    """Test 1.2: Verify AIDE2 MCTS, AST repair, and MetaHarness progression."""
    from common.domain.backtester import BacktestEngine
    from common.domain.market_data import generate_multi_era_market_data
    import importlib
    aide_mcts = importlib.import_module("02_aide2.mcts_search")
    AIDE2Search = aide_mcts.AIDE2Search
    aide_meta = importlib.import_module("02_aide2.meta_harness")
    MetaHarness = aide_meta.MetaHarness
    aide_ops = importlib.import_module("02_aide2.operators")
    DebuggingOperator, DraftingOperator, RefinementOperator = aide_ops.DebuggingOperator, aide_ops.DraftingOperator, aide_ops.RefinementOperator
    aide_tree = importlib.import_module("02_aide2.tree_node")
    StrategyNode = aide_tree.StrategyNode

    results: Dict[str, Any] = {}

    # 1. AST repair testing on diverse errors
    debugger = DebuggingOperator()
    code_zero_div = "def run(x):\n    return x / 0.0\n"
    fixed_zero = debugger.debug_and_fix(code_zero_div, "ZeroDivisionError: float division by zero")
    results["ast_repair_zero_division"] = "1e-6" in fixed_zero or "+ 1e-8" in fixed_zero or "/ 0.0" not in fixed_zero

    code_syntax = "def bad_syntax(:\n    pass\n"
    fixed_syntax = debugger.debug_and_fix(code_syntax, "SyntaxError: invalid syntax")
    results["ast_repair_syntax_error"] = "def bad_syntax():" in fixed_syntax

    code_index = "def get_p(close, t):\n    return close[t + self.window]\n"
    fixed_index = debugger.debug_and_fix(code_index, "IndexError: index out of range")
    results["ast_repair_index_error"] = "close[t + self.window]" not in fixed_index

    # 2. Inspect MetaHarness.run_meta_optimization
    harness = MetaHarness()
    gens = harness.run_meta_optimization(generations=3, iterations_per_gen=2)
    results["metaharness_gens_returned"] = len(gens)
    results["metaharness_sharpes"] = [g["best_sharpe"] for g in gens]

    # Verify if root node metrics were hardcoded to sharpe_base (1.10, 1.45, 1.82)
    # Check if child nodes in active search instances were ever evaluated
    search_instances = harness.active_search_instances
    child_eval_status = []
    for s_idx, s in enumerate(search_instances):
        for nid, node in s.nodes.items():
            if nid != s.root_id:
                child_eval_status.append({
                    "search_gen": s_idx,
                    "node_id": nid,
                    "metrics_is_none": node.metrics is None,
                    "sharpe": node.sharpe,
                })
    results["child_nodes_evaluated"] = child_eval_status

    return results


def test_ai_scientist_2_components() -> Dict[str, Any]:
    """Test 1.3: Verify The AI Scientist 2 components."""
    from common.domain.backtester import BacktestEngine
    from common.domain.market_data import generate_multi_era_market_data
    import importlib
    sci_abl = importlib.import_module("03_ai_scientist_2.ablation")
    MultiEraAblation = sci_abl.MultiEraAblation
    sci_exp = importlib.import_module("03_ai_scientist_2.experimenter")
    ExperimentManager = sci_exp.ExperimentManager
    sci_hyp = importlib.import_module("03_ai_scientist_2.hypothesis")
    HypothesisGenerator = sci_hyp.HypothesisGenerator
    sci_mass = importlib.import_module("03_ai_scientist_2.mass_loop")
    MASSLoop = sci_mass.MASSLoop
    sci_rev = importlib.import_module("03_ai_scientist_2.peer_review")
    PeerReviewBoard = sci_rev.PeerReviewBoard

    results: Dict[str, Any] = {}

    mdata = generate_multi_era_market_data(seed=42, num_days=756)
    engine = BacktestEngine()
    gen = HypothesisGenerator()
    hyp = gen.generate_hypothesis(generation=0)

    # 1. Hypothesis generated
    results["hypothesis_title"] = hyp.title
    results["hypothesis_has_formula"] = bool(hyp.formula)
    results["hypothesis_has_code"] = bool(hyp.strategy_code)

    # 2. Multi-era ablation across 3 regimes
    exp_mgr = ExperimentManager(engine, mdata)
    strat = exp_mgr.synthesize_strategy(hyp)
    ablation = MultiEraAblation(engine, mdata)
    ablation_res = ablation.run_ablation(strat)
    results["ablation_regimes_tested"] = list(ablation_res.keys())
    results["all_3_regimes_present"] = set(ablation_res.keys()) == {"bull", "choppy", "crisis"}

    # 3. Peer Reviewer personas
    reviewer = PeerReviewBoard()
    review = reviewer.evaluate(hyp, ablation_res)
    individual_reviews = review.get("individual_reviews", [])
    reviewer_ids = [r.get("reviewer_id") for r in individual_reviews]
    results["individual_reviews_count"] = len(individual_reviews)
    results["reviewer_personas"] = reviewer_ids
    # 4 distinct reviewer personas: Reviewer 1, 2, 3 + Meta-Reviewer / Area Chair
    results["has_area_chair_synthesis"] = "critique" in review and "overall_score" in review

    # 4. MASS Loop distillation
    mass = MASSLoop()
    init_priors_count = len(mass.distilled_priors)
    new_prior = mass.distill(review)
    results["mass_distilled_new_prior"] = new_prior.startswith("Distilled prior:")
    results["priors_count_increased"] = len(mass.distilled_priors) == init_priors_count + 1

    return results


def test_cli_execution_stability() -> Dict[str, Any]:
    """Test 2: Multi-CWD invocations, invalid flags, and generation counts."""
    results: Dict[str, Any] = {}
    repo_root = _assignment_root.parent.parent

    # 1. Non-standard generations on 02_aide2
    res_aide_g2 = run_cmd([sys.executable, "02_aide2/demo.py", "--generations", "2"], _assignment_root)
    results["aide2_generations_2_exit_code"] = res_aide_g2.returncode
    results["aide2_generations_2_has_index_error"] = "IndexError" in res_aide_g2.stderr or "IndexError" in res_aide_g2.stdout

    res_aide_g0 = run_cmd([sys.executable, "02_aide2/demo.py", "--generations", "0"], _assignment_root)
    results["aide2_generations_0_exit_code"] = res_aide_g0.returncode
    results["aide2_generations_0_has_index_error"] = "IndexError" in res_aide_g0.stderr or "IndexError" in res_aide_g0.stdout

    # 2. Non-standard generations on 03_ai_scientist_2
    res_sci_g0 = run_cmd([sys.executable, "03_ai_scientist_2/demo.py", "--generations", "0"], _assignment_root)
    results["ai_scientist_generations_0_exit_code"] = res_sci_g0.returncode
    results["ai_scientist_generations_0_has_error"] = (
        "ValueError" in res_sci_g0.stderr or "ValueError" in res_sci_g0.stdout
    )

    # 3. Multi-CWD invocations
    # CWD: repo_root (d:\SJSU\Fall26\Special Topics)
    cwd_tests = {}
    rel_path = "Special-Topics-in-AI/assignment-5-rsi"

    # From repo root: python -m ...
    res_root_m1 = run_cmd([sys.executable, f"{rel_path}/01_dream_rsi/demo.py", "--quiet"], repo_root)
    cwd_tests["root_to_01_script"] = res_root_m1.returncode

    # From assignment-5-rsi: python demo.py & python -m
    res_asgn_01 = run_cmd([sys.executable, "01_dream_rsi/demo.py", "--quiet"], _assignment_root)
    res_asgn_m01 = run_cmd([sys.executable, "-m", "01_dream_rsi.demo", "--quiet"], _assignment_root)
    cwd_tests["asgn_to_01_script"] = res_asgn_01.returncode
    cwd_tests["asgn_to_01_module"] = res_asgn_m01.returncode

    # From inside 01_dream_rsi/
    res_inside_01 = run_cmd([sys.executable, "demo.py", "--quiet"], _assignment_root / "01_dream_rsi")
    cwd_tests["inside_01_script"] = res_inside_01.returncode

    # From inside 02_aide2/
    res_inside_02 = run_cmd([sys.executable, "demo.py"], _assignment_root / "02_aide2")
    cwd_tests["inside_02_script"] = res_inside_02.returncode

    # From inside 03_ai_scientist_2/
    res_inside_03 = run_cmd([sys.executable, "demo.py"], _assignment_root / "03_ai_scientist_2")
    cwd_tests["inside_03_script"] = res_inside_03.returncode

    # From inside 04_comparative_benchmark/
    res_inside_04 = run_cmd([sys.executable, "demo.py"], _assignment_root / "04_comparative_benchmark")
    cwd_tests["inside_04_script"] = res_inside_04.returncode

    results["cwd_matrix"] = cwd_tests

    # 4. Live-to-mock fallback testing with invalid key / broken network
    from unittest.mock import patch
    from common.config import RSIExecutionConfig
    from common.llm_provider import GeminiLiveProvider, get_llm_provider

    cfg_live_invalid = RSIExecutionConfig(mode="live", api_key="INVALID_TEST_KEY_12345")
    provider = get_llm_provider(cfg_live_invalid)
    results["live_provider_init_class"] = type(provider).__name__

    # When live calls fail (e.g. network broken / invalid key rejection), does provider catch or raise?
    live_call_raised = False
    with patch.object(provider.client.models, "generate_content", side_effect=ConnectionError("Simulated Network Failure")):
        try:
            provider.generate_text("test prompt")
        except Exception as e:
            live_call_raised = True
            results["live_call_raised_exception"] = f"{type(e).__name__}: {str(e)}"
    results["live_call_raised_without_internal_fallback"] = live_call_raised

    return results


def main() -> None:
    print("=== Running Adversarial Stress Harness (challenger_rsi_2) ===", flush=True)
    out: Dict[str, Any] = {}

    print("\n--- 1. Testing Dream-RSI Progression Authenticity ---", flush=True)
    out["dream_rsi"] = test_dream_rsi_progression_authenticity()
    print(json.dumps(out["dream_rsi"], indent=2), flush=True)

    print("\n--- 2. Testing AIDE2 MCTS, AST, and MetaHarness ---", flush=True)
    out["aide2"] = test_aide2_mcts_ast_and_metaharness()
    print(json.dumps(out["aide2"], indent=2), flush=True)

    print("\n--- 3. Testing The AI Scientist 2 Components ---", flush=True)
    out["ai_scientist_2"] = test_ai_scientist_2_components()
    print(json.dumps(out["ai_scientist_2"], indent=2), flush=True)

    print("\n--- 4. Testing CLI Stability & Edge Cases ---", flush=True)
    out["cli_stability"] = test_cli_execution_stability()
    print(json.dumps(out["cli_stability"], indent=2), flush=True)

    out_file = _assignment_root / "scripts" / "adversarial_results.json"
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[OK] Results saved to: {out_file}", flush=True)


if __name__ == "__main__":
    main()
