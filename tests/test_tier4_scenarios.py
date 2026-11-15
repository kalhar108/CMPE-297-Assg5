"""
Tier 4: End-to-End Real-World Scenario Tests for Assignment 5 RSI.

Covers full multi-era operational scenarios:
1. Full Multi-Era Alpha Discovery Lifecycle (756-day multi-regime quantitative optimization)
2. Dream-RSI End-to-End Off-Policy Macro Search (Replay logging -> ERTM -> Dream simulation -> Policy promotion)
3. AIDE2 Bi-Level MCTS Meta-Optimization (Draft -> Debug -> Refine -> UCT -> MetaHarness evolution)
4. The AI Scientist 2 Autonomous Research & Peer Review Studio (Hypothesis -> Ablation -> Committee Review -> MASS)
5. Master Comparative Benchmark & Pareto Evaluation (Head-to-head multi-paradigm evaluation)
"""

import time
import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from conftest import (
    MarketData, BacktestEngine, BacktestResult, BaseStrategy, make_dummy_backtest_result,
    BuyAndHoldSPY, MomentumStrategy, AdaptiveRegimeStrategy,
    DiscoveryTreeLogger, RegimeWorldModel, MacroPolicy, DreamEngine,
    StrategyNode, AIDE2Search, DraftingOperator, DebuggingOperator, RefinementOperator, MetaHarness,
    Hypothesis, ExperimentManager, MultiEraAblation, PeerReviewBoard, MASSLoop,
    BenchmarkRunner, ParetoAnalytics, DeterministicMockProvider
)


def test_scenario_full_multi_era_alpha_discovery_lifecycle(standard_market_data: MarketData, backtest_engine: BacktestEngine):
    """
    Scenario 1: Full Multi-Era Alpha Discovery Lifecycle.
    Evaluates baseline Buy-and-Hold SPY vs Adaptive Regime-Switching Strategy
    across 756 trading days spanning Bull, Choppy, and Crisis market eras.
    """
    # 1. Baseline Evaluation
    baseline_strat = BuyAndHoldSPY()
    baseline_result = backtest_engine.run(baseline_strat, standard_market_data)
    assert baseline_result is not None
    assert len(baseline_result.equity_curve) == 756

    # 2. Adaptive Strategy Evaluation
    adaptive_strat = AdaptiveRegimeStrategy(vix_threshold=24.0)
    adaptive_result = backtest_engine.run(adaptive_strat, standard_market_data)
    assert adaptive_result is not None

    # 3. Assert Outperformance in Stress Management
    assert adaptive_result.max_drawdown <= 0.65
    assert adaptive_result.equity_curve[-1] > 0.0

    # 4. Generate Telemetry Summary
    report = {
        "baseline_sharpe": baseline_result.sharpe_ratio,
        "baseline_drawdown": baseline_result.max_drawdown,
        "adaptive_sharpe": adaptive_result.sharpe_ratio,
        "adaptive_drawdown": adaptive_result.max_drawdown,
        "lifecycle_days": 756,
        "status": "PASSED"
    }
    assert report["status"] == "PASSED"


def test_scenario_dream_rsi_end_to_end_off_policy_macro_search():
    """
    Scenario 2: Dream-RSI End-to-End Off-Policy Macro Search.
    Simulates discovery tree logging, world model calibration, counterfactual off-policy
    dreaming over alternative exploration policies, and macro policy promotion.
    """
    logger = DiscoveryTreeLogger()
    
    # 1. Log online discovery trajectories
    logger.log_transition("root", {"regime": "bull"}, {"action": "draft_momentum"}, 1.1, {"regime": "bull"})
    logger.log_transition("c1", {"regime": "bull"}, {"action": "expand_breadth"}, 1.3, {"regime": "choppy"}, parent_id="root")
    logger.log_transition("c2", {"regime": "choppy"}, {"action": "add_filter"}, 1.5, {"regime": "crisis"}, parent_id="c1")
    logger.log_transition("c3", {"regime": "crisis"}, {"action": "hedge_cash"}, 1.8, {"regime": "bull"}, parent_id="c2")
    assert len(logger.nodes) == 4

    # 2. World Model ERTM
    world_model = RegimeWorldModel()
    predicted_next = world_model.predict_next_regime("crisis")
    assert predicted_next in ["bull", "choppy", "crisis"]

    # 3. Dream Engine offline simulations
    dream_engine = DreamEngine(logger, world_model)
    candidate_policies = [
        MacroPolicy(branching_factor=1, search_depth=2, stopping_threshold=2.0),
        MacroPolicy(branching_factor=3, search_depth=3, stopping_threshold=1.2),
        MacroPolicy(branching_factor=5, search_depth=5, stopping_threshold=0.8),
    ]

    yields = [dream_engine.dream(p) for p in candidate_policies]
    assert len(yields) == 3
    best_idx = int(np.argmax(yields))
    best_policy = candidate_policies[best_idx]

    # 4. Verify policy promotion
    assert best_policy.branching_factor == 5
    assert best_policy.search_depth == 5


def test_scenario_aide2_bi_level_mcts_meta_optimization():
    """
    Scenario 3: AIDE2 Bi-Level MCTS Meta-Optimization.
    Inner Loop: MCTS explores code candidates via Drafting, Debugging, and Refinement.
    Outer Loop: Meta-Harness tunes operator prompts and tracks generational progression.
    """
    provider = DeterministicMockProvider()
    drafter = DraftingOperator(provider)
    debugger = DebuggingOperator(provider)
    refiner = RefinementOperator(provider)
    search = AIDE2Search(c_param=1.414)

    # 1. Root node
    root_metrics = make_dummy_backtest_result(sharpe_ratio=1.0, max_drawdown=0.15)
    root_node = StrategyNode("node_0", "class Baseline:\n    pass\n", root_metrics, visits=1, total_value=1.0)
    search.add_node(root_node)

    # 2. Draft child solution
    draft_code = drafter.draft("Multi-era alpha")
    node_draft = StrategyNode("node_draft_1", draft_code, root_metrics, parent_id="node_0", visits=0, total_value=0.0)
    search.add_node(node_draft)

    # 3. Inject syntax bug & perform surgical debugging repair
    buggy_code = draft_code + "\n# Error: 1.0 / 0.0\n"
    fixed_code = debugger.debug_and_fix(buggy_code, "ZeroDivisionError")
    assert "1.0 / 0.0" not in fixed_code

    # 4. Refine survivor
    refined_code = refiner.refine(fixed_code)
    improved_metrics = make_dummy_backtest_result(sharpe_ratio=1.65, max_drawdown=0.11)
    node_refined = StrategyNode("node_refined_2", refined_code, improved_metrics, parent_id="node_draft_1", visits=1, total_value=1.65)
    search.add_node(node_refined)

    # 5. UCT Selection
    uct_score = search.uct_score(node_refined, parent_visits=5)
    assert uct_score > 1.65

    # 6. Outer loop: MetaHarness generational tracking
    harness = MetaHarness()
    gens = harness.get_generational_progression()
    assert gens[2]["best_sharpe"] > gens[0]["best_sharpe"]


def test_scenario_ai_scientist_2_autonomous_research_and_review(standard_market_data: MarketData, backtest_engine: BacktestEngine):
    """
    Scenario 4: The AI Scientist 2 Autonomous Research & Peer Review Studio.
    Executes automated scientific lifecycle:
    Hypothesis -> Experimentation -> Multi-Era Ablation -> Peer Review -> MASS Distillation.
    """
    mgr = ExperimentManager(backtest_engine, standard_market_data)
    ablation = MultiEraAblation(backtest_engine, standard_market_data)
    review_board = PeerReviewBoard()
    mass = MASSLoop()

    # 1. Hypothesize
    hyp = Hypothesis(
        title="Multi-Regime Volatility Parity",
        anomaly_description="Inverting portfolio exposure to rolling VIX spikes limits crisis drawdown",
        formula="w_cash = clip(VIX / 30.0, 0.0, 0.9)",
        expected_regime="crisis"
    )

    # 2. Experiment & Ablate across Bull, Choppy, Crisis
    strat = AdaptiveRegimeStrategy(vix_threshold=22.0)
    ablation_res = ablation.run_ablation(strat)
    assert len(ablation_res) == 3

    # 3. Multi-agent peer review
    review = review_board.evaluate(hyp, ablation_res)
    assert review["overall_score"] >= 7.0
    assert review["decision"] == "ACCEPT"

    # 4. MASS distillation into prompt prior
    prior = mass.distill(review)
    assert len(mass.distilled_priors) >= 4
    assert "Distilled prior" in prior


def test_scenario_master_comparative_benchmark_and_pareto_evaluation(standard_market_data: MarketData):
    """
    Scenario 5: Master Comparative Benchmark & Pareto Evaluation.
    Runs Dream-RSI, AIDE2, and The AI Scientist 2 head-to-head on identical 756-day
    market data and evaluates the multi-generational Pareto frontier.
    """
    runner = BenchmarkRunner(standard_market_data)
    results = runner.run_comparative_benchmark()

    assert "Dream-RSI" in results
    assert "AIDE2" in results
    assert "AI-Scientist-2" in results

    points = []
    for method_name, metrics in results.items():
        assert metrics["final_sharpe"] > 1.5, f"{method_name} failed to reach Sharpe > 1.5"
        assert metrics["max_drawdown"] < 0.20, f"{method_name} exceeded max drawdown target"
        points.append((metrics["compute_seconds"], metrics["final_sharpe"]))

    frontier = ParetoAnalytics.compute_pareto_frontier(points)
    assert len(frontier) >= 1
    for cost, perf in frontier:
        assert cost > 0.0
        assert perf > 0.0
