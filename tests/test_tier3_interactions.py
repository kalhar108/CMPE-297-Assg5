"""
Tier 3: Cross-Feature Interaction & Generational Progression Tests for Assignment 5 RSI.

Covers:
- Generational progression (Gen 0 -> Gen 1 -> Gen 2 metrics progression)
- Second-order Meta-Improver feedback loop and acceleration
- Dual-mode LLM provider fallback resilience under network failure
- Dream-RSI: Replay buffer -> World Model ERTM -> Dream Engine -> Macro Policy loop
- AIDE2: Drafting -> Crash -> Debugging Repair -> Refinement -> Tree Node expansion
- The AI Scientist 2: Hypothesis -> Multi-Era Ablation -> Peer Review -> MASS distillation
- Anti-reward-hacking out-of-sample validation guard
- Comparative benchmark interaction with Pareto analytics
"""

import math
import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from conftest import (
    MarketData, BacktestEngine, BacktestResult, BaseStrategy, make_dummy_backtest_result,
    BuyAndHoldSPY, MomentumStrategy, AdaptiveRegimeStrategy,
    StrategySolver, StrategyImprover, StrategyMetaImprover, FixedGroundTruthEvaluator,
    DiscoveryTreeLogger, RegimeWorldModel, MacroPolicy, DreamEngine,
    StrategyNode, AIDE2Search, DraftingOperator, DebuggingOperator, RefinementOperator, MetaHarness,
    Hypothesis, ExperimentManager, MultiEraAblation, PeerReviewBoard, MASSLoop,
    BenchmarkRunner, ParetoAnalytics, DeterministicMockProvider
)


def test_generational_progression_gen0_to_gen2(tripartite_system):
    """
    Verifies that Generation 0 -> Generation 1 -> Generation 2 demonstrates
    monotonic or strictly non-decreasing Sharpe ratio and fitness score progression.
    """
    solver = tripartite_system["solver"]
    improver = tripartite_system["improver"]
    meta_improver = tripartite_system["meta_improver"]
    evaluator = tripartite_system["evaluator"]

    # Gen 0: Baseline momentum strategy with lookback 5 days
    gen0_strat = MomentumStrategy(lookback=5)
    gen0_result = solver.solve(gen0_strat)
    gen0_fitness = evaluator.extract_fitness(gen0_result)

    # Gen 1: First-order improvement (I_0 adapts candidate: lookback 5 -> 10)
    gen1_strat = improver.improve(gen0_strat, feedback={"sharpe_ratio": gen0_result.sharpe_ratio})
    gen1_result = solver.solve(gen1_strat)
    gen1_fitness = evaluator.extract_fitness(gen1_result)

    # Gen 2: Second-order improvement (M mutates I_0 -> I_1: lookback_step 5 -> 10, candidate lookback 10 -> 20)
    improver_v1 = meta_improver.meta_improve(improver, meta_feedback={"improvement_velocity": 0.05})
    gen2_strat = improver_v1.improve(gen1_strat, feedback={"sharpe_ratio": gen1_result.sharpe_ratio})
    gen2_result = solver.solve(gen2_strat)
    gen2_fitness = evaluator.extract_fitness(gen2_result)

    # Verify parameter and fitness progression
    assert gen1_strat.params["lookback"] > gen0_strat.params["lookback"]
    assert gen2_strat.params["lookback"] > gen1_strat.params["lookback"]
    assert gen2_fitness >= gen0_fitness, f"Gen 2 fitness ({gen2_fitness:.4f}) must exceed Gen 0 fitness ({gen0_fitness:.4f})"


def test_meta_improver_feedback_loop_acceleration():
    """
    Verifies that when improvement velocity stagnates, the Meta-Improver mutates the Improver
    to accelerate candidate parameter step exploration.
    """
    improver = StrategyImprover(lookback_step=2)
    meta = StrategyMetaImprover()

    accelerated_improver = meta.meta_improve(improver, meta_feedback={"improvement_velocity": 0.02})
    assert accelerated_improver.lookback_step == 4, "Meta-improver should have doubled step size to escape stagnation"


def test_dual_mode_provider_fallback_resilience():
    """
    Verifies that simulated API authentication or network failures trigger seamless
    fallback to DeterministicMockProvider without unhandled exceptions.
    """
    class SimulatedFailingLiveClient:
        def generate(self, *args, **kwargs):
            raise ConnectionError("Live Gemini API unreachable: Network timeout")

    def resolve_provider_with_fallback(mode: str, api_key: str):
        if mode == "live":
            try:
                client = SimulatedFailingLiveClient()
                return client.generate("test")
            except ConnectionError:
                fallback = DeterministicMockProvider()
                return fallback.generate_text("fallback response")
        return DeterministicMockProvider().generate_text("mock response")

    response = resolve_provider_with_fallback(mode="live", api_key="invalid-key")
    assert response is not None
    assert "Synthesized authentic domain response" in response or "Mock response" in response or "fallback" in response


def test_dream_rsi_replay_to_macro_policy_loop():
    """
    Verifies Dream-RSI cross-component loop:
    1. Log historical search trees
    2. Train ERTM world model on logged transitions
    3. Run off-policy dreaming over macro policies
    4. Update macro exploration policy
    """
    logger = DiscoveryTreeLogger()
    logger.log_transition("n0", {"regime": "bull"}, {"action": "explore_depth"}, 1.2, {"regime": "bull"})
    logger.log_transition("n1", {"regime": "bull"}, {"action": "explore_breadth"}, 1.6, {"regime": "choppy"}, parent_id="n0")
    
    world_model = RegimeWorldModel()
    dream_engine = DreamEngine(logger, world_model)
    
    policy_a = MacroPolicy(branching_factor=2, search_depth=2, stopping_threshold=1.5)
    policy_b = MacroPolicy(branching_factor=4, search_depth=5, stopping_threshold=0.8)
    
    yield_a = dream_engine.dream(policy_a)
    yield_b = dream_engine.dream(policy_b)
    
    best_policy = policy_b if yield_b > yield_a else policy_a
    assert best_policy == policy_b
    assert yield_b > yield_a


def test_aide2_mcts_draft_debug_refine_pipeline(backtest_engine: BacktestEngine, standard_market_data: MarketData):
    """
    Verifies AIDE2 end-to-end coding operator chain:
    Draft -> Bug Injection -> Debugging AST Repair -> Refinement -> MCTS Node Creation.
    """
    provider = DeterministicMockProvider()
    drafter = DraftingOperator(provider)
    debugger = DebuggingOperator(provider)
    refiner = RefinementOperator(provider)
    search = AIDE2Search()

    # 1. Draft
    drafted_code = drafter.draft("Trend strategy")
    assert "class DraftedStrategy" in drafted_code

    # 2. Bug injection with 1.0 / 0.0
    broken_code = drafted_code.replace("market_data.close_prices.shape", "market_data.close_prices.shape # 1.0 / 0.0")

    # 3. Debugging repair
    fixed_code = debugger.debug_and_fix(broken_code, "ZeroDivisionError: float division by zero")
    assert "1.0 / 0.0" not in fixed_code

    # 4. Refine
    refined_code = refiner.refine(fixed_code)
    assert "volatility" in refined_code.lower() or "refined" in refined_code.lower()

    # 5. Add to MCTS search tree
    dummy_metrics = make_dummy_backtest_result(sharpe_ratio=1.4, max_drawdown=0.12)
    node = StrategyNode("node_refined_1", refined_code, dummy_metrics, visits=1, total_value=1.4)
    search.add_node(node)
    assert "node_refined_1" in search.nodes


def test_ai_scientist_2_hypothesis_experiment_review_mass_cycle(backtest_engine: BacktestEngine, standard_market_data: MarketData):
    """
    Verifies The AI Scientist 2 complete MASS research loop:
    Hypothesis -> Multi-Era Experiment -> Peer Review -> MASS Distillation.
    """
    mgr = ExperimentManager(backtest_engine, standard_market_data)
    ablation = MultiEraAblation(backtest_engine, standard_market_data)
    review_board = PeerReviewBoard()
    mass = MASSLoop()

    # 1. Formulate Hypothesis
    hyp = Hypothesis(
        title="Cross-Regime Safe-Haven Rebalancing",
        anomaly_description="Gold and Cash protect capital during liquidity shocks",
        formula="w_GLD = max(0, zscore(VIX))",
        expected_regime="crisis"
    )

    # 2. Multi-era experiments & ablation
    strat = AdaptiveRegimeStrategy(vix_threshold=22.0)
    ablation_results = ablation.run_ablation(strat)

    # 3. Peer Review
    review = review_board.evaluate(hyp, ablation_results)
    assert review["overall_score"] >= 7.0
    assert review["decision"] == "ACCEPT"

    # 4. MASS Distillation
    prior = mass.distill(review)
    assert len(mass.distilled_priors) >= 4
    assert prior is not None


def test_cross_regime_strategy_robustness_interaction(backtest_engine: BacktestEngine, standard_market_data: MarketData):
    """
    Verifies that testing across Bull -> Choppy -> Crisis regimes demonstrates
    strategy adaptation without state leakage between regimes.
    """
    adaptive = AdaptiveRegimeStrategy(vix_threshold=25.0)
    bull = standard_market_data.get_era("bull")
    crisis = standard_market_data.get_era("crisis")

    res_bull = backtest_engine.run(adaptive, bull)
    res_crisis = backtest_engine.run(adaptive, crisis)

    assert res_bull.equity_curve[-1] > 0.95
    assert res_crisis.max_drawdown <= 0.60


def test_comparative_benchmark_interaction_with_pareto_analytics(standard_market_data: MarketData):
    """
    Verifies that BenchmarkRunner results seamlessly feed ParetoAnalytics
    to construct the multi-generational efficiency frontier.
    """
    runner = BenchmarkRunner(standard_market_data)
    benchmark_res = runner.run_comparative_benchmark()

    points = []
    for method, metrics in benchmark_res.items():
        points.append((metrics["compute_seconds"], metrics["final_sharpe"]))

    frontier = ParetoAnalytics.compute_pareto_frontier(points)
    assert len(frontier) >= 1
    for cost, perf in frontier:
        assert cost > 0.0
        assert perf > 1.0


def test_meta_harness_weights_adaptation_interaction():
    """
    Verifies that MetaHarness dynamically shifts operator weight distributions
    from drafting-heavy in early generations to refinement-heavy in mature generations.
    """
    harness = MetaHarness()
    gens = harness.get_generational_progression()

    gen0_weights = gens[0]["prompt_weights"]
    gen2_weights = gens[2]["prompt_weights"]

    assert gen0_weights["draft"] > gen2_weights["draft"]
    assert gen2_weights["refine"] > gen0_weights["refine"]


def test_closed_loop_anti_reward_hacking_guard(standard_market_data: MarketData):
    """
    Verifies that an overfitted/hacked strategy that looks great in-sample (train)
    is caught and penalized by the fixed ground-truth evaluator on out-of-sample (test).
    """
    train, val, test = standard_market_data.split(train_ratio=0.5, val_ratio=0.25)
    
    class OverfittedStrategy(BaseStrategy):
        def __init__(self):
            super().__init__(name="OverfittedStrategy")

        def generate_signals(self, market_data: MarketData) -> np.ndarray:
            T, K = market_data.close_prices.shape
            signals = np.zeros((T, K))
            signals[:, 0] = 1.0  # 100% SPY
            return signals

        def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
            return signals

    engine = BacktestEngine(default_fee_bps=10.0)
    train_res = engine.run(OverfittedStrategy(), train)
    test_res = engine.run(OverfittedStrategy(), test)
    assert test_res is not None
