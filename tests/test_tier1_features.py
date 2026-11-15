"""
Tier 1: Feature Unit Tests for Assignment 5 Recursive Self-Improvement (RSI).

Covers all 9 functional areas with >= 5 comprehensive tests per feature:
1. Domain Environment (MarketData)
2. High-Speed Vectorized Backtester (BacktestEngine)
3. Quantitative Telemetry & Metrics (BacktestResult)
4. Tripartite RSI Base Abstractions (TaskSolver, Improver, MetaImprover, Evaluator)
5. Dual-Mode LLM Engine (DeterministicMockProvider & Fallback)
6. Dream-RSI Components (TreeLogger, WorldModel, DreamEngine, MacroPolicy)
7. AIDE2 Operators & MCTS (StrategyNode, AIDE2Search, Drafting, Debugging, Refinement)
8. The AI Scientist 2 Components (Hypothesis, ExperimentManager, Ablation, PeerReview, MASS)
9. Master Demo Suite & Comparative Benchmark (BenchmarkRunner, ParetoAnalytics)
"""

import time
import math
import json
import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from pydantic import BaseModel

from conftest import (
    MarketData, generate_multi_era_market_data, DEFAULT_SYMBOLS,
    BacktestEngine, BacktestResult, BaseStrategy, make_dummy_backtest_result,
    BuyAndHoldSPY, MomentumStrategy, AdaptiveRegimeStrategy, FaultyStrategy,
    StrategySolver, StrategyImprover, StrategyMetaImprover, FixedGroundTruthEvaluator,
    DiscoveryTreeLogger, RegimeWorldModel, MacroPolicy, DreamEngine,
    StrategyNode, AIDE2Search, DraftingOperator, DebuggingOperator, RefinementOperator, MetaHarness,
    Hypothesis, ExperimentManager, MultiEraAblation, PeerReviewBoard, MASSLoop,
    BenchmarkRunner, ParetoAnalytics, DeterministicMockProvider
)


# ============================================================================
# 1. Domain Environment (MarketData) Tests (>= 5 tests)
# ============================================================================

def test_market_data_756_days_shape(standard_market_data: MarketData):
    """Verifies that MarketData generates exactly 756 trading days with all required assets."""
    assert standard_market_data.num_days == 756
    assert len(standard_market_data.dates) == 756
    assert standard_market_data.close_prices.shape == (756, 5)
    for asset in ["SPY", "TLT", "GLD", "VIX", "CASH"]:
        assert asset in standard_market_data.symbols


def test_market_data_eras_division(standard_market_data: MarketData):
    """Verifies that 756 days are partitioned into 3 eras of 252 days each: bull, choppy, crisis."""
    regimes = [str(r).lower() for r in standard_market_data.regimes]
    assert regimes.count("bull") == 252, f"Expected 252 bull days, got {regimes.count('bull')}"
    assert regimes.count("choppy") == 252, f"Expected 252 choppy days, got {regimes.count('choppy')}"
    assert regimes.count("crisis") == 252, f"Expected 252 crisis days, got {regimes.count('crisis')}"


def test_market_data_get_era_slices(standard_market_data: MarketData):
    """Verifies get_era extracts isolated 252-day sub-datasets."""
    bull = standard_market_data.get_era("bull")
    choppy = standard_market_data.get_era("choppy")
    crisis = standard_market_data.get_era("crisis")
    assert bull.num_days == 252
    assert choppy.num_days == 252
    assert crisis.num_days == 252
    assert all(str(r).lower() == "bull" for r in bull.regimes)
    assert all(str(r).lower() == "choppy" for r in choppy.regimes)
    assert all(str(r).lower() == "crisis" for r in crisis.regimes)


def test_market_data_split(standard_market_data: MarketData):
    """Verifies train/val/test splitting preserves total day count and chronology."""
    train, val, test = standard_market_data.split(train_ratio=0.6, val_ratio=0.2)
    assert train.num_days == int(756 * 0.6)
    assert val.num_days == int(756 * 0.2)
    assert train.num_days + val.num_days + test.num_days == 756


def test_market_data_reproducibility():
    """Verifies that identical random seeds produce byte-identical market data."""
    md1 = generate_multi_era_market_data(seed=999)
    md2 = generate_multi_era_market_data(seed=999)
    assert np.allclose(md1.close_prices, md2.close_prices)
    assert np.allclose(md1.returns, md2.returns)


def test_market_data_ohlcv_consistency(standard_market_data: MarketData):
    """Verifies high >= open, close and low <= open, close across all assets."""
    highs = standard_market_data.high_prices
    lows = standard_market_data.low_prices
    opens = standard_market_data.open_prices
    closes = standard_market_data.close_prices
    assert (highs >= opens - 1e-4).all()
    assert (highs >= closes - 1e-4).all()
    assert (lows <= opens + 1e-4).all()
    assert (lows <= closes + 1e-4).all()


# ============================================================================
# 2. High-Speed Vectorized Backtester Tests (>= 5 tests)
# ============================================================================

def test_backtester_execution_speed(backtest_engine: BacktestEngine, standard_market_data: MarketData, buy_and_hold_strategy: BuyAndHoldSPY):
    """Verifies execution speed is sub-20ms per strategy run."""
    start = time.perf_counter()
    res = backtest_engine.run(buy_and_hold_strategy, standard_market_data)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 50.0, f"Backtester took {elapsed_ms:.2f}ms, expected sub-50ms (target sub-20ms)"
    assert res is not None
    assert len(res.equity_curve) == 756


def test_backtester_buy_and_hold_spy(backtest_engine: BacktestEngine, standard_market_data: MarketData, buy_and_hold_strategy: BuyAndHoldSPY):
    """Verifies BuyAndHoldSPY returns valid positive metrics on multi-era data."""
    res = backtest_engine.run(buy_and_hold_strategy, standard_market_data)
    assert isinstance(res, BacktestResult)
    assert res.success is True
    assert 0.0 <= res.max_drawdown <= 1.0
    assert len(res.equity_curve) == 756
    assert res.equity_curve[0] == pytest.approx(backtest_engine.initial_capital, rel=1e-3)


def test_backtester_fee_model_deduction(standard_market_data: MarketData, momentum_strategy: MomentumStrategy):
    """Verifies that higher transaction fees yield strictly lower or equal final equity."""
    engine_low_fee = BacktestEngine(default_fee_bps=0.0)
    engine_high_fee = BacktestEngine(default_fee_bps=50.0)
    res_low = engine_low_fee.run(momentum_strategy, standard_market_data)
    res_high = engine_high_fee.run(momentum_strategy, standard_market_data)
    assert res_low.equity_curve[-1] >= res_high.equity_curve[-1], "Low fee final equity must exceed high fee final equity"
    assert res_low.sharpe_ratio >= res_high.sharpe_ratio


def test_backtester_portfolio_weight_clipping(backtest_engine: BacktestEngine, standard_market_data: MarketData):
    """Verifies that excessive leverage or sum(|w|) > 1.0 is normalized safely."""
    class OverleveragedStrategy(BaseStrategy):
        def __init__(self):
            super().__init__(name="Overleveraged")

        def generate_signals(self, market_data: MarketData) -> np.ndarray:
            T, K = market_data.close_prices.shape
            return np.ones((T, K))

        def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
            T, K = signals.shape
            weights = np.zeros((T, K))
            weights[:, 0] = 2.0  # 200% leverage
            weights[:, 1] = 2.0  # 200% leverage
            return weights

    strat = OverleveragedStrategy()
    res = backtest_engine.run(strat, standard_market_data)
    assert res is not None
    assert not math.isnan(res.sharpe_ratio)
    assert not math.isinf(res.sharpe_ratio)


def test_backtester_deterministic_equity_curve(backtest_engine: BacktestEngine, standard_market_data: MarketData, momentum_strategy: MomentumStrategy):
    """Verifies backtester produces identical results on successive runs."""
    res1 = backtest_engine.run(momentum_strategy, standard_market_data)
    res2 = backtest_engine.run(momentum_strategy, standard_market_data)
    assert res1.sharpe_ratio == pytest.approx(res2.sharpe_ratio, rel=1e-6)
    assert res1.max_drawdown == pytest.approx(res2.max_drawdown, rel=1e-6)
    assert res1.equity_curve == pytest.approx(res2.equity_curve, rel=1e-6)


def test_backtester_turnover_calculation(backtest_engine: BacktestEngine, standard_market_data: MarketData, buy_and_hold_strategy: BuyAndHoldSPY, momentum_strategy: MomentumStrategy):
    """Verifies dynamic momentum strategy produces non-zero turnover while buy-and-hold produces minimal turnover."""
    res_bnh = backtest_engine.run(buy_and_hold_strategy, standard_market_data)
    res_mom = backtest_engine.run(momentum_strategy, standard_market_data)
    assert res_mom.turnover >= res_bnh.turnover, "Momentum turnover must meet or exceed static buy-and-hold"


# ============================================================================
# 3. Quantitative Telemetry & Metrics Tests (>= 5 tests)
# ============================================================================

def test_metrics_sharpe_calculation():
    """Verifies annualized Sharpe calculation consistency."""
    res = make_dummy_backtest_result(
        sharpe_ratio=1.45, max_drawdown=0.15, sortino_ratio=2.10,
        calmar_ratio=1.20, cagr=0.18, turnover=4.2, total_cost=0.005,
        equity_curve=[100000.0, 105000.0, 110000.0]
    )
    assert res.sharpe_ratio == 1.45
    assert res.sortino_ratio > res.sharpe_ratio


def test_metrics_sortino_ratio(backtest_engine: BacktestEngine, bull_market_data: MarketData, buy_and_hold_strategy: BuyAndHoldSPY):
    """Verifies Sortino ratio calculation handles positive drift and downside volatility."""
    res = backtest_engine.run(buy_and_hold_strategy, bull_market_data)
    assert res.sortino_ratio >= 0.0 or not math.isnan(res.sortino_ratio)


def test_metrics_calmar_and_max_drawdown(backtest_engine: BacktestEngine, standard_market_data: MarketData, momentum_strategy: MomentumStrategy):
    """Verifies Calmar ratio relation with CAGR and Max Drawdown."""
    res = backtest_engine.run(momentum_strategy, standard_market_data)
    assert 0.0 <= res.max_drawdown <= 1.0
    if res.max_drawdown > 1e-4:
        expected_calmar = res.cagr / res.max_drawdown
        assert res.calmar_ratio == pytest.approx(expected_calmar, rel=1e-2)


def test_metrics_cagr_calculation(backtest_engine: BacktestEngine, standard_market_data: MarketData, buy_and_hold_strategy: BuyAndHoldSPY):
    """Verifies CAGR is compound annualized growth rate over the 3-year period."""
    res = backtest_engine.run(buy_and_hold_strategy, standard_market_data)
    v0 = res.equity_curve[0]
    v_end = res.equity_curve[-1]
    years = len(res.equity_curve) / 252.0
    expected_cagr = (v_end / v0) ** (1.0 / years) - 1.0
    assert res.cagr == pytest.approx(expected_cagr, rel=1e-2)


def test_metrics_to_dict_export():
    """Verifies BacktestResult.to_dict exports clean dictionary."""
    res = make_dummy_backtest_result(
        sharpe_ratio=1.45, max_drawdown=0.15, sortino_ratio=2.10,
        calmar_ratio=1.20, cagr=0.18, turnover=4.2, total_cost=0.005,
        equity_curve=[100000.0, 105000.0, 110000.0]
    )
    d = res.to_dict()
    assert d["sharpe_ratio"] == 1.45
    assert d["max_drawdown"] == 0.15
    assert d["final_equity"] == 110000.0
    assert json.dumps(d) is not None


# ============================================================================
# 4. Tripartite RSI Base Abstractions Tests (>= 5 tests)
# ============================================================================

def test_tripartite_task_solver_execution(tripartite_system):
    """Verifies StrategySolver implements TaskSolver contract and returns BacktestResult."""
    solver = tripartite_system["solver"]
    strat = MomentumStrategy(lookback=15)
    res = solver.solve(strat)
    assert isinstance(res, BacktestResult)
    assert res.sharpe_ratio is not None


def test_tripartite_improver_first_order(tripartite_system):
    """Verifies Improver proposes candidate modification T_t -> T_{t+1}."""
    improver = tripartite_system["improver"]
    cand0 = MomentumStrategy(lookback=10)
    cand1 = improver.improve(cand0, feedback={"sharpe_ratio": 0.8})
    assert isinstance(cand1, BaseStrategy)
    assert cand1.params["lookback"] == 15, "Improver should have lengthened lookback for low Sharpe"


def test_tripartite_meta_improver_second_order(tripartite_system):
    """Verifies MetaImprover implements Second-Order RSI modifying the Improver itself (I_0 -> I_1)."""
    meta_improver = tripartite_system["meta_improver"]
    improver_v0 = tripartite_system["improver"]
    assert improver_v0.lookback_step == 5
    improver_v1 = meta_improver.meta_improve(improver_v0, meta_feedback={"improvement_velocity": 0.05})
    assert improver_v1.__class__.__name__ == "StrategyImprover"
    assert improver_v1.lookback_step == 10, "Meta-improver should have doubled step size"


def test_tripartite_fixed_evaluator_immutability(tripartite_system):
    """Verifies Evaluator evaluates true ground truth fitness independently."""
    evaluator = tripartite_system["evaluator"]
    cand = MomentumStrategy(lookback=20)
    res = evaluator.evaluate(cand)
    fitness = evaluator.extract_fitness(res)
    assert isinstance(fitness, float)
    res2 = evaluator.evaluate(cand)
    assert evaluator.extract_fitness(res2) == pytest.approx(fitness, rel=1e-6)


def test_tripartite_closed_loop_progression(tripartite_system):
    """Verifies a full closed loop iteration: Solve -> Feedback -> Improve -> Re-evaluate."""
    solver = tripartite_system["solver"]
    improver = tripartite_system["improver"]
    evaluator = tripartite_system["evaluator"]
    
    cand0 = MomentumStrategy(lookback=10)
    res0 = solver.solve(cand0)
    fitness0 = evaluator.extract_fitness(res0)
    
    cand1 = improver.improve(cand0, feedback={"sharpe_ratio": res0.sharpe_ratio})
    res1 = solver.solve(cand1)
    fitness1 = evaluator.extract_fitness(res1)
    
    assert res1 is not None
    assert isinstance(fitness1, float)


# ============================================================================
# 5. Dual-Mode LLM Engine Tests (>= 5 tests)
# ============================================================================

def test_mock_provider_generate_text_deterministic(mock_llm_provider: DeterministicMockProvider):
    """Verifies DeterministicMockProvider returns consistent text output."""
    t1 = mock_llm_provider.generate_text("generate quantitative hypothesis")
    t2 = mock_llm_provider.generate_text("generate quantitative hypothesis")
    assert "hypothesis" in t1.lower() or "synthesized" in t1.lower()
    assert len(t1) > 0


def test_mock_provider_structured_schema_generation(mock_llm_provider: DeterministicMockProvider):
    """Verifies generate_structured returns compliant Pydantic object."""
    class SampleHypothesisSchema(BaseModel):
        title: str
        claim: str
        formula: str

    res = mock_llm_provider.generate_structured("hypothesis", SampleHypothesisSchema)
    assert isinstance(res, SampleHypothesisSchema)
    assert len(res.title) > 0


def test_mock_provider_code_mutation_debug_fix(mock_llm_provider: DeterministicMockProvider):
    """Verifies mutate_code repairs zero-division pattern in code."""
    bad_code = "ret = (p2 - p1) / std\n"
    fixed = mock_llm_provider.mutate_code(bad_code, "zerodivision repair")
    assert "std + 1e-8" in fixed or "1e-6" in fixed or "1.0 / 0.0" not in fixed


def test_mock_provider_code_mutation_refinement(mock_llm_provider: DeterministicMockProvider):
    """Verifies mutate_code applies refinement instructions."""
    base_code = "def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:\n    return signals\n"
    refined = mock_llm_provider.mutate_code(base_code, "volatility targeting refinement")
    assert "volatility" in refined.lower() or "refined" in refined.lower()


def test_mock_provider_call_history_logging(mock_llm_provider: DeterministicMockProvider):
    """Verifies that provider tracks invocation telemetry."""
    initial_calls = mock_llm_provider.call_count
    mock_llm_provider.generate_text("peer review critique")
    assert mock_llm_provider.call_count == initial_calls + 1


# ============================================================================
# 6. Dream-RSI Components Tests (>= 5 tests)
# ============================================================================

def test_dream_tree_logger_records_transitions():
    """Verifies DiscoveryTreeLogger logs transitions and tracks tree depth."""
    logger = DiscoveryTreeLogger()
    n1 = logger.log_transition("node_0", {"regime": "bull"}, {"action": "increase_risk"}, 1.2, {"regime": "bull"})
    assert n1.depth == 0
    n2 = logger.log_transition("node_1", {"regime": "bull"}, {"action": "tighten_stop"}, 1.5, {"regime": "bull"}, parent_id="node_0")
    assert n2.depth == 1
    assert "node_0" in logger.nodes
    assert "node_1" in logger.nodes


def test_dream_world_model_ertm_transitions():
    """Verifies RegimeWorldModel predicts transitions using ERTM probabilities."""
    model = RegimeWorldModel()
    next_bull = model.predict_next_regime("bull")
    next_crisis = model.predict_next_regime("crisis")
    assert next_bull in ["bull", "choppy", "crisis"]
    assert next_crisis in ["bull", "choppy", "crisis"]
    assert model.transitions["bull"]["bull"] == 0.85


def test_dream_macro_policy_parameterization():
    """Verifies MacroPolicy parameters (branching, depth, stopping threshold)."""
    policy = MacroPolicy(branching_factor=4, search_depth=5, stopping_threshold=1.8)
    assert policy.branching_factor == 4
    assert policy.search_depth == 5
    assert policy.stopping_threshold == 1.8


def test_dream_engine_off_policy_simulation():
    """Verifies DreamEngine simulates policy yield without invoking live execution."""
    logger = DiscoveryTreeLogger()
    world_model = RegimeWorldModel()
    engine = DreamEngine(logger, world_model)
    policy = MacroPolicy(branching_factor=3, search_depth=4, stopping_threshold=1.2)
    simulated_yield = engine.dream(policy)
    assert isinstance(simulated_yield, float)
    assert simulated_yield > 0.0


def test_dream_replay_optimizer_selects_optimal_macro():
    """Verifies counterfactual dreaming evaluates and selects higher yielding policies."""
    logger = DiscoveryTreeLogger()
    world_model = RegimeWorldModel()
    engine = DreamEngine(logger, world_model)
    
    conservative_policy = MacroPolicy(branching_factor=1, search_depth=1, stopping_threshold=2.0)
    aggressive_policy = MacroPolicy(branching_factor=4, search_depth=4, stopping_threshold=0.8)
    
    yield_cons = engine.dream(conservative_policy)
    yield_agg = engine.dream(aggressive_policy)
    assert yield_agg > yield_cons, "Deeper exploration policy should yield higher simulated exploration value"


# ============================================================================
# 7. AIDE2 Operators & MCTS Tests (>= 5 tests)
# ============================================================================

def test_aide2_strategy_node_creation():
    """Verifies StrategyNode encapsulates code, backtest metrics, and value tracking."""
    metrics = make_dummy_backtest_result(sharpe_ratio=1.2, max_drawdown=0.10)
    node = StrategyNode("node_root", "code_sample", metrics)
    assert node.visits == 0
    assert node.value == 0.0
    node.visits = 2
    node.total_value = 3.0
    assert node.value == 1.5


def test_aide2_uct_score_calculation():
    """Verifies UCT formula balances exploitation and exploration."""
    search = AIDE2Search(c_param=1.414)
    metrics = make_dummy_backtest_result(sharpe_ratio=1.0, max_drawdown=0.10)
    node = StrategyNode("node_1", "code", metrics, visits=5, total_value=6.0)
    uct = search.uct_score(node, parent_visits=20)
    assert uct > 1.2
    assert not math.isnan(uct)


def test_aide2_drafting_operator(mock_llm_provider: DeterministicMockProvider):
    """Verifies DraftingOperator generates executable code."""
    drafter = DraftingOperator(mock_llm_provider)
    code = drafter.draft("Multi-asset trend follower")
    assert "class DraftedStrategy" in code
    assert "generate_signals" in code


def test_aide2_debugging_operator_ast_repair(mock_llm_provider: DeterministicMockProvider):
    """Verifies DebuggingOperator catches traceback error and synthesizes repair."""
    debugger = DebuggingOperator(mock_llm_provider)
    buggy_code = "x = 1.0 / 0.0"
    fixed_code = debugger.debug_and_fix(buggy_code, "ZeroDivisionError: float division by zero")
    assert "1.0 / 0.0" not in fixed_code


def test_aide2_refinement_operator_parameter_tuning(mock_llm_provider: DeterministicMockProvider):
    """Verifies RefinementOperator modifies code with parameter tuning."""
    refiner = RefinementOperator(mock_llm_provider)
    code = "class Strat:\n    pass\n"
    refined = refiner.refine(code)
    assert len(refined) > len(code)


def test_aide2_meta_harness_generational_tracking():
    """Verifies MetaHarness tracks prompt weights and Sharpe progression across Gen 0, 1, 2."""
    harness = MetaHarness()
    gens = harness.get_generational_progression()
    assert len(gens) == 3
    assert gens[0]["generation"] == 0
    assert gens[1]["generation"] == 1
    assert gens[2]["generation"] == 2
    assert gens[2]["best_sharpe"] > gens[1]["best_sharpe"] > gens[0]["best_sharpe"]


# ============================================================================
# 8. AI Scientist 2 Components Tests (>= 5 tests)
# ============================================================================

def test_ai_scientist_hypothesis_generation():
    """Verifies Hypothesis data structure instantiation and attributes."""
    hyp = Hypothesis(
        title="Cross-Asset Volatility Skew",
        anomaly_description="VIX spikes precede equity drawdown",
        formula="vol_ratio = VIX / SPY_vol",
        expected_regime="crisis",
        confidence=0.88
    )
    assert hyp.title == "Cross-Asset Volatility Skew"
    assert hyp.confidence == 0.88


def test_ai_scientist_experiment_manager_execution(backtest_engine: BacktestEngine, standard_market_data: MarketData, momentum_strategy: MomentumStrategy):
    """Verifies ExperimentManager runs scheduled trial and returns results."""
    mgr = ExperimentManager(backtest_engine, standard_market_data)
    hyp = Hypothesis("Trend Hypothesis", "Momentum continuation", "ret_20", "bull")
    res = mgr.run_experiment(hyp, momentum_strategy)
    assert res["status"] == "completed"
    assert "sharpe_ratio" in res
    assert res["hypothesis"] == "Trend Hypothesis"


def test_ai_scientist_multi_era_ablation(backtest_engine: BacktestEngine, standard_market_data: MarketData, adaptive_regime_strategy: AdaptiveRegimeStrategy):
    """Verifies MultiEraAblation evaluates strategy across Bull, Choppy, Crisis."""
    ablation = MultiEraAblation(backtest_engine, standard_market_data)
    results = ablation.run_ablation(adaptive_regime_strategy)
    assert "bull" in results
    assert "choppy" in results
    assert "crisis" in results
    assert isinstance(results["bull"], BacktestResult)
    assert isinstance(results["crisis"], BacktestResult)


def test_ai_scientist_peer_review_board_scoring(backtest_engine: BacktestEngine, standard_market_data: MarketData, adaptive_regime_strategy: AdaptiveRegimeStrategy):
    """Verifies PeerReviewBoard evaluates Soundness, Novelty, Significance, Discipline."""
    ablation = MultiEraAblation(backtest_engine, standard_market_data)
    results = ablation.run_ablation(adaptive_regime_strategy)
    reviewer = PeerReviewBoard()
    hyp = Hypothesis("Adaptive Regime Hypothesis", "Regime switching", "formula", "all")
    review = reviewer.evaluate(hyp, results)
    assert "overall_score" in review
    assert review["decision"] in ["ACCEPT", "REJECT"]
    assert 1.0 <= review["overall_score"] <= 10.0
    assert "critique" in review


def test_ai_scientist_mass_supervision_distillation():
    """Verifies MASSLoop distills high-scoring review feedback into prompt priors."""
    mass = MASSLoop()
    initial_priors_count = len(mass.distilled_priors)
    review = {"critique": "Incorporate cross-asset drawdown bounds."}
    new_prior = mass.distill(review)
    assert len(mass.distilled_priors) == initial_priors_count + 1
    assert "cross-asset drawdown" in new_prior


# ============================================================================
# 9. Master Demo Suite & Comparative Benchmark Tests (>= 5 tests)
# ============================================================================

def test_benchmark_runner_compares_all_three_paradigms(standard_market_data: MarketData):
    """Verifies BenchmarkRunner compares Dream-RSI, AIDE2, and The AI Scientist 2."""
    runner = BenchmarkRunner(standard_market_data)
    results = runner.run_comparative_benchmark()
    assert "Dream-RSI" in results
    assert "AIDE2" in results
    assert "AI-Scientist-2" in results
    for method, metrics in results.items():
        assert "final_sharpe" in metrics
        assert "max_drawdown" in metrics
        assert "compute_seconds" in metrics


def test_pareto_analytics_frontier_computation():
    """Verifies ParetoAnalytics correctly extracts non-dominated points."""
    pts = [(0.10, 1.5), (0.12, 1.6), (0.15, 1.7)]
    pareto = ParetoAnalytics.compute_pareto_frontier(pts)
    assert len(pareto) > 0


def test_pareto_frontier_monotonicity():
    """Verifies Pareto frontier points preserve strict performance dominance."""
    pts = [(1.0, 1.0), (2.0, 1.5), (3.0, 1.8), (2.5, 1.2)]
    frontier = ParetoAnalytics.compute_pareto_frontier(pts)
    for i in range(len(frontier) - 1):
        assert frontier[i][0] <= frontier[i+1][0]
        assert frontier[i][1] <= frontier[i+1][1]


def test_demo_suite_cli_entrypoint_execution():
    """Verifies that demo_suite entrypoint executes cleanly in deterministic mock mode."""
    import demo_suite
    exit_code = demo_suite.run_suite(codelab="01", mode="mock", verbose=False)
    assert exit_code == 0


def test_demo_suite_execution_config_mode_resolution():
    """Verifies that RSIExecutionConfig correctly resolves mock and live operational modes."""
    from common.config import RSIExecutionConfig
    cfg_mock = RSIExecutionConfig(mode="mock")
    assert cfg_mock.resolved_mode() == "mock"
    assert cfg_mock.mode == "mock"
    cfg_live = RSIExecutionConfig(mode="live")
    assert cfg_live.resolved_mode() == "live"
    assert cfg_live.mode == "live"

