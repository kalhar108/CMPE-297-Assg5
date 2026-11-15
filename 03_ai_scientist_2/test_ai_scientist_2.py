"""Comprehensive unit and integration test suite for 03_ai_scientist_2 modules.

Tests:
1. Hypothesis data structure & serialization
2. HypothesisGenerator dual-mode generation, deduplication & generational progression
3. ExperimentTrial data structure & ExperimentManager execution
4. Dynamic strategy synthesis and compilation
5. MultiEraAblation multi-era partition backtesting & ASCII table formatting
6. PeerReviewBoard 4-criteria scoring, individual assessments & decision gating
7. MASSLoop prompt prior distillation & multi-generational self-improvement campaign
8. End-to-end integration across all components
"""

import os
import sys
import pytest
import numpy as np

# Ensure module path resolution
_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_dir, ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from common.config import RSIExecutionConfig
from common.domain.backtester import BacktestEngine, BacktestResult
from common.domain.market_data import MarketData, generate_multi_era_market_data
from common.domain.strategy_base import BaseStrategy, EqualWeightStrategy, MomentumFactor
from common.llm_provider import DeterministicMockProvider

from hypothesis import Hypothesis, HypothesisGenerator
from experimenter import ExperimentManager, ExperimentTrial, ParameterizedAdaptiveStrategy
from ablation import MultiEraAblation
from peer_review import PeerReviewBoard, ReviewerAssessment
from mass_loop import MASSLoop


@pytest.fixture
def market_data():
    return generate_multi_era_market_data(seed=42, num_days=756)


@pytest.fixture
def backtest_engine():
    return BacktestEngine(default_fee_bps=10.0)


# ==============================================================================
# 1. Hypothesis Tests
# ==============================================================================

def test_hypothesis_instantiation_and_serialization():
    hyp = Hypothesis(
        title="Test Skew Hypothesis",
        anomaly_description="VIX skew predicts drawdown",
        formula="Signal = VIX / SPY_vol",
        expected_regime="crisis",
        confidence=0.89,
        rationale="Microstructure liquidity withdrawal.",
        parameters={"lookback": 20},
        generation=1,
    )
    assert hyp.title == "Test Skew Hypothesis"
    assert hyp.confidence == 0.89
    assert hyp.expected_regime == "crisis"
    assert hyp.generation == 1

    d = hyp.to_dict()
    assert d["title"] == "Test Skew Hypothesis"
    assert d["confidence"] == 0.89
    assert "parameters" in d
    assert "strategy_code_len" in d

    card = hyp.format_card()
    assert "Test Skew Hypothesis" in card
    assert "Formula" in card


def test_hypothesis_generator_mock_and_generational_progression():
    provider = DeterministicMockProvider()
    generator = HypothesisGenerator(llm_provider=provider)

    h0 = generator.generate_hypothesis(generation=0, variant_index=0)
    assert h0.generation == 0
    assert len(generator.archive) == 1

    h1 = generator.generate_hypothesis(generation=1, variant_index=0)
    assert h1.generation == 1
    assert len(generator.archive) == 2

    h2 = generator.generate_hypothesis(generation=2, variant_index=0)
    assert h2.generation == 2
    assert "semivariance" in h2.title.lower() or "skew" in h2.title.lower()
    assert len(generator.archive) == 3


def test_hypothesis_generator_update_priors():
    generator = HypothesisGenerator()
    initial_len = len(generator.priors)
    generator.update_priors(["New prior: Always hedge beta in crises."])
    assert len(generator.priors) == initial_len + 1
    assert "Always hedge beta" in generator.priors[-1]


# ==============================================================================
# 2. Experimenter Tests
# ==============================================================================

def test_experiment_manager_run_with_explicit_strategy(backtest_engine, market_data):
    mgr = ExperimentManager(backtest_engine, market_data)
    hyp = Hypothesis("Momentum Hypothesis", "Price trend continuation", "ret_20", "bull")
    strat = MomentumFactor(lookback=20)
    res = mgr.run_experiment(hyp, strat)

    assert res["status"] == "completed"
    assert res["hypothesis"] == "Momentum Hypothesis"
    assert "sharpe_ratio" in res
    assert "max_drawdown" in res
    assert res["execution_time_ms"] < 20.0
    assert len(mgr.history) == 1


def test_experiment_manager_synthesize_strategy(backtest_engine, market_data):
    mgr = ExperimentManager(backtest_engine, market_data)
    hyp = Hypothesis(
        title="Asymmetric Downside Semivariance",
        anomaly_description="Semivariance protects capital",
        formula="w = mom / semivar",
        expected_regime="crisis",
        parameters={"vix_threshold": 24.0},
    )
    strat = mgr.synthesize_strategy(hyp)
    assert isinstance(strat, BaseStrategy)
    res = mgr.run_experiment(hyp, strat)
    assert res["status"] == "completed"


def test_progressive_trials_scheduling(backtest_engine, market_data):
    mgr = ExperimentManager(backtest_engine, market_data)
    hyp1 = Hypothesis("Hypo 1", "desc 1", "formula 1", "bull")
    hyp2 = Hypothesis("Hypo 2", "desc 2", "formula 2", "choppy")
    trials = mgr.schedule_progressive_trials([hyp1, hyp2])
    assert len(trials) == 2
    assert len(mgr.history) == 2


# ==============================================================================
# 3. Multi-Era Ablation Tests
# ==============================================================================

def test_multi_era_ablation_partitions(backtest_engine, market_data):
    ablation = MultiEraAblation(backtest_engine, market_data)
    strat = EqualWeightStrategy()
    results = ablation.run_ablation(strat)

    assert "bull" in results
    assert "choppy" in results
    assert "crisis" in results
    assert isinstance(results["bull"], BacktestResult)
    assert isinstance(results["choppy"], BacktestResult)
    assert isinstance(results["crisis"], BacktestResult)

    table = ablation.format_ablation_table(results)
    assert "BULL" in table
    assert "CHOPPY" in table
    assert "CRISIS" in table


def test_multi_era_ablation_hypothesis_report(backtest_engine, market_data):
    ablation = MultiEraAblation(backtest_engine, market_data)
    hyp = Hypothesis("Adaptive Regime Hypothesis", "desc", "formula", "crisis")
    strat = ParameterizedAdaptiveStrategy(vix_threshold=24.0)
    rep = ablation.run_hypothesis_ablation(hyp, strat)

    assert "mean_sharpe" in rep
    assert "worst_drawdown" in rep
    assert "crisis_resilience" in rep
    assert rep["worst_drawdown"] <= 1.0


# ==============================================================================
# 4. Peer Review Board Tests
# ==============================================================================

def test_peer_review_board_scoring_and_critique(backtest_engine, market_data):
    ablation = MultiEraAblation(backtest_engine, market_data)
    strat = ParameterizedAdaptiveStrategy(vix_threshold=24.0)
    results = ablation.run_ablation(strat)

    reviewer = PeerReviewBoard(accept_threshold=7.0)
    hyp = Hypothesis("Adaptive Semivariance", "desc", "formula", "all", generation=2)
    review = reviewer.evaluate(hyp, results)

    assert "overall_score" in review
    assert review["decision"] in ["ACCEPT", "REJECT"]
    assert 1.0 <= review["overall_score"] <= 10.0
    assert "critique" in review
    assert "soundness" in review
    assert "novelty" in review
    assert "significance" in review
    assert "risk_discipline" in review
    assert len(review["individual_reviews"]) == 3

    card = reviewer.format_review_card(review)
    assert "PEER REVIEW COMMITTEE SCORECARD" in card


def test_peer_review_rejects_catastrophic_strategy(backtest_engine, market_data):
    reviewer = PeerReviewBoard(accept_threshold=7.5)
    hyp = Hypothesis("Naive Overfitting", "desc", "formula", "bull", generation=0)

    # Mock catastrophic results
    catastrophic_res = {
        "bull": BacktestResult(sharpe_ratio=-1.5, max_drawdown=0.60, sortino_ratio=-1.0, calmar_ratio=-1.0, cagr=-0.50, turnover=20.0, total_cost=0.1, equity_curve=[100000.0]*252, daily_returns=[0.0]*251, weights_history=[[0.2]*5]*252, execution_time_ms=1.0, success=True),
        "choppy": BacktestResult(sharpe_ratio=-2.0, max_drawdown=0.55, sortino_ratio=-2.0, calmar_ratio=-2.0, cagr=-0.40, turnover=30.0, total_cost=0.1, equity_curve=[100000.0]*252, daily_returns=[0.0]*251, weights_history=[[0.2]*5]*252, execution_time_ms=1.0, success=True),
        "crisis": BacktestResult(sharpe_ratio=-3.0, max_drawdown=0.85, sortino_ratio=-3.0, calmar_ratio=-3.0, cagr=-0.80, turnover=40.0, total_cost=0.1, equity_curve=[100000.0]*252, daily_returns=[0.0]*251, weights_history=[[0.2]*5]*252, execution_time_ms=1.0, success=True),
    }

    review = reviewer.evaluate(hyp, catastrophic_res)
    assert review["decision"] == "REJECT"
    assert review["overall_score"] < 7.0


# ==============================================================================
# 5. MASSLoop Tests
# ==============================================================================

def test_mass_loop_distillation():
    mass = MASSLoop()
    initial_count = len(mass.distilled_priors)
    assert initial_count >= 3

    review = {"critique": "Incorporate cross-asset drawdown bounds to protect capital."}
    prior = mass.distill(review)

    assert len(mass.distilled_priors) == initial_count + 1
    assert "cross-asset drawdown" in prior
    assert "Distilled prior:" in prior


def test_mass_loop_end_to_end_campaign(backtest_engine, market_data):
    mass = MASSLoop()
    campaign = mass.run_discovery_campaign(
        engine=backtest_engine,
        market_data=market_data,
        num_generations=3,
        hypotheses_per_gen=2,
    )

    assert campaign["num_generations"] == 3
    assert campaign["total_hypotheses_evaluated"] == 6
    assert len(campaign["generations_summary"]) == 3

    # Generational progression check: Generation 2 score should exceed Generation 0 score
    gen0_score = campaign["generations_summary"][0]["avg_peer_score"]
    gen2_score = campaign["generations_summary"][2]["avg_peer_score"]
    assert gen2_score >= gen0_score
    assert campaign["peer_score_delta"] >= 0.0

    # Best candidate is identified
    assert "best_hypothesis" in campaign
    assert "best_composite_score" in campaign
    assert campaign["best_composite_score"] >= 7.0
