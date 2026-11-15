"""
Shared test configuration, fixtures, synthetic market data generators,
and genuine domain/codelab imports for Assignment 5 Recursive Self-Improvement (RSI).

Guarantees 100% offline, deterministic execution without external API keys or network dependencies.
Directly imports genuine components from common/ and all four RSI codelabs:
  - 01_dream_rsi: RegimeWorldModel, DreamEngine, MacroPolicy, DiscoveryTreeLogger
  - 02_aide2: StrategyNode, AIDE2Search, DraftingOperator, DebuggingOperator, RefinementOperator, MetaHarness
  - 03_ai_scientist_2: Hypothesis, HypothesisGenerator, ExperimentManager, MultiEraAblation, PeerReviewBoard, MASSLoop
  - 04_comparative_benchmark: BenchmarkRunner, ParetoAnalytics
"""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, Type, TypeVar

import numpy as np
import pandas as pd
import pytest

# Ensure assignment root is on sys.path
_assignment_root = Path(__file__).resolve().parent.parent
if str(_assignment_root) not in sys.path:
    sys.path.insert(0, str(_assignment_root))

# Fix Windows Proactor event loop policy for test stability
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Sets environment variables ensuring deterministic offline mock execution."""
    os.environ["RSI_MODE"] = "mock"
    os.environ["RSI_EXECUTION_MODE"] = "mock"
    os.environ["GEMINI_API_KEY"] = "mock-gemini-api-key-test-12345"
    os.environ["GOOGLE_API_KEY"] = "mock-google-api-key-test-12345"
    os.environ["HARNESS_MODEL"] = "gemini-3.8-flash"
    yield


# ============================================================================
# 1. Core Domain Contracts & Financial Backtesting Engine
# ============================================================================

from common.domain.market_data import (
    DEFAULT_SYMBOLS,
    MarketData,
    MarketRegime,
    generate_multi_era_market_data,
)
from common.domain.backtester import BacktestEngine, BacktestResult
from common.domain.strategy_base import (
    BaseStrategy,
    DynamicCodeStrategy,
    EqualWeightStrategy,
    MovingAverageCrossover,
)
import common.domain.metrics as metrics_mod
from common.domain.metrics import (
    cagr,
    max_drawdown,
    annualized_sharpe,
    annualized_sortino,
    calmar_ratio,
    turnover,
    transaction_costs,
    compute_portfolio_metrics,
)

# Contract Aliases
MarketDataGenerator = generate_multi_era_market_data
MarketEnvironment = MarketData


@dataclass
class BacktestConfig:
    """Configuration telemetry for backtest execution."""
    default_fee_bps: float = 10.0
    risk_free_rate: float = 0.02
    initial_capital: float = 100_000.0


class FinancialMetrics:
    """Namespace providing standard financial metrics calculation methods."""
    cagr = staticmethod(cagr)
    max_drawdown = staticmethod(max_drawdown)
    annualized_sharpe = staticmethod(annualized_sharpe)
    annualized_sortino = staticmethod(annualized_sortino)
    calmar_ratio = staticmethod(calmar_ratio)
    turnover = staticmethod(turnover)
    transaction_costs = staticmethod(transaction_costs)
    compute_portfolio_metrics = staticmethod(compute_portfolio_metrics)


# Enhance MarketData with train/val/test split convenience method if not present
def _split_market_data(
    self: MarketData, train_ratio: float = 0.6, val_ratio: float = 0.2
) -> Tuple[MarketData, MarketData, MarketData]:
    n = self.num_days
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))

    def slice_md(start: int, end: int) -> MarketData:
        return MarketData(
            symbols=list(self.symbols),
            dates=self.dates[start:end],
            close_prices=self.close_prices[start:end].copy(),
            open_prices=self.open_prices[start:end].copy(),
            high_prices=self.high_prices[start:end].copy(),
            low_prices=self.low_prices[start:end].copy(),
            volumes=self.volumes[start:end].copy(),
            returns=self.returns[start:end].copy(),
            regimes=self.regimes[start:end],
        )

    return slice_md(0, n_train), slice_md(n_train, n_val), slice_md(n_val, n)


if not hasattr(MarketData, "split"):
    MarketData.split = _split_market_data  # type: ignore

if not hasattr(MarketData, "data"):
    setattr(MarketData, "data", property(lambda self: self.to_dataframe()))


# Wrapper to ensure BacktestEngine accepts fee_bps or default_fee_bps gracefully
_orig_engine_init = BacktestEngine.__init__


def _flexible_engine_init(self, *args, **kwargs):
    if "fee_bps" in kwargs and "default_fee_bps" not in kwargs:
        kwargs["default_fee_bps"] = kwargs.pop("fee_bps")
    _orig_engine_init(self, *args, **kwargs)


BacktestEngine.__init__ = _flexible_engine_init  # type: ignore


def make_dummy_backtest_result(
    sharpe_ratio: float = 1.45,
    max_drawdown: float = 0.15,
    sortino_ratio: float = 2.10,
    calmar_ratio: float = 1.20,
    cagr: float = 0.18,
    turnover: float = 4.2,
    total_cost: float = 0.005,
    equity_curve: Optional[List[float]] = None,
) -> BacktestResult:
    """Helper to instantiate a complete BacktestResult conforming to all dataclass fields."""
    eq = equity_curve or [100000.0, 105000.0, 110000.0]
    return BacktestResult(
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        cagr=cagr,
        turnover=turnover,
        total_cost=total_cost,
        equity_curve=eq,
        daily_returns=[0.0] * (len(eq) - 1),
        weights_history=[[0.2] * 5] * len(eq),
        execution_time_ms=1.5,
        success=True,
    )


# ============================================================================
# 2. Standard Test Strategies Subclassing BaseStrategy
# ============================================================================

class BuyAndHoldSPY(BaseStrategy):
    """100% allocation to SPY."""

    def __init__(self):
        super().__init__(name="BuyAndHoldSPY")

    def generate_signals(self, market_data: MarketData) -> np.ndarray:
        T = market_data.num_days
        K = len(market_data.symbols)
        signals = np.zeros((T, K), dtype=float)
        spy_idx = market_data.get_symbol_index("SPY") if hasattr(market_data, "get_symbol_index") else 0
        signals[:, spy_idx] = 1.0
        return signals

    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        T, K = signals.shape
        weights = np.zeros((T, K), dtype=float)
        weights[:, 0] = 1.0
        return weights


class MomentumStrategy(BaseStrategy):
    """Trend-following strategy with configurable lookback."""

    def __init__(self, lookback: int = 20):
        super().__init__(name="MomentumStrategy", params={"lookback": lookback})
        self.lookback = lookback

    def generate_signals(self, market_data: MarketData) -> np.ndarray:
        close = market_data.close_prices
        T, K = close.shape
        signals = np.zeros((T, K), dtype=float)
        for t in range(self.lookback, T):
            rets = (close[t] - close[t - self.lookback]) / (close[t - self.lookback] + 1e-8)
            best_idx = int(np.argmax(rets[: K - 1]))
            if rets[best_idx] > 0:
                signals[t, best_idx] = 1.0
            else:
                signals[t, K - 1] = 1.0  # Cash
        return signals

    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        T, K = signals.shape
        weights = np.zeros((T, K), dtype=float)
        for t in range(T):
            sig_sum = np.sum(signals[t])
            if sig_sum > 0:
                weights[t] = signals[t] / sig_sum
            else:
                weights[t, -1] = 1.0
        return weights


class AdaptiveRegimeStrategy(BaseStrategy):
    """Regime-adaptive strategy: Equity in Bull, Bonds/Gold in Choppy, Cash/VIX in Crisis."""

    def __init__(self, vix_threshold: float = 25.0):
        super().__init__(name="AdaptiveRegimeStrategy", params={"vix_threshold": vix_threshold})
        self.vix_threshold = vix_threshold

    def generate_signals(self, market_data: MarketData) -> np.ndarray:
        close = market_data.close_prices
        regimes = market_data.regimes
        T, K = close.shape
        signals = np.zeros((T, K), dtype=float)
        vix_idx = market_data.get_symbol_index("VIX") if hasattr(market_data, "get_symbol_index") else 3
        cash_idx = market_data.get_symbol_index("CASH") if hasattr(market_data, "get_symbol_index") else 4
        spy_idx = market_data.get_symbol_index("SPY") if hasattr(market_data, "get_symbol_index") else 0
        tlt_idx = market_data.get_symbol_index("TLT") if hasattr(market_data, "get_symbol_index") else 1
        gld_idx = market_data.get_symbol_index("GLD") if hasattr(market_data, "get_symbol_index") else 2

        for t in range(T):
            vix = close[t, vix_idx]
            regime = str(regimes[t]).lower()
            if regime == "crisis" or vix > self.vix_threshold:
                signals[t, cash_idx] = 0.8
                signals[t, vix_idx] = 0.2
            elif regime == "choppy":
                signals[t, tlt_idx] = 0.5
                signals[t, gld_idx] = 0.5
            else:
                signals[t, spy_idx] = 0.8
                signals[t, gld_idx] = 0.2
        return signals

    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        T, K = signals.shape
        weights = np.zeros((T, K), dtype=float)
        for t in range(T):
            s = np.sum(signals[t])
            weights[t] = signals[t] / max(s, 1e-8)
        return weights


class FaultyStrategy(BaseStrategy):
    """Strategy designed for boundary and error-recovery testing."""

    def __init__(self, fault_type: str = "zero_division"):
        super().__init__(name="FaultyStrategy", params={"fault_type": fault_type})
        self.fault_type = fault_type

    def generate_signals(self, market_data: MarketData) -> np.ndarray:
        if self.fault_type == "zero_division":
            _ = 1.0 / 0.0
        elif self.fault_type == "nan_weights":
            T, K = market_data.close_prices.shape
            return np.full((T, K), np.nan)
        T, K = market_data.close_prices.shape
        return np.zeros((T, K))

    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        if self.fault_type == "nan_weights":
            return np.full_like(signals, np.nan)
        return np.zeros_like(signals)


# ============================================================================
# 3. Tripartite RSI Base Abstractions
# ============================================================================

from common.rsi_base import (
    Evaluator,
    GenerationRecord,
    Improver,
    MetaImprover,
    TaskSolver,
)


class StrategySolver(TaskSolver[BaseStrategy, BacktestResult]):
    def __init__(self, engine: BacktestEngine, market_data: MarketData):
        super().__init__(name="StrategySolver")
        self.engine = engine
        self.market_data = market_data

    def solve(self, candidate: BaseStrategy, **kwargs) -> BacktestResult:
        return self.engine.run(candidate, self.market_data)


class StrategyImprover(Improver[BaseStrategy]):
    def __init__(self, name: str = "StrategyImprover", lookback_step: int = 5):
        super().__init__(name=name)
        self.lookback_step = lookback_step

    def improve(self, candidate: BaseStrategy, feedback: Any, **kwargs) -> BaseStrategy:
        self.record_mutation()
        curr = candidate.params.get("lookback", 10)
        new_lookback = curr + self.lookback_step
        return MomentumStrategy(lookback=new_lookback)


class StrategyMetaImprover(MetaImprover[StrategyImprover]):
    def __init__(self):
        super().__init__(name="StrategyMetaImprover")

    def meta_improve(
        self,
        improver: StrategyImprover,
        history=None,
        meta_feedback: Any = None,
        **kwargs,
    ) -> StrategyImprover:
        step = improver.lookback_step * 2
        return StrategyImprover(name=f"{improver.name}_v2", lookback_step=step)


class FixedGroundTruthEvaluator(Evaluator[BaseStrategy, BacktestResult]):
    def __init__(self, test_data: MarketData, default_fee_bps: float = 10.0):
        super().__init__(name="FixedGroundTruthEvaluator")
        self.engine = BacktestEngine(default_fee_bps=default_fee_bps)
        self.test_data = test_data

    def evaluate(self, candidate: BaseStrategy) -> BacktestResult:
        if hasattr(self, "record_eval"):
            self.record_eval()
        return self.engine.run(candidate, self.test_data)

    def extract_fitness(self, result: BacktestResult) -> float:
        return float(result.sharpe_ratio - (result.max_drawdown * 2.0))


# ============================================================================
# 4. LLM Provider Infrastructure
# ============================================================================

from common.llm_provider import (
    BaseModelProvider,
    DeterministicMockProvider,
    get_llm_provider,
)


# ============================================================================
# 5. Genuine Codelab Classes (Imported from 01-04 modules)
# ============================================================================

# Codelab 01: Dream-RSI
_dream_wm_mod = importlib.import_module("01_dream_rsi.world_model")
_dream_engine_mod = importlib.import_module("01_dream_rsi.dream_engine")
_dream_logger_mod = importlib.import_module("01_dream_rsi.tree_logger")
_dream_opt_mod = importlib.import_module("01_dream_rsi.policy_optimizer")

RegimeWorldModel = _dream_wm_mod.RegimeWorldModel
MarkovWorldModel = RegimeWorldModel
DreamEngine = _dream_engine_mod.DreamEngine
CounterfactualDreamer = DreamEngine
MacroPolicy = _dream_engine_mod.MacroPolicy
DiscoveryTreeLogger = _dream_logger_mod.DiscoveryTreeLogger
ExperienceBuffer = DiscoveryTreeLogger
DiscoveryNode = _dream_logger_mod.DiscoveryNode
PolicyOptimizer = _dream_opt_mod.PolicyOptimizer
OptimizationResult = _dream_opt_mod.OptimizationResult

# Codelab 02: AIDE2
_aide2_ops_mod = importlib.import_module("02_aide2.operators")
_aide2_tree_mod = importlib.import_module("02_aide2.tree_node")
_aide2_search_mod = importlib.import_module("02_aide2.mcts_search")
_aide2_harness_mod = importlib.import_module("02_aide2.meta_harness")

DraftingOperator = _aide2_ops_mod.DraftingOperator
DebuggingOperator = _aide2_ops_mod.DebuggingOperator
RefinementOperator = _aide2_ops_mod.RefinementOperator


class ASTMutationEngine:
    """Namespace providing AST code transformation operators for AIDE2."""
    DraftingOperator = DraftingOperator
    DebuggingOperator = DebuggingOperator
    RefinementOperator = RefinementOperator


StrategyNode = _aide2_tree_mod.StrategyNode
CodeDiff = _aide2_tree_mod.CodeDiff
CodeTree = _aide2_tree_mod.CodeTree
AIDE2Search = _aide2_search_mod.AIDE2Search
CodeTreeMCTS = AIDE2Search
MetaHarness = _aide2_harness_mod.MetaHarness
BiLevelMetaHarness = MetaHarness

# Codelab 03: The AI Scientist 2
_sc_hypo_mod = importlib.import_module("03_ai_scientist_2.hypothesis")
_sc_exp_mod = importlib.import_module("03_ai_scientist_2.experimenter")
_sc_ablation_mod = importlib.import_module("03_ai_scientist_2.ablation")
_sc_review_mod = importlib.import_module("03_ai_scientist_2.peer_review")
_sc_mass_mod = importlib.import_module("03_ai_scientist_2.mass_loop")

Hypothesis = _sc_hypo_mod.Hypothesis
HypothesisGenerator = _sc_hypo_mod.HypothesisGenerator
GenerationAgent = HypothesisGenerator
ExperimentTrial = _sc_exp_mod.ExperimentTrial
ExperimentManager = _sc_exp_mod.ExperimentManager
MultiEraAblation = _sc_ablation_mod.MultiEraAblation
ReviewerAssessment = _sc_review_mod.ReviewerAssessment
ReviewerAgent = ReviewerAssessment
PeerReviewBoard = _sc_review_mod.PeerReviewBoard
MetaReviewAgent = PeerReviewBoard
MASSLoop = _sc_mass_mod.MASSLoop
MultiAgentScientificSearch = MASSLoop

# Codelab 04: Comparative Benchmark & Pareto Analytics
_bench_runner_mod = importlib.import_module("04_comparative_benchmark.benchmark_runner")
_bench_pareto_mod = importlib.import_module("04_comparative_benchmark.pareto_analytics")

BenchmarkRunner = _bench_runner_mod.BenchmarkRunner
BenchmarkResultDict = _bench_runner_mod.BenchmarkResultDict
BenchmarkRunResult = _bench_runner_mod.BenchmarkRunResult
ParetoAnalytics = _bench_pareto_mod.ParetoAnalytics


# ============================================================================
# 6. Pytest Fixtures (Genuine Component Instances)
# ============================================================================

@pytest.fixture(scope="session")
def standard_market_data() -> MarketData:
    """Generates authentic multi-era 756-day market data across Bull, Choppy, and Crisis eras."""
    return generate_multi_era_market_data(seed=42, num_days=756)


@pytest.fixture(scope="session")
def bull_market_data(standard_market_data: MarketData) -> MarketData:
    """Isolates the 252-day Bull trend market era."""
    return standard_market_data.get_era("bull")


@pytest.fixture(scope="session")
def choppy_market_data(standard_market_data: MarketData) -> MarketData:
    """Isolates the 252-day Choppy / mean-reverting market era."""
    return standard_market_data.get_era("choppy")


@pytest.fixture(scope="session")
def crisis_market_data(standard_market_data: MarketData) -> MarketData:
    """Isolates the 252-day Crisis / liquidity shock market era."""
    return standard_market_data.get_era("crisis")


@pytest.fixture(scope="session")
def backtest_engine() -> BacktestEngine:
    """Instantiates a genuine BacktestEngine configured with 10 bps default transaction friction."""
    return BacktestEngine(default_fee_bps=10.0)


@pytest.fixture(scope="session")
def mock_llm_provider() -> DeterministicMockProvider:
    """Instantiates a genuine DeterministicMockProvider for offline deterministic execution."""
    return DeterministicMockProvider()


@pytest.fixture
def buy_and_hold_strategy() -> BuyAndHoldSPY:
    return BuyAndHoldSPY()


@pytest.fixture
def momentum_strategy() -> MomentumStrategy:
    return MomentumStrategy(lookback=20)


@pytest.fixture
def adaptive_regime_strategy() -> AdaptiveRegimeStrategy:
    return AdaptiveRegimeStrategy(vix_threshold=25.0)


@pytest.fixture
def tripartite_system(backtest_engine: BacktestEngine, standard_market_data: MarketData):
    """Instantiates the complete tripartite RSI system: Solver, Improver, MetaImprover, Evaluator."""
    train, val, test = standard_market_data.split()
    solver = StrategySolver(backtest_engine, train)
    improver = StrategyImprover(lookback_step=5)
    meta_improver = StrategyMetaImprover()
    evaluator = FixedGroundTruthEvaluator(test, default_fee_bps=10.0)
    return {
        "solver": solver,
        "improver": improver,
        "meta_improver": meta_improver,
        "evaluator": evaluator,
        "train_data": train,
        "val_data": val,
        "test_data": test,
    }


# Genuine Codelab Fixtures
@pytest.fixture
def discovery_tree_logger() -> DiscoveryTreeLogger:
    """Returns a genuine DiscoveryTreeLogger instance."""
    return DiscoveryTreeLogger()


@pytest.fixture
def regime_world_model() -> RegimeWorldModel:
    """Returns a genuine RegimeWorldModel instance."""
    return RegimeWorldModel()


@pytest.fixture
def dream_engine(
    discovery_tree_logger: DiscoveryTreeLogger,
    regime_world_model: RegimeWorldModel,
) -> DreamEngine:
    """Returns a genuine DreamEngine instance."""
    return DreamEngine(logger=discovery_tree_logger, world_model=regime_world_model)


@pytest.fixture
def macro_policy() -> MacroPolicy:
    """Returns a genuine MacroPolicy instance."""
    return MacroPolicy(branching_factor=3, search_depth=3, stopping_threshold=1.2)


@pytest.fixture
def aide2_search() -> AIDE2Search:
    """Returns a genuine AIDE2Search MCTS instance."""
    return AIDE2Search(c_param=1.414)


@pytest.fixture
def aide2_meta_harness() -> MetaHarness:
    """Returns a genuine MetaHarness instance."""
    return MetaHarness()


@pytest.fixture
def hypothesis_generator(mock_llm_provider: DeterministicMockProvider) -> HypothesisGenerator:
    """Returns a genuine HypothesisGenerator instance."""
    return HypothesisGenerator(llm_provider=mock_llm_provider)


@pytest.fixture
def experiment_manager(
    backtest_engine: BacktestEngine,
    standard_market_data: MarketData,
) -> ExperimentManager:
    """Returns a genuine ExperimentManager instance."""
    return ExperimentManager(engine=backtest_engine, market_data=standard_market_data)


@pytest.fixture
def multi_era_ablation(
    backtest_engine: BacktestEngine,
    standard_market_data: MarketData,
) -> MultiEraAblation:
    """Returns a genuine MultiEraAblation instance."""
    return MultiEraAblation(engine=backtest_engine, market_data=standard_market_data)


@pytest.fixture
def peer_review_board() -> PeerReviewBoard:
    """Returns a genuine PeerReviewBoard instance."""
    return PeerReviewBoard()


@pytest.fixture
def mass_loop() -> MASSLoop:
    """Returns a genuine MASSLoop instance."""
    return MASSLoop()


@pytest.fixture
def benchmark_runner(standard_market_data: MarketData) -> BenchmarkRunner:
    """Returns a genuine BenchmarkRunner instance."""
    return BenchmarkRunner(market_data=standard_market_data, mode="mock")


@pytest.fixture
def pareto_analytics() -> Type[ParetoAnalytics]:
    """Returns the genuine ParetoAnalytics class."""
    return ParetoAnalytics
