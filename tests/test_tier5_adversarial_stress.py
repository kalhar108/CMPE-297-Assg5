"""
Tier 5: Adversarial Stress Testing & Empirical Vulnerability Suite.

Adversarial stress-testing targeting four critical architectural vulnerability surfaces:
1. Extreme Market Data Anomalies (flat prices, zero volatility, 99% crash shocks, negative prices, NaN/Inf return handling, single-day series)
2. Backtest Engine Stress (zero trade frequency, 100% turnover churn, extreme fee drag 100 bps vs 0 bps, cash depletion and insolvency guards)
3. Dream-RSI World Model Stress (singular transition matrices, zero-observation regimes with/without Laplace smoothing, extreme macro policy branching B=100, D=100, tau=0.0)
4. AIDE2 Code Tree Stress (infinite recursion syntax bombs, extreme AST mutations, non-numeric returns, over-leveraged weight clipping, dead-end MCTS recovery)
"""

from __future__ import annotations

import importlib
import math
import time
import pytest
import numpy as np

# Core domain imports
from common.domain.market_data import MarketData, generate_multi_era_market_data, DEFAULT_SYMBOLS
from common.domain.backtester import BacktestEngine, BacktestResult
from common.domain.strategy_base import BaseStrategy, DynamicCodeStrategy
from common.llm_provider import DeterministicMockProvider

# Concrete Codelab 1 implementations
dream_wm_mod = importlib.import_module("01_dream_rsi.world_model")
dream_engine_mod = importlib.import_module("01_dream_rsi.dream_engine")
dream_logger_mod = importlib.import_module("01_dream_rsi.tree_logger")

RegimeWorldModel = dream_wm_mod.RegimeWorldModel
DreamEngine = dream_engine_mod.DreamEngine
MacroPolicy = dream_engine_mod.MacroPolicy
DiscoveryTreeLogger = dream_logger_mod.DiscoveryTreeLogger

# Concrete Codelab 2 implementations
aide2_tree_mod = importlib.import_module("02_aide2.tree_node")
aide2_search_mod = importlib.import_module("02_aide2.mcts_search")
aide2_ops_mod = importlib.import_module("02_aide2.operators")

StrategyNode = aide2_tree_mod.StrategyNode
AIDE2Search = aide2_search_mod.AIDE2Search
DebuggingOperator = aide2_ops_mod.DebuggingOperator


class BuyAndHoldSPY(BaseStrategy):
    """100% allocation to SPY."""
    def __init__(self):
        super().__init__(name="BuyAndHoldSPY")
    def generate_signals(self, market_data: MarketData) -> np.ndarray:
        T, K = market_data.close_prices.shape
        signals = np.zeros((T, K), dtype=float)
        signals[:, 0] = 1.0
        return signals
    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        T, K = signals.shape
        weights = np.zeros((T, K), dtype=float)
        weights[:, 0] = 1.0
        return weights


# ==============================================================================
# 1. Extreme Market Data Anomalies
# ==============================================================================

def test_stress_flat_prices_zero_volatility():
    """Stress-tests metrics calculation when all prices are identical and volatility is exactly zero.
    
    Verifies that zero standard deviation does not raise ZeroDivisionError in Sharpe, Sortino, or MDD.
    """
    num_days = 252
    flat_md = generate_multi_era_market_data(num_days=num_days)
    flat_md.close_prices[:] = 100.0
    flat_md.open_prices[:] = 100.0
    flat_md.high_prices[:] = 100.0
    flat_md.low_prices[:] = 100.0
    flat_md.returns[:] = 0.0

    engine = BacktestEngine(default_fee_bps=10.0)
    strat = BuyAndHoldSPY()
    res = engine.run(strat, flat_md)

    assert res.success is True
    assert res.sharpe_ratio == 0.0
    assert res.sortino_ratio <= 0.0  # Due to underperforming 2% risk-free rate
    assert res.max_drawdown == 0.0
    assert not math.isnan(res.sharpe_ratio)
    assert not math.isnan(res.max_drawdown)
    assert not math.isnan(res.cagr)


def test_stress_single_day_series():
    """Stress-tests the minimum time-series boundary condition (T=1).
    
    Verifies that 1-day input data does not trigger index out of bounds or negative slicing errors.
    """
    single_day_md = generate_multi_era_market_data(num_days=1)
    engine = BacktestEngine(default_fee_bps=10.0)
    strat = BuyAndHoldSPY()
    res = engine.run(strat, single_day_md)

    assert res.success is True
    assert len(res.equity_curve) == 1
    assert res.equity_curve[0] == pytest.approx(100_000.0)
    assert res.turnover == 0.0
    assert not math.isnan(res.max_drawdown)


def test_stress_extreme_99pct_market_crash():
    """Stress-tests portfolio survival under an immediate 99.0% single-day flash crash.
    
    Verifies that equity drops near zero, MDD approaches 0.99, and metrics remain finite.
    """
    crash_md = generate_multi_era_market_data(num_days=10)
    # Day 5: 99% crash in equities
    crash_md.close_prices[5:, 0] = crash_md.close_prices[4, 0] * 0.01
    crash_md.returns[1:] = (crash_md.close_prices[1:] - crash_md.close_prices[:-1]) / (crash_md.close_prices[:-1] + 1e-8)

    engine = BacktestEngine(default_fee_bps=0.0)
    strat = BuyAndHoldSPY()
    res = engine.run(strat, crash_md)

    assert res.success is True
    assert res.max_drawdown >= 0.98
    assert res.final_equity <= 2_000.0
    assert not math.isnan(res.max_drawdown)
    assert not math.isinf(res.max_drawdown)
    assert res.sharpe_ratio < 0.0


def test_stress_negative_prices_commodity_shock():
    """Stress-tests negative commodity price regimes (e.g., WTI crude April 2020: -$37.63).
    
    Verifies that negative prices do not cause unhandled runtime exceptions in return calculations.
    """
    neg_md = generate_multi_era_market_data(num_days=15)
    # Asset 2 (GLD/Commodities) drops to negative prices on days 5..8
    neg_md.close_prices[5:9, 2] = -37.63
    neg_md.returns[1:] = (neg_md.close_prices[1:] - neg_md.close_prices[:-1]) / (neg_md.close_prices[:-1] + 1e-8)

    class CommodityTrader(BaseStrategy):
        def __init__(self):
            super().__init__(name="CommodityTrader")
        def generate_signals(self, md: MarketData) -> np.ndarray:
            return np.ones(md.close_prices.shape)
        def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
            w = np.zeros(signals.shape)
            w[:, 2] = 0.5  # 50% commodity
            w[:, -1] = 0.5 # 50% cash
            return w

    engine = BacktestEngine(default_fee_bps=10.0)
    res = engine.run(CommodityTrader(), neg_md)

    assert res.success is True
    assert len(res.equity_curve) == 15
    assert not math.isnan(res.final_equity)


def test_stress_nan_inf_return_corruption_vulnerability():
    """Empirically tests and documents engine behavior when market data returns contain NaN or Inf.
    
    Verifies that non-finite returns (NaN / Inf) are caught and rejected by the input validation guard,
    preventing corruption of downstream metrics and equity curves.
    """
    nan_md = generate_multi_era_market_data(num_days=20)
    nan_md.returns[5, 0] = np.nan
    nan_md.returns[10, 1] = np.inf

    engine = BacktestEngine(default_fee_bps=10.0)
    strat = BuyAndHoldSPY()
    res = engine.run(strat, nan_md)

    assert res is not None
    assert res.success is False
    assert "NaN or Inf" in str(res.error_message)
    assert not any(math.isnan(x) for x in res.equity_curve)


# ==============================================================================
# 2. Backtest Engine Stress & Insolvency
# ==============================================================================

def test_stress_zero_trade_frequency_756_days():
    """Stress-tests a pure buy-and-hold cash strategy across all 756 trading days.
    
    Verifies that zero trades yield exactly 0.0 turnover, zero transaction costs, and positive growth.
    """
    class PureCashStrategy(BaseStrategy):
        def __init__(self):
            super().__init__(name="PureCashStrategy")
        def generate_signals(self, md: MarketData) -> np.ndarray:
            return np.zeros(md.close_prices.shape)
        def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
            w = np.zeros(signals.shape)
            w[:, -1] = 1.0  # 100% Cash
            return w

    md = generate_multi_era_market_data(num_days=756)
    engine = BacktestEngine(default_fee_bps=10.0)
    res = engine.run(PureCashStrategy(), md)

    assert res.success is True
    assert res.turnover == 0.0
    assert res.total_cost == 0.0
    assert res.final_equity > engine.initial_capital  # Cash earns positive risk-free yield
    assert res.max_drawdown == 0.0


def test_stress_100pct_daily_turnover_churn():
    """Stress-tests high-frequency churning strategy alternating 100% allocation daily.
    
    With 756 days, flipping 100% between SPY and TLT daily produces 504.0 (50,400%) annualized turnover.
    """
    class DailyChurnStrategy(BaseStrategy):
        def __init__(self):
            super().__init__(name="DailyChurnStrategy")
        def generate_signals(self, md: MarketData) -> np.ndarray:
            return np.ones(md.close_prices.shape)
        def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
            T, K = signals.shape
            w = np.zeros((T, K))
            for t in range(T):
                if t % 2 == 0:
                    w[t, 0] = 1.0  # 100% SPY
                else:
                    w[t, 1] = 1.0  # 100% TLT
            return w

    md = generate_multi_era_market_data(num_days=756)
    engine = BacktestEngine(default_fee_bps=10.0)
    res = engine.run(DailyChurnStrategy(), md)

    assert res.success is True
    assert res.turnover == pytest.approx(504.0, rel=1e-3)
    assert res.total_cost == pytest.approx(1.51, rel=1e-2)  # ~151% of capital paid in fees


def test_stress_extreme_fee_drag_monotonicity():
    """Verifies that increasing fee drag (0 bps -> 10 bps -> 100 bps -> 1000 bps) strictly reduces equity."""
    class ModerateTurnoverStrategy(BaseStrategy):
        def __init__(self):
            super().__init__(name="ModerateTurnoverStrategy")
        def generate_signals(self, md: MarketData) -> np.ndarray:
            return np.ones(md.close_prices.shape)
        def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
            T, K = signals.shape
            w = np.zeros((T, K))
            for t in range(T):
                asset_idx = (t // 5) % (K - 1)
                w[t, asset_idx] = 0.90
                w[t, -1] = 0.10
            return w

    md = generate_multi_era_market_data(num_days=252)
    fees = [0.0, 10.0, 100.0, 1000.0]
    final_equities = []

    for fee in fees:
        engine = BacktestEngine(default_fee_bps=fee)
        res = engine.run(ModerateTurnoverStrategy(), md)
        final_equities.append(res.final_equity)

    # Strictly monotonic decline in final equity as fees increase
    for i in range(len(final_equities) - 1):
        assert final_equities[i] > final_equities[i + 1]


def test_stress_cash_depletion_and_insolvency_guard():
    """Empirically tests cash depletion and insolvency behavior under catastrophic cumulative returns.
    
    Exposes whether negative equity flips back to positive on subsequent negative return days.
    """
    class AllInEquityStrategy(BaseStrategy):
        def __init__(self):
            super().__init__(name="AllInEquityStrategy")
        def generate_signals(self, md: MarketData) -> np.ndarray:
            return np.ones(md.close_prices.shape)
        def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
            w = np.zeros(signals.shape)
            w[:, 0] = 1.0  # 100% in asset 0
            return w

    md = generate_multi_era_market_data(num_days=5)
    # Day 1: -150% crash (instant insolvency)
    md.returns[1, 0] = -1.5
    # Day 2: -120% crash
    md.returns[2, 0] = -1.2

    engine = BacktestEngine(default_fee_bps=0.0)
    res = engine.run(AllInEquityStrategy(), md)

    assert res.success is True
    # Day 1: Account becomes insolvent; insolvency guard clamps equity to 0.0
    assert res.equity_curve[1] == 0.0
    # Day 2: Insolvency guard ensures equity stays at 0.0 (no negative inversion to positive wealth)
    assert res.equity_curve[2] == 0.0
    assert res.final_equity == 0.0


# ==============================================================================
# 3. Dream-RSI World Model Stress
# ==============================================================================

def test_stress_singular_transition_matrix_absorbing():
    """Stress-tests ERTM world model when transition matrix has an absorbing state (P(crisis|crisis) = 1.0).
    
    Verifies that sampling from the absorbing state deterministically remains in that state without NaN.
    """
    wm = RegimeWorldModel()
    # Force 'crisis' to be an absorbing state
    wm.transitions["crisis"] = {"bull": 0.0, "choppy": 0.0, "crisis": 1.0}

    rng = np.random.RandomState(42)
    sampled = [wm.sample_next_regime("crisis", rng=rng) for _ in range(50)]
    assert all(r == "crisis" for r in sampled)


def test_stress_zero_probability_row_vulnerability():
    """Empirically tests world model resilience when a transition row sums to 0.0.
    
    Verifies that when all probabilities in a row are 0.0, the world model safely falls back
    to a uniform prior rather than triggering ZeroDivisionError or NaN in numpy choice.
    """
    wm = RegimeWorldModel()
    wm.transitions["bull"] = {"bull": 0.0, "choppy": 0.0, "crisis": 0.0}

    sampled = wm.sample_next_regime("bull")
    assert sampled in ["bull", "choppy", "crisis"]


def test_stress_zero_observation_regime_calibration():
    """Tests world model calibration when market data contains zero observations of certain regimes.
    
    Verifies that Laplace smoothing (smoothing=1.0) prevents division by zero,
    and zero smoothing (smoothing=0.0) safely falls back to uniform prior (1/3) without ZeroDivisionError.
    """
    # 10 days of market data only contains 'bull' regime
    md_bull_only = generate_multi_era_market_data(num_days=10)

    # With default Laplace smoothing=1.0: calibration succeeds with uniform pseudo-counts
    wm_smoothed = RegimeWorldModel()
    diag_smoothed = wm_smoothed.fit_from_market_data(md_bull_only, laplace_smoothing=1.0)
    assert diag_smoothed["status"] == "calibrated"
    assert wm_smoothed.transitions["crisis"]["crisis"] == pytest.approx(0.3333, abs=1e-2)

    # With laplace_smoothing=0.0: unobserved regimes safely fall back to uniform prior (1/3) without ZeroDivisionError
    wm_zero = RegimeWorldModel()
    diag_zero = wm_zero.fit_from_market_data(md_bull_only, laplace_smoothing=0.0)
    assert wm_zero.transitions["crisis"]["crisis"] == pytest.approx(0.3333, abs=1e-2)


def test_stress_extreme_macro_policy_branching_budget():
    """Stress-tests Dream-RSI with extreme macro policy parameters: B=100, D=100, tau=0.0.
    
    Verifies that structural yield remains positive and finite, early stopping triggers gracefully,
    and execution completes within milliseconds without out-of-memory errors.
    """
    logger = DiscoveryTreeLogger()
    wm = RegimeWorldModel()
    engine = DreamEngine(logger, wm)

    extreme_policy = MacroPolicy(
        branching_factor=100,
        search_depth=100,
        stopping_threshold=0.0,
        novelty_weight=0.5,
        name="UltraExtremeBranching",
    )

    t0 = time.perf_counter()
    dream_yield = engine.dream(extreme_policy)
    sim_result = engine.simulate_trajectory(extreme_policy, num_simulations=3)
    elapsed = time.perf_counter() - t0

    assert dream_yield > 0.0
    assert not math.isnan(dream_yield)
    assert not math.isinf(dream_yield)
    assert sim_result["total_nodes_explored"] == 300  # 100 * 3 simulations with early stop at depth 0
    assert elapsed < 0.5  # Sub-500ms execution
    assert len(sim_result["sample_path"]) == 1  # Stopped at depth 0 due to tau=0.0


def test_stress_extreme_macro_policy_no_stopping_large_scale():
    """Stress-tests Dream-RSI with large search tree without early stopping: B=50, D=50, tau=100.0.
    
    Executes 2,500 tree expansion steps in simulation. Verifies sub-second performance.
    """
    logger = DiscoveryTreeLogger()
    wm = RegimeWorldModel()
    engine = DreamEngine(logger, wm)

    large_policy = MacroPolicy(
        branching_factor=50,
        search_depth=50,
        stopping_threshold=100.0,
        novelty_weight=0.1,
        name="LargeTreeNoStop",
    )

    t0 = time.perf_counter()
    sim_result = engine.simulate_trajectory(large_policy, num_simulations=1)
    elapsed = time.perf_counter() - t0

    assert sim_result["total_nodes_explored"] == 2500  # 50 * 50
    assert elapsed < 1.0  # Runs in under 1.0 second
    assert len(sim_result["sample_path"]) == 50


# ==============================================================================
# 4. AIDE2 Code Tree Stress
# ==============================================================================

def test_stress_ast_infinite_recursion_syntax_bomb():
    """Stress-tests AIDE2 sandbox execution against an infinite recursion bomb.
    
    Verifies that RecursionError is caught cleanly, BacktestResult records failure with
    Sharpe=-999.0, reward is assigned -10.0, and the node is marked as a terminal dead end.
    """
    recursion_code = """
import numpy as np
from common.domain.strategy_base import BaseStrategy

class InfiniteRecursionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="InfiniteRecursionStrategy")
    def generate_signals(self, market_data):
        # Recursive stack overflow bomb
        return self.generate_signals(market_data)
    def compute_weights(self, signals, current_weights):
        return np.zeros_like(signals)
"""
    search = AIDE2Search()
    bomb_node = StrategyNode(node_id="bomb_node", code=recursion_code, metrics=None)
    search.add_node(bomb_node)

    eval_result = search.evaluate_node(bomb_node)
    reward = search.compute_reward(eval_result)

    assert eval_result.success is False
    assert "RecursionError" in eval_result.error_message
    assert eval_result.sharpe_ratio == -999.0
    assert reward == -10.0


def test_stress_ast_malformed_syntax_bomb():
    """Stress-tests AIDE2 sandbox against syntactically invalid Python scripts.
    
    Verifies that SyntaxError / IndentationError are intercepted without crashing MCTS.
    """
    malformed_code = "class MalformedStrategy(:\n    def broken_indent(\nreturn None"
    search = AIDE2Search()
    broken_node = StrategyNode(node_id="syntax_bomb", code=malformed_code, metrics=None)
    search.add_node(broken_node)

    eval_result = search.evaluate_node(broken_node)
    assert eval_result.success is False
    assert eval_result.sharpe_ratio == -999.0
    assert "SyntaxError" in eval_result.error_message or "IndentationError" in eval_result.error_message


def test_stress_ast_non_numeric_return_values():
    """Stress-tests strategy methods returning string or None instead of numeric numpy arrays.
    
    Verifies that type errors are caught and recorded as failed backtests.
    """
    string_code = """
import numpy as np
from common.domain.strategy_base import BaseStrategy

class StringStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="StringStrategy")
    def generate_signals(self, market_data):
        return "BUY_EVERYTHING"
    def compute_weights(self, signals, current_weights):
        return "100_PERCENT_LEVERAGE"
"""
    search = AIDE2Search()
    str_node = StrategyNode(node_id="string_return", code=string_code, metrics=None)
    search.add_node(str_node)

    eval_result = search.evaluate_node(str_node)
    assert eval_result.success is False
    assert eval_result.sharpe_ratio == -999.0
    assert "AttributeError" in eval_result.error_message or "TypeError" in eval_result.error_message


def test_stress_ast_over_leveraged_weights_sanitization():
    """Stress-tests BacktestEngine weight normalization against an over-leveraged strategy (5,000% leverage).
    
    Verifies that BacktestEngine clamps and normalizes weights row-wise to sum <= 1.0.
    """
    class OverLeveragedStrategy(BaseStrategy):
        def __init__(self):
            super().__init__(name="OverLeveragedStrategy")
        def generate_signals(self, md: MarketData) -> np.ndarray:
            return np.ones(md.close_prices.shape)
        def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
            # 50.0 allocation across all assets (massive over-leverage)
            return np.full_like(signals, 10.0)

    md = generate_multi_era_market_data(num_days=20)
    engine = BacktestEngine(default_fee_bps=10.0)
    strat = OverLeveragedStrategy()
    res = engine.run(strat, md)

    assert res.success is True
    # BacktestEngine must have normalized weights to sum to 1.0
    weights_arr = np.array(res.weights_history)
    row_sums = np.sum(np.abs(weights_arr), axis=1)
    assert np.all(row_sums <= 1.0001)


def test_stress_mcts_all_branches_dead_end_recovery():
    """Stress-tests AIDE2 MCTS adaptive backtracking when every child node fails evaluation.
    
    Verifies that MCTS step loop marks failed nodes as terminal and backtracks cleanly without infinite loops.
    """
    search = AIDE2Search(max_depth=3)
    root = StrategyNode(node_id="root_failing", code="def invalid(:\n    pass", metrics=None)
    search.add_node(root)

    # Step through 5 iterations where every candidate is broken
    for i in range(5):
        eval_node = search.step()
        assert eval_node.is_terminal is True
        assert eval_node.metrics.success is False

    summary = search.get_tree_summary()
    assert summary["total_nodes"] == 5  # root evaluated on step 0, then 4 expanded children
    assert summary["best_sharpe"] == -999.0
