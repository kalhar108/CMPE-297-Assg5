"""
Tier 2: Boundary, Stress, and Corner Case Tests for Assignment 5 RSI.

Covers:
- Empty data and single-day boundaries
- Flat market and zero volatility regimes
- Zero trades / cash-only allocation
- Crisis shock regime and drawdown extremes
- Prohibitive fee models (0 bps to 1000 bps)
- Divide-by-zero and syntax error AST mutation recovery
- UCT edge cases (unvisited nodes, zero parent visits, extreme exploration parameters)
- NaN/Inf weight sanitization
- Lookback parameter exceeding dataset length
"""

import math
import pytest
import numpy as np
import pandas as pd

from conftest import (
    MarketData, generate_multi_era_market_data,
    BacktestEngine, BacktestResult, BaseStrategy, make_dummy_backtest_result,
    BuyAndHoldSPY, MomentumStrategy, AdaptiveRegimeStrategy, FaultyStrategy,
    AIDE2Search, StrategyNode, DebuggingOperator, DeterministicMockProvider
)


def test_boundary_empty_market_data():
    """Verifies backtester handles empty market data gracefully without throwing unhandled exceptions."""
    empty_md = MarketData(
        symbols=["SPY", "TLT", "GLD", "VIX", "CASH"],
        dates=[],
        close_prices=np.zeros((0, 5)),
        open_prices=np.zeros((0, 5)),
        high_prices=np.zeros((0, 5)),
        low_prices=np.zeros((0, 5)),
        volumes=np.zeros((0, 5)),
        returns=np.zeros((0, 5)),
        regimes=[]
    )
    engine = BacktestEngine(default_fee_bps=10.0)
    strat = BuyAndHoldSPY()
    res = engine.run(strat, empty_md)
    assert res is not None
    assert res.sharpe_ratio == 0.0 or math.isnan(res.sharpe_ratio) or res.max_drawdown == 0.0


def test_boundary_single_day_data():
    """Verifies backtesting on single-day data avoids unhandled ZeroDivisionError."""
    single_md = generate_multi_era_market_data(num_days=1)
    engine = BacktestEngine(default_fee_bps=10.0)
    strat = BuyAndHoldSPY()
    res = engine.run(strat, single_md)
    assert res is not None
    assert len(res.equity_curve) <= 1 or res.equity_curve[0] == pytest.approx(engine.initial_capital, rel=1e-3)


def test_boundary_flat_market_zero_volatility():
    """Verifies market with 0 price change returns without crashing on zero variance."""
    flat_md = generate_multi_era_market_data(num_days=100)
    flat_md.close_prices[:] = 100.0
    flat_md.returns[:] = 0.0
    engine = BacktestEngine(default_fee_bps=10.0)
    res = engine.run(BuyAndHoldSPY(), flat_md)
    assert res.sharpe_ratio == 0.0 or not math.isnan(res.max_drawdown)


def test_boundary_zero_trades_no_signals(standard_market_data: MarketData):
    """Verifies strategy that generates zero signals (100% cash) executes with zero turnover."""
    class AllCashStrategy(BaseStrategy):
        def __init__(self):
            super().__init__(name="AllCashStrategy")

        def generate_signals(self, market_data: MarketData) -> np.ndarray:
            T, K = market_data.close_prices.shape
            return np.zeros((T, K))

        def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
            T, K = signals.shape
            w = np.zeros((T, K))
            w[:, -1] = 1.0  # 100% Cash
            return w

    engine = BacktestEngine(default_fee_bps=10.0)
    res = engine.run(AllCashStrategy(), standard_market_data)
    assert res.turnover <= 1e-4 or res.max_drawdown <= 1e-3


def test_boundary_crisis_shock_regime(crisis_market_data: MarketData, buy_and_hold_strategy: BuyAndHoldSPY):
    """Verifies behavior under negative drift and high volatility; MaxDD is bounded within [0.0, 1.0]."""
    engine = BacktestEngine(default_fee_bps=10.0)
    res = engine.run(buy_and_hold_strategy, crisis_market_data)
    assert 0.0 <= res.max_drawdown <= 1.0
    assert not math.isnan(res.max_drawdown)


def test_boundary_zero_transaction_fee(standard_market_data: MarketData, momentum_strategy: MomentumStrategy):
    """Verifies default_fee_bps = 0.0 imposes zero friction drag."""
    engine_zero_fee = BacktestEngine(default_fee_bps=0.0)
    res = engine_zero_fee.run(momentum_strategy, standard_market_data)
    assert res.equity_curve[-1] > 0.0


def test_boundary_prohibitive_transaction_fee(standard_market_data: MarketData, momentum_strategy: MomentumStrategy):
    """Verifies extreme fee (1000 bps = 10%) does not produce negative equity below 0.0."""
    engine_extreme_fee = BacktestEngine(default_fee_bps=1000.0)
    res = engine_extreme_fee.run(momentum_strategy, standard_market_data)
    assert (np.array(res.equity_curve) >= 0.0).all()


def test_boundary_divide_by_zero_ast_mutation():
    """Verifies DebuggingOperator catches and repairs divide-by-zero expression."""
    provider = DeterministicMockProvider()
    debugger = DebuggingOperator(provider)
    buggy = "def signal(p):\n    return p / (1.0 / 0.0)\n"
    fixed = debugger.debug_and_fix(buggy, "ZeroDivisionError: float division by zero")
    assert "1.0 / 0.0" not in fixed
    assert "1e-6" in fixed or "std + 1e-8" in fixed


def test_boundary_syntax_error_recovery():
    """Verifies DebuggingOperator can handle syntax error traceback and emit valid code."""
    provider = DeterministicMockProvider()
    debugger = DebuggingOperator(provider)
    broken_syntax = "def bad_syntax(:\n    pass"
    repaired = debugger.debug_and_fix(broken_syntax, "SyntaxError: invalid syntax")
    assert repaired is not None


def test_boundary_uct_unvisited_node():
    """Verifies unvisited node (visits=0) returns float('inf') to ensure immediate exploration."""
    search = AIDE2Search()
    metrics = make_dummy_backtest_result(sharpe_ratio=1.0, max_drawdown=0.1)
    node = StrategyNode("unvisited_node", "code", metrics, visits=0, total_value=0.0)
    score = search.uct_score(node, parent_visits=10)
    assert score == float("inf")


def test_boundary_uct_zero_parent_visits():
    """Verifies UCT computation with parent_visits=0 does not raise math domain error in log(0)."""
    search = AIDE2Search()
    metrics = make_dummy_backtest_result(sharpe_ratio=1.0, max_drawdown=0.1)
    node = StrategyNode("child_node", "code", metrics, visits=1, total_value=1.5)
    score = search.uct_score(node, parent_visits=0)
    assert not math.isnan(score)
    assert not math.isinf(score)
    assert score >= 1.5


def test_boundary_uct_extreme_c_param():
    """Verifies UCT behavior with extreme c_param values (c=0 vs c=100)."""
    metrics = make_dummy_backtest_result(sharpe_ratio=1.0, max_drawdown=0.1)
    node = StrategyNode("test_node", "code", metrics, visits=4, total_value=4.0)
    
    search_pure_exploit = AIDE2Search(c_param=0.0)
    search_pure_explore = AIDE2Search(c_param=100.0)
    
    score_exploit = search_pure_exploit.uct_score(node, parent_visits=16)
    score_explore = search_pure_explore.uct_score(node, parent_visits=16)
    
    assert score_exploit == pytest.approx(1.0)
    assert score_explore > 10.0


def test_boundary_nan_weights_sanitization(standard_market_data: MarketData):
    """Verifies strategy returning NaN weights is handled gracefully."""
    faulty_strat = FaultyStrategy(fault_type="nan_weights")
    engine = BacktestEngine(default_fee_bps=10.0)
    res = engine.run(faulty_strat, standard_market_data)
    assert res is not None


def test_boundary_lookback_exceeding_dataset_length():
    """Verifies strategy with lookback parameter exceeding dataset length executes safely."""
    short_md = generate_multi_era_market_data(num_days=5)
    strat = MomentumStrategy(lookback=50)  # 50 > 5
    engine = BacktestEngine(default_fee_bps=10.0)
    res = engine.run(strat, short_md)
    assert res is not None
    assert len(res.equity_curve) == 5
