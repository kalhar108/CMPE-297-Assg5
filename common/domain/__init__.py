"""Quantitative Alpha Domain Engine.

Provides:
- Multi-era 756-day market data generation across Bull, Choppy, and Crisis regimes.
- Vectorized NumPy backtesting engine with strict 10 bps transaction friction.
- Institutional risk/return metrics: Sharpe, Sortino, Calmar, Max Drawdown, CAGR, Turnover.
- Strategy AST base class and baseline trading strategies.
"""

from .market_data import (
    DEFAULT_SYMBOLS,
    MarketData,
    MarketRegime,
    generate_multi_era_market_data,
)
from .backtester import (
    BacktestEngine,
    BacktestResult,
)
from .metrics import (
    annualized_sharpe,
    annualized_sortino,
    cagr,
    calmar_ratio,
    compute_portfolio_metrics,
    max_drawdown,
    transaction_costs,
    turnover,
)
from .strategy_base import (
    BaseStrategy,
    DynamicCodeStrategy,
    EqualWeightStrategy,
    MeanReversionStrategy,
    MomentumFactor,
    MovingAverageCrossover,
)

__all__ = [
    # Market Data
    "DEFAULT_SYMBOLS",
    "MarketData",
    "MarketRegime",
    "generate_multi_era_market_data",
    # Backtester
    "BacktestEngine",
    "BacktestResult",
    # Metrics
    "cagr",
    "max_drawdown",
    "annualized_sharpe",
    "annualized_sortino",
    "calmar_ratio",
    "turnover",
    "transaction_costs",
    "compute_portfolio_metrics",
    # Strategies
    "BaseStrategy",
    "EqualWeightStrategy",
    "MovingAverageCrossover",
    "MomentumFactor",
    "MeanReversionStrategy",
    "DynamicCodeStrategy",
]
