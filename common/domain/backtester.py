"""High-speed vectorized NumPy backtesting engine for quantitative alpha strategies.

Executes backtests in sub-20ms per strategy with strict transaction cost modeling
(10 basis points default friction), zero-lookahead temporal alignment, and
institutional risk/return telemetry.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .market_data import MarketData
from .metrics import compute_portfolio_metrics
from .strategy_base import BaseStrategy


@dataclass
class BacktestResult:
    """Comprehensive performance result emitted by the BacktestEngine."""

    sharpe_ratio: float
    max_drawdown: float
    sortino_ratio: float
    calmar_ratio: float
    cagr: float
    turnover: float
    total_cost: float
    equity_curve: List[float]
    daily_returns: List[float]
    weights_history: List[List[float]]
    execution_time_ms: float
    success: bool = True
    error_message: Optional[str] = None
    traceback_log: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else 0.0

    @property
    def total_return(self) -> float:
        if not self.equity_curve or self.equity_curve[0] <= 0:
            return 0.0
        return (self.equity_curve[-1] / self.equity_curve[0]) - 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes backtest metrics and operational telemetry to a dictionary."""
        return {
            "success": self.success,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "cagr": self.cagr,
            "turnover": self.turnover,
            "total_cost": self.total_cost,
            "total_return": round(self.total_return, 4),
            "final_equity": round(self.final_equity, 2),
            "execution_time_ms": round(self.execution_time_ms, 2),
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    def summary_table(self) -> str:
        """Emits an ASCII table summarizing key financial metrics."""
        return (
            f"+-----------------------------------------------------+\n"
            f"|              Backtest Performance Summary           |\n"
            f"+-------------------------+---------------------------+\n"
            f"| Metric                  | Value                     |\n"
            f"+-------------------------+---------------------------+\n"
            f"| Sharpe Ratio (Ann.)     | {self.sharpe_ratio:>25.4f} |\n"
            f"| Sortino Ratio (Ann.)    | {self.sortino_ratio:>25.4f} |\n"
            f"| Maximum Drawdown (MDD)  | {self.max_drawdown * 100:>24.2f}% |\n"
            f"| CAGR                    | {self.cagr * 100:>24.2f}% |\n"
            f"| Calmar Ratio            | {self.calmar_ratio:>25.4f} |\n"
            f"| Annualized Turnover     | {self.turnover * 100:>24.2f}% |\n"
            f"| Total Transaction Cost  | {self.total_cost * 100:>24.4f}% |\n"
            f"| Total Cumulative Return | {self.total_return * 100:>24.2f}% |\n"
            f"| Execution Wall Time     | {self.execution_time_ms:>23.2f}ms |\n"
            f"+-------------------------+---------------------------+"
        )


class BacktestEngine:
    """Vectorized NumPy backtesting engine."""

    def __init__(
        self,
        default_fee_bps: float = 10.0,
        risk_free_rate: float = 0.02,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.default_fee_bps = default_fee_bps
        self.risk_free_rate = risk_free_rate
        self.initial_capital = initial_capital

    def run(
        self,
        strategy: BaseStrategy,
        data: MarketData,
        fee_bps: Optional[float] = None,
    ) -> BacktestResult:
        """Executes a vectorized backtest for a strategy on given market data.

        Guarantees sub-20ms execution per run via vectorized NumPy operations.
        Catches execution exceptions, recording detailed tracebacks for self-debugging.

        Args:
            strategy: BaseStrategy instance to evaluate.
            data: MarketData instance containing historical market prices and returns.
            fee_bps: Optional fee in basis points (defaults to self.default_fee_bps).

        Returns:
            BacktestResult object containing performance metrics and equity curve.
        """
        start_time = time.perf_counter()
        active_fee_bps = fee_bps if fee_bps is not None else self.default_fee_bps
        fee_rate = active_fee_bps / 10000.0

        try:
            # 0. Input Validation
            if np.any(np.isnan(data.returns)) or np.any(np.isinf(data.returns)):
                raise ValueError("MarketData returns contain NaN or Inf values; execution rejected.")

            # 1. Signal Generation
            signals = strategy.generate_signals(data)

            # 2. Portfolio Weight Allocation
            T, K = data.close_prices.shape
            current_weights = np.zeros((T, K), dtype=float)
            target_weights = strategy.compute_weights(signals, current_weights)

            # Validate dimensions and bounds
            if target_weights.shape != (T, K):
                raise ValueError(
                    f"Weight shape mismatch: expected ({T}, {K}), got {target_weights.shape}"
                )

            # Check and clamp weights: sum <= 1.0 (with cash slack if needed)
            weight_sums = np.sum(np.abs(target_weights), axis=1)
            over_leverage = weight_sums > 1.0001
            if np.any(over_leverage):
                # Normalize over-leveraged rows
                target_weights[over_leverage] = (
                    target_weights[over_leverage] / weight_sums[over_leverage, None]
                )

            # 3. Zero-Lookahead Return Execution
            # Weights computed at time t realize returns at time t+1
            realized_weights = target_weights[:-1]  # shape: (T-1, K)
            asset_returns = data.returns[1:]        # shape: (T-1, K)

            gross_step_returns = np.sum(realized_weights * asset_returns, axis=1)  # shape: (T-1,)

            # 4. Transaction Cost Friction (10 bps fee per turnover)
            # Turnover between time t and time t-1
            turnovers = np.sum(np.abs(target_weights[1:] - target_weights[:-1]), axis=1)  # shape: (T-1,)
            cost_friction = turnovers * fee_rate

            net_step_returns = gross_step_returns - cost_friction

            # 5. Cumulative Equity Curve
            # Insolvency guard in cumulative equity curve calculation
            equity_curve = [float(self.initial_capital)]
            current_equity = float(self.initial_capital)
            for ret in net_step_returns:
                if current_equity <= 0.0:
                    current_equity = 0.0
                else:
                    current_equity = max(0.0, current_equity * (1.0 + float(ret)))
                equity_curve.append(current_equity)
            equity_curve = np.array(equity_curve, dtype=float)

            # 6. Quantitative Metrics Computation
            metrics = compute_portfolio_metrics(
                equity_curve=equity_curve,
                daily_returns=net_step_returns,
                weights=target_weights,
                fee_bps=active_fee_bps,
                risk_free_rate=self.risk_free_rate,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return BacktestResult(
                sharpe_ratio=metrics["sharpe_ratio"],
                max_drawdown=metrics["max_drawdown"],
                sortino_ratio=metrics["sortino_ratio"],
                calmar_ratio=metrics["calmar_ratio"],
                cagr=metrics["cagr"],
                turnover=metrics["turnover"],
                total_cost=metrics["total_cost"],
                equity_curve=equity_curve.tolist(),
                daily_returns=net_step_returns.tolist(),
                weights_history=target_weights.tolist(),
                execution_time_ms=elapsed_ms,
                success=True,
                metadata={
                    "strategy_name": strategy.name,
                    "num_days": T,
                    "num_assets": K,
                    "fee_bps": active_fee_bps,
                },
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            tb = traceback.format_exc()
            return BacktestResult(
                sharpe_ratio=-999.0,
                max_drawdown=1.0,
                sortino_ratio=-999.0,
                calmar_ratio=-999.0,
                cagr=-1.0,
                turnover=0.0,
                total_cost=0.0,
                equity_curve=[self.initial_capital],
                daily_returns=[],
                weights_history=[],
                execution_time_ms=elapsed_ms,
                success=False,
                error_message=f"{type(e).__name__}: {str(e)}",
                traceback_log=tb,
                metadata={
                    "strategy_name": strategy.name,
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )
