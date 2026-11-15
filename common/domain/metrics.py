"""Quantitative risk and return telemetry metrics for the Alpha-RSI engine.

Calculates standard institutional performance statistics:
- Annualized Cumulative Growth Rate (CAGR)
- Annualized Sharpe Ratio (excess return over risk-free rate / volatility)
- Annualized Sortino Ratio (downside semivariance penalty only)
- Maximum Drawdown (peak-to-trough decline)
- Calmar Ratio (CAGR / Max Drawdown)
- Annualized Portfolio Turnover
- Transaction Cost Drag (modeling basis points fee per turn)
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np


def cagr(equity_curve: np.ndarray, num_days: Optional[int] = None) -> float:
    """Calculates Compound Annual Growth Rate (CAGR).

    Args:
        equity_curve: 1D array of portfolio equity values over time.
        num_days: Optional explicit number of trading days; defaults to len(equity_curve).

    Returns:
        Annualized growth rate as a decimal (e.g. 0.15 = +15% per year).
    """
    if len(equity_curve) < 2:
        return 0.0

    v0 = float(equity_curve[0])
    v_end = float(equity_curve[-1])

    if v0 <= 0 or v_end <= 0:
        return -1.0  # Total loss

    T = num_days if num_days is not None else len(equity_curve)
    if T <= 1:
        return 0.0

    years = T / 252.0
    try:
        return float((v_end / v0) ** (1.0 / years) - 1.0)
    except (ZeroDivisionError, OverflowError):
        return 0.0


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Calculates Maximum Drawdown (MDD) as a positive decimal fraction.

    Args:
        equity_curve: 1D array of portfolio equity values over time.

    Returns:
        Maximum peak-to-trough percentage drop (e.g., 0.18 = 18% drawdown).
    """
    if len(equity_curve) < 2:
        return 0.0

    peaks = np.maximum.accumulate(equity_curve)
    # Guard against zero peak
    peaks = np.maximum(peaks, 1e-8)
    drawdowns = (peaks - equity_curve) / peaks
    mdd = float(np.max(drawdowns))
    return max(0.0, mdd)


def annualized_sharpe(
    returns: np.ndarray, risk_free_rate: float = 0.02
) -> float:
    """Calculates Annualized Sharpe Ratio against a risk-free benchmark.

    Args:
        returns: 1D array of daily portfolio percentage returns.
        risk_free_rate: Annualized risk-free rate (e.g. 0.02 = 2.0%).

    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / 252.0
    excess_returns = returns - daily_rf

    mean_excess = np.mean(excess_returns)
    std_returns = np.std(returns, ddof=1) if len(returns) > 1 else np.std(returns)

    if std_returns < 1e-8 or np.isnan(std_returns):
        return 0.0

    sharpe = float(np.sqrt(252.0) * (mean_excess / std_returns))
    return sharpe if not np.isnan(sharpe) else 0.0


def annualized_sortino(
    returns: np.ndarray, risk_free_rate: float = 0.02
) -> float:
    """Calculates Annualized Sortino Ratio focusing strictly on downside volatility.

    Args:
        returns: 1D array of daily portfolio percentage returns.
        risk_free_rate: Annualized risk-free rate (e.g. 0.02 = 2.0%).

    Returns:
        Annualized Sortino ratio.
    """
    if len(returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / 252.0
    excess_returns = returns - daily_rf
    mean_excess = np.mean(excess_returns)

    # Downside deviations: only deviations below 0 are penalized
    downside_deviations = np.minimum(excess_returns, 0.0)
    downside_variance = np.mean(downside_deviations**2)
    downside_std = np.sqrt(downside_variance)

    if downside_std < 1e-8 or np.isnan(downside_std):
        # If no downside, high sortino if mean excess > 0, else 0
        return float(np.sqrt(252.0) * mean_excess / 1e-6) if mean_excess > 0 else 0.0

    sortino = float(np.sqrt(252.0) * (mean_excess / downside_std))
    return sortino if not np.isnan(sortino) else 0.0


def calmar_ratio(cagr_val: float, mdd_val: float, epsilon: float = 1e-4) -> float:
    """Calculates Calmar Ratio (CAGR / Max Drawdown).

    Args:
        cagr_val: Compound annual growth rate.
        mdd_val: Maximum drawdown (positive decimal).
        epsilon: Small constant to prevent division by zero.

    Returns:
        Calmar ratio.
    """
    denom = max(mdd_val, epsilon)
    calmar = cagr_val / denom
    return float(calmar) if not np.isnan(calmar) else 0.0


def turnover(weights: np.ndarray, num_days: Optional[int] = None) -> float:
    """Calculates Annualized Portfolio Turnover.

    Args:
        weights: 2D array of portfolio weights of shape (T, K).
        num_days: Optional number of trading days for annualization.

    Returns:
        Annualized turnover rate (e.g., 2.5 = 250% annual turnover).
    """
    if len(weights) < 2:
        return 0.0

    T = num_days if num_days is not None else len(weights)
    daily_turnover = np.sum(np.abs(weights[1:] - weights[:-1]), axis=1)
    total_turnover = np.sum(daily_turnover)

    # Annualize: (total_turnover / (T - 1)) * 252
    annualized = float((total_turnover / max(T - 1, 1)) * 252.0)
    return annualized if not np.isnan(annualized) else 0.0


def transaction_costs(weights: np.ndarray, fee_bps: float = 10.0) -> float:
    """Computes total transaction costs incurred across the simulation.

    Args:
        weights: 2D array of portfolio weights of shape (T, K).
        fee_bps: Transaction cost in basis points (10 bps = 0.0010).

    Returns:
        Total cumulative transaction cost fraction.
    """
    if len(weights) < 2:
        return 0.0

    fee_rate = fee_bps / 10000.0
    daily_traded_volume = np.sum(np.abs(weights[1:] - weights[:-1]), axis=1)
    total_cost = float(np.sum(daily_traded_volume) * fee_rate)
    return total_cost


def compute_portfolio_metrics(
    equity_curve: np.ndarray,
    daily_returns: np.ndarray,
    weights: np.ndarray,
    fee_bps: float = 10.0,
    risk_free_rate: float = 0.02,
) -> Dict[str, float]:
    """Computes the full institutional suite of portfolio performance metrics.

    Args:
        equity_curve: 1D array of portfolio equity values.
        daily_returns: 1D array of net daily returns.
        weights: 2D array of portfolio weights over time.
        fee_bps: Fee in basis points per unit turnover.
        risk_free_rate: Annualized benchmark risk-free rate.

    Returns:
        Dictionary of computed performance metrics.
    """
    cagr_val = cagr(equity_curve)
    mdd_val = max_drawdown(equity_curve)
    sharpe_val = annualized_sharpe(daily_returns, risk_free_rate=risk_free_rate)
    sortino_val = annualized_sortino(daily_returns, risk_free_rate=risk_free_rate)
    calmar_val = calmar_ratio(cagr_val, mdd_val)
    turnover_val = turnover(weights)
    total_cost_val = transaction_costs(weights, fee_bps=fee_bps)

    return {
        "cagr": round(cagr_val, 6),
        "sharpe_ratio": round(sharpe_val, 4),
        "sortino_ratio": round(sortino_val, 4),
        "max_drawdown": round(mdd_val, 4),
        "calmar_ratio": round(calmar_val, 4),
        "turnover": round(turnover_val, 4),
        "total_cost": round(total_cost_val, 6),
    }
