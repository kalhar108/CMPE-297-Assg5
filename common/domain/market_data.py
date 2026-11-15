"""Multi-era 756-day daily OHLCV market data generator.

Generates standardized financial time-series data across three distinct regimes:
- Era 1 (Days 1–252): Bull Trend (high drift, low volatility)
- Era 2 (Days 253–504): Choppy / Mean-Reverting (zero drift, moderate volatility)
- Era 3 (Days 505–756): Crisis / Stress Regime (negative drift, extreme volatility spikes)

Universe includes 5 standardized assets:
- SPY: US Large-Cap Equities
- TLT: Long-Term US Treasury Bonds (Flight to Safety)
- GLD: Gold / Commodity Inflation & Chaos Hedge
- VIX: Realized/Implied Volatility Proxy
- CASH: Risk-Free Short Rate Baseline (2.0% annualized)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd


class MarketRegime(str, Enum):
    """Macroeconomic market regimes."""
    BULL = "bull"
    CHOPPY = "choppy"
    CRISIS = "crisis"


DEFAULT_SYMBOLS = ["SPY", "TLT", "GLD", "VIX", "CASH"]


@dataclass
class MarketData:
    """Standardized multi-asset daily market data structure."""

    symbols: List[str]
    dates: List[str]
    close_prices: np.ndarray  # shape: (T, K)
    open_prices: np.ndarray   # shape: (T, K)
    high_prices: np.ndarray   # shape: (T, K)
    low_prices: np.ndarray    # shape: (T, K)
    volumes: np.ndarray       # shape: (T, K)
    returns: np.ndarray       # shape: (T, K), daily relative returns
    regimes: List[str]        # length T

    @property
    def num_days(self) -> int:
        return len(self.dates)

    @property
    def num_assets(self) -> int:
        return len(self.symbols)

    def get_symbol_index(self, symbol: str) -> int:
        """Returns the 0-based column index for an asset symbol."""
        if symbol not in self.symbols:
            raise KeyError(f"Symbol '{symbol}' not in market universe: {self.symbols}")
        return self.symbols.index(symbol)

    def get_era(self, era: Union[int, str, MarketRegime]) -> MarketData:
        """Slices market data for a specific era partition.

        Era 1 / 'bull': Days 0..251 (Train)
        Era 2 / 'choppy': Days 252..503 (Validation)
        Era 3 / 'crisis': Days 504..755 (Stress test)
        """
        era_str = str(era).lower().replace("marketregime.", "")
        if era_str in ("1", "bull"):
            start, end = 0, 252
        elif era_str in ("2", "choppy"):
            start, end = 252, 504
        elif era_str in ("3", "crisis"):
            start, end = 504, 756
        else:
            raise ValueError(f"Unknown era identifier: {era}. Must be 1, 2, 3 or 'bull', 'choppy', 'crisis'.")

        # Clamp bounds
        start = min(start, self.num_days)
        end = min(end, self.num_days)

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

    def to_dataframe(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Converts market data to a pandas DataFrame."""
        if symbol is not None:
            idx = self.get_symbol_index(symbol)
            return pd.DataFrame(
                {
                    "open": self.open_prices[:, idx],
                    "high": self.high_prices[:, idx],
                    "low": self.low_prices[:, idx],
                    "close": self.close_prices[:, idx],
                    "volume": self.volumes[:, idx],
                    "return": self.returns[:, idx],
                    "regime": self.regimes,
                },
                index=pd.to_datetime(self.dates),
            )

        # Multi-index or Close DataFrame
        df = pd.DataFrame(
            self.close_prices,
            columns=self.symbols,
            index=pd.to_datetime(self.dates),
        )
        return df


def generate_multi_era_market_data(
    seed: int = 42,
    num_days: int = 756,
    symbols: Optional[List[str]] = None,
) -> MarketData:
    """Generates synthetic multi-era daily market data using seeded geometric Brownian motion and jump-diffusion.

    Guarantees 100% deterministic reproducibility across runs.
    """
    rng = np.random.RandomState(seed)
    symbols = symbols or list(DEFAULT_SYMBOLS)
    K = len(symbols)

    # Asset initial prices
    init_prices = {
        "SPY": 350.0,
        "TLT": 140.0,
        "GLD": 170.0,
        "VIX": 15.0,
        "CASH": 100.0,
    }
    p0 = np.array([init_prices.get(s, 100.0) for s in symbols], dtype=float)

    # 3-Era configuration: (days, regime_name, annual_drifts, annual_vols, corr_matrix)
    # Era 1: Bull (0..252)
    # Era 2: Choppy (252..504)
    # Era 3: Crisis (504..756)
    dt = 1.0 / 252.0

    close_arr = np.zeros((num_days, K), dtype=float)
    open_arr = np.zeros((num_days, K), dtype=float)
    high_arr = np.zeros((num_days, K), dtype=float)
    low_arr = np.zeros((num_days, K), dtype=float)
    vol_arr = np.zeros((num_days, K), dtype=float)
    regimes: List[str] = []

    # Initial state
    current_prices = p0.copy()

    for t in range(num_days):
        if t < 252:
            regime = "bull"
            # Bull parameters: SPY strong drift, TLT modest, GLD modest, VIX low mean-reverting
            mu = np.array([0.18, 0.04, 0.06, 0.0, 0.020])
            sigma = np.array([0.12, 0.09, 0.12, 0.35, 0.0001])
            # SPY and TLT slightly negative correlation
            corr = np.array([
                [1.00, -0.25,  0.05, -0.65, 0.0],
                [-0.25, 1.00,  0.20,  0.20, 0.0],
                [0.05,  0.20,  1.00, -0.10, 0.0],
                [-0.65, 0.20, -0.10,  1.00, 0.0],
                [0.00,  0.00,  0.00,  0.00, 1.0],
            ])
            jump_prob = 0.0
            jump_scale = 0.0

        elif t < 504:
            regime = "choppy"
            # Choppy parameters: zero drift, moderate volatility, sector rotation
            mu = np.array([0.00, 0.01, 0.03, 0.0, 0.025])
            sigma = np.array([0.20, 0.14, 0.16, 0.45, 0.0001])
            corr = np.array([
                [1.00, -0.10,  0.10, -0.70, 0.0],
                [-0.10, 1.00,  0.25,  0.15, 0.0],
                [0.10,  0.25,  1.00, -0.05, 0.0],
                [-0.70, 0.15, -0.05,  1.00, 0.0],
                [0.00,  0.00,  0.00,  0.00, 1.0],
            ])
            jump_prob = 0.02
            jump_scale = -0.015

        else:
            regime = "crisis"
            # Crisis parameters: steep negative drift in equities, extreme vol, gold/bonds hedge flight
            mu = np.array([-0.38, 0.14, 0.18, 0.60, 0.030])
            sigma = np.array([0.45, 0.22, 0.25, 0.85, 0.0001])
            corr = np.array([
                [1.00, -0.55, -0.30, -0.85, 0.0],
                [-0.55, 1.00,  0.40,  0.50, 0.0],
                [-0.30, 0.40,  1.00,  0.25, 0.0],
                [-0.85, 0.50,  0.25,  1.00, 0.0],
                [0.00,  0.00,  0.00,  0.00, 1.0],
            ])
            jump_prob = 0.08
            jump_scale = -0.045

        regimes.append(regime)

        # Cholesky decomposition for correlated shocks
        L = np.linalg.cholesky(corr[:K, :K])
        z = rng.standard_normal(K)
        correlated_shocks = L @ z

        # Daily drift and diffusion
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * correlated_shocks
        daily_log_returns = drift + diffusion

        # Jump shock in choppy/crisis
        if jump_prob > 0 and rng.rand() < jump_prob:
            daily_log_returns[0] += jump_scale  # SPY drops
            daily_log_returns[1] += abs(jump_scale) * 0.5  # TLT rises
            daily_log_returns[3] += abs(jump_scale) * 3.0  # VIX surges

        # Update price
        current_prices = current_prices * np.exp(daily_log_returns)

        # Enforce VIX boundary conditions (mean reversion floor 10, cap 90)
        vix_idx = symbols.index("VIX")
        target_vix = 14.0 if regime == "bull" else (22.0 if regime == "choppy" else 45.0)
        current_prices[vix_idx] = 0.90 * current_prices[vix_idx] + 0.10 * target_vix
        current_prices[vix_idx] = np.clip(current_prices[vix_idx], 10.0, 95.0)

        # Cash asset grows deterministically at risk-free rate
        cash_idx = symbols.index("CASH")
        current_prices[cash_idx] = p0[cash_idx] * (1.0 + 0.02 * (t / 252.0))

        close_arr[t] = current_prices

        # Synthetic Open, High, Low, Volume
        noise_open = rng.normal(0.0, 0.002, K)
        noise_hl = np.abs(rng.normal(0.0, 0.005, K))

        if t == 0:
            open_arr[t] = p0 * (1.0 + noise_open)
        else:
            open_arr[t] = close_arr[t - 1] * (1.0 + noise_open)

        high_arr[t] = np.maximum(open_arr[t], close_arr[t]) * (1.0 + noise_hl)
        low_arr[t] = np.minimum(open_arr[t], close_arr[t]) * (1.0 - noise_hl)

        # Base volume with volatility multiplier
        base_vols = np.array([50_000_000, 15_000_000, 10_000_000, 5_000_000, 1_000_000])[:K]
        vol_arr[t] = base_vols * (1.0 + 3.0 * np.abs(daily_log_returns))

    # Calculate daily relative returns
    rets = np.zeros_like(close_arr)
    rets[1:] = (close_arr[1:] - close_arr[:-1]) / (close_arr[:-1] + 1e-8)

    # Generate dates (starting 2024-01-02, 756 business days)
    date_range = pd.date_range(start="2024-01-02", periods=num_days, freq="B")
    date_strings = [d.strftime("%Y-%m-%d") for d in date_range]

    return MarketData(
        symbols=symbols,
        dates=date_strings,
        close_prices=close_arr,
        open_prices=open_arr,
        high_prices=high_arr,
        low_prices=low_arr,
        volumes=vol_arr,
        returns=rets,
        regimes=regimes,
    )
