"""Base strategy AST and executable interface for quantitative alpha discovery.

Provides:
- BaseStrategy: Abstract base class with signal generation, portfolio allocation, and AST representation.
- Baseline Strategies:
  1. EqualWeightStrategy: Benchmark 1/N allocation.
  2. MovingAverageCrossover: Dual moving average trend-following alpha (vectorized sub-1ms).
  3. MomentumFactor: Cross-sectional and time-series momentum ranking (vectorized sub-1ms).
  4. MeanReversionStrategy: Bollinger / Z-Score mean-reverting alpha with volatility gating (vectorized sub-3ms).
- DynamicCodeStrategy: Compiles, validates, and runs code synthesized dynamically by LLM agents.
"""

from __future__ import annotations

import ast
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from scipy import signal as sp_signal

from .market_data import MarketData


class BaseStrategy(ABC):
    """Abstract base class for all algorithmic trading strategies."""

    def __init__(
        self,
        name: str = "BaseStrategy",
        params: Optional[Dict[str, Any]] = None,
        code: Optional[str] = None,
    ) -> None:
        self.name = name
        self.params = params or {}
        self._code = code

    @property
    def code(self) -> str:
        """Returns the Python source code string defining this strategy."""
        if self._code is not None:
            return self._code
        try:
            return inspect.getsource(self.__class__)
        except Exception:
            return f"# Source code unavailable for {self.name}"

    @code.setter
    def code(self, value: str) -> None:
        self._code = value

    @abstractmethod
    def generate_signals(self, market_data: MarketData) -> np.ndarray:
        """Generates raw directional or alpha signals for all assets.

        Args:
            market_data: Multi-era daily market data.

        Returns:
            numpy array of shape (T, K) containing signal values.
        """
        pass

    @abstractmethod
    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        """Maps raw signals to normalized portfolio weights subject to risk constraints.

        Args:
            signals: Array of shape (T, K) containing generated signals.
            current_weights: Array of shape (T, K) representing current asset weights.

        Returns:
            numpy array of shape (T, K) containing target normalized weights.
            Must satisfy sum(weights[t, :]) <= 1.0 and weights[t, k] >= 0.0.
        """
        pass

    def get_ast(self) -> Optional[ast.AST]:
        """Parses the strategy's source code into a Python Abstract Syntax Tree."""
        try:
            return ast.parse(self.code)
        except Exception:
            return None

    def get_ast_summary(self) -> Dict[str, Any]:
        """Extracts structured summary of the strategy's AST."""
        tree = self.get_ast()
        if tree is None:
            return {"error": "Failed to parse AST"}

        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        num_statements = sum(1 for node in ast.walk(tree) if isinstance(node, ast.stmt))

        return {
            "name": self.name,
            "classes": classes,
            "methods": functions,
            "ast_nodes": len(list(ast.walk(tree))),
            "statements": num_statements,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes strategy metadata to a dictionary."""
        return {
            "name": self.name,
            "params": self.params,
            "code_length": len(self.code),
            "ast_summary": self.get_ast_summary(),
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}' params={self.params}>"


# ==============================================================================
# Baseline Strategy Implementations
# ==============================================================================

class EqualWeightStrategy(BaseStrategy):
    """Baseline equal-weight benchmark portfolio strategy (1/N across risk assets)."""

    def __init__(self, risk_assets: Optional[List[str]] = None, cash_buffer: float = 0.05) -> None:
        super().__init__(
            name="EqualWeightStrategy",
            params={"cash_buffer": cash_buffer},
        )
        self.risk_assets = risk_assets
        self.cash_buffer = cash_buffer

    def generate_signals(self, market_data: MarketData) -> np.ndarray:
        T, K = market_data.close_prices.shape
        signals = np.ones((T, K), dtype=float)
        cash_idx = market_data.get_symbol_index("CASH") if "CASH" in market_data.symbols else K - 1
        signals[:, cash_idx] = 0.0
        return signals

    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        T, K = signals.shape
        weights = np.zeros((T, K), dtype=float)
        risk_asset_count = K - 1  # Excluding CASH
        target_per_asset = (1.0 - self.cash_buffer) / max(risk_asset_count, 1)

        weights[:, :risk_asset_count] = target_per_asset
        weights[:, -1] = self.cash_buffer
        return weights


class MovingAverageCrossover(BaseStrategy):
    """Dual Exponential Moving Average (EMA) trend-following alpha strategy (vectorized)."""

    def __init__(self, fast_window: int = 10, slow_window: int = 30) -> None:
        super().__init__(
            name="MovingAverageCrossover",
            params={"fast_window": fast_window, "slow_window": slow_window},
        )
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, market_data: MarketData) -> np.ndarray:
        close = market_data.close_prices
        T, K = close.shape

        alpha_fast = 2.0 / (self.fast_window + 1.0)
        alpha_slow = 2.0 / (self.slow_window + 1.0)

        # High-speed IIR filter implementation of EMA
        ema_fast = sp_signal.lfilter([alpha_fast], [1.0, -(1.0 - alpha_fast)], close, axis=0)
        ema_slow = sp_signal.lfilter([alpha_slow], [1.0, -(1.0 - alpha_slow)], close, axis=0)

        diff = ema_fast - ema_slow
        signals = np.where(diff > 0.0, 1.0, 0.0)

        # Do not trade CASH asset; reserve it for slack
        if "CASH" in market_data.symbols:
            cash_idx = market_data.get_symbol_index("CASH")
            signals[:, cash_idx] = 0.0

        return signals

    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        T, K = signals.shape
        weights = np.zeros((T, K), dtype=float)

        active = np.maximum(signals[:, :K - 1], 0.0)
        counts = np.sum(active > 0, axis=1, keepdims=True)
        alloc = np.where(counts > 0, 0.90 / np.maximum(counts, 1), 0.0)

        weights[:, :K - 1] = active * alloc
        weights[:, K - 1] = np.where(counts.squeeze(axis=1) > 0, 0.10, 1.0)
        return weights


class MomentumFactor(BaseStrategy):
    """Cross-sectional and time-series momentum ranking alpha strategy (vectorized)."""

    def __init__(self, lookback: int = 20, top_k: int = 2) -> None:
        super().__init__(
            name="MomentumFactor",
            params={"lookback": lookback, "top_k": top_k},
        )
        self.lookback = lookback
        self.top_k = top_k

    def generate_signals(self, market_data: MarketData) -> np.ndarray:
        close = market_data.close_prices
        T, K = close.shape
        signals = np.zeros((T, K), dtype=float)

        if T <= self.lookback:
            return signals

        past_close = close[:-self.lookback]
        curr_close = close[self.lookback:]
        mom = (curr_close - past_close) / (past_close + 1e-8)

        risk_mom = mom[:, :K - 1]
        order = np.argsort(risk_mom, axis=1)

        top_mask = np.zeros_like(risk_mom, dtype=bool)
        rows = np.arange(len(risk_mom))
        for k_idx in range(1, min(self.top_k, K - 1) + 1):
            cols = order[:, -k_idx]
            top_mask[rows, cols] = True

        valid_top = top_mask & (risk_mom > 0.0)
        signals[self.lookback:, :K - 1] = np.where(valid_top, 1.0, 0.0)
        return signals

    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        T, K = signals.shape
        weights = np.zeros((T, K), dtype=float)

        active = np.maximum(signals[:, :K - 1], 0.0)
        counts = np.sum(active > 0, axis=1, keepdims=True)
        alloc = np.where(counts > 0, 0.85 / np.maximum(counts, 1), 0.0)

        weights[:, :K - 1] = active * alloc
        weights[:, K - 1] = np.where(counts.squeeze(axis=1) > 0, 0.15, 1.0)
        return weights


class MeanReversionStrategy(BaseStrategy):
    """Z-Score mean-reverting alpha strategy with volatility threshold gating (vectorized)."""

    def __init__(self, window: int = 20, z_threshold: float = 1.5, vol_filter: float = 0.35) -> None:
        super().__init__(
            name="MeanReversionStrategy",
            params={"window": window, "z_threshold": z_threshold, "vol_filter": vol_filter},
        )
        self.window = window
        self.z_threshold = z_threshold
        self.vol_filter = vol_filter

    def generate_signals(self, market_data: MarketData) -> np.ndarray:
        close = market_data.close_prices
        T, K = close.shape
        signals = np.zeros((T, K), dtype=float)

        if T <= self.window:
            return signals

        # Sliding window view over close prices: shape (T - window, K, window)
        windows = np.lib.stride_tricks.sliding_window_view(close[:-1], window_shape=self.window, axis=0)
        means = np.mean(windows, axis=2)
        stds = np.std(windows, axis=2) + 1e-8
        curr = close[self.window:]
        z = (curr - means) / stds

        # Rolling returns over window
        diffs = (windows[:, :, 1:] - windows[:, :, :-1]) / (windows[:, :, :-1] + 1e-8)
        realized_vols = np.std(diffs, axis=2) * np.sqrt(252.0)

        buy_cond = (realized_vols[:, :K - 1] < self.vol_filter) & (z[:, :K - 1] < -self.z_threshold)
        signals[self.window:, :K - 1] = np.where(buy_cond, 1.0, 0.0)
        return signals

    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        T, K = signals.shape
        weights = np.zeros((T, K), dtype=float)

        active = np.maximum(signals[:, :K - 1], 0.0)
        counts = np.sum(active > 0, axis=1, keepdims=True)
        alloc = np.where(counts > 0, 0.80 / np.maximum(counts, 1), 0.0)

        weights[:, :K - 1] = active * alloc
        weights[:, K - 1] = np.where(counts.squeeze(axis=1) > 0, 0.20, 1.0)
        return weights


class DynamicCodeStrategy(BaseStrategy):
    """Wraps dynamically synthesized Python code into an executable BaseStrategy instance."""

    def __init__(self, code: str, strategy_class_name: Optional[str] = None) -> None:
        super().__init__(name="DynamicCodeStrategy", code=code)
        self.strategy_class_name = strategy_class_name
        self._compiled_strategy: Optional[BaseStrategy] = None
        self._compile()

    def _compile(self) -> None:
        """Compiles and instantiates the dynamic strategy code."""
        cleaned_code = self.code.strip()
        if cleaned_code.startswith("```"):
            lines = cleaned_code.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned_code = "\n".join(lines).strip()

        local_scope: Dict[str, Any] = {
            "BaseStrategy": BaseStrategy,
            "MarketData": MarketData,
            "np": np,
            "numpy": np,
        }
        exec(cleaned_code, local_scope, local_scope)

        target_class = None
        if self.strategy_class_name and self.strategy_class_name in local_scope:
            target_class = local_scope[self.strategy_class_name]
        else:
            for val in local_scope.values():
                if (
                    isinstance(val, type)
                    and issubclass(val, BaseStrategy)
                    and val not in (BaseStrategy, DynamicCodeStrategy)
                ):
                    target_class = val
                    break

        if target_class is None:
            raise ValueError("No valid BaseStrategy subclass found in dynamically compiled code.")

        self._compiled_strategy = target_class()
        self.name = self._compiled_strategy.name
        self.params = self._compiled_strategy.params

    def generate_signals(self, market_data: MarketData) -> np.ndarray:
        if self._compiled_strategy is None:
            raise RuntimeError("Dynamic strategy has not been compiled.")
        return self._compiled_strategy.generate_signals(market_data)

    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        if self._compiled_strategy is None:
            raise RuntimeError("Dynamic strategy has not been compiled.")
        return self._compiled_strategy.compute_weights(signals, current_weights)
