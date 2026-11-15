"""Automated progressive experiment manager scheduling and executing backtesting trials.

Implements:
- ExperimentTrial: Telemetry and audit record for a single backtest experiment.
- ExperimentManager: Progressive experiment orchestrator managing strategy synthesis,
  vectorized execution, and performance logging across multi-era market regimes.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# Handle relative vs absolute imports cleanly
try:
    from common.domain.backtester import BacktestEngine, BacktestResult
    from common.domain.market_data import MarketData
    from common.domain.strategy_base import (
        BaseStrategy,
        DynamicCodeStrategy,
        EqualWeightStrategy,
        MeanReversionStrategy,
        MomentumFactor,
        MovingAverageCrossover,
    )
    from common.llm_provider import BaseModelProvider
except ImportError:
    try:
        from ..common.domain.backtester import BacktestEngine, BacktestResult
        from ..common.domain.market_data import MarketData
        from ..common.domain.strategy_base import (
            BaseStrategy,
            DynamicCodeStrategy,
            EqualWeightStrategy,
            MeanReversionStrategy,
            MomentumFactor,
            MovingAverageCrossover,
        )
        from ..common.llm_provider import BaseModelProvider
    except ImportError:
        BacktestEngine = None  # type: ignore
        BacktestResult = None  # type: ignore
        MarketData = None  # type: ignore
        BaseStrategy = None  # type: ignore
        DynamicCodeStrategy = None  # type: ignore
        BaseModelProvider = None  # type: ignore

try:
    from .hypothesis import Hypothesis
except ImportError:
    from hypothesis import Hypothesis  # type: ignore

logger = logging.getLogger("03_ai_scientist_2.experimenter")


@dataclass
class ExperimentTrial:
    """Audit record and performance telemetry for an experimental backtesting run."""

    trial_id: str
    hypothesis: Hypothesis
    strategy_name: str
    status: str = "completed"
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    cagr: float = 0.0
    turnover: float = 0.0
    total_cost: float = 0.0
    execution_time_ms: float = 0.0
    result: Optional[Any] = None
    timestamp: float = field(default_factory=time.time)
    logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes trial to dictionary."""
        return {
            "trial_id": self.trial_id,
            "hypothesis": self.hypothesis.title,
            "strategy_name": self.strategy_name,
            "status": self.status,
            "sharpe_ratio": round(float(self.sharpe_ratio), 4),
            "max_drawdown": round(float(self.max_drawdown), 4),
            "sortino_ratio": round(float(self.sortino_ratio), 4),
            "calmar_ratio": round(float(self.calmar_ratio), 4),
            "cagr": round(float(self.cagr), 4),
            "turnover": round(float(self.turnover), 4),
            "total_cost": round(float(self.total_cost), 4),
            "execution_time_ms": round(float(self.execution_time_ms), 2),
            "timestamp": self.timestamp,
        }


class ExperimentManager:
    """Automated progressive experiment manager scheduling and executing multi-era backtesting trials.

    Responsibilities:
    - Schedules and runs backtesting trials for formulated hypotheses.
    - Dynamically synthesizes and compiles executable strategies from hypotheses when needed.
    - Captures high-speed execution metrics (Sharpe, Drawdown, Sortino, Turnover, Fees).
    - Maintains an immutable audit history of experimental trials.
    """

    def __init__(
        self,
        engine: BacktestEngine,
        market_data: MarketData,
        llm_provider: Optional[BaseModelProvider] = None,
    ) -> None:
        self.engine = engine
        self.market_data = market_data
        self.llm_provider = llm_provider
        self.history: List[ExperimentTrial] = []

    def run_experiment(
        self,
        hypothesis: Hypothesis,
        strategy: Optional[BaseStrategy] = None,
    ) -> Dict[str, Any]:
        """Executes a backtest trial for a given hypothesis and strategy.

        Args:
            hypothesis: The research hypothesis to evaluate.
            strategy: Optional explicit strategy instance. If None, automatically
                      synthesizes an executable strategy from the hypothesis.

        Returns:
            Dictionary with keys expected by test contracts:
            'status', 'sharpe_ratio', 'max_drawdown', 'hypothesis', etc.
        """
        trial_id = f"trial_{uuid.uuid4().hex[:8]}"
        t_start = time.perf_counter()

        # If no strategy provided, compile or synthesize one from hypothesis
        exec_strategy: BaseStrategy
        if strategy is not None:
            exec_strategy = strategy
        else:
            exec_strategy = self.synthesize_strategy(hypothesis)

        logger.info(f"Running experiment {trial_id} for '{hypothesis.title}' with {exec_strategy.name}")

        try:
            res: BacktestResult = self.engine.run(exec_strategy, self.market_data)
            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            trial = ExperimentTrial(
                trial_id=trial_id,
                hypothesis=hypothesis,
                strategy_name=exec_strategy.name,
                status="completed",
                sharpe_ratio=res.sharpe_ratio,
                max_drawdown=res.max_drawdown,
                sortino_ratio=res.sortino_ratio,
                calmar_ratio=res.calmar_ratio,
                cagr=res.cagr,
                turnover=res.turnover,
                total_cost=res.total_cost,
                execution_time_ms=t_elapsed_ms,
                result=res,
                logs=[f"Successfully executed {exec_strategy.name} across {self.market_data.num_days} days."],
            )
            self.history.append(trial)

            return {
                "status": "completed",
                "sharpe_ratio": res.sharpe_ratio,
                "max_drawdown": res.max_drawdown,
                "sortino_ratio": res.sortino_ratio,
                "calmar_ratio": res.calmar_ratio,
                "cagr": res.cagr,
                "turnover": res.turnover,
                "total_cost": res.total_cost,
                "execution_time_ms": t_elapsed_ms,
                "hypothesis": hypothesis.title,
                "strategy_name": exec_strategy.name,
                "trial_id": trial_id,
                "result": res,
            }

        except Exception as err:
            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.error(f"Experiment {trial_id} failed: {err}")
            fail_trial = ExperimentTrial(
                trial_id=trial_id,
                hypothesis=hypothesis,
                strategy_name=getattr(exec_strategy, "name", "UnknownStrategy"),
                status="failed",
                sharpe_ratio=-999.0,
                max_drawdown=1.0,
                execution_time_ms=t_elapsed_ms,
                logs=[f"Error: {err}"],
            )
            self.history.append(fail_trial)

            return {
                "status": "failed",
                "sharpe_ratio": -999.0,
                "max_drawdown": 1.0,
                "hypothesis": hypothesis.title,
                "error": str(err),
                "trial_id": trial_id,
            }

    def schedule_progressive_trials(
        self,
        hypotheses: List[Hypothesis],
    ) -> List[Dict[str, Any]]:
        """Progressively executes a sequence of trials, logging progress."""
        results: List[Dict[str, Any]] = []
        for i, hyp in enumerate(hypotheses, 1):
            logger.info(f"Progressive Trial [{i}/{len(hypotheses)}]: '{hyp.title}'")
            out = self.run_experiment(hyp)
            results.append(out)
        return results

    def synthesize_strategy(self, hypothesis: Hypothesis) -> BaseStrategy:
        """Instantiates or compiles an executable BaseStrategy corresponding to the hypothesis."""
        # 1. If explicit strategy code is attached, compile dynamically
        if hypothesis.strategy_code:
            try:
                return DynamicCodeStrategy(hypothesis.strategy_code)
            except Exception as err:
                logger.warning(f"Dynamic compilation failed ({err}); falling back to parameterized mapping.")

        # 2. Map based on formula / tags / expected regime
        formula_lower = hypothesis.formula.lower()
        title_lower = hypothesis.title.lower()

        if "semivariance" in formula_lower or "semivariance" in title_lower or "skew" in formula_lower:
            # High sophistication: Downside semivariance / safe haven rebalancer
            vix_thresh = float(hypothesis.parameters.get("vix_threshold", 24.0))
            return ParameterizedAdaptiveStrategy(
                vix_threshold=vix_thresh,
                name=f"Synthesized_{hypothesis.title.replace(' ', '')[:24]}",
            )
        elif "zscore" in formula_lower or "reversion" in title_lower:
            window = int(hypothesis.parameters.get("window", 20))
            z_thresh = float(hypothesis.parameters.get("z_thresh", 1.5))
            return MeanReversionStrategy(window=window, z_threshold=z_thresh)
        elif "ema" in formula_lower or "crossover" in title_lower:
            fast = int(hypothesis.parameters.get("fast", 12))
            slow = int(hypothesis.parameters.get("slow", 26))
            return MovingAverageCrossover(fast_period=fast, slow_period=slow)
        else:
            # Default to momentum factor
            lookback = int(hypothesis.parameters.get("lookback", 20))
            return MomentumFactor(lookback=lookback)


class ParameterizedAdaptiveStrategy(BaseStrategy):
    """Regime-adaptive alpha strategy allocating to Equities in calm markets and safe havens in crises."""

    def __init__(self, vix_threshold: float = 24.0, name: str = "ParameterizedAdaptiveStrategy") -> None:
        super().__init__(name=name, params={"vix_threshold": vix_threshold})
        self.vix_threshold = vix_threshold

    def generate_signals(self, market_data: MarketData) -> np.ndarray:
        close = market_data.close_prices
        T, K = close.shape
        signals = np.zeros((T, K), dtype=float)

        vix_idx = market_data.get_symbol_index("VIX") if hasattr(market_data, "get_symbol_index") else 3
        spy_idx = market_data.get_symbol_index("SPY") if hasattr(market_data, "get_symbol_index") else 0
        tlt_idx = market_data.get_symbol_index("TLT") if hasattr(market_data, "get_symbol_index") else 1
        gld_idx = market_data.get_symbol_index("GLD") if hasattr(market_data, "get_symbol_index") else 2
        cash_idx = market_data.get_symbol_index("CASH") if hasattr(market_data, "get_symbol_index") else 4

        for t in range(20, T):
            vix_val = close[t, vix_idx]
            if vix_val > self.vix_threshold:
                # Crisis shock regime: flight to safe havens
                signals[t, tlt_idx] = 0.50
                signals[t, gld_idx] = 0.30
                signals[t, cash_idx] = 0.20
            else:
                # Normal/Bull regime: equity momentum
                mom_spy = (close[t, spy_idx] - close[t-20, spy_idx]) / (close[t-20, spy_idx] + 1e-8)
                if mom_spy > 0:
                    signals[t, spy_idx] = 0.80
                    signals[t, tlt_idx] = 0.10
                    signals[t, cash_idx] = 0.10
                else:
                    signals[t, tlt_idx] = 0.50
                    signals[t, cash_idx] = 0.50
        return signals

    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        T, K = signals.shape
        weights = np.zeros((T, K), dtype=float)
        for t in range(T):
            pos = np.maximum(signals[t], 0.0)
            s = np.sum(pos)
            if s > 1e-6:
                weights[t] = pos / s
            else:
                weights[t, -1] = 1.0  # Cash
        return weights
