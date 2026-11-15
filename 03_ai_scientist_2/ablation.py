"""Multi-era ablation testing engine evaluating hypotheses across market stress regimes.

Implements:
- MultiEraAblation: Evaluates quantitative trading strategies across 3 distinct regimes:
  1. Era 1 (Days 1–252): Bull Trend (high positive drift, low volatility).
  2. Era 2 (Days 253–504): Choppy / Mean-Reverting (near-zero drift, moderate volatility).
  3. Era 3 (Days 505–756): Crisis / Liquidity Shock (negative drift, extreme volatility spikes).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

# Handle relative vs absolute imports cleanly
try:
    from common.domain.backtester import BacktestEngine, BacktestResult
    from common.domain.market_data import MarketData
    from common.domain.strategy_base import BaseStrategy
except ImportError:
    try:
        from ..common.domain.backtester import BacktestEngine, BacktestResult
        from ..common.domain.market_data import MarketData
        from ..common.domain.strategy_base import BaseStrategy
    except ImportError:
        BacktestEngine = None  # type: ignore
        BacktestResult = None  # type: ignore
        MarketData = None  # type: ignore
        BaseStrategy = None  # type: ignore

try:
    from .hypothesis import Hypothesis
except ImportError:
    from hypothesis import Hypothesis  # type: ignore

logger = logging.getLogger("03_ai_scientist_2.ablation")


class MultiEraAblation:
    """Multi-era ablation testing engine evaluating strategies across Bull, Choppy, and Crisis market stress regimes.

    Ablation across stress regimes verifies whether an empirical anomaly:
    - Truly captures structural alpha rather than in-sample curve fitting.
    - Demonstrates downside survival during liquidity crises (Era 3).
    - Preserves capital and minimizes turnover friction during whipsaws (Era 2).
    """

    def __init__(self, engine: BacktestEngine, market_data: MarketData) -> None:
        self.engine = engine
        self.market_data = market_data

    def run_ablation(self, strategy: BaseStrategy) -> Dict[str, BacktestResult]:
        """Runs the strategy across all 3 market eras and returns BacktestResult per era.

        Args:
            strategy: BaseStrategy instance to evaluate.

        Returns:
            Dict with keys 'bull', 'choppy', 'crisis' mapping to BacktestResult instances.
        """
        eras = ["bull", "choppy", "crisis"]
        results: Dict[str, BacktestResult] = {}

        for era in eras:
            try:
                era_data = self.market_data.get_era(era)
                res = self.engine.run(strategy, era_data)
                results[era] = res
            except Exception as err:
                logger.error(f"Ablation run for era '{era}' failed: {err}")
                # Fallback safe result if an era fails
                results[era] = BacktestResult(
                    sharpe_ratio=-99.0,
                    max_drawdown=1.0,
                    sortino_ratio=-99.0,
                    calmar_ratio=-99.0,
                    cagr=-1.0,
                    turnover=0.0,
                    total_cost=0.0,
                    equity_curve=[100000.0] * 252,
                    daily_returns=[0.0] * 251,
                    weights_history=[[0.0] * len(self.market_data.symbols)] * 252,
                    execution_time_ms=0.0,
                    success=False,
                )

        return results

    def run_hypothesis_ablation(
        self,
        hypothesis: Hypothesis,
        strategy: BaseStrategy,
    ) -> Dict[str, Any]:
        """Runs a complete hypothesis ablation study comparing expected vs observed behavior."""
        ablation_dict = self.run_ablation(strategy)

        bull_res = ablation_dict["bull"]
        choppy_res = ablation_dict["choppy"]
        crisis_res = ablation_dict["crisis"]

        # Compute multi-era resilience diagnostics
        mean_sharpe = float(np.mean([bull_res.sharpe_ratio, choppy_res.sharpe_ratio, crisis_res.sharpe_ratio]))
        worst_drawdown = float(max([bull_res.max_drawdown, choppy_res.max_drawdown, crisis_res.max_drawdown]))

        # Crisis resilience index: high if crisis drawdown is bounded (< 20%) and sharpe is non-catastrophic
        crisis_resilience = float(np.clip(1.0 - crisis_res.max_drawdown, 0.0, 1.0) * (1.0 if crisis_res.sharpe_ratio > -0.5 else 0.5))

        # Check alignment with hypothesis expectation
        exp = hypothesis.expected_regime.lower()
        regime_aligned = True
        if exp == "bull" and bull_res.sharpe_ratio < 0.5:
            regime_aligned = False
        elif exp == "crisis" and crisis_res.max_drawdown > 0.25:
            regime_aligned = False

        summary_table = self.format_ablation_table(ablation_dict)

        return {
            "hypothesis": hypothesis.title,
            "expected_regime": hypothesis.expected_regime,
            "regime_aligned": regime_aligned,
            "mean_sharpe": mean_sharpe,
            "worst_drawdown": worst_drawdown,
            "crisis_resilience": crisis_resilience,
            "ablation_results": ablation_dict,
            "summary_table": summary_table,
        }

    def format_ablation_table(self, ablation_dict: Dict[str, BacktestResult]) -> str:
        """Formats ablation results into an ASCII telemetry table."""
        header = f"+----------+------------+---------------+------------+--------------+------------+\n" \
                 f"| Regime   | Sharpe     | Max Drawdown  | CAGR       | Sortino      | Turnover   |\n" \
                 f"+----------+------------+---------------+------------+--------------+------------+"
        rows = []
        for era in ["bull", "choppy", "crisis"]:
            r = ablation_dict.get(era)
            if r is not None:
                rows.append(
                    f"| {era.upper():<8} | {r.sharpe_ratio:>10.2f} | {r.max_drawdown*100:>12.2f}% | "
                    f"{r.cagr*100:>9.2f}% | {r.sortino_ratio:>12.2f} | {r.turnover:>10.2f} |"
                )
            else:
                rows.append(f"| {era.upper():<8} |        N/A |           N/A |        N/A |          N/A |        N/A |")
        footer = f"+----------+------------+---------------+------------+--------------+------------+"
        return "\n".join([header] + rows + [footer])
