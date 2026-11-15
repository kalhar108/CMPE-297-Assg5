"""Empirical Regime Transition Model (ERTM) / World Model for Dream-RSI.

Models macro regime dynamics, transition probabilities P(s_{t+1} | s_t), and
expected policy response distributions across multi-era market regimes:
- Bull (high drift, low volatility)
- Choppy (zero drift, moderate volatility)
- Crisis (negative drift, extreme volatility spikes)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from common.domain.market_data import MarketData, MarketRegime
except ImportError:
    MarketData = Any  # type: ignore[assignment, misc]
    MarketRegime = Any  # type: ignore[assignment, misc]

try:
    from .tree_logger import DiscoveryTreeLogger
except (ImportError, ValueError):
    from tree_logger import DiscoveryTreeLogger


class RegimeWorldModel:
    """Empirical Regime Transition Model (ERTM) acting as the Dream-RSI World Model (W).

    Learns transition dynamics from market data and exploration logs to enable
    off-policy counterfactual imagination over candidate exploration policies.
    """

    def __init__(self) -> None:
        # Default canonical transition matrix matching benchmark research specifications
        self.transitions: Dict[str, Dict[str, float]] = {
            "bull": {"bull": 0.85, "choppy": 0.12, "crisis": 0.03},
            "choppy": {"bull": 0.20, "choppy": 0.70, "crisis": 0.10},
            "crisis": {"bull": 0.05, "choppy": 0.25, "crisis": 0.70},
        }

        # Baseline annualized empirical statistics per regime
        self.regime_stats: Dict[str, Dict[str, float]] = {
            "bull": {
                "annual_drift": 0.18,
                "annual_volatility": 0.12,
                "expected_sharpe": 1.45,
                "jump_probability": 0.00,
            },
            "choppy": {
                "annual_drift": 0.00,
                "annual_volatility": 0.20,
                "expected_sharpe": 0.35,
                "jump_probability": 0.02,
            },
            "crisis": {
                "annual_drift": -0.38,
                "annual_volatility": 0.45,
                "expected_sharpe": -0.65,
                "jump_probability": 0.08,
            },
        }

        self.observation_count: int = 0
        self.is_fitted: bool = False

    def predict_next_regime(self, current_regime: str) -> str:
        """Predicts the most probable next regime state given the current regime.

        Args:
            current_regime: Current market regime identifier ('bull', 'choppy', 'crisis').

        Returns:
            The regime name with the highest transition probability.
        """
        probs = self.transitions.get(
            current_regime.lower(),
            {"bull": 0.33, "choppy": 0.33, "crisis": 0.34},
        )
        return max(probs.items(), key=lambda x: x[1])[0]

    def transition_probability(self, from_regime: str, to_regime: str) -> float:
        """Returns the transition probability P(to_regime | from_regime)."""
        row = self.transitions.get(from_regime.lower(), {})
        return row.get(to_regime.lower(), 0.0)

    def sample_next_regime(
        self,
        current_regime: str,
        rng: Optional[np.random.RandomState] = None,
    ) -> str:
        """Probabilistically samples the next regime according to transition probabilities."""
        local_rng = rng or np.random.RandomState()
        probs_dict = self.transitions.get(
            current_regime.lower(),
            {"bull": 0.334, "choppy": 0.333, "crisis": 0.333},
        )
        regimes = list(probs_dict.keys())
        probabilities = np.array([probs_dict[r] for r in regimes], dtype=float)
        p_sum = float(np.sum(probabilities))
        if p_sum <= 0.0 or np.isnan(p_sum) or np.isinf(p_sum):
            probabilities = np.full_like(probabilities, 1.0 / len(probabilities))
        else:
            probabilities /= p_sum
        return str(local_rng.choice(regimes, p=probabilities))

    def fit_from_market_data(
        self,
        market_data: Any,
        laplace_smoothing: float = 1.0,
    ) -> Dict[str, Any]:
        """Calibrates empirical transition matrix and dynamics from historical market data.

        Args:
            market_data: MarketData instance containing historical regime tags and returns.
            laplace_smoothing: Pseudocount parameter to prevent zero probabilities.

        Returns:
            Dictionary containing calibration diagnostics.
        """
        regimes: List[str] = [str(r).lower() for r in market_data.regimes]
        T = len(regimes)
        if T < 2:
            return {"status": "insufficient_data", "samples": T}

        known_regimes = ["bull", "choppy", "crisis"]
        counts: Dict[str, Dict[str, float]] = {
            r_from: {r_to: laplace_smoothing for r_to in known_regimes}
            for r_from in known_regimes
        }

        # Count empirical transitions
        for t in range(T - 1):
            rf = regimes[t]
            rt = regimes[t + 1]
            if rf in counts and rt in counts[rf]:
                counts[rf][rt] += 1.0

        # Normalize to probability distributions
        updated_transitions: Dict[str, Dict[str, float]] = {}
        for rf in known_regimes:
            total = sum(counts[rf].values())
            if total <= 0.0:
                updated_transitions[rf] = {
                    rt: round(1.0 / len(known_regimes), 4) for rt in known_regimes
                }
            else:
                updated_transitions[rf] = {
                    rt: round(counts[rf][rt] / total, 4) for rt in known_regimes
                }

        # Compute empirical return and volatility stats per regime
        returns_arr = getattr(market_data, "returns", None)
        if returns_arr is not None and len(returns_arr) == T:
            # Asset 0 is typically SPY (market proxy)
            spy_returns = returns_arr[:, 0]
            for r in known_regimes:
                mask = np.array([reg == r for reg in regimes])
                if np.any(mask):
                    r_rets = spy_returns[mask]
                    mean_daily = float(np.mean(r_rets))
                    std_daily = float(np.std(r_rets)) + 1e-8
                    ann_drift = mean_daily * 252.0
                    ann_vol = std_daily * np.sqrt(252.0)
                    exp_sharpe = ann_drift / ann_vol
                    self.regime_stats[r] = {
                        "annual_drift": round(ann_drift, 4),
                        "annual_volatility": round(ann_vol, 4),
                        "expected_sharpe": round(exp_sharpe, 4),
                        "jump_probability": round(float(np.mean(r_rets < -0.02)), 4),
                    }

        self.transitions = updated_transitions
        self.observation_count += T - 1
        self.is_fitted = True

        return {
            "status": "calibrated",
            "samples": T,
            "transitions": self.transitions,
            "regime_stats": self.regime_stats,
        }

    def fit_from_tree_logger(
        self,
        logger: DiscoveryTreeLogger,
        laplace_smoothing: float = 1.0,
    ) -> Dict[str, Any]:
        """Enriches the world model with state transitions logged during exploration."""
        known_regimes = ["bull", "choppy", "crisis"]
        counts: Dict[str, Dict[str, float]] = {
            r_from: {r_to: laplace_smoothing for r_to in known_regimes}
            for r_from in known_regimes
        }

        transitions_found = 0
        for node in logger.nodes.values():
            s_reg = node.state.get("regime", "").lower()
            ns_reg = node.next_state.get("regime", "").lower()
            if s_reg in counts and ns_reg in counts[s_reg]:
                counts[s_reg][ns_reg] += 1.0
                transitions_found += 1

        if transitions_found > 0:
            for rf in known_regimes:
                total = sum(counts[rf].values())
                if total <= 0.0:
                    self.transitions[rf] = {
                        rt: round(1.0 / len(known_regimes), 4) for rt in known_regimes
                    }
                else:
                    self.transitions[rf] = {
                        rt: round(counts[rf][rt] / total, 4) for rt in known_regimes
                    }
            self.observation_count += transitions_found
            self.is_fitted = True

        return {
            "status": "updated_from_logger",
            "transitions_found": transitions_found,
            "observation_count": self.observation_count,
        }

    def predict_reward_distribution(
        self,
        regime: str,
        action_type: str = "default",
    ) -> Tuple[float, float]:
        """Predicts the expected mean reward and variance for a candidate action in a given regime.

        Enables off-policy dreaming to estimate outcomes of counterfactual actions
        without invoking expensive backtests.
        """
        reg_clean = regime.lower()
        stats = self.regime_stats.get(
            reg_clean,
            {"expected_sharpe": 0.5, "annual_volatility": 0.20},
        )
        base_sharpe = stats["expected_sharpe"]

        # Action-specific adjustments
        act_lower = action_type.lower()
        if "hedge" in act_lower or "cash" in act_lower:
            # Defensive actions outperform in crisis, underperform in bull
            if reg_clean == "crisis":
                expected_reward = base_sharpe + 1.2
            elif reg_clean == "choppy":
                expected_reward = base_sharpe + 0.3
            else:
                expected_reward = base_sharpe - 0.4
        elif "momentum" in act_lower or "trend" in act_lower:
            # Momentum excels in bull, suffers in choppy/crisis
            if reg_clean == "bull":
                expected_reward = base_sharpe + 0.6
            elif reg_clean == "choppy":
                expected_reward = base_sharpe - 0.3
            else:
                expected_reward = base_sharpe - 0.9
        elif "mean_reversion" in act_lower or "zscore" in act_lower:
            # Mean reversion excels in choppy markets
            if reg_clean == "choppy":
                expected_reward = base_sharpe + 0.8
            else:
                expected_reward = base_sharpe - 0.2
        else:
            expected_reward = base_sharpe

        uncertainty = stats.get("annual_volatility", 0.20) * 0.5
        return (float(expected_reward), float(uncertainty))

    def get_regime_dynamics(self, regime: str) -> Dict[str, float]:
        """Returns the macroeconomic dynamics for a given regime."""
        return dict(self.regime_stats.get(regime.lower(), {}))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the world model state."""
        return {
            "transitions": self.transitions,
            "regime_stats": self.regime_stats,
            "observation_count": self.observation_count,
            "is_fitted": self.is_fitted,
        }

    def summary(self) -> str:
        """Formats the transition matrix into an ASCII table."""
        lines = [
            "+-------------------------------------------------------+",
            "|           Empirical Regime Transition Matrix (ERTM)   |",
            "+------------+---------------+---------------+----------+",
            "| From \\ To  | Bull          | Choppy        | Crisis   |",
            "+------------+---------------+---------------+----------+",
        ]
        for r_from in ["bull", "choppy", "crisis"]:
            p_bull = self.transitions.get(r_from, {}).get("bull", 0.0)
            p_chop = self.transitions.get(r_from, {}).get("choppy", 0.0)
            p_cris = self.transitions.get(r_from, {}).get("crisis", 0.0)
            lines.append(
                f"| {r_from.capitalize():<10} | {p_bull:>11.2%}   | {p_chop:>11.2%}   | {p_cris:>6.2%}   |"
            )
        lines.append("+------------+---------------+---------------+----------+")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<RegimeWorldModel fitted={self.is_fitted} "
            f"observations={self.observation_count}>"
        )
