"""Pre-seeded deterministic responses and fixtures for RSI Suite.

Provides realistic, authentic mock outputs for strategy drafting, code debugging/repair,
code refinement, hypothesis formulation, and peer review in the Quantitative Alpha domain.
100% offline, zero network dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ==============================================================================
# 1. Strategy Drafting Mock Data
# ==============================================================================

MOCK_DRAFT_STRATEGIES: List[Dict[str, Any]] = [
    {
        "strategy_name": "AdaptiveTrendMomentum",
        "rationale": "Combines fast/slow exponential moving average crossovers with rolling price momentum to capture trending regimes while avoiding lag.",
        "formula": "Signal = sign(EMA(Close, 12) - EMA(Close, 26)) * (Close / Close[t-20] - 1.0)",
        "code": '''"""Adaptive Trend Momentum Strategy"""
import numpy as np
from common.domain.strategy_base import BaseStrategy

class AdaptiveTrendMomentum(BaseStrategy):
    def __init__(self, fast_window: int = 12, slow_window: int = 26, mom_window: int = 20):
        super().__init__(
            name="AdaptiveTrendMomentum",
            params={"fast_window": fast_window, "slow_window": slow_window, "mom_window": mom_window}
        )
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.mom_window = mom_window

    def generate_signals(self, market_data) -> np.ndarray:
        # close shape: (T, K)
        close = market_data.close_prices
        T, K = close.shape
        signals = np.zeros((T, K), dtype=float)

        # Simple exponential smoothing approximation
        alpha_fast = 2.0 / (self.fast_window + 1.0)
        alpha_slow = 2.0 / (self.slow_window + 1.0)

        ema_fast = np.copy(close)
        ema_slow = np.copy(close)
        for t in range(1, T):
            ema_fast[t] = alpha_fast * close[t] + (1 - alpha_fast) * ema_fast[t-1]
            ema_slow[t] = alpha_slow * close[t] + (1 - alpha_slow) * ema_slow[t-1]

        # Momentum over lookback
        for t in range(self.mom_window, T):
            mom = (close[t] - close[t - self.mom_window]) / (close[t - self.mom_window] + 1e-8)
            trend = np.sign(ema_fast[t] - ema_slow[t])
            signals[t] = trend * np.clip(mom * 5.0, -1.0, 1.0)

        return signals

    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        # Long-only allocation proportional to positive signals, remainder in cash (last asset)
        T, K = signals.shape
        weights = np.zeros((T, K), dtype=float)
        for t in range(T):
            sig = np.maximum(signals[t, :K-1], 0.0)
            s_sum = np.sum(sig)
            if s_sum > 1e-6:
                weights[t, :K-1] = 0.85 * (sig / s_sum)
                weights[t, K-1] = 0.15  # Cash buffer
            else:
                weights[t, K-1] = 1.0   # 100% Cash when no bullish signals
        return weights
''',
        "target_regime": "bull_and_breakout",
    },
    {
        "strategy_name": "VolatilityFilteredReversion",
        "rationale": "Exploits mean-reversion during choppy range-bound markets, gating risk when rolling realized volatility breaches threshold.",
        "formula": "Signal = -ZScore(Close, 20) * (1.0 if Vol < VolThreshold else 0.0)",
        "code": '''"""Volatility Filtered Reversion Strategy"""
import numpy as np
from common.domain.strategy_base import BaseStrategy

class VolatilityFilteredReversion(BaseStrategy):
    def __init__(self, window: int = 20, z_threshold: float = 1.5, vol_cap: float = 0.25):
        super().__init__(
            name="VolatilityFilteredReversion",
            params={"window": window, "z_threshold": z_threshold, "vol_cap": vol_cap}
        )
        self.window = window
        self.z_threshold = z_threshold
        self.vol_cap = vol_cap

    def generate_signals(self, market_data) -> np.ndarray:
        close = market_data.close_prices
        T, K = close.shape
        signals = np.zeros((T, K), dtype=float)

        for t in range(self.window, T):
            window_slice = close[t - self.window:t]
            mean = np.mean(window_slice, axis=0)
            std = np.std(window_slice, axis=0) + 1e-8
            z = (close[t] - mean) / std

            # Realized annual vol estimate
            rets = np.diff(window_slice, axis=0) / (window_slice[:-1] + 1e-8)
            vol = np.std(rets, axis=0) * np.sqrt(252)

            # Reversion signal: buy depressed, sell extended, shut off in panic vol
            for k in range(K - 1):
                if vol[k] < self.vol_cap:
                    if z[k] < -self.z_threshold:
                        signals[t, k] = 1.0
                    elif z[k] > self.z_threshold:
                        signals[t, k] = -0.5
        return signals

    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        T, K = signals.shape
        weights = np.zeros((T, K), dtype=float)
        for t in range(T):
            pos = np.maximum(signals[t, :K-1], 0.0)
            s_sum = np.sum(pos)
            if s_sum > 1e-6:
                weights[t, :K-1] = 0.80 * (pos / s_sum)
                weights[t, K-1] = 0.20
            else:
                weights[t, K-1] = 1.0
        return weights
''',
        "target_regime": "choppy_range_bound",
    },
]

# ==============================================================================
# 2. Code Debugging / Repair Mock Data
# ==============================================================================

MOCK_DEBUG_REPAIRS: Dict[str, Dict[str, str]] = {
    "ZeroDivisionError": {
        "bug_pattern": "std = np.std(window_slice, axis=0)",
        "diagnostics": "ZeroDivisionError detected: Rolling standard deviation becomes zero on flat price windows, causing division by zero when computing z-scores.",
        "patch": "std = np.std(window_slice, axis=0) + 1e-8",
        "repaired_code_snippet": """            mean = np.mean(window_slice, axis=0)
            std = np.std(window_slice, axis=0) + 1e-8  # Added epsilon to prevent division by zero
            z_score = (close[t] - mean) / std""",
    },
    "LookaheadBias": {
        "bug_pattern": "realized_weights = weights[t]",
        "diagnostics": "LookaheadBias detected: Realizing return at time t using allocation computed at time t using price at time t violates temporal causality.",
        "patch": "realized_weights = weights[t-1]",
        "repaired_code_snippet": """        # Ensure weights lag by 1 time step to eliminate lookahead bias
        realized_weights = weights[:-1]
        step_returns = (realized_weights * returns[1:]).sum(axis=1)""",
    },
    "IndexError": {
        "bug_pattern": "close[t + self.window]",
        "diagnostics": "IndexError / Out of bounds detected: Accessing index (t + window) exceeds array bounds T.",
        "patch": "close[t - self.window:t]",
        "repaired_code_snippet": """        for t in range(self.window, T):
            window_slice = close[t - self.window:t]  # Correct backward lookback window""",
    },
}

# ==============================================================================
# 3. Code Refinement Mock Data
# ==============================================================================

MOCK_REFINEMENTS: List[Dict[str, Any]] = [
    {
        "refinement_type": "VolatilityTargeting",
        "description": "Scales position weights inversely by realized volatility to equalize risk contribution across market regimes.",
        "code_patch": """        # Volatility targeting refinement
        target_vol = 0.12  # 12% annualized target
        vol_scaler = np.clip(target_vol / (realized_vol + 1e-6), 0.2, 1.5)
        refined_weights = base_weights * vol_scaler
        # Re-normalize to ensure leverage <= 1.0
        refined_weights = refined_weights / max(np.sum(refined_weights), 1.0)""",
    },
    {
        "refinement_type": "DownsideSemivarianceDamping",
        "description": "Applies asymmetric penalty only to downside returns, avoiding penalizing upside volatility.",
        "code_patch": """        # Downside semivariance dampener
        downside_rets = np.minimum(rets, 0.0)
        semi_var = np.sqrt(np.mean(downside_rets**2, axis=0)) * np.sqrt(252) + 1e-8
        risk_weight = 1.0 / (semi_var / np.mean(semi_var))
        refined_weights = base_weights * np.clip(risk_weight, 0.3, 2.0)""",
    },
]

# ==============================================================================
# 4. Hypothesis Formulation Mock Data
# ==============================================================================

MOCK_HYPOTHESES: List[Dict[str, Any]] = [
    {
        "title": "Downside Semivariance Regulated Momentum in Asymmetric Volatility Regimes",
        "claim": "Traditional time-series momentum suffers catastrophic crashes during liquidity shocks because standard volatility scaling penalizes upside drift equally with downside panic. Conditioning momentum allocations on downside semivariance maintains full participation in bull trends while swiftly de-risking before crash drawdowns.",
        "mathematical_formulation": "w_i = (Mom_{i, 20} / \\sigma_{down, i}) \\cdot \\mathbb{I}(\\Delta VIX_{5d} < \\theta_{panic})",
        "economic_rationale": "Liquidity providers withdraw liquidity asymmetrically when left-tail downside shocks emerge, creating autocorrelation in negative returns. Downside semivariance isolates left-tail hazards without truncating upside momentum runs.",
        "expected_regime_behavior": {
            "bull_era": "Full 80-90% equity participation, outperforming equal weight by +4-6% CAGR.",
            "choppy_era": "Moderate turnover with tight trailing stops; preserves capital in range-bound whipsaws.",
            "crisis_era": "Rapid de-risking into sovereign bonds (TLT) and cash within 2-3 trading days of vol spike, bounding Max Drawdown to < 12%.",
        },
    },
    {
        "title": "Cross-Asset Volatility Transmission and Flight-to-Safety Alpha",
        "claim": "Volatility spikes in equity proxies (VIX) precede maximum drawdown in equities by 2-5 days, creating a systematic arbitrage window where capital can be rotated into negative-correlation safe havens (TLT and GLD) prior to market recovery.",
        "mathematical_formulation": "w_{safe} = \\text{sigmoid}\\Big( \\frac{VIX_t - \\text{EMA}(VIX, 20)}{\\text{ATR}(VIX, 10)} \\Big)",
        "economic_rationale": "Institutional mandates enforce value-at-risk (VaR) de-leveraging when volatility crosses risk budgets. This forced liquidation creates temporary price depressions in risk assets and predictable surges into liquid sovereign safe havens.",
        "expected_regime_behavior": {
            "bull_era": "Safe-haven weight remains near 10-15%, equity allocation dominates.",
            "choppy_era": "Dynamic hedging mitigates sector rotation drawdowns.",
            "crisis_era": "Flips to 70%+ bonds/gold/cash allocation, generating positive returns during equity collapse.",
        },
    },
]

# ==============================================================================
# 5. Peer Review Scorecard Mock Data
# ==============================================================================

MOCK_PEER_REVIEWS: List[Dict[str, Any]] = [
    {
        "reviewer_id": "Reviewer_1_Methodology",
        "soundness_score": 9,
        "novelty_score": 8,
        "statistical_significance_score": 9,
        "risk_discipline_score": 9,
        "overall_recommendation": "Strong Accept",
        "critique": (
            "The mathematical formulation of downside semivariance gating is mathematically rigorous. "
            "The 756-day multi-era evaluation across Bull, Choppy, and Crisis partitions definitively proves "
            "that the alpha signal is not an artifact of in-sample curve fitting. Transaction fees of 10 bps "
            "are properly accounted for, and zero-lookahead temporal causality is strictly preserved."
        ),
        "strengths": [
            "Rigorous separation of upside vs downside volatility.",
            "Sub-20ms backtest execution speed enables extensive Pareto frontier exploration.",
            "Excellent resilience in Era 3 Crisis regime with Maximum Drawdown bounded below 12%.",
        ],
        "weaknesses": [
            "Turnover in Era 2 Choppy regime is slightly elevated (approx 180% annualized).",
            "Consider adding an explicit turnover damping term to further optimize net returns.",
        ],
    },
    {
        "reviewer_id": "Reviewer_2_Theory",
        "soundness_score": 8,
        "novelty_score": 9,
        "statistical_significance_score": 8,
        "risk_discipline_score": 9,
        "overall_recommendation": "Accept",
        "critique": (
            "The economic theory linking liquidity withdrawal to asymmetric downside semivariance is compelling. "
            "The generational progression curves illustrate genuine recursive improvement from Generation 0 (Sharpe 0.45) "
            "to Generation 2 (Sharpe 1.75+). Highly suitable for publication and educational demonstration."
        ),
        "strengths": [
            "Clear causal link to market microstructure and institutional VaR constraints.",
            "Consistent generational improvement under fixed evaluator $E$.",
        ],
        "weaknesses": [
            "Sensitivity analysis across lookback window hyperparameters (W=10 to W=30) should be documented.",
        ],
    },
]

MOCK_META_REVIEW: Dict[str, Any] = {
    "composite_score": 8.75,
    "decision": "Accept with Minor Revision",
    "summary": "Both independent reviewers confirm high theoretical novelty and exemplary empirical rigor. The multi-era backtesting across Bull, Choppy, and Stress regimes proves robust out-of-sample generalization with verified Sharpe > 1.60 and controlled drawdown under 10 bps friction.",
    "action_items": [
        "Incorporate turnover damping parameter to minimize friction in choppy regimes.",
        "Verify sensitivity bounds on lookback windows.",
    ],
}


def get_mock_strategy(index: int = 0) -> Dict[str, Any]:
    """Retrieves a mock strategy candidate."""
    return MOCK_DRAFT_STRATEGIES[index % len(MOCK_DRAFT_STRATEGIES)]


def get_mock_repair(error_type: str) -> Dict[str, str]:
    """Retrieves mock repair details by error type."""
    return MOCK_DEBUG_REPAIRS.get(error_type, MOCK_DEBUG_REPAIRS["ZeroDivisionError"])


def get_mock_hypothesis(index: int = 0) -> Dict[str, Any]:
    """Retrieves a mock research hypothesis."""
    return MOCK_HYPOTHESES[index % len(MOCK_HYPOTHESES)]


def get_mock_peer_review() -> Dict[str, Any]:
    """Retrieves a full mock peer review committee package."""
    return {
        "reviews": MOCK_PEER_REVIEWS,
        "meta_review": MOCK_META_REVIEW,
    }
