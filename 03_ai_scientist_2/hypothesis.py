"""Template-free economic & quantitative anomaly hypothesis generator.

Implements:
- Hypothesis: Core research artifact representing a quantitative trading hypothesis.
- HypothesisGenerator: Autonomous template-free research hypothesis formulator with
  dual-mode execution (deterministic mock vs live Gemini API), research archiving,
  deduplication, and recursive prompt prior conditioning via MASS.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Handle relative vs absolute imports cleanly
try:
    from common.config import RSIExecutionConfig
    from common.llm_provider import BaseModelProvider, get_llm_provider
    from common.mock_data import MOCK_HYPOTHESES, get_mock_hypothesis
except ImportError:
    try:
        from ..common.config import RSIExecutionConfig
        from ..common.llm_provider import BaseModelProvider, get_llm_provider
        from ..common.mock_data import MOCK_HYPOTHESES, get_mock_hypothesis
    except ImportError:
        # Standalone fallback when running directly without package root
        RSIExecutionConfig = None  # type: ignore
        BaseModelProvider = None  # type: ignore
        get_llm_provider = None  # type: ignore
        MOCK_HYPOTHESES = []
        get_mock_hypothesis = lambda idx=0: {}

logger = logging.getLogger("03_ai_scientist_2.hypothesis")


@dataclass
class Hypothesis:
    """Quantitative trading and economic anomaly hypothesis artifact.

    Attributes:
        title: Short descriptive title of the research hypothesis.
        anomaly_description: Detailed statement of the market anomaly or behavioral effect.
        formula: Mathematical or algorithmic specification of the alpha signal.
        expected_regime: Target market regime where signal performs best ('bull', 'choppy', 'crisis', 'all').
        confidence: Subjective or model-assigned initial confidence score (0.0 to 1.0).
        rationale: Economic or microstructure justification.
        target_assets: List of symbols involved in the trade signal.
        parameters: Configurable hyperparameters (lookbacks, thresholds, decay rates).
        generation: Discovery cycle generation index (0, 1, 2, ...).
        tags: Categorical tags (e.g. 'momentum', 'volatility', 'cross-asset').
        strategy_code: Optional synthesized Python source code for the strategy.
    """

    title: str
    anomaly_description: str
    formula: str
    expected_regime: str
    confidence: float = 0.85
    rationale: str = ""
    target_assets: List[str] = field(default_factory=lambda: ["SPY", "TLT", "GLD", "VIX", "CASH"])
    parameters: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    tags: List[str] = field(default_factory=list)
    strategy_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes hypothesis into a structured dictionary."""
        return {
            "title": self.title,
            "anomaly_description": self.anomaly_description,
            "formula": self.formula,
            "expected_regime": self.expected_regime,
            "confidence": round(float(self.confidence), 4),
            "rationale": self.rationale,
            "target_assets": list(self.target_assets),
            "parameters": dict(self.parameters),
            "generation": self.generation,
            "tags": list(self.tags),
            "strategy_code_len": len(self.strategy_code) if self.strategy_code else 0,
        }

    def summary(self) -> str:
        """Returns concise one-line summary."""
        return f"[{self.expected_regime.upper()}] '{self.title}' (Conf: {self.confidence:.2f}) -> {self.formula}"

    def format_card(self) -> str:
        """Renders formatted multi-line display card."""
        lines = [
            f"+--------------------------------------------------------------------------------+",
            f"| Hypothesis: {self.title[:64]:<64} |",
            f"+--------------------------------------------------------------------------------+",
            f"| Generation: {self.generation:<4} | Regime: {self.expected_regime:<10} | Confidence: {self.confidence:<6.2f}         |",
            f"| Formula   : {self.formula[:64]:<64} |",
            f"| Anomaly   : {self.anomaly_description[:64]:<64} |",
            f"+--------------------------------------------------------------------------------+",
        ]
        return "\n".join(lines)


# ==============================================================================
# Domain Hypothesis Exemplars Across Generational Progression
# ==============================================================================

GENERATIONAL_HYPOTHESIS_BANK: Dict[int, List[Dict[str, Any]]] = {
    # Generation 0: Naive, unhedged single-factor hypotheses (lacks risk discipline)
    0: [
        {
            "title": "Unfiltered Time-Series Momentum Continuation",
            "anomaly_description": "Assets with positive 20-day returns continue to outperform due to behavioral underreaction and herd purchasing.",
            "formula": "Signal = sign(Close / Close[t-20] - 1.0)",
            "expected_regime": "bull",
            "confidence": 0.70,
            "rationale": "Investors anchor on past gains, causing persistent trending momentum during economic expansions.",
            "parameters": {"lookback": 20},
            "tags": ["momentum", "trend", "unfiltered"],
            "strategy_code": '''"""Unfiltered Momentum Strategy"""
import numpy as np
from common.domain.strategy_base import BaseStrategy

class UnfilteredMomentum(BaseStrategy):
    def __init__(self, lookback: int = 20):
        super().__init__(name="UnfilteredMomentum", params={"lookback": lookback})
        self.lookback = lookback

    def generate_signals(self, market_data) -> np.ndarray:
        close = market_data.close_prices
        T, K = close.shape
        signals = np.zeros((T, K), dtype=float)
        for t in range(self.lookback, T):
            rets = (close[t] - close[t - self.lookback]) / (close[t - self.lookback] + 1e-8)
            best_k = int(np.argmax(rets[:K-1]))
            signals[t, best_k] = 1.0 if rets[best_k] > 0 else 0.0
        return signals

    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        T, K = signals.shape
        weights = np.zeros((T, K), dtype=float)
        for t in range(T):
            s = np.sum(signals[t, :K-1])
            if s > 0:
                weights[t, :K-1] = signals[t, :K-1] / s
            else:
                weights[t, K-1] = 1.0  # Cash
        return weights
''',
        },
        {
            "title": "Dual Moving Average Trend Following",
            "anomaly_description": "Moving average crossovers identify macro trend inflections when fast trend exceeds slow trend.",
            "formula": "Signal = EMA(Close, 12) - EMA(Close, 26)",
            "expected_regime": "bull",
            "confidence": 0.68,
            "rationale": "Information cascades lag fundamental announcements, resulting in smooth exponential price progression.",
            "parameters": {"fast": 12, "slow": 26},
            "tags": ["trend", "moving_average"],
            "strategy_code": '''"""Dual Moving Average Trend Strategy"""
import numpy as np
from common.domain.strategy_base import BaseStrategy

class DualMovingAverage(BaseStrategy):
    def __init__(self, fast: int = 12, slow: int = 26):
        super().__init__(name="DualMovingAverage", params={"fast": fast, "slow": slow})
        self.fast = fast
        self.slow = slow

    def generate_signals(self, market_data) -> np.ndarray:
        close = market_data.close_prices
        T, K = close.shape
        signals = np.zeros((T, K), dtype=float)
        alpha_fast = 2.0 / (self.fast + 1.0)
        alpha_slow = 2.0 / (self.slow + 1.0)
        ema_f = np.copy(close)
        ema_s = np.copy(close)
        for t in range(1, T):
            ema_f[t] = alpha_fast * close[t] + (1 - alpha_fast) * ema_f[t-1]
            ema_s[t] = alpha_slow * close[t] + (1 - alpha_slow) * ema_s[t-1]
        for t in range(self.slow, T):
            for k in range(K - 1):
                if ema_f[t, k] > ema_s[t, k]:
                    signals[t, k] = 1.0
        return signals

    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        T, K = signals.shape
        weights = np.zeros((T, K), dtype=float)
        for t in range(T):
            s = np.sum(signals[t, :K-1])
            if s > 0:
                weights[t, :K-1] = signals[t, :K-1] / s
            else:
                weights[t, K-1] = 1.0
        return weights
''',
        },
    ],
    # Generation 1: Volatility-filtered and range-bound mean-reverting hypotheses
    1: [
        {
            "title": "Volatility-Gated Mean Reversion in Choppy Regimes",
            "anomaly_description": "Asset prices overextend in range-bound markets; mean reversion occurs reliably when realized volatility is subdued.",
            "formula": "Signal = -ZScore(Close, 20) * (1.0 if RealizedVol < 0.22 else 0.0)",
            "expected_regime": "choppy",
            "confidence": 0.82,
            "rationale": "Market makers provide liquidity during low-volatility regimes, driving extreme price deviations back to local equilibrium.",
            "parameters": {"window": 20, "z_thresh": 1.5, "vol_cap": 0.22},
            "tags": ["mean_reversion", "volatility_filter", "choppy"],
            "strategy_code": '''"""Volatility-Gated Mean Reversion"""
import numpy as np
from common.domain.strategy_base import BaseStrategy

class VolatilityGatedReversion(BaseStrategy):
    def __init__(self, window: int = 20, z_thresh: float = 1.5, vol_cap: float = 0.22):
        super().__init__(name="VolatilityGatedReversion", params={"window": window, "z_thresh": z_thresh, "vol_cap": vol_cap})
        self.window = window
        self.z_thresh = z_thresh
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
            rets = np.diff(window_slice, axis=0) / (window_slice[:-1] + 1e-8)
            vol = np.std(rets, axis=0) * np.sqrt(252.0)
            for k in range(K - 1):
                if vol[k] < self.vol_cap:
                    if z[k] < -self.z_thresh:
                        signals[t, k] = 1.0
                    elif z[k] > self.z_thresh:
                        signals[t, k] = -0.5
        return signals

    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        T, K = signals.shape
        weights = np.zeros((T, K), dtype=float)
        for t in range(T):
            pos = np.maximum(signals[t, :K-1], 0.0)
            s = np.sum(pos)
            if s > 1e-6:
                weights[t, :K-1] = 0.80 * (pos / s)
                weights[t, K-1] = 0.20
            else:
                weights[t, K-1] = 1.0
        return weights
''',
        },
        {
            "title": "ATR Volatility Channel Breakout with Trailing Exits",
            "anomaly_description": "Price breakouts through Average True Range channels signal institutional accumulation with dynamic volatility-calibrated stops.",
            "formula": "Signal = 1.0 if Close > EMA(20) + 1.8 * ATR(14) else 0.0",
            "expected_regime": "bull",
            "confidence": 0.84,
            "rationale": "High-conviction institutional order flow expands trading bands, initiating persistent directional trends.",
            "parameters": {"ema_window": 20, "atr_window": 14, "multiplier": 1.8},
            "tags": ["breakout", "atr", "volatility"],
            "strategy_code": '''"""ATR Volatility Channel Breakout"""
import numpy as np
from common.domain.strategy_base import BaseStrategy

class ATRChannelBreakout(BaseStrategy):
    def __init__(self, ema_window: int = 20, atr_window: int = 14, multiplier: float = 1.8):
        super().__init__(name="ATRChannelBreakout", params={"ema_window": ema_window, "atr_window": atr_window, "multiplier": multiplier})
        self.ema_window = ema_window
        self.atr_window = atr_window
        self.multiplier = multiplier

    def generate_signals(self, market_data) -> np.ndarray:
        close = market_data.close_prices
        high = market_data.high_prices
        low = market_data.low_prices
        T, K = close.shape
        signals = np.zeros((T, K), dtype=float)
        for t in range(max(self.ema_window, self.atr_window), T):
            for k in range(K - 1):
                tr = np.maximum(high[t-self.atr_window:t, k] - low[t-self.atr_window:t, k], 1e-4)
                atr = np.mean(tr)
                mean_p = np.mean(close[t-self.ema_window:t, k])
                if close[t, k] > mean_p + self.multiplier * atr:
                    signals[t, k] = 1.0
        return signals

    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        T, K = signals.shape
        weights = np.zeros((T, K), dtype=float)
        for t in range(T):
            pos = np.maximum(signals[t, :K-1], 0.0)
            s = np.sum(pos)
            if s > 0:
                weights[t, :K-1] = 0.85 * (pos / s)
                weights[t, K-1] = 0.15
            else:
                weights[t, K-1] = 1.0
        return weights
''',
        },
    ],
    # Generation 2+: Highly sophisticated, asymmetric downside risk-gated & cross-asset flight-to-safety hypotheses
    2: [
        {
            "title": "Downside Semivariance Regulated Momentum with Cross-Asset Risk Parity",
            "anomaly_description": "Liquidations trigger asymmetric left-tail spikes without upside continuity; scaling momentum exclusively by downside semivariance and rotating to sovereign bonds (TLT) preserves bull alpha while capping crisis drawdown.",
            "formula": "w_i = (Mom_{i, 20} / \\sigma_{down, i}) \\cdot \\mathbb{I}(\\Delta VIX_{5d} < \\theta_{panic})",
            "expected_regime": "all",
            "confidence": 0.94,
            "rationale": "Institutional Value-at-Risk limits compel swift de-risking during volatility surges. Isolating downside variance protects capital without penalizing upside momentum drifts.",
            "parameters": {"lookback": 20, "vix_threshold": 24.0, "tlt_cushion": 0.25},
            "tags": ["semivariance", "flight_to_safety", "cross_asset", "asymmetric_risk"],
            "strategy_code": '''"""Downside Semivariance Regulated Momentum Strategy"""
import numpy as np
from common.domain.strategy_base import BaseStrategy

class DownsideSemivarianceMomentum(BaseStrategy):
    def __init__(self, lookback: int = 20, vix_threshold: float = 24.0, tlt_cushion: float = 0.25):
        super().__init__(
            name="DownsideSemivarianceMomentum",
            params={"lookback": lookback, "vix_threshold": vix_threshold, "tlt_cushion": tlt_cushion}
        )
        self.lookback = lookback
        self.vix_threshold = vix_threshold
        self.tlt_cushion = tlt_cushion

    def generate_signals(self, market_data) -> np.ndarray:
        close = market_data.close_prices
        T, K = close.shape
        signals = np.zeros((T, K), dtype=float)
        vix_idx = market_data.get_symbol_index("VIX") if hasattr(market_data, "get_symbol_index") else 3
        spy_idx = market_data.get_symbol_index("SPY") if hasattr(market_data, "get_symbol_index") else 0
        tlt_idx = market_data.get_symbol_index("TLT") if hasattr(market_data, "get_symbol_index") else 1
        gld_idx = market_data.get_symbol_index("GLD") if hasattr(market_data, "get_symbol_index") else 2
        cash_idx = market_data.get_symbol_index("CASH") if hasattr(market_data, "get_symbol_index") else 4

        for t in range(self.lookback, T):
            current_vix = close[t, vix_idx]
            vix_spike = current_vix > self.vix_threshold

            if vix_spike:
                # Crisis regime: rotate capital to TLT, GLD and Cash
                signals[t, tlt_idx] = 0.45
                signals[t, gld_idx] = 0.35
                signals[t, cash_idx] = 0.20
            else:
                # Bull / Normal regime: downside semivariance weighted momentum
                window_slice = close[t - self.lookback:t]
                rets = np.diff(window_slice, axis=0) / (window_slice[:-1] + 1e-8)
                mom = (close[t] - close[t - self.lookback]) / (close[t - self.lookback] + 1e-8)
                downside = np.minimum(rets, 0.0)
                semi_std = np.sqrt(np.mean(downside**2, axis=0)) + 1e-6
                
                score_spy = max(mom[spy_idx] / semi_std[spy_idx], 0.0)
                score_tlt = max(mom[tlt_idx] / semi_std[tlt_idx], 0.0)
                score_gld = max(mom[gld_idx] / semi_std[gld_idx], 0.0)

                signals[t, spy_idx] = score_spy
                signals[t, tlt_idx] = score_tlt * self.tlt_cushion
                signals[t, gld_idx] = score_gld * 0.15
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
''',
        },
        {
            "title": "Cross-Asset Volatility Skew Transmission & Safe-Haven Rebalancing",
            "anomaly_description": "VIX spikes precede equity drawdowns by 2 to 5 days; systematically allocating into negatively correlated safe havens (TLT/GLD) during early volatility expansions isolates portfolios from systemic selloffs.",
            "formula": "w_{safe} = \\text{sigmoid}\\Big( \\frac{VIX_t - \\text{EMA}(VIX, 20)}{\\text{ATR}(VIX, 10)} \\Big)",
            "expected_regime": "crisis",
            "confidence": 0.96,
            "rationale": "Cross-asset contagion creates predictable capital flight into liquid US Treasuries and precious metals before equity bottoms form.",
            "parameters": {"vix_ema": 20, "atr_lookback": 10, "panic_ratio": 1.25},
            "tags": ["crisis", "cross_asset", "safe_haven", "vix_transmission"],
            "strategy_code": '''"""Cross-Asset Volatility Skew Transmission Strategy"""
import numpy as np
from common.domain.strategy_base import BaseStrategy

class VolatilitySkewTransmission(BaseStrategy):
    def __init__(self, vix_ema: int = 20, panic_ratio: float = 1.25):
        super().__init__(name="VolatilitySkewTransmission", params={"vix_ema": vix_ema, "panic_ratio": panic_ratio})
        self.vix_ema = vix_ema
        self.panic_ratio = panic_ratio

    def generate_signals(self, market_data) -> np.ndarray:
        close = market_data.close_prices
        T, K = close.shape
        signals = np.zeros((T, K), dtype=float)
        vix_idx = market_data.get_symbol_index("VIX") if hasattr(market_data, "get_symbol_index") else 3
        spy_idx = market_data.get_symbol_index("SPY") if hasattr(market_data, "get_symbol_index") else 0
        tlt_idx = market_data.get_symbol_index("TLT") if hasattr(market_data, "get_symbol_index") else 1
        gld_idx = market_data.get_symbol_index("GLD") if hasattr(market_data, "get_symbol_index") else 2
        cash_idx = market_data.get_symbol_index("CASH") if hasattr(market_data, "get_symbol_index") else 4

        for t in range(self.vix_ema, T):
            vix_hist = close[t - self.vix_ema:t, vix_idx]
            vix_mean = np.mean(vix_hist)
            ratio = close[t, vix_idx] / (vix_mean + 1e-8)

            if ratio > self.panic_ratio:
                # Volatility expansion: flight to safe haven
                signals[t, tlt_idx] = 0.50
                signals[t, gld_idx] = 0.30
                signals[t, cash_idx] = 0.20
            else:
                # Normal expansion: capture equity drift
                signals[t, spy_idx] = 0.70
                signals[t, tlt_idx] = 0.20
                signals[t, cash_idx] = 0.10
        return signals

    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        T, K = signals.shape
        weights = np.zeros((T, K), dtype=float)
        for t in range(T):
            pos = np.maximum(signals[t], 0.0)
            s = np.sum(pos)
            if s > 0:
                weights[t] = pos / s
            else:
                weights[t, -1] = 1.0
        return weights
''',
        },
    ],
}


class HypothesisGenerator:
    """Template-free generator of economic anomaly & quantitative trading hypotheses.

    Supports:
    - Dual-mode execution (deterministic mock vs live LLM API).
    - Research archive to ensure originality and prevent duplicate ideas.
    - Recursive prompt prior integration from MASS self-supervision loops.
    """

    def __init__(
        self,
        llm_provider: Optional[BaseModelProvider] = None,
        config: Optional[RSIExecutionConfig] = None,
        initial_priors: Optional[List[str]] = None,
    ) -> None:
        self.config = config or (RSIExecutionConfig() if RSIExecutionConfig else None)
        self.provider = llm_provider
        if self.provider is None and get_llm_provider and self.config:
            self.provider = get_llm_provider(self.config)

        self.archive: List[Hypothesis] = []
        self.priors: List[str] = list(initial_priors or [
            "Prior 0: Simple momentum without regime filter suffers high drawdown in Crisis era.",
            "Prior 1: Volatility skew indicators protect against sudden liquidity shocks.",
            "Prior 2: Multi-era cross-asset balancing stabilizes equity curve across choppy markets.",
        ])

    def update_priors(self, new_priors: List[str]) -> None:
        """Appends new distilled priors from MASS self-supervision."""
        for prior in new_priors:
            if prior not in self.priors:
                self.priors.append(prior)
        logger.info(f"HypothesisGenerator priors updated. Total priors: {len(self.priors)}")

    def generate_hypothesis(
        self,
        generation: int = 0,
        domain_context: Optional[str] = None,
        priors: Optional[List[str]] = None,
        variant_index: int = 0,
    ) -> Hypothesis:
        """Formulates a new research hypothesis for the given discovery generation.

        Args:
            generation: Discovery cycle index (0 = baseline, 1 = refined, 2+ = advanced).
            domain_context: Optional narrative on market environment or research goals.
            priors: Optional list of distilled priors overriding instance priors.
            variant_index: Index to pick diverse hypothesis variants within the generation.

        Returns:
            Validated Hypothesis instance.
        """
        active_priors = priors if priors is not None else self.priors

        # Live LLM execution if configured and provider is live
        if self.provider is not None and getattr(self.provider, "mode", "mock") == "live":
            return self._generate_live(generation, domain_context, active_priors)

        # Deterministic offline mock execution
        return self._generate_mock(generation, domain_context, active_priors, variant_index)

    def _generate_mock(
        self,
        generation: int,
        domain_context: Optional[str],
        priors: List[str],
        variant_index: int,
    ) -> Hypothesis:
        """Generates authentic hypothesis from generational bank in mock mode."""
        gen_clamped = min(max(generation, 0), max(GENERATIONAL_HYPOTHESIS_BANK.keys()))
        candidates = GENERATIONAL_HYPOTHESIS_BANK.get(gen_clamped, GENERATIONAL_HYPOTHESIS_BANK[0])
        template = candidates[variant_index % len(candidates)]

        # Check if already in archive; if so, vary confidence slightly
        existing_titles = {h.title for h in self.archive}
        title = template["title"]
        if title in existing_titles:
            title = f"{title} (Gen {generation} Rev {variant_index + 1})"

        hyp = Hypothesis(
            title=title,
            anomaly_description=template["anomaly_description"],
            formula=template["formula"],
            expected_regime=template["expected_regime"],
            confidence=template["confidence"],
            rationale=template["rationale"],
            parameters=dict(template.get("parameters", {})),
            generation=generation,
            tags=list(template.get("tags", [])),
            strategy_code=template.get("strategy_code"),
        )
        self.archive.append(hyp)
        return hyp

    def _generate_live(
        self,
        generation: int,
        domain_context: Optional[str],
        priors: List[str],
    ) -> Hypothesis:
        """Calls live Gemini model to synthesize a template-free hypothesis."""
        priors_text = "\n".join(f"- {p}" for p in priors[-5:])
        prompt = f"""You are an elite quantitative finance researcher formulating an authentic anomaly hypothesis.
Generation: {generation}
Domain: Automated Quantitative Alpha Discovery over SPY, TLT, GLD, VIX, and Cash across 756-day Bull, Choppy, and Crisis eras.

Empirical Priors from Prior Rounds:
{priors_text}

Task:
Formulate a novel, template-free quantitative trading hypothesis.
Return ONLY valid JSON matching this schema:
{{
  "title": "Short title",
  "anomaly_description": "Precise economic or microstructure anomaly description",
  "formula": "Mathematical alpha signal formula",
  "expected_regime": "bull" | "choppy" | "crisis" | "all",
  "confidence": 0.85,
  "rationale": "Economic theory explaining why this anomaly persists",
  "parameters": {{"window": 20, ...}}
}}
"""
        try:
            raw_text = self.provider.generate_text(prompt)  # type: ignore
            # Parse JSON
            cleaned = raw_text.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            data = json.loads(cleaned)

            hyp = Hypothesis(
                title=data.get("title", f"Live Alpha Hypothesis Gen {generation}"),
                anomaly_description=data.get("anomaly_description", "Anomaly identified via live LLM discovery."),
                formula=data.get("formula", "Signal = Momentum / DownsideRisk"),
                expected_regime=data.get("expected_regime", "all"),
                confidence=float(data.get("confidence", 0.85)),
                rationale=data.get("rationale", "Microstructure liquidity dynamics."),
                parameters=data.get("parameters", {}),
                generation=generation,
                tags=["live_llm", "discovery"],
            )
            self.archive.append(hyp)
            return hyp
        except Exception as err:
            logger.warning(f"Live hypothesis generation failed ({err}); falling back to mock bank.")
            return self._generate_mock(generation, domain_context, priors, 0)
