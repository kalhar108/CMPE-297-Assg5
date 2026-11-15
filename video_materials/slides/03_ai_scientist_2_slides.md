---
marp: true
theme: default
paginate: true
header: "Codelab 3: The AI Scientist 2 | Automated Discovery & MASS"
footer: "Special Topics in AI | Assignment 5: Alpha-RSI Discovery"
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 24px;
    padding: 40px;
  }
  h1 {
    color: #1a365d;
    font-size: 38px;
  }
  h2 {
    color: #2b6cb0;
    font-size: 30px;
  }
  pre {
    background-color: #f7fafc;
    border-left: 4px solid #3182ce;
  }
---

# Codelab 3: The AI Scientist 2
## Automated Scientific Discovery & Multi-Agent Self-Supervision (MASS)

**Course**: Section 10 — State-of-the-Art RSI Paradigms  
**Module**: `03_ai_scientist_2/`  
**Key Principle**: End-to-end scientific method: Hypothesis $\to$ Ablation $\to$ Review $\to$ MASS Loop  
**Execution Runtime**: Python 3.10+ / Multi-Agent Protocol / Pydantic

---

# Why Direct Code Search is Not Enough

- **The Limitation of Program Search**:
  - Code search (like AIDE2) optimizes *implementations*, but lacks high-level scientific reasoning about *why* a phenomenon occurs.
  - Blind search easily discovers spurious correlations that look great in backtests but collapse out-of-sample.

- **The AI Scientist 2 Solution**:
  - Lift the problem to the level of the **scientific literature lifecycle**:
    1. Formulate grounded economic anomaly hypotheses.
    2. Design stress-testing experiments and multi-era ablations.
    3. Simulate a rigorous peer review board to critique methodology.
    4. Distill peer review critiques into persistent meta-memory priors (**MASS**).

---

# The 4-Stage Autonomous Scientific Lifecycle

```
+---------------------------------------------------------------------------------+
|                       The AI Scientist 2 Architecture                           |
+---------------------------------------------------------------------------------+
                                         |
     Stage A: Hypothesis Generation      |  - Template-free synthesis of economic
                                         |    anomalies & mathematical formulas
                                         v
     Stage B: Progressive Experimenter   |  - Automated trial execution across
              & Multi-Era Ablation       |    Bull (1), Choppy (2), and Crisis (3)
                                         v
     Stage C: Multi-Agent Peer Review    |  - Multi-agent committee evaluates
              Board                      |    Soundness, Novelty, Significance, Risk
                                         v
     Stage D: MASS Self-Supervision      |  - Distills review critiques into priors
              Loop                       |  - Directly updates next-generation prompt
```

---

# Stage A: Template-Free Hypothesis Generation

Implemented in `03_ai_scientist_2/hypothesis.py`:

```python
class Hypothesis(BaseModel):
    title: str
    anomaly_description: str
    formula: str
    expected_regime: str
    confidence: float
```

### Generation 0 Baseline Hypothesis:
```
+--------------------------------------------------------------------------------+
| Hypothesis: Unfiltered Time-Series Momentum Continuation                       |
+--------------------------------------------------------------------------------+
| Generation: 0    | Regime: bull       | Confidence: 0.70                       |
| Formula   : Signal = sign(Close / Close[t-20] - 1.0)                           |
| Anomaly   : Assets with positive 20-day returns continue to outperform due     |
|             to investor behavioral anchoring.                                  |
+--------------------------------------------------------------------------------+
```

---

# Stage B: Multi-Era Stress Ablation

Implemented in `03_ai_scientist_2/ablation.py`:

- Evaluates the strategy across the 3 distinct market eras:

```
+----------+------------+---------------+------------+--------------+------------+
| Regime   | Sharpe     | Max Drawdown  | CAGR       | Sortino      | Turnover   |
+----------+------------+---------------+------------+--------------+------------+
| BULL     |      -2.08 |        38.48% |    -34.61% |        -2.64 |     116.46 |
| CHOPPY   |      -0.87 |        33.53% |    -26.19% |        -1.24 |      70.28 |
| CRISIS   |      -0.02 |        46.08% |    -14.04% |        -0.03 |      58.23 |
+----------+------------+---------------+------------+--------------+------------+
```

- **Diagnosis**: Simple momentum suffers severe turnover drag and huge drawdowns in all three eras!

---

# Stage C: Simulated Multi-Agent Peer Review Board

Implemented in `03_ai_scientist_2/peer_review.py`:

```
+--------------------------------------------------------------------------------+
| PEER REVIEW COMMITTEE SCORECARD: Unfiltered Time-Series Momentum Continuation |
+--------------------------------------------------------------------------------+
| Composite Score : 6.25  / 10.00   | Final Decision: [REJECT]                   |
| - Soundness     : 5.50  / 10.00   | - Novelty     : 6.50  / 10.00              |
| - Significance  : 8.00  / 10.00   | - Risk Discip.: 5.00  / 10.00              |
+--------------------------------------------------------------------------------+
| Committee Critique:                                                            |
| Excessive transaction turnover and lack of volatility filtering leads to       |
| severe drawdowns during regime shifts. Lacks cross-asset hedge balance.        |
+--------------------------------------------------------------------------------+
```

- The committee provides explicit qualitative feedback on *why* the hypothesis failed!

---

# Stage D: Multi-Agent Self-Supervision (MASS)

Implemented in `03_ai_scientist_2/mass_loop.py`:

```python
class MASSLoop:
    def __init__(self):
        self.distilled_priors: List[str] = [
            "Simple momentum without regime filter suffers high drawdown in Crisis era.",
            "Volatility skew indicators protect against sudden liquidity shocks.",
            "Multi-era cross-asset balancing stabilizes equity curve across choppy markets."
        ]

    def distill(self, review: Dict[str, Any]) -> str:
        """Extracts core critique lessons and appends to persistent prompt priors."""
        prior = self._extract_key_lesson(review["critique"])
        self.distilled_priors.append(prior)
        return prior
```

- These priors are injected into the generator prompt for Generation 1 and 2!

---

# Generational Progression: Verified Results

```
+------+------------+------------------+-------------------+---------------+-----------------+
| Gen  | Candidates | Avg Review Score | Best Review Score | Best Sharpe   | Worst Crisis MDD|
+------+------------+------------------+-------------------+---------------+-----------------+
| 0    | 2          |           6.44/10 |            6.62/10 |        0.15 |          46.08% |
| 1    | 2          |           7.67/10 |            7.67/10 |        0.39 |          17.54% |
| 2    | 2          |           7.96/10 |            8.00/10 |        0.50 |          36.01% |
+------+------------+------------------+-------------------+---------------+-----------------+
```

### Measurable Self-Supervised Progress:
- **Peer Review Score**: **$6.44 \to 7.96 / 10.0$** ($+1.52$ point improvement).
- **Best Sharpe Ratio**: **$0.15 \to 0.50$** ($+0.35$ delta).
- **Crisis Drawdown**: Cut from $46.08\%$ down to $17.54\%$ in Gen 1!

---

# The Winning Scientific Discovery Artifact

The peer-reviewed accepted strategy synthesized by The AI Scientist 2:

- **Title**: *Cross-Asset Volatility Skew Transmission & Safe-Haven Rebalancing*
- **Mathematical Formula**:
  $$w_{\text{safe}} = \text{sigmoid}\left( \frac{\text{VIX}_t - \text{EMA}(\text{VIX}, 20)}{\text{ATR}(\text{VIX}, 10)} \right)$$
- **Economic Rationale**:
  - VIX volatility expansions precede equity liquidations by 2–5 days.
  - Dynamically shifting equity weights into negatively correlated safe-havens (`TLT`, `GLD`) during volatility spikes protects capital.
- **Committee Decision**: **ACCEPT** (Score: **8.00 / 10.00**)

---

# Key Takeaways: Codelab 3 (The AI Scientist 2)

1. **Reasoning Above Raw Code**:
   - Focuses on formulating sound scientific principles rather than trial-and-error code patching.
2. **Multi-Era Ablation Exposes Flaws**:
   - Stress-testing across Bull, Choppy, and Crisis eras prevents overfitting.
3. **Multi-Agent Self-Supervision (MASS)**:
   - Peer review critiques become persistent memory priors, driving clear generational improvement.

*Next up: Codelab 4 — Comparative Benchmark & Pareto Analysis!*
