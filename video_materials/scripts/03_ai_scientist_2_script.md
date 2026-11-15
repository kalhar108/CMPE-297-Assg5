# Narration Script: Codelab 3 — The AI Scientist 2
## Automated Scientific Discovery & Multi-Agent Self-Supervision (MASS)

**Target Slide Deck**: `video_materials/slides/03_ai_scientist_2_slides.md`  
**Video Target Duration**: 8 minutes 45 seconds  
**Tone**: Intellectual, analytical, scientific, visionary  
**Formatting Keys**: `[SLIDE: X]` = Advance slide; `[PAUSE]` = 1-2 second breath/pacing; `[TERMINAL]` = Live demo command; `[HIGHLIGHT: ...]` = On-screen emphasis

---

### [00:00 - 00:55] Introduction to The AI Scientist 2

[SLIDE: 1 - Title Slide: Codelab 3 The AI Scientist 2]
"Welcome to Codelab 3 of our Recursive Self-Improvement series! Today, we are exploring **The AI Scientist 2**: autonomous scientific discovery powered by Multi-Agent Self-Supervision. [PAUSE]

In Codelabs 1 and 2, we looked at self-improvement through the lens of world models and discrete program tree search. But in many high-stakes domains—like quantitative finance, biology, and materials science—code generation is only the downstream execution. The real intellectual work lies in **the scientific method**: formulating sound theoretical hypotheses, designing controlled experiments, stress-testing across adversarial regimes, and subjecting results to rigorous peer review. [PAUSE]

The AI Scientist 2 automates this entire scientific lifecycle."

---

### [00:55 - 02:00] The 4-Stage Autonomous Scientific Lifecycle

[SLIDE: 2 - Why Direct Code Search is Not Enough]
"When an agent blindly searches code space without an underlying hypothesis, it easily falls victim to data snooping: finding mathematical formulas that fit noise in historical data but fail miserably in the real world. [PAUSE]

[SLIDE: 3 - The 4-Stage Autonomous Scientific Lifecycle]
The AI Scientist 2 structures the research process into four rigorous stages:
- **Stage A: Template-Free Hypothesis Generation**. Formulating novel economic anomaly theories and mathematical representations without static templates.
- **Stage B: Progressive Experimentation & Multi-Era Ablation**. Running controlled backtests across Bull, Choppy, and Crisis stress regimes.
- **Stage C: Simulated Multi-Agent Peer Review**. A committee of specialist agents scoring Soundness, Novelty, Statistical Significance, and Risk Discipline.
- **Stage D: Multi-Agent Self-Supervision (MASS)**. Distilling review feedback into persistent memory priors that directly improve the next generation of hypotheses."

---

### [02:00 - 03:15] Stage A: Template-Free Hypothesis Generation

[SLIDE: 4 - Stage A: Template-Free Hypothesis Generation]
"Let's look at `03_ai_scientist_2/hypothesis.py`. [PAUSE]

Unlike older systems that plug numbers into static templates, our `HypothesisGenerator` synthesizes structured Pydantic objects containing:
- The title of the anomaly;
- The economic behavioral rationale—explaining *why* market participants create this inefficiency;
- The mathematical signal formula;
- The expected regime where the effect should be strongest;
- And an initial confidence score. [PAUSE]

In Generation 0, the agent synthesizes a baseline theory: *Unfiltered Time-Series Momentum Continuation*, hypothesizing that assets with positive 20-day returns will continue trending due to investor anchoring."

---

### [03:15 - 04:30] Stage B: Multi-Era Stress Ablation

[SLIDE: 5 - Stage B: Multi-Era Stress Ablation]
"Now, look at `ablation.py`. Instead of merely running an aggregate backtest, the system performs a multi-era stress ablation. [PAUSE]

Notice the ablation breakdown across our three market eras:
- In the **Bull era**, the strategy achieves a Sharpe of -2.08 due to massive turnover friction (turnover of 116.46!).
- In the **Choppy era**, it loses -26.19% from false breakout whipsaws.
- In the **Crisis era**, it suffers a brutal **46.08% maximum drawdown**! [HIGHLIGHT: Crisis MDD 46.08%]

The ablation proves definitively that naive momentum without volatility filtering or regime awareness is catastrophic in stress regimes."

---

### [04:30 - 05:45] Stage C: Simulated Peer Review & Decision

[SLIDE: 6 - Stage C: Simulated Multi-Agent Peer Review Board]
"Next, the hypothesis and ablation report are submitted to our simulated Peer Review Board in `peer_review.py`. [PAUSE]

The review committee evaluates four independent scientific dimensions:
- **Soundness**: 5.50 / 10.0
- **Novelty**: 6.50 / 10.0
- **Statistical Significance**: 8.00 / 10.0
- **Risk Discipline**: 5.00 / 10.0
- **Composite Score**: **6.25 / 10.00**
- **Final Decision**: **REJECT**! [PAUSE]

The committee issues a detailed narrative critique:
*Excessive transaction turnover and complete lack of volatility filtering leads to severe drawdown during regime shifts. Lacks cross-asset hedge balance.*"

---

### [05:45 - 07:00] Stage D: The MASS Feedback Loop

[SLIDE: 7 - Stage D: Multi-Agent Self-Supervision (MASS)]
"In standard research workflows, a rejected paper is discarded. But in The AI Scientist 2, rejection is the fuel for recursive self-improvement! [PAUSE]

In `mass_loop.py`, the **Multi-Agent Self-Supervision (MASS)** engine distills the committee's critique into actionable memory priors:
1. *Simple momentum without regime filter suffers high drawdown in Crisis era.*
2. *Volatility skew indicators protect against sudden liquidity shocks.*
3. *Multi-era cross-asset balancing stabilizes equity curves across choppy markets.*

These distilled priors are appended to the system's persistent memory and injected directly into the prompt context for Generation 1 and Generation 2!"

---

### [07:00 - 08:00] Live Demonstration & Winning Discovery

[TERMINAL: Run `python 03_ai_scientist_2/demo.py`]
"Let's look at the generational progression table from our live demo run: [PAUSE]

[SLIDE: 8 - Generational Progression: Verified Results]
- In **Generation 0**: Average review score was 6.44 / 10.0, best Sharpe was 0.15, and Crisis drawdown was 46.08%.
- In **Generation 1**: Armed with the distilled priors, the agent designs strategies with cross-asset volatility hedges. Review score jumps to 7.67 / 10.0, Sharpe rises to 0.39, and Crisis drawdown plunges from 46% down to **17.54%**!
- In **Generation 2**: Review score reaches **7.96 / 10.0**, with peak score hitting **8.00 / 10.0** and Sharpe rising to **0.50**! [PAUSE]

[SLIDE: 9 - The Winning Scientific Discovery Artifact]
And here is the winning discovery approved with an **ACCEPT** by the committee:
**Cross-Asset Volatility Skew Transmission & Safe-Haven Rebalancing**!
It uses the formula:
$$w_{\text{safe}} = \text{sigmoid}\left( \frac{\text{VIX}_t - \text{EMA}(\text{VIX}, 20)}{\text{ATR}(\text{VIX}, 10)} \right)$$
dynamically rotating capital into long Treasuries and Gold whenever implied volatility spikes."

---

### [08:00 - 08:45] Conclusion & Key Insights

[SLIDE: 10 - Key Takeaways: Codelab 3]
"To wrap up Codelab 3: The AI Scientist 2 demonstrates that self-improvement can operate at the conceptual level of scientific inquiry. By combining template-free hypothesis generation, multi-era ablation, and simulated peer review with MASS memory distillation, the agent achieved a **+1.52 point gain in scientific rigor** and discovered robust, multi-era trading alpha. [PAUSE]

In our final video, we bring all three codelabs together in Codelab 4 for a head-to-head comparative benchmark and Pareto frontier analysis!"
