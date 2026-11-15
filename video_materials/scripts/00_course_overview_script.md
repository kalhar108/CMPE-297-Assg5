# Narration Script: Course Overview & Theoretical Foundations of RSI
## Section 10: State-of-the-Art Recursive Self-Improvement (RSI) Codelabs

**Target Slide Deck**: `video_materials/slides/00_course_overview_slides.md`  
**Video Target Duration**: 8 minutes 30 seconds  
**Tone**: Authoritative, engaging, rigorous, accessible academic delivery  
**Formatting Keys**: `[SLIDE: X]` = Advance slide; `[PAUSE]` = 1-2 second breath/pacing; `[HIGHLIGHT: ...]` = On-screen emphasis

---

### [00:00 - 00:45] Title & Introduction

[SLIDE: 1 - Title Slide]
"Hello, and welcome to Section 10 of our course on Recursive Self-Improvement in Artificial Intelligence. Today, we are exploring one of the most exciting frontiers in modern machine learning: how autonomous systems can recursively inspect, optimize, and elevate their own problem-solving procedures. [PAUSE]

In this comprehensive codelab series, we will examine three state-of-the-art paradigms of Recursive Self-Improvement—Dream-RSI, AIDE2, and The AI Scientist 2—grounded in a unified, mathematically rigorous domain: Automated Quantitative Alpha Feature Discovery. By the end of this walkthrough, you will understand not just the theoretical mechanics of second-order meta-improvement, but see verified execution traces proving that these systems measurably improve across generations."

---

### [00:45 - 01:50] The RSI Course Map & Section 10 Pinnacle

[SLIDE: 2 - Course Map: Where Does Section 10 Fit?]
"Let's look at where Section 10 sits in the broader RSI Course Map. Earlier in the course, we established the foundations of LLM harnesses, learned how to build deterministic evaluators, explored first-order AST mutation operators, and implemented multi-agent memory frameworks. [PAUSE]

Section 10 represents the ultimate synthesis: the pinnacle codelabs. Here, we move beyond simple prompt engineering or static tool calling. We are constructing autonomous architectures that actively measure their own discovery performance and systematically rewrite their search topologies, prompt priors, and execution policies. This is where AI transitions from a passive tool into an actively self-improving researcher."

---

### [01:50 - 03:15] The Tripartite Architecture ($S, I, E$)

[SLIDE: 3 - The Core Challenge of Recursive Self-Improvement]
"To understand why self-improvement is non-trivial, consider standard optimization. In conventional machine learning, a human engineer designs the optimization algorithm. The algorithm produces candidate solutions. But the human's design introduces an impenetrable ceiling: the human's own heuristic biases and search assumptions. [PAUSE]

[SLIDE: 4 - The Tripartite Architecture (S, I, E)]
To break through this ceiling, our system establishes a formal tripartite architecture. [HIGHLIGHT: S, I, E]

First, we define the **Task Solver ($S$)**. This is the operational engine that executes our target domain—in our case, an executable Python strategy AST that maps daily market price action into portfolio allocation weights.

Second, we define the **Improver ($I$)**. The improver observes the solver's execution, reads the diagnostic telemetry, and mutates the solver to produce candidate $S_{t+1}$.

Third, and most importantly, we have the **Evaluator ($E$)**. The evaluator is our immutable ground-truth anchor. It evaluates the solver on unseen data and under real-world transaction friction. By keeping the Evaluator completely independent from the Improver, we mathematically eliminate the risk of reward hacking."

---

### [03:15 - 04:30] The Frozen Weights Paradigm

[SLIDE: 5 - The Frozen Weights Paradigm]
"Now, let's address the single biggest misconception in AI self-improvement: the belief that recursive self-improvement requires modifying the underlying neural network weights $\theta$. [PAUSE]

In modern agentic engineering, **the foundation model weights remain strictly frozen**! [HIGHLIGHT: theta_LLM = constant]

Why? Because modifying neural weights in an iterative loop frequently leads to catastrophic forgetting, reward collapse, and training instability. Instead, modern RSI operates in the cognitive and programmatic space:
- We mutate the **code and AST representations** directly.
- We adapt the **search topology** via Monte Carlo Tree Search.
- We calibrate **surrogate world models** that allow the system to simulate rollouts in imagination.
- And we accumulate **meta-memory priors** from structured peer review critiques.

Because the system modifies its own prompts, code representations, and search topologies, its problem-solving capability compounds exponentially without altering a single floating-point parameter in the underlying LLM."

---

### [04:30 - 05:45] The Problem Domain: Alpha-RSI

[SLIDE: 6 - The Problem Domain: Alpha-RSI]
"To prove that our self-improvement loops are genuine, we evaluate all methods on a unified, verifiable benchmark domain: **Alpha-RSI**. [PAUSE]

Why quantitative trading? Because financial markets offer zero room for hallucination. A backtest is either mathematically profitable or it is not. A strategy either survives a liquidity crisis or it blows up. [PAUSE]

Our universe consists of five fundamental assets:
- `SPY`, tracking broad equities;
- `TLT`, tracking 20-year long-term Treasuries;
- `GLD`, tracking physical gold;
- `VIX`, measuring market implied volatility;
- and `CASH`, our risk-free reserve.

Our backtest engine is built from scratch in pure vectorized NumPy, executing in under 20 milliseconds per strategy, and deducting a realistic 10 basis points transaction friction on every portfolio turnover."

---

### [05:45 - 06:45] The 756-Day Multi-Era Stress Environment

[SLIDE: 7 - The 756-Day Multi-Era Stress Environment]
"Our domain features a 756-day market history—three full trading years—partitioned into three distinct market regimes:
- **Era 1 (Days 1 to 252)** is a roaring **Bull Market**: positive drift of 25% annualized and low volatility of 12%.
- **Era 2 (Days 253 to 504)** is a **Choppy, Sideways Regime**: zero drift, frequent false breakouts, and oscillating mean-reversion.
- **Era 3 (Days 505 to 756)** is a severe **Crisis Regime**: 35% annualized liquidation, massive volatility spikes up to 45%, and systemic liquidity shocks.

Any strategy that naively overfits to the bull market will be liquidated in the crisis regime. Our RSI agents must discover strategies that adapt dynamically across all three regimes."

---

### [06:45 - 07:45] Overview of the Three Paradigms

[SLIDE: 8 - Three State-of-the-Art Paradigms]
"In this series, we walk through three distinct paradigms of self-improvement: [PAUSE]

[SLIDE: 9 - Paradigm 1: Dream-RSI]
In **Codelab 1 (Dream-RSI)**, we explore how an agent can build an Empirical Regime Transition Model and dream in imagination—simulating counterfactual macro exploration policies offline without spending a single live API token.

[SLIDE: 10 - Paradigm 2: AIDE2]
In **Codelab 2 (AIDE2)**, we dive into program synthesis as Monte Carlo Tree Search, watching specialized drafting, debugging, and refinement operators synthesize code, intercept runtime errors like ZeroDivisionError, and perform bi-level meta-optimization.

[SLIDE: 11 - Paradigm 3: The AI Scientist 2]
In **Codelab 3 (The AI Scientist 2)**, we elevate self-improvement to the level of the scientific method: generating template-free hypotheses, running multi-era ablations, submitting to simulated peer review, and distilling critiques into persistent prompt priors."

---

### [07:45 - 08:30] Pareto Frontier & Verification Roadmap

[SLIDE: 12 - Cross-Paradigm Benchmark & Pareto Frontier]
"In **Codelab 4**, we bring all three methods together on the same starting grid, analyzing their Pareto frontiers across Sharpe ratio and compute latency. [PAUSE]

[SLIDE: 13 - Verification Suite]
Every line of code you will see is backed by our 4-tier automated test harness—78 automated pytest cases passing 100% in under two seconds.

[SLIDE: 14 - Roadmap]
Without further ado, let's open our terminals and begin with Codelab 1: Dream-RSI!"
