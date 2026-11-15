---
marp: true
theme: default
paginate: true
header: "Recursive Self-Improvement (RSI) Course Suite | Section 10 Overview"
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

# Recursive Self-Improvement (RSI)
## State-of-the-Art Autonomous Systems & Second-Order Meta-Discovery

**Course Section**: Section 10 Codelabs (RSI Course Map)  
**Domain**: Quantitative Alpha Discovery (`Alpha-RSI`)  
**Three SOTA Paradigms**: Dream-RSI, AIDE2, The AI Scientist 2  
**Presenter**: Autonomous AI Research Team

---

# Course Map: Where Does Section 10 Fit?

- **Sections 1–3**: Foundations of LLM Harnesses, Deterministic Evaluators, Self-Correction
- **Sections 4–6**: First-Order Optimization, AST Mutation Operators, Search Spaces
- **Sections 7–9**: Multi-Agent Architectures, In-Context Memory, Tool Use
- **Section 10 (Pinnacle)**: **State-of-the-Art Recursive Self-Improvement**
  - Going beyond prompt tuning: *autonomous systems improving their own search procedures*
  - Moving from first-order parameter updates to **second-order meta-improvement**
  - Concrete validation on non-trivial, adversarial problem domains

---

# The Core Challenge of Recursive Self-Improvement

- **Traditional Optimization**:
  - Human designs algorithm $\to$ Algorithm produces candidate solutions
  - Bottleneck: Human-imposed algorithmic heuristics and fixed search biases

- **First-Order Self-Improvement**:
  - LLM mutates candidate solution: $S_t \to S_{t+1}$
  - Stagnation: Search efficiency plateaus as initial prompts run out of leverage

- **Second-Order Recursive Self-Improvement (RSI)**:
  - System inspects its own search trajectory and mutates **the improver itself**:
  $$\mathcal{M}(I_t) = I_{t+1}$$
  - The discovery process itself accelerates over time!

---

# The Tripartite Architecture ($S, I, E$)

A robust RSI system separates three mathematical entities:

1. **Task Solver ($S \in \mathcal{S}$)**
   - Executes domain tasks (e.g., Python trading strategy calculating portfolio weights $w_t$).
2. **Improver ($I \in \mathcal{I}$)**
   - Mutates and refines solver candidates based on feedback: $S_{t+1} = I(S_t, \text{Feedback}_t)$.
3. **Evaluator ($E: \mathcal{S} \to \mathbb{R}$)**
   - **Fixed ground-truth judge**: Completely immutable and separate from $I$.
   - Prevents **reward hacking** and evaluation gaming.

---

# The Frozen Weights Paradigm

> **Crucial Insight**: Modern RSI does *not* require modifying base LLM weights!

$$\theta_{\text{LLM}} = \text{constant}$$

### Where Does Recursive Self-Improvement Live?
- **Code Space**: AST transformations, bug eradication, algorithmic synthesis.
- **Search Topology**: MCTS tree growth, pruning dead-ends, adaptive UCT exploration ($c$).
- **Surrogate World Models**: Learning empirical transition dynamics ($\mathcal{W}$) for offline dreaming.
- **Persistent Meta-Memory**: Accumulating peer review critiques into prompt priors (MASS).

---

# The Problem Domain: Alpha-RSI

We benchmark all RSI paradigms on **Automated Quantitative Alpha Discovery**:

- **Objective**: Synthesize executable Python strategies that generate excess risk-adjusted returns (Sharpe ratio) while controlling drawdown across market crises.
- **Universe**: `SPY` (Equities), `TLT` (Treasuries), `GLD` (Gold), `VIX` (Volatility), `CASH`.
- **High-Speed Vectorized Backtest**: Sub-20ms per strategy execution.
- **Strict Realistic Friction**: **10 bps (0.10%) transaction fee** per turnover.
- **Leverage Normalization**: Maximum gross leverage bounded at $\sum |w_i| \le 1.0$.

---

# The 756-Day Multi-Era Stress Environment

Strategies are stress-tested across 3 consecutive years with distinct market regimes:

```
+-------------------+--------------------+--------------------+
| Era 1: Bull Trend | Era 2: Choppy      | Era 3: Crisis      |
| Days 1–252        | Days 253–504       | Days 505–756       |
+-------------------+--------------------+--------------------+
| Drift: +25% / yr  | Drift: 0% / yr     | Drift: -35% / yr   |
| Realized Vol: 12% | Realized Vol: 18%  | Realized Vol: 45%  |
| Steady uptrend    | False breakouts    | Liquidity shock    |
+-------------------+--------------------+--------------------+
```

*Strategies overfitting to Era 1 suffer catastrophic liquidation in Era 3!*

---

# Three State-of-the-Art Paradigms

```
+---------------------------------------------------------------------------------+
|                                 The Three Paradigms                             |
+---------------------+---------------------------+-------------------------------+
| Codelab 1: Dream-RSI| Codelab 2: AIDE2          | Codelab 3: The AI Scientist 2 |
+---------------------+---------------------------+-------------------------------+
| World Models &      | Code Tree MCTS &          | Automated Discovery &         |
| Off-Policy Dreaming | Bi-Level Meta-Harness     | MASS Peer Review Distillation |
| (Simulation Space)  | (Program AST Space)       | (Scientific Literature Space) |
+---------------------+---------------------------+-------------------------------+
```

Each paradigm tackles self-improvement through a distinct, principled lens!

---

# Paradigm 1: Dream-RSI (World Models)

- **Philosophy**: Inspired by world models in RL (Dreamer, MuZero).
- **Core Mechanism**:
  1. Record exploration trajectory in an immutable replay buffer.
  2. Fit an **Empirical Regime Transition Model (ERTM)**: $P(s_{t+1} \mid s_t)$.
  3. **Dream in Imagination**: Simulate counterfactual macro search policies ($B, D, \tau$) offline without live LLM API calls.
  4. Redeploy optimized macro policy online.
- **Key Result**: **+0.7087 Sharpe gain**, 19.7% drawdown reduction in **55.6 milliseconds**!

---

# Paradigm 2: AIDE2 (Code Tree MCTS)

- **Philosophy**: Program synthesis as tree search over discrete code space.
- **Core Mechanism**:
  1. Tree graph where nodes are full Python strategies.
  2. MCTS guided by Upper Confidence Bounds for Trees (UCT).
  3. Specialized operators: **Drafting**, **Debugging** (AST repair), **Refinement**.
  4. **Bi-Level Meta-Harness**: Outer loop adaptively mutates prompt weights and exploration parameter $c$ across generations.
- **Key Result**: **+0.72 Sharpe gain** ($1.10 \to 1.82$), automated recovery from `ZeroDivisionError` via AST surgery!

---

# Paradigm 3: The AI Scientist 2 (Scientific Discovery)

- **Philosophy**: Emulating the human scientific method and peer review loop.
- **Core Mechanism**:
  1. **Template-Free Hypothesis Generation**: Formulates novel economic anomalies.
  2. **Multi-Era Ablation**: Stress-tests across Bull, Choppy, and Crisis eras.
  3. **Simulated Peer Review Board**: Scores Soundness, Novelty, Significance, Discipline.
  4. **Multi-Agent Self-Supervision (MASS)**: Distills review critiques into persistent memory prompt priors for subsequent generations.
- **Key Result**: Review score jumps from **6.44 to 7.96/10**, producing accepted volatility transmission alpha!

---

# Cross-Paradigm Benchmark & Pareto Frontier

```
Sharpe Ratio
     ^
2.0  |                                          * [AIDE2 Gen 2: Sharpe 1.82, 8.2s]
     |
1.5  |                            * [AIDE2 Gen 1: Sharpe 1.45]
     |         * [Dream-RSI Gen 1: Sharpe 1.19, 0.05s]
1.0  |
     |
0.5  |  * [AI Scientist 2: Sharpe 0.50, 0.30s]
     |
0.0  +-------------------------------------------------------------------->
     0.01s            0.10s            1.0s             10.0s     Wall Time
```

- **Dream-RSI**: Dominates ultra-fast low-latency simulation.
- **AIDE2**: Dominates peak absolute performance via deep tree search.
- **AI Scientist 2**: Dominates qualitative interpretability & scientific reasoning.

---

# Verification Suite: 100% Deterministic Passing

- **78 Automated Pytest Cases** running in sub-2 seconds:
  - **Tier 1**: 49 Unit & Feature Coverage tests
  - **Tier 2**: 14 Boundary & Stress Regime tests
  - **Tier 3**: 10 Cross-Feature Generational Interaction tests
  - **Tier 4**: 5 Real-World End-to-End Scenarios
  - **Master**: Unified E2E assertion runner
- **Dual-Mode Execution**:
  - `RSI_MODE=mock`: 100% offline, zero network, fully reproducible.
  - `RSI_MODE=live`: Seamless integration with Gemini 3.8 Flash (`google-genai`).

---

# Section 10 Codelabs Series Roadmap

In the upcoming video segments, we dive step-by-step into each codelab:

1. **Codelab 1**: Dream-RSI — Building ERTM world models and off-policy dreaming.
2. **Codelab 2**: AIDE2 — Implementing code tree MCTS, AST debuggers, and meta-harnesses.
3. **Codelab 3**: The AI Scientist 2 — Hypothesis generation, ablation, and MASS loops.
4. **Codelab 4**: Comparative Benchmark — Running Pareto analytics and multi-paradigm trade-offs.

*Let's dive into Codelab 1: Dream-RSI!*
