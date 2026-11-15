# Narration Script: Codelab 2 — AIDE2
## Code Tree MCTS & Bi-Level Meta-Optimization

**Target Slide Deck**: `video_materials/slides/02_aide2_slides.md`  
**Video Target Duration**: 8 minutes 15 seconds  
**Tone**: High-energy, deeply technical, practical software engineering focus  
**Formatting Keys**: `[SLIDE: X]` = Advance slide; `[PAUSE]` = 1-2 second breath/pacing; `[TERMINAL]` = Live demo command; `[HIGHLIGHT: ...]` = On-screen emphasis

---

### [00:00 - 00:50] Introduction to AIDE2

[SLIDE: 1 - Title Slide: Codelab 2 AIDE2]
"Welcome back to Section 10! In this video, we are unpacking Codelab 2: **AIDE2**—combining program synthesis with Monte Carlo Tree Search and bi-level meta-optimization. [PAUSE]

If you've ever built a code-generation agent with large language models, you know the frustration of linear prompting. You ask the LLM to write a script. It makes a syntax error. You ask it to fix the error, and it introduces a logical bug. Pretty soon, the chat history is thousands of tokens long, polluted with failed attempts, and the model enters an endless loop of broken code. [PAUSE]

AIDE2 fundamentally redesigns code generation by treating program synthesis not as a conversation, but as a formal **Tree Search over Program Space**."

---

### [00:50 - 02:00] The Bi-Level Architecture

[SLIDE: 2 - Why Linear Code Generation Fails]
"In tree search, a bad line of code is simply a branch on a tree. If a branch fails, you don't stay on it—you backtrack to a known working parent node and explore an alternative path! [PAUSE]

[SLIDE: 3 - AIDE2 Architecture: Bi-Level Optimization]
AIDE2 implements this through a **Bi-Level Optimization loop**:
- In the **Inner Loop**, Monte Carlo Tree Search explores code space using three specialized operators: Drafting, Debugging, and Refinement. Every node is evaluated by our vectorized backtester, and node values are propagated using Upper Confidence Bounds for Trees (UCT).
- In the **Outer Loop**, our Meta-Harness acts as a second-order improver: it monitors the tree's health, identifies whether the search is stagnating, and dynamically shifts operator prompt weights and exploration coefficients across generations."

---

### [02:00 - 03:15] Code Structures: StrategyNode & Operators

[SLIDE: 4 - Data Structure: The StrategyNode]
"Let's look at the implementation in `02_aide2/`. [PAUSE]

In `tree_node.py`, the fundamental unit is the `StrategyNode`. Each node holds:
- The full, executable Python source code for the strategy;
- The ground-truth backtest metrics—Sharpe, Max Drawdown, and CAGR;
- Pointers to parent and child nodes;
- And the MCTS statistics: visit count $N$ and cumulative value $Q$. [PAUSE]

[SLIDE: 5 - The Three Specialized Operators]
In `operators.py`, we implement three discrete code-mutation engines:
1. The **Drafting Operator** synthesizes new strategy classes from scratch, adhering to our `BaseStrategy` interface.
2. The **Debugging Operator** handles compiler errors and runtime exceptions by parsing tracebacks and performing surgical AST repairs.
3. The **Refinement Operator** performs fine-grained indicator parameter tuning, adding volatility filters and cash buffers."

---

### [03:15 - 04:30] Live Execution: Drafting & Error Interception

[TERMINAL: Run `python 02_aide2/demo.py`]
"Let's execute the AIDE2 demonstration live! [PAUSE]

[SLIDE: 6 - Step A: Drafting Baseline Alpha Code]
First, Step A fires the Drafting Operator. It generates a clean momentum strategy class called `DraftedStrategy`. The strategy is immediately passed to our backtester:
- Backtest execution time: **17.68 milliseconds**;
- Initial Sharpe ratio: **0.8712**;
- Max Drawdown: **13.76%**. [PAUSE]

[SLIDE: 7 - Step B & C: Automated Exception Interception & AST Repair]
Now, watch Step B! We simulate what happens when a mutated strategy introduces a dangerous calculation: a division by zero during volatility normalization `(p2 - p1) / std`.

The backtester immediately catches the `ZeroDivisionError: float division by zero`.
Instead of crashing or quitting, AIDE2 intercepts the traceback and passes it to the `DebuggingOperator`.
The debugger inspects the AST, diagnoses the root cause, and applies surgical repair: replacing the naked denominator with `volatility + 1e-8`.
The repaired strategy is recompiled and re-tested—it runs flawlessly with zero runtime exceptions!"

---

### [04:30 - 05:45] MCTS Tree Search, UCT & Backtracking

[SLIDE: 8 - Step D: UCT Selection & Adaptive Backtracking]
"Next, look at Step D: MCTS Tree Search and UCT selection. [PAUSE]

The search computes UCT scores for each candidate branch:
$$\text{UCT}(n) = \frac{Q(n)}{N(n)} + c \sqrt{\frac{\ln N(\text{parent})}{N(n)}}$$

In our run:
- Branch 1 (Refined Node) achieved a high UCT score of **1.872**;
- Branch 2 (Flawed Node) experienced a catastrophic $-62\%$ drawdown, producing a negative UCT of **-0.673**! [PAUSE]

Watch what happens: AIDE2 flags Branch 2 as a `DEAD-END`, prunes it from the tree graph, and triggers **adaptive backtracking**, returning selection to the root node to expand high-conviction branches. This keeps the search focused entirely on viable strategy spaces."

---

### [05:45 - 07:15] Bi-Level Meta-Harness Progression

[SLIDE: 9 - Step E: Bi-Level Meta-Harness Self-Improvement]
"Now, look at Step E: the outer meta-optimization loop in action across three full generations: [PAUSE]

Look at this progression table:
- In **Generation 0**, the meta-harness starts with broad exploration: $c = 1.41$, and operator weights favoring Drafting ($w_{\text{draft}} = 0.5$). The best strategy achieves a Sharpe of **1.10** with 22.0% drawdown.
- In **Generation 1**, the outer loop detects viable prototypes. It lowers exploration to $c = 1.10$ and increases Refinement to $0.5$. Sharpe jumps to **1.45**, and drawdown drops to 17.0%.
- In **Generation 2**, the meta-harness tightens exploration further to $c = 0.80$, with 60% of compute dedicated to Refinement. The peak strategy hits a phenomenal Sharpe ratio of **1.82** with only 12.0% drawdown! [HIGHLIGHT: Sharpe 1.82]

Over the full run, AIDE2 delivers a net recursive self-improvement delta of **+0.72 Sharpe—a 65.5% performance gain**!"

---

### [07:15 - 08:15] Conclusion & Key Insights

[SLIDE: 10 - Key Takeaways: Codelab 2]
"To summarize Codelab 2: AIDE2 demonstrates that code generation is best modeled as a tree search problem. With automated traceback interception, surgical AST repair, and a bi-level meta-harness adapting search heuristics, autonomous agents can reliably synthesize complex, high-performing software systems. [PAUSE]

In the next video, we ascend to Codelab 3: **The AI Scientist 2**, where we explore automated scientific discovery and multi-agent peer review loops!"
