# YouTube Video Walkthrough & Recording Choreography Guide
## Step-by-Step Live Demonstration Script & Visual Pacing Manual

**Target Audience**: On-Camera Presenter / Technical Instructor  
**Target Video Format**: Multi-Part YouTube Playlist (5 Episodes) or Single Master 35-Minute Video  
**Companion Assets**: Slide Decks (`slides/00..04`), Narration Scripts (`scripts/00..04`), Setup Guide (`recording_setup_guide.md`)

---

## Master Production Schedule

| Episode # | Topic | Slide Deck | Script | Target Run Time | Key Demo Command |
|---|---|---|---|:---:|---|
| **Part 0** | Course Overview & Theoretical Foundations | `00_course_overview_slides.md` | `00_course_overview_script.md` | 8m 30s | `pytest tests/ -v` |
| **Part 1** | Codelab 1: Dream-RSI (World Models) | `01_dream_rsi_slides.md` | `01_dream_rsi_script.md` | 7m 45s | `python 01_dream_rsi/demo.py` |
| **Part 2** | Codelab 2: AIDE2 (Code Tree MCTS) | `02_aide2_slides.md` | `02_aide2_script.md` | 8m 15s | `python 02_aide2/demo.py` |
| **Part 3** | Codelab 3: The AI Scientist 2 (MASS) | `03_ai_scientist_2_slides.md` | `03_ai_scientist_2_script.md` | 8m 45s | `python 03_ai_scientist_2/demo.py` |
| **Part 4** | Codelab 4: Comparative Benchmark | `04_comparative_benchmark_slides.md` | `04_comparative_benchmark_script.md` | 7m 30s | Master benchmark summary |
| **TOTAL** | **Full Master Walkthrough** | **All 5 Decks** | **All 5 Scripts** | **~40m 45s** | **Full System Suite** |

---

## Part 0: Course Overview & Theoretical Foundations

### Visual Layout:
- **00:00 - 05:45**: Full-Screen Presentation Slides (`00_course_overview_slides.md`).
- **05:45 - 07:45**: Split-Screen: Slides on Left (40%), VS Code / Terminal on Right (60%) displaying `common/domain/market_data.py`.
- **07:45 - 08:30**: Full-Screen Terminal running automated test suite.

### Choreography Steps:
1. **[00:00] Opening Hook**: Begin with slide 1. Deliver opening lines with confident, energetic cadence.
2. **[01:50] The Tripartite Diagram**: On slide 4, point your cursor at the $S$, $I$, and $E$ boxes. Emphasize that $E$ is **immutable** to prevent reward hacking.
3. **[03:15] Frozen Weights Clarification**: On slide 5, pause for dramatic effect. Emphasize: *"The model weights are frozen. The self-improvement lives in the search topology, program AST, and meta-memory."*
4. **[07:45] Terminal Verification Cue**:
   - Switch active window to Terminal.
   - Run the command:
     ```bash
     python -m pytest tests/ -v
     ```
   - Narrate while tests scroll: *"Watch this: 78 tests passing across all 4 tiers in under 2 seconds!"*
   - Highlight the green summary line `78 passed in 1.70s`.

---

## Part 1: Codelab 1 — Dream-RSI (World Models & Off-Policy Dreaming)

### Visual Layout:
- **00:00 - 04:15**: Full-Screen Slides (`01_dream_rsi_slides.md`).
- **04:15 - 07:15**: Full-Screen Terminal (`python 01_dream_rsi/demo.py`).
- **07:15 - 07:45**: Split-Screen: Slide 11 (Discovery Tree) on Left, Terminal on Right.

### Choreography Steps:
1. **[00:00] Conceptual Introduction**: Frame the problem of live LLM search costs.
2. **[03:00] Off-Policy Dreaming Math**: Explain that the world model ERTM matrix provides the transition probabilities for counterfactual trajectory evaluation.
3. **[04:15] Live Terminal Execution**:
   - Clear terminal screen: `cls` or `clear`.
   - Run the command:
     ```bash
     python 01_dream_rsi/demo.py
     ```
   - As the output streams, narrate in sync with the 5 phases:
     * Phase 1: Point to `Gen 0 Best Strategy: MomentumFactor (Sharpe: 0.4856)`.
     * Phase 2: Highlight the ERTM Transition Matrix on screen.
     * Phase 3: Highlight the Dream Optimization table: `Simulated Dream Yield: 0.8794 -> 4.3042 (+3.4248 Yield Delta)` in **0.09ms**!
     * Phase 5: Point to the final summary box: `Generation 1 Optimized Sharpe: 1.1943` (+0.7087 net improvement).
4. **[07:15] Tree Topology Walkthrough**: Review the ASCII discovery tree, showing how the search pruned the negative branch ($r=-0.264$) and concentrated on `node_g1_root` ($r=1.194$).

---

## Part 2: Codelab 2 — AIDE2 (Code Tree MCTS & Bi-Level Meta-Optimization)

### Visual Layout:
- **00:00 - 03:15**: Full-Screen Slides (`02_aide2_slides.md`).
- **03:15 - 07:15**: Full-Screen Terminal (`python 02_aide2/demo.py`).
- **07:15 - 08:15**: Split-Screen: Slide 9 (Bi-Level Table) on Left, VS Code on Right showing `02_aide2/operators.py`.

### Choreography Steps:
1. **[00:00] Linear Prompting Frustration**: Discuss how linear chat histories cause LLMs to get stuck in hallucination loops.
2. **[01:00] Tree Search Metaphor**: Introduce the tree structure where nodes are complete Python programs.
3. **[03:15] Live Terminal Execution**:
   - Clear terminal screen.
   - Run the command:
     ```bash
     python 02_aide2/demo.py
     ```
4. **[03:45] Traceback Interception & AST Repair (Crucial Moment)**:
   - When the `ZeroDivisionError: float division by zero` error log appears in red, slow down your delivery:
     *"Look at that: the candidate code crashed with a division by zero in the volatility calculation!"*
   - Show how the Debugging Operator intercepts the traceback, extracts line 38, and performs surgical AST repair: `volatility + 1e-8`.
   - Show the post-repair re-test immediately passing with Sharpe 0.8712.
5. **[05:45] Meta-Harness Table**:
   - Highlight the generational table showing Sharpe progression: **1.10 $\to$ 1.45 $\to$ 1.82** (+65.5% gain).
   - Point out how the exploration parameter $c$ anneals from $1.41$ down to $0.80$.

---

## Part 3: Codelab 3 — The AI Scientist 2 (Automated Discovery & MASS)

### Visual Layout:
- **00:00 - 02:00**: Full-Screen Slides (`03_ai_scientist_2_slides.md`).
- **02:00 - 04:30**: Split-Screen: Slide on Left, Terminal on Right.
- **04:30 - 08:00**: Full-Screen Terminal (`python 03_ai_scientist_2/demo.py`).
- **08:00 - 08:45**: Full-Screen Slide 9 (Winning Artifact).

### Choreography Steps:
1. **[00:00] Scientific Inquiry Introduction**: Contrast raw code mutations with the formal scientific method.
2. **[03:15] Multi-Era Ablation**: Point out the stress ablation table. Emphasize that naive momentum suffered a **46.08% drawdown in the Crisis era**!
3. **[04:30] Live Terminal Execution**:
   - Clear terminal screen.
   - Run the command:
     ```bash
     python 03_ai_scientist_2/demo.py
     ```
4. **[05:15] Peer Review Rejection**:
   - Highlight the Peer Review Scorecard: `Score: 6.25 / 10.0 [REJECT]`.
   - Read the committee's critique verbatim.
5. **[06:00] MASS Distillation**:
   - Explain how the MASS loop extracts critique lessons and saves them to persistent memory priors.
6. **[07:00] Winning Discovery Artifact**:
   - Showcase the final accepted discovery: *Cross-Asset Volatility Skew Transmission & Safe-Haven Rebalancing*.
   - Point to the Peer Review Committee score: **8.00 / 10.00 [ACCEPT]**!

---

## Part 4: Codelab 4 — Comparative Benchmark & Grand Synthesis

### Visual Layout:
- **00:00 - 03:15**: Full-Screen Slides (`04_comparative_benchmark_slides.md`).
- **03:15 - 05:45**: Slide 4 (The Pareto Frontier Chart).
- **05:45 - 07:30**: Slide 7 (The Unified Super-RSI Architecture) and Final Sign-Off.

### Choreography Steps:
1. **[00:00] Unified Comparison Setup**: Emphasize that all three methods competed on the exact same 756-day dataset with 10 bps friction.
2. **[01:50] Architectural Comparison Matrix**: Compare latency (55.6ms vs 8.21s vs 0.30s) and peak Sharpe (1.19 vs 1.82 vs 0.50).
3. **[03:15] The Pareto Frontier Walkthrough**:
   - Use cursor to trace the Pareto frontier curve.
   - Explain when an engineer should choose Dream-RSI (ultra-low latency), AIDE2 (peak code performance), or The AI Scientist 2 (explainable scientific grounding).
4. **[05:45] The Unified Vision**:
   - Walk through the three-tier Super-RSI diagram: AI Scientist at top, AIDE2 in middle, Dream-RSI at execution.
5. **[07:00] Outro & Call to Action**:
   - Direct viewers to check the repository, star the project on GitHub, and inspect `WALKTHROUGH.md` and `README.md`.
   - Deliver clear, confident sign-off.
