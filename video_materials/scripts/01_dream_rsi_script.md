# Narration Script: Codelab 1 — Dream-RSI
## World Models, Replay Simulators & Off-Policy Macro Dreaming

**Target Slide Deck**: `video_materials/slides/01_dream_rsi_slides.md`  
**Video Target Duration**: 7 minutes 45 seconds  
**Tone**: Technical, focused, demonstrative, energetic  
**Formatting Keys**: `[SLIDE: X]` = Advance slide; `[PAUSE]` = 1-2 second breath/pacing; `[TERMINAL]` = Live demo command; `[HIGHLIGHT: ...]` = On-screen emphasis

---

### [00:00 - 00:50] Introduction to Dream-RSI

[SLIDE: 1 - Title Slide: Codelab 1 Dream-RSI]
"Welcome back! In this video, we are diving deep into Codelab 1: **Dream-RSI**. [PAUSE]

One of the greatest bottlenecks in modern agentic search is the cost of live exploration. Every time an agent decides to try a new search path, it typically makes multiple API calls to a large language model and executes heavy environment trials. If we want an agent to explore hundreds of alternative search strategies, doing so live would burn through API budgets and take hours of wall-clock time. [PAUSE]

Dream-RSI solves this through a brilliant insight borrowed from world models in reinforcement learning: **what if the agent could dream?**"

---

### [00:50 - 01:45] The 4-Phase Dream-RSI Architecture

[SLIDE: 2 - The Problem: High Cost of Live Search]
"Instead of testing every search heuristic live, Dream-RSI splits the self-improvement lifecycle into four distinct phases: [PAUSE]

[SLIDE: 3 - Dream-RSI System Architecture]
1. **Phase 1: Online Baseline Deployment**. The agent executes an initial baseline search policy in the live environment, recording every observation and result.
2. **Phase 2: World Model Calibration**. The agent builds an Empirical Regime Transition Model and logs the trajectory into an immutable replay buffer.
3. **Phase 3: Off-Policy Macro Dreaming**. Here is where the magic happens: the agent simulates dozens of counterfactual exploration policies entirely *inside its imagination*, optimizing its macro search hyperparameters with zero API cost.
4. **Phase 4: Online Redeployment**. The agent promotes the winning macro policy and deploys it back online, achieving higher Sharpe ratios and lower drawdowns with fewer live evaluations."

---

### [01:45 - 03:00] Deep Dive: The Four Core Code Components

[SLIDE: 4 - Component 1: Discovery Tree Logger]
"Let's look at the source code in `01_dream_rsi/`. [PAUSE]

First, in `tree_logger.py`, we implement the `DiscoveryTreeLogger`. Every time an action is taken, the logger records a structured `TrajectoryNode` containing the regime state, the action taken, the backtested reward, and pointers to parent nodes. This forms an immutable directed graph representing our search history. [PAUSE]

[SLIDE: 5 - Component 2: Empirical Regime Transition Model (ERTM)]
Second, in `world_model.py`, we have the `RegimeWorldModel`. It analyzes the 756 days of market data and computes the empirical regime transition probabilities: the chance that a Bull market transitions to Choppy, or Choppy to Crisis. This gives the agent a calibrated surrogate model of market dynamics. [PAUSE]

[SLIDE: 6 - Component 3: Macro Exploration Policy]
Third, in `policy_optimizer.py`, we parameterize the search procedure itself into a `MacroPolicy`. It governs four critical levers:
- **Branching Factor ($B$)**: How many parallel strategies to spawn.
- **Search Depth ($D$)**: How deep to explore along promising branches.
- **Stopping Threshold ($\tau$)**: When to terminate exploration.
- **Novelty Weight ($w_n$)**: How much bonus to assign to under-explored regimes."

---

### [03:00 - 04:15] Component 4: The Off-Policy Dream Engine

[SLIDE: 7 - Component 4: The Off-Policy Dream Engine]
"Now, let's look at `dream_engine.py`. This is where off-policy dreaming occurs. [PAUSE]

The `DreamEngine` takes candidate macro policies and simulates their expected yield over historical trajectory subgraphs. Because it uses the ERTM transition matrix to weight rewards across regimes, it can accurately predict which macro policy will find the best alpha strategies. [PAUSE]

And look at the execution speed: simulating an entire tree of candidate policies takes **just 0.09 milliseconds**! That's tens of thousands of times faster than making live LLM calls, with zero token consumption."

---

### [04:15 - 06:00] Live Code Demonstration

[TERMINAL: Run `python 01_dream_rsi/demo.py`]
"Let's see Dream-RSI run live in the terminal! [PAUSE]

[SLIDE: 8 - Demonstration: Phase 1 & 2 Execution]
In Phase 1, our Generation 0 baseline policy starts with conservative branching: $B=2, D=2$. It runs four online backtests, discovering a baseline momentum strategy with a Sharpe ratio of **0.4856** and a Maximum Drawdown of **31.31%**.

In Phase 2, the ERTM World Model is calibrated, showing a 42.8% probability of remaining in Bull and a 14.3% chance of transitioning directly to Crisis. Four transitions are logged into the Replay Buffer. [PAUSE]

[SLIDE: 9 - Demonstration: Phase 3 Dreaming Optimization]
Now watch Phase 3: the Dreaming Loop!
The agent tests multiple candidate policies. It discovers that an aggressive deep search policy with $B=5, D=5$ and an early stopping threshold of $0.80$ increases simulated dream yield from **0.8794 to 4.3042**—a massive $+3.42$ yield improvement delta!
And the wall-clock time for this optimization? **0.09 milliseconds**!"

---

### [06:00 - 07:15] Verified Generational Progression

[SLIDE: 10 - Demonstration: Phase 4 & 5 Progression]
"In Phase 4, the agent redeploys this winning policy online for Generation 1. [PAUSE]

Look at the verified progression summary:
- **Baseline Gen 0 Sharpe**: 0.4856
- **Optimized Gen 1 Sharpe**: **1.1943**!
- That is a net Sharpe improvement of **+0.7087**! [HIGHLIGHT: Net Sharpe +0.7087]
- Maximum Drawdown dropped from **31.31% down to 11.59%**—a 19.7% tail-risk reduction!
- And because the macro policy had an optimized stopping threshold, it found this winning strategy in only 2 evaluations instead of 4—a **50% compute reduction**! [PAUSE]

[SLIDE: 11 - The Discovery Tree Topology]
Here on screen is the final Discovery Tree. You can see how Generation 1 immediately branched into the high-reward node `node_g1_root` with reward $1.194$, completely ignoring the negative branches that bogged down Generation 0."

---

### [07:15 - 07:45] Conclusion & Next Steps

[SLIDE: 12 - Key Takeaways: Codelab 1]
"To wrap up Codelab 1: Dream-RSI proves that an autonomous agent can decouple exploration policy optimization from live API costs. By dreaming counterfactual search paths inside a surrogate world model, the agent achieved more than double the risk-adjusted return while halving search compute. [PAUSE]

Join us in the next video, where we dive into **Codelab 2: AIDE2**, exploring program synthesis as Monte Carlo Tree Search and surgical AST debugging!"
