---
marp: true
theme: default
paginate: true
header: "Codelab 1: Dream-RSI | World Models & Off-Policy Dreaming"
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

# Codelab 1: Dream-RSI
## World Models, Replay Simulators & Off-Policy Macro Dreaming

**Course**: Section 10 — State-of-the-Art RSI Paradigms  
**Module**: `01_dream_rsi/`  
**Key Principle**: Optimizing exploration policies in imagination without live LLM calls  
**Execution Runtime**: Sub-60ms offline execution

---

# The Problem: The High Cost of Live Search

- **Standard Search Problem**:
  - LLM agents exploring search spaces must invoke live model APIs and execute live environment trials for every proposed branch.
  - Costly in compute, API quotas, and wall-clock latency ($1$–$5$ seconds per branch).
- **The Dream-RSI Solution**:
  - Ground discovery in an **Empirical World Model** learned from observed transitions.
  - Decouple policy evaluation from live execution: **simulate alternative search paths inside the world model** ("dreaming").
  - Only redeploy to the live environment once the macro policy is optimized!

---

# Dream-RSI System Architecture

```
+---------------------------------------------------------------------------------+
|                         Dream-RSI 4-Phase Lifecycle                             |
+---------------------------------------------------------------------------------+
                                         |
     [Phase 1] Online Deployment         |  - Executes initial baseline policy
               (Generation 0)            |  - Logs all transitions & backtests
                                         v
     [Phase 2] World Model Calibration   |  - Calibrates ERTM transition matrix
               & Trajectory Indexing     |  - Indexes replay buffer graph
                                         v
     [Phase 3] Off-Policy Dreaming       |  - Simulates counterfactual macro policies
               (In Imagination)          |  - Evaluates yield without live API calls
                                         v
     [Phase 4] Online Redeployment       |  - Deploys optimized macro policy online
               (Generation 1)            |  - Measurable recursive Sharpe progression
```

---

# Component 1: Discovery Tree Logger

Implemented in `01_dream_rsi/tree_logger.py`:

```python
class DiscoveryTreeLogger:
    def log_transition(
        self,
        node_id: str,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Dict[str, Any],
        parent_id: Optional[str] = None
    ) -> TrajectoryNode:
        """Records state, action, backtested reward, and tree topology."""
        node = TrajectoryNode(node_id, state, action, reward, next_state, parent_id)
        self.nodes[node_id] = node
        return node
```

- Creates an **immutable directed acyclic graph (DAG)** of all explored strategy nodes.
- Preserves both high-performing branches and negative discovery dead-ends.

---

# Component 2: Empirical Regime Transition Model (ERTM)

Implemented in `01_dream_rsi/world_model.py`:

- Quantifies the environment's empirical dynamics across regimes:
$$P(s_{t+1} = \text{Regime}_j \mid s_t = \text{Regime}_i)$$

### Calibrated ERTM Matrix (756 Trading Days):
```
+------------+---------------+---------------+----------+
| From \ To  | Bull          | Choppy        | Crisis   |
+------------+---------------+---------------+----------+
| Bull       |      42.86%   |      42.86%   | 14.29%   |
| Choppy     |      33.33%   |      33.33%   | 33.33%   |
| Crisis     |      33.33%   |      33.33%   | 33.33%   |
+------------+---------------+---------------+----------+
```
- Provides the probabilistic transition foundation for simulated rollouts.

---

# Component 3: Macro Exploration Policy

Implemented in `01_dream_rsi/policy_optimizer.py`:

```python
@dataclass
class MacroPolicy:
    name: str = "MacroPolicy"
    branching_factor: int = 2      # B: Parallel candidates per expansion
    search_depth: int = 2          # D: Lookahead depth in mutation space
    stopping_threshold: float = 1.5# tau: Sharpe threshold for early exit
    novelty_weight: float = 0.20   # w_n: Bonus for exploring under-visited regimes
```

- Controls **how** the first-order search operates.
- Improving this policy is a **second-order meta-improvement**: $\mathcal{M}(I) = I'$.

---

# Component 4: The Off-Policy Dream Engine

Implemented in `01_dream_rsi/dream_engine.py`:

```python
class DreamEngine:
    def dream(self, policy: MacroPolicy) -> float:
        """Simulates macro policy yield over the historical replay graph."""
        total_simulated_yield = 0.0
        for rollout in range(policy.branching_factor):
            simulated_depth = min(policy.search_depth, self.tree_max_depth)
            prob_regime = self.world_model.predict_transition_prob(self.current_regime)
            node_reward = self.sample_historical_node_reward(simulated_depth)
            
            # Simulated reward weighted by regime transition dynamics
            total_simulated_yield += node_reward * prob_regime * (1.0 + policy.novelty_weight)
        return total_simulated_yield
```

- Evaluates alternative candidate policies in **0.09 milliseconds**!
- Zero network requests, zero LLM tokens consumed.

---

# Demonstration: Phase 1 & 2 Execution

Let's inspect the live demo output:

```
[PHASE 1] GENERATION 0: Baseline Exploration Policy Online Execution
------------------------------------------------------------------------
[*] Deployed Gen 0 Macro Exploration Policy: <MacroPolicy B=2 D=2 tau=1.50 nov=0.20>
    Searching strategy candidate space using Gen 0 branching (B=2, D=2)...
    Completed 4 online backtests in Gen 0.
    Gen 0 Best Strategy: MomentumFactor (Sharpe: 0.4856, MDD: 31.31%)

[PHASE 2] WORLD MODEL CALIBRATION & REPLAY BUFFER LOGGING
------------------------------------------------------------------------
[*] ERTM World Model calibrated from 756 trading days.
[*] Replay Buffer indexed: 4 transitions stored.
```

- Baseline Sharpe is modest ($0.4856$), with high drawdown ($31.31\%$) during the Crisis regime.

---

# Demonstration: Phase 3 Dreaming Optimization

```
[PHASE 3] OFF-POLICY DREAMING: Optimizing Macro Policy in Imagination
------------------------------------------------------------------------
    Simulating counterfactual exploration policies over the replay simulator...
+---------------------------------------------------------+
|            Macro Exploration Policy Optimization        |
+-------------------------+---------------+---------------+
| Parameter               | Initial (Gen 0) | Best (Gen 1)  |
+-------------------------+---------------+---------------+
| Branching Factor (B)    |             1 |             5 |
| Search Depth (D)        |             1 |             5 |
| Stopping Threshold (tau)|          2.00 |          0.80 |
| Novelty Weight (w_n)    |          0.20 |          0.20 |
| Simulated Dream Yield   |        0.8794 |        4.3042 |
| Yield Improvement Delta |               |       +3.4248 |
+-------------------------+---------------+---------------+
    Dream Optimization Wall Time: 0.09ms (Zero live API/backtest cost)
    Promoting Best Macro Policy: Deep_Aggressive -> Generation 1
```

- The Dream Engine discovers that deeper exploration ($D=5, B=5$) yields $+3.42$ simulated yield.

---

# Demonstration: Phase 4 & 5 Progression

```
[PHASE 5] VERIFIED RECURSIVE SELF-IMPROVEMENT PROGRESSION
========================================================================
 Generation 0 Baseline Sharpe  :   0.4856 (Drawdown: 31.31%)
 Generation 1 Optimized Sharpe :   1.1943 (Drawdown: 11.59%)
 Net Sharpe Improvement Delta  :  +0.7087 (POSITIVE PROGRESSION)
 Maximum Drawdown Reduction    :   +19.72% (Reduced Tail Risk)
 Search Compute Reduction      :     50.0% (4 evals -> 2 evals)
 Total Replay Buffer Nodes     :        6 transitions
 Total Wall-Clock Execution    :    55.59ms
========================================================================
```

- **Net Sharpe Improvement**: $+0.7087$ (more than doubled!).
- **Risk Reduction**: Max drawdown slashed from $31.3\%$ to $11.6\%$.
- **Compute Efficiency**: Achieved with $50\%$ fewer live online evaluations!

---

# The Discovery Tree Topology

```
=== Discovery Tree: DreamRSI_ReplayBuffer (Total Nodes: 6) ===
[*] Root [node_g0_root] reward=0.163
  +-- [node_g0_b1] r=0.282 | shorten_lookback_5_20 (depth=1)
  +-- [node_g0_b2] r=0.486 | cross_sectional_momentum (depth=1)
  |   \-- [node_g0_b2_child] r=-0.264 | concentrate_top1_lookback10 (depth=2)
  \-- [node_g1_root] r=1.194 | optimized_macro_search (depth=1)
      \-- [node_g1_refined] r=1.194 | reduce_cash_buffer_5pct (depth=2)
```

- Notice how Gen 1 immediately explores high-reward topology branch `node_g1_root` ($r=1.194$), pruning negative branch `node_g0_b2_child` ($r=-0.264$).

---

# Key Takeaways: Codelab 1 (Dream-RSI)

1. **Decoupled Optimization**:
   - Live interaction builds the world model; dream simulation optimizes the search procedure.
2. **Zero-Cost Imagination**:
   - Counterfactual exploration runs in sub-millisecond memory speed, avoiding LLM token waste.
3. **Verified Self-Improvement**:
   - The agent demonstrated measurable, reproducible Sharpe and Drawdown gains.

*Next up: Codelab 2 — AIDE2 and Code-Space Tree Search!*
