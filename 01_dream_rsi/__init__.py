"""Codelab 1: Dream-RSI (World Models & Off-Policy Dreaming).

Demonstrates off-policy recursive self-improvement where an agent:
1. Logs real exploration search trees into an immutable replay buffer.
2. Calibrates an Empirical Regime Transition Model (ERTM) world model.
3. Simulates counterfactual exploration policies via off-policy dreaming at zero model cost.
4. Optimizes macro search hyperparameters (branching factor B, depth D, stopping threshold tau).
5. Redeploys the improved macro policy online, achieving higher Sharpe ratio and compute reduction.
"""

try:
    from .dream_engine import (
        DreamEngine,
        MacroPolicy,
    )
    from .policy_optimizer import (
        OptimizationResult,
        PolicyOptimizer,
    )
    from .tree_logger import (
        DiscoveryNode,
        DiscoveryTreeLogger,
    )
    from .world_model import (
        RegimeWorldModel,
    )
except (ImportError, ValueError):
    from dream_engine import (
        DreamEngine,
        MacroPolicy,
    )
    from policy_optimizer import (
        OptimizationResult,
        PolicyOptimizer,
    )
    from tree_logger import (
        DiscoveryNode,
        DiscoveryTreeLogger,
    )
    from world_model import (
        RegimeWorldModel,
    )

# Semantic alias
ERTMWorldModel = RegimeWorldModel

__all__ = [
    # Discovery Tree Logging
    "DiscoveryNode",
    "DiscoveryTreeLogger",
    # World Model
    "RegimeWorldModel",
    "ERTMWorldModel",
    # Off-Policy Dreaming Engine
    "MacroPolicy",
    "DreamEngine",
    # Policy Optimization
    "PolicyOptimizer",
    "OptimizationResult",
]
