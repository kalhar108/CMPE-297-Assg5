"""Codelab 2: AIDE2 (Code Tree MCTS & Bi-Level Meta-Optimization).

Demonstrates Monte Carlo Tree Search (MCTS) over the space of executable trading strategies
coupled with atomic coding operators (Drafting, Debugging, Refinement) and bi-level
recursive self-improvement.
"""

from .mcts_search import AIDE2Search
from .meta_harness import MetaHarness
from .operators import DebuggingOperator, DraftingOperator, RefinementOperator
from .tree_node import CodeDiff, CodeTree, StrategyNode

__all__ = [
    "StrategyNode",
    "CodeDiff",
    "CodeTree",
    "DraftingOperator",
    "DebuggingOperator",
    "RefinementOperator",
    "AIDE2Search",
    "MetaHarness",
]
