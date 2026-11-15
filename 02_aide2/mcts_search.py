"""Monte Carlo Tree Search (MCTS) Engine over Strategy Code Space for AIDE2.

Provides:
- AIDE2Search: Full MCTS implementation using Upper Confidence Bound for Trees (UCT),
  dynamic branch pruning, adaptive backtracking from dead ends, and tripartite operator chaining.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from common.domain.backtester import BacktestEngine, BacktestResult
from common.domain.market_data import MarketData, generate_multi_era_market_data
from common.domain.strategy_base import DynamicCodeStrategy
from common.llm_provider import BaseModelProvider, get_llm_provider

try:
    from .operators import DebuggingOperator, DraftingOperator, RefinementOperator
    from .tree_node import CodeTree, StrategyNode
except (ImportError, ValueError):
    from operators import DebuggingOperator, DraftingOperator, RefinementOperator
    from tree_node import CodeTree, StrategyNode

logger = logging.getLogger(__name__)


class AIDE2Search:
    """Monte Carlo Tree Search engine over executable Python trading strategy code space."""

    def __init__(
        self,
        c_param: float = 1.414,
        max_depth: int = 8,
        pruning_threshold: float = -50.0,
        provider: Optional[BaseModelProvider] = None,
        drafter: Optional[DraftingOperator] = None,
        debugger: Optional[DebuggingOperator] = None,
        refiner: Optional[RefinementOperator] = None,
        backtest_engine: Optional[BacktestEngine] = None,
        market_data: Optional[MarketData] = None,
    ) -> None:
        self.c_param = float(c_param)
        self.max_depth = max_depth
        self.pruning_threshold = pruning_threshold

        self.provider = provider or get_llm_provider()
        self.drafter = drafter or DraftingOperator(self.provider)
        self.debugger = debugger or DebuggingOperator(self.provider)
        self.refiner = refiner or RefinementOperator(self.provider)

        self.backtest_engine = backtest_engine or BacktestEngine(default_fee_bps=10.0)
        self.market_data = market_data or generate_multi_era_market_data(seed=42, num_days=756)

        self.tree = CodeTree()
        # Direct reference for backwards compatibility with tests expecting search.nodes
        self.nodes = self.tree.nodes
        self.root_id: Optional[str] = None
        self.step_count: int = 0
        self.pruned_nodes: List[str] = []

    def uct_score(self, node: StrategyNode, parent_visits: int) -> float:
        """Computes Upper Confidence Bound for Trees (UCT) score.
        
        Formula:
            UCT(s) = Q(s) / N(s) + c * sqrt(ln(max(1, N(parent))) / N(s))
        
        Special cases:
            - If node.visits == 0: returns float('inf') ensuring unvisited nodes are explored first.
            - If parent_visits == 0: avoids math domain error on log(0) and returns pure exploitation.
        """
        if node.visits == 0:
            return float("inf")

        exploitation = node.value
        exploration = self.c_param * math.sqrt(
            math.log(max(1, parent_visits)) / node.visits
        )
        return exploitation + exploration

    def add_node(self, node: StrategyNode) -> None:
        """Registers a strategy node into the search tree."""
        self.tree.add_node(node)
        if self.root_id is None:
            self.root_id = node.node_id

    def evaluate_node(self, node: StrategyNode) -> BacktestResult:
        """Compiles and backtests the strategy code in an isolated execution sandbox."""
        try:
            dynamic_strat = DynamicCodeStrategy(node.code)
            result = self.backtest_engine.run(dynamic_strat, self.market_data)
            node.metrics = result
            return result
        except Exception as e:
            logger.debug("Evaluation error for node %s: %s", node.node_id, e)
            failed_result = BacktestResult(
                sharpe_ratio=-999.0,
                max_drawdown=1.0,
                sortino_ratio=-999.0,
                calmar_ratio=-999.0,
                cagr=-1.0,
                turnover=0.0,
                total_cost=0.0,
                equity_curve=[self.backtest_engine.initial_capital],
                daily_returns=[],
                weights_history=[],
                execution_time_ms=0.0,
                success=False,
                error_message=f"{type(e).__name__}: {str(e)}",
            )
            node.metrics = failed_result
            return failed_result

    def compute_reward(self, result: BacktestResult) -> float:
        """Computes scalar reward for backtest performance balancing Sharpe and Drawdown."""
        if not result.success:
            return -10.0
        # Institutional objective: reward Sharpe while penalizing extreme drawdown
        return float(result.sharpe_ratio - 0.5 * result.max_drawdown)

    def select(self) -> StrategyNode:
        """Traverses the tree using UCT selection with adaptive backtracking from dead ends."""
        if self.root_id is None or self.root_id not in self.nodes:
            raise ValueError("MCTS tree has no root node. Add a root node before selection.")

        curr = self.nodes[self.root_id]

        while curr.children_ids and curr.depth < self.max_depth:
            # Filter non-terminal/unpruned children
            active_children = [
                self.nodes[cid] for cid in curr.children_ids
                if cid in self.nodes and not self.nodes[cid].is_terminal
            ]

            if not active_children:
                # All children are dead ends; mark curr as terminal and backtrack to parent
                curr.is_terminal = True
                if curr.parent_id and curr.parent_id in self.nodes:
                    curr = self.nodes[curr.parent_id]
                    continue
                else:
                    return curr

            # Pick child with highest UCT score
            best_child = max(
                active_children,
                key=lambda child: self.uct_score(child, parent_visits=curr.visits),
            )

            # If best child is unvisited, select it for immediate rollout/expansion
            if best_child.visits == 0:
                return best_child

            curr = best_child

        return curr

    def expand(
        self,
        node: StrategyNode,
        operator_type: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> StrategyNode:
        """Applies an operator to spawn a new child strategy node."""
        self.step_count += 1
        new_node_id = f"node_{self.step_count}"

        # Determine operator if unspecified
        if operator_type is None:
            if node.metrics is not None and not getattr(node.metrics, "success", True):
                operator_type = "debug"
            elif node.operator == "root" and not node.children_ids:
                operator_type = "draft"
            else:
                operator_type = "refine"

        # Apply operator
        if operator_type == "draft":
            new_code = self.drafter.draft(instruction or "Multi-asset quantitative trend alpha")
            diff_desc = "Drafted initial candidate strategy"
        elif operator_type == "debug":
            error_trace = getattr(node.metrics, "traceback_log", "") or getattr(node.metrics, "error_message", "") or "ZeroDivisionError"
            new_code = self.debugger.debug_and_fix(node.code, error_trace)
            diff_desc = f"Repaired error: {getattr(node.metrics, 'error_message', 'traceback bug')}"
        else:  # refine
            new_code = self.refiner.refine(node.code, instruction=instruction)
            diff_desc = "Refined strategy parameters and risk filters"

        # Initialize child node with initial placeholder metrics
        child = StrategyNode(
            node_id=new_node_id,
            code=new_code,
            metrics=None,
            parent_id=node.node_id,
            operator=operator_type,
            diff_summary=diff_desc,
            depth=node.depth + 1,
            generation=node.generation,
        )

        self.add_node(child)
        return child

    def backpropagate(self, node: StrategyNode, reward: float) -> None:
        """Propagates evaluation reward up the tree to the root."""
        curr: Optional[StrategyNode] = node
        while curr is not None:
            curr.update(reward)
            curr = self.nodes.get(curr.parent_id) if curr.parent_id else None

    def prune_dead_ends(self, threshold: Optional[float] = None) -> List[str]:
        """Prunes branches with severe underperformance or catastrophic drawdowns."""
        thresh = threshold if threshold is not None else self.pruning_threshold
        pruned_ids: List[str] = []

        for node_id, node in list(self.nodes.items()):
            if node_id == self.root_id:
                continue
            if node.metrics is not None and node.sharpe < thresh and not node.is_terminal:
                pruned = self.tree.prune_branch(node_id)
                pruned_ids.extend(pruned)

        self.pruned_nodes.extend(pruned_ids)
        return pruned_ids

    def step(self) -> StrategyNode:
        """Executes one full MCTS cycle: Selection -> Expansion -> Evaluation -> Backpropagation."""
        selected_node = self.select()

        # If selected node was already evaluated and has children, expand a new child
        if selected_node.visits > 0:
            child = self.expand(selected_node)
            eval_target = child
        else:
            eval_target = selected_node

        # Evaluate candidate
        eval_result = self.evaluate_node(eval_target)
        reward = self.compute_reward(eval_result)

        # Backpropagate reward
        self.backpropagate(eval_target, reward)

        # Auto-prune catastrophic dead-ends
        if not eval_result.success or eval_result.sharpe_ratio < self.pruning_threshold:
            eval_target.is_terminal = True

        return eval_target

    def search(self, iterations: int = 5) -> StrategyNode:
        """Runs MCTS search for a specified number of iterations and returns the best strategy found."""
        for _ in range(iterations):
            self.step()

        best = self.tree.best_node()
        if best is None:
            if self.root_id and self.root_id in self.nodes:
                return self.nodes[self.root_id]
            raise RuntimeError("MCTS search finished with empty tree.")
        return best

    def get_tree_summary(self) -> Dict[str, Any]:
        """Returns structured metrics summarizing tree search exploration."""
        best = self.tree.best_node()
        return {
            "total_nodes": len(self.nodes),
            "root_id": self.root_id,
            "best_node_id": best.node_id if best else None,
            "best_sharpe": best.sharpe if best else 0.0,
            "best_mdd": best.max_drawdown if best else 1.0,
            "pruned_nodes_count": len(self.pruned_nodes),
            "max_depth_explored": max((n.depth for n in self.nodes.values()), default=0),
        }
