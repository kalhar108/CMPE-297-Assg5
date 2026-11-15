"""Strategy Code Tree Data Structure for AIDE2.

Provides:
- StrategyNode: Tree node encapsulating complete executable strategy code,
  backtest telemetry, parent/child links, visit count (N), reward sum (Q),
  and AST inspection.
- CodeDiff: Represents mutation/patch operations connecting tree nodes.
- CodeTree: Complete tree graph managing node lineage, branch pruning,
  and ASCII/JSON visualization.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class CodeDiff:
    """Represents a code transformation edge connecting a parent node to a child node."""

    parent_id: Optional[str]
    child_id: str
    operator: str  # e.g., 'draft', 'debug', 'refine'
    description: str
    unified_diff: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "operator": self.operator,
            "description": self.description,
            "diff_preview": self.unified_diff[:300] if self.unified_diff else "",
            "timestamp": self.timestamp,
        }


@dataclass
class StrategyNode:
    """A node in the AIDE2 code exploration tree.
    
    Each node represents a complete, executable Python trading strategy script,
    along with its validation telemetry, MCTS statistics (visits N, total value Q),
    and parent/child lineage.
    """

    node_id: str
    code: str
    metrics: Any  # BacktestResult or Dict containing backtest metrics
    parent_id: Optional[str] = None
    visits: int = 0
    total_value: float = 0.0
    children_ids: List[str] = field(default_factory=list)
    operator: str = "root"
    diff_summary: str = ""
    depth: int = 0
    generation: int = 0
    is_terminal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> float:
        """Returns mean exploitation value Q / N (0.0 if unvisited)."""
        if self.visits > 0:
            return self.total_value / self.visits
        return 0.0

    @property
    def q_value(self) -> float:
        """Alias for self.value."""
        return self.value

    @property
    def visit_count(self) -> int:
        """Alias for self.visits."""
        return self.visits

    @property
    def sharpe(self) -> float:
        """Extracts the Sharpe ratio from node metrics if available."""
        if self.metrics is None:
            return 0.0
        if hasattr(self.metrics, "sharpe_ratio"):
            return float(self.metrics.sharpe_ratio)
        if isinstance(self.metrics, dict) and "sharpe_ratio" in self.metrics:
            return float(self.metrics["sharpe_ratio"])
        return 0.0

    @property
    def max_drawdown(self) -> float:
        """Extracts the maximum drawdown from node metrics if available."""
        if self.metrics is None:
            return 1.0
        if hasattr(self.metrics, "max_drawdown"):
            return float(self.metrics.max_drawdown)
        if isinstance(self.metrics, dict) and "max_drawdown" in self.metrics:
            return float(self.metrics["max_drawdown"])
        return 1.0

    def update(self, reward: float) -> None:
        """Updates MCTS visit count and accumulated reward value."""
        self.visits += 1
        self.total_value += float(reward)

    def code_hash(self) -> str:
        """Computes deterministic SHA-256 hash of normalized code for deduplication."""
        normalized = "\n".join(
            line.strip() for line in self.code.splitlines() if line.strip()
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def compute_diff(self, parent_code: str) -> str:
        """Computes unified diff against parent code."""
        parent_lines = parent_code.splitlines(keepends=True)
        child_lines = self.code.splitlines(keepends=True)
        diff = list(
            difflib.unified_diff(
                parent_lines,
                child_lines,
                fromfile=f"parent_{self.parent_id or 'none'}.py",
                tofile=f"child_{self.node_id}.py",
                n=3,
            )
        )
        return "".join(diff)

    def get_ast(self) -> Optional[ast.AST]:
        """Parses the node's code into an Abstract Syntax Tree."""
        try:
            return ast.parse(self.code)
        except Exception:
            return None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes node metadata and telemetry to a dictionary."""
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "visits": self.visits,
            "total_value": round(self.total_value, 4),
            "mean_value": round(self.value, 4),
            "sharpe_ratio": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "operator": self.operator,
            "diff_summary": self.diff_summary,
            "depth": self.depth,
            "generation": self.generation,
            "is_terminal": self.is_terminal,
            "children_count": len(self.children_ids),
            "code_hash": self.code_hash(),
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"<StrategyNode id='{self.node_id}' op='{self.operator}' "
            f"N={self.visits} Q={self.value:.3f} Sharpe={self.sharpe:.2f}>"
        )


class CodeTree:
    """Hierarchical graph data structure managing strategy nodes and mutation edges."""

    def __init__(self, root: Optional[StrategyNode] = None) -> None:
        self.nodes: Dict[str, StrategyNode] = {}
        self.edges: List[CodeDiff] = []
        self.root_id: Optional[str] = None
        self.best_node_id: Optional[str] = None

        if root is not None:
            self.add_node(root)

    def add_node(self, node: StrategyNode) -> None:
        """Registers a node in the tree and updates parent/child linkages."""
        self.nodes[node.node_id] = node

        if self.root_id is None:
            self.root_id = node.node_id

        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.node_id not in parent.children_ids:
                parent.children_ids.append(node.node_id)
            node.depth = parent.depth + 1

            # Record mutation edge
            edge = CodeDiff(
                parent_id=node.parent_id,
                child_id=node.node_id,
                operator=node.operator,
                description=node.diff_summary or f"Applied {node.operator} operator",
                unified_diff=node.compute_diff(parent.code),
            )
            self.edges.append(edge)

        # Update best node tracker
        if self.best_node_id is None:
            self.best_node_id = node.node_id
        else:
            current_best = self.nodes[self.best_node_id]
            if node.sharpe > current_best.sharpe:
                self.best_node_id = node.node_id

    def get_node(self, node_id: str) -> Optional[StrategyNode]:
        """Retrieves a node by identifier."""
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> List[StrategyNode]:
        """Returns all direct children of the given node."""
        parent = self.nodes.get(node_id)
        if parent is None:
            return []
        return [self.nodes[cid] for cid in parent.children_ids if cid in self.nodes]

    def get_lineage(self, node_id: str) -> List[StrategyNode]:
        """Returns the ancestry path from root to the given node."""
        path: List[StrategyNode] = []
        curr: Optional[StrategyNode] = self.nodes.get(node_id)
        while curr is not None:
            path.append(curr)
            curr = self.nodes.get(curr.parent_id) if curr.parent_id else None
        return list(reversed(path))

    def best_node(self) -> Optional[StrategyNode]:
        """Returns the node with highest Sharpe ratio in the tree."""
        if not self.nodes:
            return None
        return max(self.nodes.values(), key=lambda n: n.sharpe)

    def prune_branch(self, node_id: str) -> List[str]:
        """Marks a node and all of its descendants as terminal/pruned dead ends.
        
        Returns list of pruned node IDs.
        """
        pruned: List[str] = []
        to_visit = [node_id]

        while to_visit:
            curr_id = to_visit.pop()
            if curr_id in self.nodes:
                node = self.nodes[curr_id]
                node.is_terminal = True
                pruned.append(curr_id)
                to_visit.extend(node.children_ids)

        return pruned

    def render_ascii(self) -> str:
        """Renders an ASCII visualization of the strategy tree topology."""
        if not self.root_id or self.root_id not in self.nodes:
            return "[Empty Tree]"

        lines: List[str] = []

        def _traverse(node_id: str, prefix: str = "", is_last: bool = True) -> None:
            node = self.nodes[node_id]
            connector = "\\-- " if is_last else "+-- "
            status = "[DEAD-END]" if node.is_terminal else ""
            best_mark = " (BEST)" if node_id == self.best_node_id else ""
            line = (
                f"{prefix}{connector}[{node.node_id}] op={node.operator} "
                f"N={node.visits} Q={node.value:.2f} Sharpe={node.sharpe:.2f} "
                f"{status}{best_mark}"
            )
            lines.append(line)

            new_prefix = prefix + ("    " if is_last else "|   ")
            children = [cid for cid in node.children_ids if cid in self.nodes]
            for idx, cid in enumerate(children):
                _traverse(cid, new_prefix, idx == len(children) - 1)

        _traverse(self.root_id)
        return "\n".join(lines)

    def export_json(self) -> str:
        """Serializes entire tree topology to JSON string."""
        payload = {
            "root_id": self.root_id,
            "best_node_id": self.best_node_id,
            "total_nodes": len(self.nodes),
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
        }
        return json.dumps(payload, indent=2)

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes.values())
