"""Discovery Tree Logger and Immutable Replay Buffer for Dream-RSI.

Logs exploration trajectories, state visits, actions, rewards, and tree topologies
into an immutable replay buffer. This forms the empirical foundation for offline
world-model training and counterfactual policy dreaming.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class DiscoveryNode:
    """Represents a discrete node in the exploration discovery tree."""

    node_id: str
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Dict[str, Any]
    depth: int = 0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_leaf(self) -> bool:
        """Returns True if this node has no child branches."""
        return len(self.children_ids) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the discovery node to a dictionary."""
        return {
            "node_id": self.node_id,
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        return (
            f"<DiscoveryNode id='{self.node_id}' depth={self.depth} "
            f"reward={self.reward:.4f} children={len(self.children_ids)}>"
        )


class DiscoveryTreeLogger:
    """Logs online exploration trajectories and builds an immutable replay buffer.

    Enables off-policy dreaming by indexing past state transitions, action selections,
    and ground-truth backtest fitness evaluations into an offline queryable graph.
    """

    def __init__(self, name: str = "AlphaDiscoveryLogger") -> None:
        self.name = name
        self.nodes: Dict[str, DiscoveryNode] = {}
        self.replay_buffer: List[Dict[str, Any]] = []

    def log_transition(
        self,
        node_id: str,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Dict[str, Any],
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DiscoveryNode:
        """Records an exploration transition into the discovery tree and replay buffer.

        Args:
            node_id: Unique string identifier for the discovery node.
            state: Pre-action state representation (e.g. regime, strategy params, current metrics).
            action: Action executed (e.g. parameter mutation, filter addition, operator type).
            reward: Scalar fitness reward (e.g. backtest annualized Sharpe ratio).
            next_state: Post-action state representation.
            parent_id: Optional parent node ID representing the branch source.
            metadata: Additional contextual telemetry (e.g. execution wall time, AST diff).

        Returns:
            The recorded DiscoveryNode instance.
        """
        node = DiscoveryNode(
            node_id=node_id,
            state=state,
            action=action,
            reward=float(reward),
            next_state=next_state,
            parent_id=parent_id,
            metadata=metadata or {},
        )

        if parent_id and parent_id in self.nodes:
            parent_node = self.nodes[parent_id]
            node.depth = parent_node.depth + 1
            if node_id not in parent_node.children_ids:
                parent_node.children_ids.append(node_id)
        else:
            node.depth = 0

        self.nodes[node_id] = node

        # Append to immutable replay buffer
        self.replay_buffer.append(node.to_dict())
        return node

    def get_node(self, node_id: str) -> Optional[DiscoveryNode]:
        """Retrieves a node by its unique identifier."""
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> List[DiscoveryNode]:
        """Returns all direct child nodes branched from the given node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]

    def get_path_to_root(self, node_id: str) -> List[DiscoveryNode]:
        """Traces the lineage path from the specified node back to the root."""
        path: List[DiscoveryNode] = []
        curr = self.nodes.get(node_id)
        visited = set()
        while curr is not None and curr.node_id not in visited:
            path.append(curr)
            visited.add(curr.node_id)
            curr = self.nodes.get(curr.parent_id) if curr.parent_id else None
        return list(reversed(path))

    def get_root_nodes(self) -> List[DiscoveryNode]:
        """Returns all nodes with depth 0 (no parent or missing parent)."""
        return [n for n in self.nodes.values() if n.depth == 0 or n.parent_id is None]

    def get_leaf_nodes(self) -> List[DiscoveryNode]:
        """Returns all terminal nodes in the current tree topology."""
        return [n for n in self.nodes.values() if n.is_leaf()]

    def best_node(self) -> Optional[DiscoveryNode]:
        """Returns the discovery node with the highest empirical reward."""
        if not self.nodes:
            return None
        return max(self.nodes.values(), key=lambda n: n.reward)

    def max_depth(self) -> int:
        """Returns the maximum depth reached in the exploration tree."""
        if not self.nodes:
            return 0
        return max(n.depth for n in self.nodes.values())

    def total_transitions(self) -> int:
        """Returns the total number of recorded exploration steps."""
        return len(self.nodes)

    def get_replay_buffer(self) -> List[Dict[str, Any]]:
        """Returns a snapshot of the immutable replay buffer."""
        return list(self.replay_buffer)

    def export_replay_buffer(self) -> List[Dict[str, Any]]:
        """Exports the complete serialized replay buffer."""
        return [n.to_dict() for n in self.nodes.values()]

    def export_json(self, filepath: str) -> None:
        """Persists the replay buffer to a JSON file."""
        data = {
            "name": self.name,
            "total_nodes": len(self.nodes),
            "max_depth": self.max_depth(),
            "best_reward": self.best_node().reward if self.best_node() else 0.0,
            "nodes": [n.to_dict() for n in self.nodes.values()],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_json(self, filepath: str) -> None:
        """Restores tree state from a JSON export."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for raw in data.get("nodes", []):
            node = DiscoveryNode(
                node_id=raw["node_id"],
                state=raw["state"],
                action=raw["action"],
                reward=float(raw["reward"]),
                next_state=raw["next_state"],
                depth=raw.get("depth", 0),
                parent_id=raw.get("parent_id"),
                children_ids=raw.get("children_ids", []),
                timestamp=raw.get("timestamp", datetime.now(timezone.utc).isoformat()),
                metadata=raw.get("metadata", {}),
            )
            self.nodes[node.node_id] = node
            self.replay_buffer.append(node.to_dict())

    def render_ascii_tree(self, root_id: Optional[str] = None) -> str:
        """Renders an ASCII visualization of the discovery tree topology."""
        roots = [self.nodes[root_id]] if (root_id and root_id in self.nodes) else self.get_root_nodes()
        if not roots:
            return "[Empty Discovery Tree]"

        lines: List[str] = [f"=== Discovery Tree: {self.name} (Total Nodes: {len(self.nodes)}) ==="]

        def _traverse(node: DiscoveryNode, prefix: str, is_tail: bool):
            connector = "\\-- " if is_tail else "+-- "
            action_desc = node.action.get("action", node.action.get("type", "step"))
            lines.append(
                f"{prefix}{connector}[{node.node_id}] "
                f"r={node.reward:.3f} | {action_desc} (depth={node.depth})"
            )
            children = self.get_children(node.node_id)
            for idx, child in enumerate(children):
                next_prefix = prefix + ("    " if is_tail else "|   ")
                _traverse(child, next_prefix, idx == len(children) - 1)

        for r in roots:
            lines.append(f"[*] Root [{r.node_id}] reward={r.reward:.3f}")
            children = self.get_children(r.node_id)
            for idx, child in enumerate(children):
                _traverse(child, "  ", idx == len(children) - 1)

        return "\n".join(lines)

    def summary(self) -> Dict[str, Any]:
        """Provides a statistical summary of the exploration tree."""
        rewards = [n.reward for n in self.nodes.values()] if self.nodes else [0.0]
        return {
            "name": self.name,
            "total_nodes": len(self.nodes),
            "max_depth": self.max_depth(),
            "best_reward": max(rewards),
            "min_reward": min(rewards),
            "avg_reward": sum(rewards) / max(len(rewards), 1),
            "root_count": len(self.get_root_nodes()),
            "leaf_count": len(self.get_leaf_nodes()),
        }

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} name='{self.name}' "
            f"nodes={len(self.nodes)} max_depth={self.max_depth()}>"
        )
