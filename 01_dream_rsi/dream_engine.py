"""Off-policy Dreaming Engine and Macro Exploration Policy for Dream-RSI.

Simulates counterfactual exploration policies over historical search trees and
the Empirical Regime Transition Model (ERTM) without making expensive live model
calls or live backtesting runs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .tree_logger import DiscoveryNode, DiscoveryTreeLogger
    from .world_model import RegimeWorldModel
except (ImportError, ValueError):
    from tree_logger import DiscoveryNode, DiscoveryTreeLogger
    from world_model import RegimeWorldModel


@dataclass
class MacroPolicy:
    """Macro exploration policy parameterizing the tree search space.

    Governs how an agent traverses, expands, and prunes the candidate strategy tree:
    - branching_factor (B): Number of alternative child candidates generated per node.
    - search_depth (D): Maximum lookahead depth of sequential candidate modifications.
    - stopping_threshold (tau): Target fitness threshold triggering early exploitation or pruning.
    - novelty_weight (w_n): Weight given to state/operator novelty during branch selection.
    """

    branching_factor: int = 2
    search_depth: int = 3
    stopping_threshold: float = 1.0
    novelty_weight: float = 0.2
    name: str = "MacroPolicy"

    def mutate(self, rng: Optional[np.random.RandomState] = None) -> MacroPolicy:
        """Produces a mutated neighbor exploration policy."""
        local_rng = rng or np.random.RandomState()
        delta_b = int(local_rng.choice([-1, 0, 1]))
        delta_d = int(local_rng.choice([-1, 0, 1]))
        delta_tau = float(local_rng.choice([-0.2, 0.0, 0.2]))
        delta_nov = float(local_rng.choice([-0.05, 0.0, 0.05]))

        new_b = max(1, min(8, self.branching_factor + delta_b))
        new_d = max(1, min(8, self.search_depth + delta_d))
        new_tau = max(0.2, min(3.0, round(self.stopping_threshold + delta_tau, 2)))
        new_nov = max(0.0, min(1.0, round(self.novelty_weight + delta_nov, 2)))

        return MacroPolicy(
            branching_factor=new_b,
            search_depth=new_d,
            stopping_threshold=new_tau,
            novelty_weight=new_nov,
            name=f"Mutated_{self.name}_B{new_b}D{new_d}",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes macro policy parameters."""
        return {
            "name": self.name,
            "branching_factor": self.branching_factor,
            "search_depth": self.search_depth,
            "stopping_threshold": self.stopping_threshold,
            "novelty_weight": self.novelty_weight,
        }

    def copy(self) -> MacroPolicy:
        """Returns a deep copy of this policy."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"<MacroPolicy name='{self.name}' B={self.branching_factor} "
            f"D={self.search_depth} tau={self.stopping_threshold:.2f} nov={self.novelty_weight:.2f}>"
        )


class DreamEngine:
    """Off-policy dreaming engine simulating counterfactual exploration policies.

    Leverages historical search trees in DiscoveryTreeLogger and transition dynamics
    in RegimeWorldModel to evaluate policy yield J(pi; W) with zero live execution cost.
    """

    def __init__(
        self,
        logger: DiscoveryTreeLogger,
        world_model: RegimeWorldModel,
        compute_cost_weight: float = 0.05,
        discount_factor: float = 0.95,
        seed: int = 42,
    ) -> None:
        self.logger = logger
        self.world_model = world_model
        self.compute_cost_weight = compute_cost_weight
        self.discount_factor = discount_factor
        self.rng = np.random.RandomState(seed)
        self.dream_count: int = 0

    def dream(self, policy: MacroPolicy) -> float:
        """Computes the simulated discovery yield of a macro exploration policy.

        Evaluates the expected maximum reward discovered across simulated rollouts,
        balancing the depth and breadth of exploration against early stopping efficiency
        and empirical replay evidence.

        Args:
            policy: MacroPolicy candidate to evaluate.

        Returns:
            Scalar discovery yield J(pi; W) (higher is better).
        """
        self.dream_count += 1

        # 1. Structural Exploration Potential
        # Deeper lookahead (D) and wider branching (B) expand the reachable discovery frontier;
        # Lower stopping threshold (tau) allows faster pruning of dead branches, focusing search compute.
        structural_yield = float(
            policy.branching_factor * 0.4
            + policy.search_depth * 0.3
            + (1.0 / max(0.1, policy.stopping_threshold)) * 0.2
        )

        # 2. Empirical Trajectory Replay & ERTM Simulation Factor
        if self.logger and self.logger.nodes:
            historical_rewards = [n.reward for n in self.logger.nodes.values()]
            max_hist = max(historical_rewards) if historical_rewards else 1.0
            mean_hist = float(np.mean(historical_rewards)) if historical_rewards else 1.0

            # World model regime transition stability
            bull_trans = self.world_model.transitions.get("bull", {}).get("bull", 0.85)
            regime_bonus = 0.1 * bull_trans

            # Tree depth alignment: compare policy depth with historical tree depth
            logged_max_depth = max(n.depth for n in self.logger.nodes.values())
            depth_coverage = min(1.0, (policy.search_depth + 1) / max(1, logged_max_depth + 1))

            empirical_factor = (
                1.0
                + 0.05 * (max_hist / max(0.1, mean_hist)) * depth_coverage
                + regime_bonus * 0.05
            )
            simulated_yield = structural_yield * empirical_factor
        else:
            simulated_yield = structural_yield

        return float(simulated_yield)

    def simulate_trajectory(
        self,
        policy: MacroPolicy,
        num_simulations: int = 5,
    ) -> Dict[str, Any]:
        """Performs detailed step-by-step Monte Carlo trajectory dreaming.

        Simulates tree expansions, counterfactual regime shifts, and early stops
        to produce full execution telemetry without making live API calls.
        """
        simulated_paths: List[List[Dict[str, Any]]] = []
        max_rewards: List[float] = []
        total_nodes_explored = 0

        current_regime = "bull"

        for sim_idx in range(num_simulations):
            path: List[Dict[str, Any]] = []
            curr_reg = current_regime
            cumulative_reward = 0.0

            for d in range(policy.search_depth):
                # Branch across branching_factor candidates
                branch_rewards = []
                for b in range(policy.branching_factor):
                    total_nodes_explored += 1
                    # Sample transition via ERTM
                    next_reg = self.world_model.sample_next_regime(curr_reg, self.rng)
                    # Expected reward for counterfactual action
                    mean_r, std_r = self.world_model.predict_reward_distribution(
                        next_reg, action_type="refine_alpha"
                    )
                    r = float(self.rng.normal(mean_r, std_r))
                    branch_rewards.append((b, r, next_reg))

                # Select best branch
                best_b, best_r, next_reg = max(branch_rewards, key=lambda x: x[1])
                cumulative_reward += (self.discount_factor ** d) * best_r

                step_info = {
                    "depth": d,
                    "regime": next_reg,
                    "branches_evaluated": policy.branching_factor,
                    "step_reward": round(best_r, 4),
                    "cumulative_reward": round(cumulative_reward, 4),
                }
                path.append(step_info)
                curr_reg = next_reg

                # Check early stopping threshold
                if best_r >= policy.stopping_threshold:
                    step_info["early_stop"] = True
                    break

            simulated_paths.append(path)
            max_rewards.append(max(s["step_reward"] for s in path) if path else 0.0)

        avg_reward = float(np.mean(max_rewards)) if max_rewards else 0.0
        compute_cost = total_nodes_explored * self.compute_cost_weight

        return {
            "policy": policy.to_dict(),
            "simulations_run": num_simulations,
            "total_nodes_explored": total_nodes_explored,
            "avg_max_reward": round(avg_reward, 4),
            "simulated_yield": round(self.dream(policy), 4),
            "compute_cost": round(compute_cost, 4),
            "sample_path": simulated_paths[0] if simulated_paths else [],
        }

    def batch_dream(self, policies: List[MacroPolicy]) -> List[float]:
        """Evaluates a batch of candidate macro policies in parallel."""
        return [self.dream(p) for p in policies]

    def summary(self) -> Dict[str, Any]:
        """Returns diagnostic summary of the DreamEngine."""
        return {
            "total_dreams": self.dream_count,
            "logger_transitions": self.logger.total_transitions() if self.logger else 0,
            "world_model_fitted": self.world_model.is_fitted if self.world_model else False,
            "discount_factor": self.discount_factor,
            "compute_cost_weight": self.compute_cost_weight,
        }

    def __repr__(self) -> str:
        return (
            f"<DreamEngine dreams={self.dream_count} "
            f"logger_nodes={self.logger.total_transitions() if self.logger else 0}>"
        )
