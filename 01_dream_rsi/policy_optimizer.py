"""Macro Exploration Policy Optimizer for Dream-RSI.

Tunes macro search hyperparameters—branching factor (B), search depth (D),
stopping threshold (tau), and novelty weight (w_n)—via off-policy dream evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .dream_engine import DreamEngine, MacroPolicy
except (ImportError, ValueError):
    from dream_engine import DreamEngine, MacroPolicy


@dataclass
class OptimizationResult:
    """Encapsulates the outcome of a macro policy optimization run."""

    best_policy: MacroPolicy
    best_yield: float
    initial_policy: MacroPolicy
    initial_yield: float
    yield_delta: float
    evaluated_candidates: List[Tuple[MacroPolicy, float]] = field(default_factory=list)
    iterations: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """Returns a high-level summary of the optimization results."""
        return {
            "initial_policy": self.initial_policy.to_dict(),
            "initial_yield": round(self.initial_yield, 4),
            "best_policy": self.best_policy.to_dict(),
            "best_yield": round(self.best_yield, 4),
            "yield_delta": round(self.yield_delta, 4),
            "total_evaluated": len(self.evaluated_candidates),
            "iterations": self.iterations,
        }

    def summary_table(self) -> str:
        """Emits an ASCII table summarizing the policy optimization."""
        return (
            "+---------------------------------------------------------+\n"
            "|            Macro Exploration Policy Optimization        |\n"
            "+-------------------------+---------------+---------------+\n"
            "| Parameter               | Initial (Gen 0) | Best (Gen 1)  |\n"
            "+-------------------------+---------------+---------------+\n"
            f"| Branching Factor (B)    | {self.initial_policy.branching_factor:>13} | {self.best_policy.branching_factor:>13} |\n"
            f"| Search Depth (D)        | {self.initial_policy.search_depth:>13} | {self.best_policy.search_depth:>13} |\n"
            f"| Stopping Threshold (tau)| {self.initial_policy.stopping_threshold:>13.2f} | {self.best_policy.stopping_threshold:>13.2f} |\n"
            f"| Novelty Weight (w_n)    | {self.initial_policy.novelty_weight:>13.2f} | {self.best_policy.novelty_weight:>13.2f} |\n"
            f"| Simulated Dream Yield   | {self.initial_yield:>13.4f} | {self.best_yield:>13.4f} |\n"
            f"| Yield Improvement Delta |               | {self.yield_delta:>+13.4f} |\n"
            "+-------------------------+---------------+---------------+"
        )


class PolicyOptimizer:
    """Optimizes macro search policies via offline dream simulations.

    Finds the optimal balance between breadth, depth, and early pruning
    to maximize discovery rate when redeployed online.
    """

    def __init__(
        self,
        dream_engine: DreamEngine,
        search_space: Optional[Dict[str, List[Any]]] = None,
        seed: int = 42,
    ) -> None:
        self.dream_engine = dream_engine
        self.search_space = search_space or {
            "branching_factor": [1, 2, 3, 4, 5],
            "search_depth": [1, 2, 3, 4, 5],
            "stopping_threshold": [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
            "novelty_weight": [0.0, 0.1, 0.2, 0.4],
        }
        self.rng = np.random.RandomState(seed)

    def evaluate_policy(self, policy: MacroPolicy) -> float:
        """Evaluates a single macro policy via dream simulation."""
        return self.dream_engine.dream(policy)

    def optimize_grid(
        self,
        candidates: Optional[List[MacroPolicy]] = None,
    ) -> OptimizationResult:
        """Evaluates candidate policies across a specified list or structured grid.

        Args:
            candidates: Optional explicit list of candidate policies. If None,
                        constructs a representative grid across parameter ranges.

        Returns:
            OptimizationResult containing best policy and evaluation records.
        """
        if candidates is None:
            candidates = [
                MacroPolicy(branching_factor=1, search_depth=1, stopping_threshold=2.0, name="Conservative"),
                MacroPolicy(branching_factor=2, search_depth=2, stopping_threshold=1.5, name="Baseline_G0"),
                MacroPolicy(branching_factor=3, search_depth=3, stopping_threshold=1.2, name="Balanced"),
                MacroPolicy(branching_factor=4, search_depth=4, stopping_threshold=1.0, name="Growth"),
                MacroPolicy(branching_factor=5, search_depth=5, stopping_threshold=0.8, name="Aggressive_Frontier"),
            ]

        if not candidates:
            raise ValueError("Candidate policy list cannot be empty.")

        evaluated: List[Tuple[MacroPolicy, float]] = []
        for p in candidates:
            score = self.evaluate_policy(p)
            evaluated.append((p, score))

        best_policy, best_yield = max(evaluated, key=lambda x: x[1])
        init_policy = candidates[0]
        init_yield = evaluated[0][1]

        return OptimizationResult(
            best_policy=best_policy,
            best_yield=best_yield,
            initial_policy=init_policy,
            initial_yield=init_yield,
            yield_delta=best_yield - init_yield,
            evaluated_candidates=evaluated,
            iterations=len(candidates),
        )

    def optimize_evolutionary(
        self,
        initial_policy: Optional[MacroPolicy] = None,
        generations: int = 4,
        population_size: int = 6,
    ) -> OptimizationResult:
        """Evolves macro exploration policies using generational mutation and selection.

        Args:
            initial_policy: Starting policy (Generation 0). Defaults to standard baseline.
            generations: Number of evolutionary refinement generations.
            population_size: Number of candidate policies per generation.

        Returns:
            OptimizationResult tracking the evolutionary progression.
        """
        curr_policy = initial_policy or MacroPolicy(
            branching_factor=2, search_depth=2, stopping_threshold=1.5, name="Gen0_Baseline"
        )
        init_yield = self.evaluate_policy(curr_policy)

        population: List[MacroPolicy] = [curr_policy]
        # Seed initial diverse population
        for i in range(population_size - 1):
            population.append(curr_policy.mutate(self.rng))

        all_evaluated: List[Tuple[MacroPolicy, float]] = []
        history: List[Dict[str, Any]] = []

        best_overall_policy = curr_policy
        best_overall_yield = init_yield

        for gen in range(generations):
            gen_scores = [(p, self.evaluate_policy(p)) for p in population]
            all_evaluated.extend(gen_scores)

            # Rank population by simulated discovery yield
            gen_scores.sort(key=lambda x: x[1], reverse=True)
            top_policy, top_yield = gen_scores[0]

            if top_yield > best_overall_yield:
                best_overall_policy = top_policy
                best_overall_yield = top_yield

            history.append({
                "generation": gen,
                "best_policy": top_policy.to_dict(),
                "best_yield": round(top_yield, 4),
                "avg_yield": round(float(np.mean([s[1] for s in gen_scores])), 4),
            })

            # Next generation: retain elite top 2, mutate rest
            elites = [p for p, _ in gen_scores[:2]]
            next_pop = list(elites)
            while len(next_pop) < population_size:
                parent = elites[int(self.rng.choice(len(elites)))]
                mutated = parent.mutate(self.rng)
                mutated.name = f"Gen{gen + 1}_Policy_{len(next_pop)}"
                next_pop.append(mutated)

            population = next_pop

        return OptimizationResult(
            best_policy=best_overall_policy,
            best_yield=best_overall_yield,
            initial_policy=curr_policy,
            initial_yield=init_yield,
            yield_delta=best_overall_yield - init_yield,
            evaluated_candidates=all_evaluated,
            iterations=generations * population_size,
            history=history,
        )

    def find_pareto_frontier(
        self,
        candidates: List[MacroPolicy],
    ) -> List[MacroPolicy]:
        """Identifies non-dominated policies balancing discovery yield vs compute cost (B * D)."""
        scored = []
        for p in candidates:
            y = self.evaluate_policy(p)
            cost = p.branching_factor * p.search_depth
            scored.append((p, y, cost))

        pareto: List[MacroPolicy] = []
        for i, (p_i, y_i, c_i) in enumerate(scored):
            is_dominated = False
            for j, (p_j, y_j, c_j) in enumerate(scored):
                if i != j:
                    if y_j >= y_i and c_j <= c_i and (y_j > y_i or c_j < c_i):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto.append(p_i)

        return pareto
