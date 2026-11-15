"""Tripartite Recursive Self-Improvement (RSI) Base Abstractions.

Defines the core formal objects established in Theme 09 and Section 10 of the RSI curriculum:
1. TaskSolver (S): Executes/solves the domain problem for a given candidate solution.
2. Improver (I): Proposes modifications to candidate solutions based on feedback.
3. MetaImprover (M): Proposes modifications to the Improver procedure itself (second-order recursion).
4. Evaluator (E): Fixed, authoritative ground-truth validation mechanism guarding against reward hacking.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")  # Candidate solution type (e.g., BaseStrategy or code string)
R = TypeVar("R")  # Evaluation result type (e.g., BacktestResult)
I = TypeVar("I", bound="Improver[Any]")  # Improver type for MetaImprover


@dataclass
class GenerationRecord(Generic[T, R]):
    """Immutable snapshot recording a single candidate evaluation in an RSI loop."""
    generation: int
    candidate: T
    result: R
    fitness_score: float
    improver_name: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskSolver(Generic[T, R], ABC):
    """Abstract Task Solver (S).

    Solves the domain problem or executes a candidate against domain tasks,
    producing an evaluation result R.
    """

    def __init__(self, name: str = "BaseTaskSolver", generation: int = 0) -> None:
        self.name = name
        self.generation = generation

    @abstractmethod
    def solve(self, candidate: T, **kwargs: Any) -> R:
        """Executes candidate T and returns domain result R."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}' gen={self.generation}>"


class Improver(Generic[T], ABC):
    """Abstract Improver (I).

    Proposes candidate modification T_t -> T_{t+1} based on evaluation feedback.
    This represents first-order self-improvement.
    """

    def __init__(self, name: str = "BaseImprover", generation: int = 0) -> None:
        self.name = name
        self.generation = generation
        self.mutation_count: int = 0

    @abstractmethod
    def improve(self, candidate: T, feedback: Any, **kwargs: Any) -> T:
        """Proposes a modified candidate T_{t+1} guided by feedback."""
        pass

    def record_mutation(self) -> None:
        """Tracks number of mutations performed by this improver."""
        self.mutation_count += 1

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}' gen={self.generation} mutations={self.mutation_count}>"


class MetaImprover(Generic[I], ABC):
    """Abstract Meta-Improver (M).

    Proposes improvements to the improver procedure itself: I_0 -> I_1.
    This represents second-order (recursive) self-improvement:
    improving the improvement procedure.
    """

    def __init__(self, name: str = "BaseMetaImprover", generation: int = 0) -> None:
        self.name = name
        self.generation = generation

    @abstractmethod
    def meta_improve(self, improver: I, history: List[GenerationRecord[Any, Any]], **kwargs: Any) -> I:
        """Proposes an improved improver instance I_{t+1} based on trajectory history."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}' gen={self.generation}>"


class Evaluator(Generic[T, R], ABC):
    """Abstract Ground-Truth Evaluator (E).

    Fixed, authoritative evaluation mechanism that scores candidate fitness.
    Crucially, E remains immutable and external to the agent's self-modifications,
    guarding against capability collapse and reward hacking.
    """

    def __init__(self, name: str = "GroundTruthEvaluator") -> None:
        self.name = name
        self.evaluation_count: int = 0

    @abstractmethod
    def evaluate(self, candidate: T) -> R:
        """Computes authoritative evaluation result R for candidate T."""
        pass

    @abstractmethod
    def extract_fitness(self, result: R) -> float:
        """Extracts a scalar fitness score from result R (higher is better)."""
        pass

    def compare(self, result_a: R, result_b: R) -> float:
        """Compares two results. Returns positive if result_a > result_b, negative if worse."""
        return self.extract_fitness(result_a) - self.extract_fitness(result_b)

    def record_eval(self) -> None:
        """Increments total evaluation calls."""
        self.evaluation_count += 1

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}' total_evals={self.evaluation_count}>"


class RSIHistory(Generic[T, R]):
    """Tracks generational history, trajectory progression, and Pareto efficiency."""

    def __init__(self) -> None:
        self.records: List[GenerationRecord[T, R]] = []

    def append(self, record: GenerationRecord[T, R]) -> None:
        """Appends a new generation record."""
        self.records.append(record)

    def best_record(self) -> Optional[GenerationRecord[T, R]]:
        """Returns the record with highest fitness score."""
        if not self.records:
            return None
        return max(self.records, key=lambda r: r.fitness_score)

    def initial_record(self) -> Optional[GenerationRecord[T, R]]:
        """Returns the initial baseline record (Generation 0)."""
        if not self.records:
            return None
        return self.records[0]

    def improvement_delta(self) -> float:
        """Calculates fitness improvement from initial baseline to best candidate."""
        if len(self.records) < 2:
            return 0.0
        init_fit = self.records[0].fitness_score
        best_fit = self.best_record().fitness_score  # type: ignore[union-attr]
        return best_fit - init_fit

    def progression_summary(self) -> Dict[str, Any]:
        """Returns a high-level summary dictionary of the RSI trajectory."""
        best = self.best_record()
        init = self.initial_record()
        return {
            "total_generations": len(self.records),
            "initial_fitness": init.fitness_score if init else 0.0,
            "best_fitness": best.fitness_score if best else 0.0,
            "improvement_delta": self.improvement_delta(),
            "best_generation": best.generation if best else 0,
            "best_improver": best.improver_name if best else "none",
        }

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)
