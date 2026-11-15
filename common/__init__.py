"""Shared Core Foundation for the Recursive Self-Improvement (RSI) Suite.

Provides:
- RSIExecutionConfig & environment sensing (mock vs live).
- Dual-mode LLM provider interface (Google Gemini & Deterministic Mock).
- Tripartite RSI abstractions: TaskSolver, Improver, MetaImprover, Evaluator.
- Quantitative Alpha Domain: 756-day multi-era market data, vectorized backtester, metrics, strategies.
"""

from .config import (
    ExecutionMode,
    RSIExecutionConfig,
)
from .llm_provider import (
    BaseModelProvider,
    DeterministicMockProvider,
    GeminiLiveProvider,
    get_llm_provider,
)
from .rsi_base import (
    Evaluator,
    GenerationRecord,
    Improver,
    MetaImprover,
    RSIHistory,
    TaskSolver,
)

__all__ = [
    # Config
    "ExecutionMode",
    "RSIExecutionConfig",
    # LLM Providers
    "BaseModelProvider",
    "DeterministicMockProvider",
    "GeminiLiveProvider",
    "get_llm_provider",
    # Tripartite RSI Abstractions
    "TaskSolver",
    "Improver",
    "MetaImprover",
    "Evaluator",
    "GenerationRecord",
    "RSIHistory",
]
