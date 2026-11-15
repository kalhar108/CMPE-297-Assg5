"""The AI Scientist 2: Automated Scientific Discovery & Multi-Agent Self-Supervision (MASS).

Codelab 3 of the Recursive Self-Improvement (RSI) Course Map (Section 10).
Applied to Quantitative Feature Discovery and Algorithmic Trading Signal Engineering.

Provides:
- Hypothesis, HypothesisGenerator: Template-free quantitative and economic anomaly formulators.
- ExperimentTrial, ExperimentManager: Automated progressive experiment managers.
- MultiEraAblation: Multi-era stress ablation evaluating across Bull, Choppy, and Crisis regimes.
- ReviewerAssessment, PeerReviewBoard: Simulated multi-agent academic peer review committee.
- MASSLoop: Multi-Agent Self-Supervision feedback loop distilling review critique into research priors.
"""

from __future__ import annotations

# Handle relative vs absolute imports cleanly
try:
    from .ablation import MultiEraAblation
    from .experimenter import ExperimentManager, ExperimentTrial
    from .hypothesis import Hypothesis, HypothesisGenerator
    from .mass_loop import MASSLoop
    from .peer_review import PeerReviewBoard, ReviewerAssessment
except ImportError:
    from ablation import MultiEraAblation  # type: ignore
    from experimenter import ExperimentManager, ExperimentTrial  # type: ignore
    from hypothesis import Hypothesis, HypothesisGenerator  # type: ignore
    from mass_loop import MASSLoop  # type: ignore
    from peer_review import PeerReviewBoard, ReviewerAssessment  # type: ignore

__all__ = [
    "Hypothesis",
    "HypothesisGenerator",
    "ExperimentTrial",
    "ExperimentManager",
    "MultiEraAblation",
    "ReviewerAssessment",
    "PeerReviewBoard",
    "MASSLoop",
]
