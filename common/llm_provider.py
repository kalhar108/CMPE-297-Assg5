"""Dual-Mode LLM Provider abstraction for the RSI Suite.

Supports live Google Gemini generation via the modern `google-genai` SDK (v1.59.0)
and 100% deterministic, offline mock execution returning authentic,
schema-valid data structures without external network dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, get_type_hints

from pydantic import BaseModel

from .config import RSIExecutionConfig
from .mock_data import (
    MOCK_DEBUG_REPAIRS,
    MOCK_DRAFT_STRATEGIES,
    MOCK_HYPOTHESES,
    MOCK_PEER_REVIEWS,
    MOCK_REFINEMENTS,
    get_mock_hypothesis,
    get_mock_peer_review,
    get_mock_repair,
    get_mock_strategy,
)

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class BaseModelProvider(ABC):
    """Abstract base class for all RSI LLM providers."""

    @abstractmethod
    def generate_text(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generates plain text response synchronously."""
        pass

    @abstractmethod
    def generate_structured(
        self, prompt: str, schema: Type[T], system_instruction: Optional[str] = None
    ) -> T:
        """Generates a structured object validated against a Pydantic schema."""
        pass

    @abstractmethod
    def mutate_code(
        self, code: str, instruction: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Mutates or repairs a Python code string according to instructions and context."""
        pass


class GeminiLiveProvider(BaseModelProvider):
    """Live LLM provider backed by Google Gemini via the `google-genai` SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.8-flash",
        temperature: float = 0.2,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("API_KEY")
        )
        if not self.api_key:
            raise ValueError("GeminiLiveProvider requires GOOGLE_API_KEY or GEMINI_API_KEY.")
        self.model_name = model_name
        self.temperature = temperature

        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system_instruction,
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    def generate_structured(
        self, prompt: str, schema: Type[T], system_instruction: Optional[str] = None
    ) -> T:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=schema,
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        raw_text = response.text or "{}"
        return schema.model_validate_json(raw_text)

    def mutate_code(
        self, code: str, instruction: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        system_instruction = (
            "You are an expert quantitative finance and algorithmic trading software engineer. "
            "Modify the provided Python strategy code precisely according to the instruction. "
            "Emit ONLY executable Python code, without markdown explanation or chat commentary."
        )
        context_str = f"\nContext/Diagnostics: {json.dumps(context)}" if context else ""
        prompt = (
            f"Instruction: {instruction}\n"
            f"{context_str}\n\n"
            f"Existing Code:\n```python\n{code}\n```\n\n"
            f"Provide the complete revised Python code:"
        )
        raw = self.generate_text(prompt, system_instruction=system_instruction)
        # Strip markdown fences if present
        cleaned = re.sub(r"^```python\s*", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
        return cleaned.strip()


class DeterministicMockProvider(BaseModelProvider):
    """100% offline, deterministic mock provider for testing and offline execution."""

    def __init__(self, config: Optional[RSIExecutionConfig] = None) -> None:
        self.config = config or RSIExecutionConfig(mode="mock")
        self.call_count = 0

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        self.call_count += 1
        p_lower = prompt.lower()

        # Intent: Drafting
        if "draft" in p_lower or "synthesize" in p_lower or "generate strategy" in p_lower:
            strat = get_mock_strategy(self.call_count)
            return (
                f"### Strategy Proposal: {strat['strategy_name']}\n\n"
                f"**Rationale**: {strat['rationale']}\n"
                f"**Formula**: {strat['formula']}\n\n"
                f"```python\n{strat['code']}\n```"
            )

        # Intent: Debugging / Repair
        if "debug" in p_lower or "traceback" in p_lower or "zerodivision" in p_lower or "error" in p_lower:
            repair = get_mock_repair("ZeroDivisionError")
            return (
                f"### Diagnostic Report\n\n"
                f"{repair['diagnostics']}\n\n"
                f"**Applied Patch**:\n```python\n{repair['repaired_code_snippet']}\n```"
            )

        # Intent: Refinement
        if "refine" in p_lower or "tune" in p_lower or "optimize" in p_lower:
            refinement = MOCK_REFINEMENTS[self.call_count % len(MOCK_REFINEMENTS)]
            return (
                f"### Refinement Report: {refinement['refinement_type']}\n\n"
                f"**Optimization**: {refinement['description']}\n\n"
                f"```python\n{refinement['code_patch']}\n```"
            )

        # Intent: Hypothesis Formulation
        if "hypothesis" in p_lower or "conjecture" in p_lower or "research idea" in p_lower:
            hypo = get_mock_hypothesis(self.call_count)
            return (
                f"### Research Hypothesis: {hypo['title']}\n\n"
                f"**Claim**: {hypo['claim']}\n\n"
                f"**Mathematical Formulation**: {hypo['mathematical_formulation']}\n\n"
                f"**Economic Rationale**: {hypo['economic_rationale']}"
            )

        # Intent: Peer Review
        if "peer review" in p_lower or "review" in p_lower or "critique" in p_lower:
            pkg = get_mock_peer_review()
            r1 = pkg["reviews"][0]
            return (
                f"### Multi-Agent Peer Review Evaluation\n\n"
                f"**Recommendation**: {r1['overall_recommendation']}\n"
                f"**Soundness**: {r1['soundness_score']}/10 | **Novelty**: {r1['novelty_score']}/10\n"
                f"**Risk Discipline**: {r1['risk_discipline_score']}/10\n\n"
                f"**Critique**: {r1['critique']}"
            )

        return (
            f"[DeterministicMockProvider] Synthesized authentic domain response for step #{self.call_count}."
        )

    def generate_structured(
        self, prompt: str, schema: Type[T], system_instruction: Optional[str] = None
    ) -> T:
        self.call_count += 1
        p_lower = prompt.lower()
        fields = getattr(schema, "model_fields", {})

        # 1. Match Hypothesis Schema
        if "title" in fields and ("claim" in fields or "hypothesis" in fields):
            hypo = get_mock_hypothesis(self.call_count)
            payload: Dict[str, Any] = {}
            for field_name in fields:
                if field_name in hypo:
                    payload[field_name] = hypo[field_name]
                elif field_name == "title":
                    payload[field_name] = hypo["title"]
                elif field_name in ("claim", "description", "statement"):
                    payload[field_name] = hypo["claim"]
                elif field_name in ("formula", "mathematical_formulation"):
                    payload[field_name] = hypo["mathematical_formulation"]
                elif field_name == "rationale":
                    payload[field_name] = hypo["economic_rationale"]
                elif field_name == "regime_behavior":
                    payload[field_name] = hypo["expected_regime_behavior"]
                else:
                    payload[field_name] = self._synthesize_field_default(fields[field_name])
            return schema.model_validate(payload)

        # 2. Match Peer Review / Scorecard Schema
        if "soundness_score" in fields or "overall_recommendation" in fields or "composite_score" in fields:
            review = MOCK_PEER_REVIEWS[self.call_count % len(MOCK_PEER_REVIEWS)]
            payload = {}
            for field_name in fields:
                if field_name in review:
                    payload[field_name] = review[field_name]
                elif field_name == "composite_score":
                    payload[field_name] = float(review.get("soundness_score", 8.5))
                elif field_name in ("summary", "critique"):
                    payload[field_name] = review.get("critique", "Methodology is statistically sound.")
                else:
                    payload[field_name] = self._synthesize_field_default(fields[field_name])
            return schema.model_validate(payload)

        # 3. Match Strategy Proposal Schema
        if "strategy_name" in fields or "code" in fields:
            strat = get_mock_strategy(self.call_count)
            payload = {}
            for field_name in fields:
                if field_name in strat:
                    payload[field_name] = strat[field_name]
                elif field_name in ("name", "title"):
                    payload[field_name] = strat["strategy_name"]
                elif field_name in ("code", "source_code", "script"):
                    payload[field_name] = strat["code"]
                elif field_name == "params":
                    payload[field_name] = {"fast_window": 12, "slow_window": 26}
                else:
                    payload[field_name] = self._synthesize_field_default(fields[field_name])
            return schema.model_validate(payload)

        # 4. General fallback: construct model with intelligent typed defaults
        payload = {}
        for field_name, field_info in fields.items():
            payload[field_name] = self._synthesize_field_default(field_info)
        return schema.model_validate(payload)

    def _synthesize_field_default(self, field_info: Any) -> Any:
        """Synthesizes an appropriate authentic default value for a schema field."""
        annotation = getattr(field_info, "annotation", str)
        if annotation is int:
            return 42
        if annotation is float:
            return 1.75
        if annotation is bool:
            return True
        if annotation is list or getattr(annotation, "__origin__", None) is list:
            return ["Item A", "Item B"]
        if annotation is dict or getattr(annotation, "__origin__", None) is dict:
            return {"key": "value"}
        return "Synthesized Mock Value"

    def mutate_code(
        self, code: str, instruction: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        self.call_count += 1
        inst_lower = instruction.lower()
        ctx_err = ""
        if context:
            ctx_err = str(context.get("error", "")).lower()

        # Check for division by zero error
        if "zerodivision" in inst_lower or "zerodivision" in ctx_err or "division by zero" in inst_lower:
            if "np.std" in code and "+ 1e-8" not in code:
                return code.replace("np.std(window_slice, axis=0)", "np.std(window_slice, axis=0) + 1e-8")
            if "/ std" in code and "1e-8" not in code:
                return code.replace("/ std", "/ (std + 1e-8)")

        # Check for lookahead / index shift
        if "lookahead" in inst_lower or "lag" in inst_lower or "temporal" in inst_lower:
            if "weights[t]" in code:
                return code.replace("weights[t]", "weights[t-1]")
            if "realized_weights = weights" in code:
                return code.replace("realized_weights = weights", "realized_weights = weights[:-1]")

        # Check for volatility targeting refinement
        if "volatility" in inst_lower or "target vol" in inst_lower or "scale" in inst_lower:
            if "# Volatility targeting refinement" not in code and "def compute_weights" in code:
                refined_snippet = (
                    "    def compute_weights(self, signals: np.ndarray, current_weights: np.ndarray) -> np.ndarray:\n"
                    "        # Volatility targeting refinement\n"
                    "        T, K = signals.shape\n"
                    "        weights = np.zeros((T, K), dtype=float)\n"
                    "        for t in range(T):\n"
                    "            pos = np.maximum(signals[t, :K-1], 0.0)\n"
                    "            s_sum = np.sum(pos)\n"
                    "            if s_sum > 1e-6:\n"
                    "                vol_factor = 0.85  # Target volatility cap\n"
                    "                weights[t, :K-1] = vol_factor * (pos / s_sum)\n"
                    "                weights[t, K-1] = 1.0 - vol_factor\n"
                    "            else:\n"
                    "                weights[t, K-1] = 1.0\n"
                    "        return weights\n"
                )
                # Replace compute_weights block
                idx = code.find("def compute_weights")
                if idx != -1:
                    return code[:idx] + refined_snippet

        # Parameter tuning: tune window parameter if requested
        if "window" in inst_lower or "fast_window" in inst_lower:
            if "fast_window: int = 12" in code:
                return code.replace("fast_window: int = 12", "fast_window: int = 10")

        # Fallback: append comment indicating refinement
        return code + f"\n# Refined via instruction: {instruction}\n"


def get_llm_provider(config: Optional[RSIExecutionConfig] = None) -> BaseModelProvider:
    """Factory function resolving execution mode with automatic live-to-mock fallback."""
    cfg = config or RSIExecutionConfig()
    active_mode = cfg.resolved_mode()

    if active_mode == "live":
        try:
            return GeminiLiveProvider(
                api_key=cfg.get_api_key(),
                model_name=cfg.model_name,
                temperature=cfg.temperature,
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize GeminiLiveProvider (%s). Falling back gracefully to DeterministicMockProvider.",
                e,
            )
            return DeterministicMockProvider(config=cfg)

    return DeterministicMockProvider(config=cfg)
