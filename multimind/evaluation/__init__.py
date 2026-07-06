"""
Evaluation module for RAG system evaluation.

Hallucination detection is stdlib-only and imported eagerly; the evaluator
classes need transformers/sentence-transformers and resolve lazily.
"""

from __future__ import annotations

from typing import Any

from multimind._lazy import lazy_attr

from .hallucination import (
    Claim,
    ClaimReport,
    GroundingReport,
    GuardedModel,
    HallucinationDetector,
    HallucinationError,
    SentenceGrounding,
    detect_hallucinations,
)

_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    "AdvancedEvaluator": ("multimind.evaluation.advanced_evaluation", "finetune"),
    "EvaluationMetrics": ("multimind.evaluation.advanced_evaluation", "finetune"),
    "EvaluationMetric": ("multimind.evaluation.evaluation", "finetune"),
    "RAGEvaluation": ("multimind.evaluation.evaluation", "finetune"),
    "RAGEvaluator": ("multimind.evaluation.evaluation", "finetune"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_path, extras_group = _LAZY_ATTRS[name]
        value = lazy_attr(name, module_path, extras_group)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'multimind.evaluation' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))


__all__ = [
    "RAGEvaluator",
    "RAGEvaluation",
    "EvaluationMetric",
    "AdvancedEvaluator",
    "EvaluationMetrics",
    "HallucinationDetector",
    "HallucinationError",
    "GuardedModel",
    "GroundingReport",
    "SentenceGrounding",
    "Claim",
    "ClaimReport",
    "detect_hallucinations",
]
