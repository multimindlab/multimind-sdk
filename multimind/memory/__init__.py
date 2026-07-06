"""
Memory management module for maintaining conversation history and context.

Lightweight memory types are imported eagerly; the rest resolve lazily via
PEP 562 ``__getattr__`` so heavy optional deps (numpy, torch, redis, …) only
load when actually used.
"""

from __future__ import annotations

from typing import Any

from multimind._lazy import lazy_attr

from .base import BaseMemory
from .buffer import BufferMemory
from .summary import SummaryMemory
from .summary_buffer import SummaryBufferMemory
from .token_aware import TokenAwareMemory
from .utils import MemoryUtils

# Each entry maps a public name to (dotted_module_path, extras_group_or_None).
_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    "ActiveLearningMemory": ("multimind.memory.active_learning", None),
    "AdapterMemory": ("multimind.memory.adapter", "finetune"),
    "AdaptiveMemory": ("multimind.memory.adaptive", "memory"),
    "AdvancedMemory": ("multimind.memory.hybrid_memory", "finetune"),
    "AssociativeMemory": ("multimind.memory.associative", None),
    "AutobiographicalMemory": ("multimind.memory.autobiographical", None),
    "BayesianMemory": ("multimind.memory.bayesian", "finetune"),
    "BufferWindowMemory": ("multimind.memory.buffer_window", None),
    "CausalMemory": ("multimind.memory.causal", None),
    "ChatMemory": ("multimind.memory.chat_memory", None),
    "CognitiveScratchpadMemory": ("multimind.memory.cognitive_scratchpad", None),
    "CombinedMemory": ("multimind.memory.combined", None),
    "ConsensusMemory": ("multimind.memory.consensus", "memory"),
    "ContextualMemory": ("multimind.memory.contextual", "memory"),
    "DeclarativeMemory": ("multimind.memory.declarative", None),
    "DNCMemory": ("multimind.memory.dnc", "memory"),
    "EmotionalMemory": ("multimind.memory.emotional", None),
    "EntityMemory": ("multimind.memory.entity", None),
    "EpisodicMemory": ("multimind.memory.episodic", None),
    "EventSourcedMemory": ("multimind.memory.event_sourced", None),
    "ExplicitMemory": ("multimind.memory.explicit", "memory"),
    "FastWeightMemory": ("multimind.memory.hebbian", "finetune"),
    "FederatedMemory": ("multimind.memory.federated", "finetune"),
    "ForgettingCurveMemory": ("multimind.memory.forgetting_curve", "memory"),
    "GenerativeMemory": ("multimind.memory.generative", "memory"),
    "HierarchicalMemory": ("multimind.memory.hierarchical", None),
    "HTMMemory": ("multimind.memory.htm", "memory"),
    "HybridMemory": ("multimind.memory.hybrid", None),
    "ImplicitMemory": ("multimind.memory.implicit", "memory"),
    "KnowledgeGraphMemory": ("multimind.memory.knowledge_graph", None),
    "MetaMemory": ("multimind.memory.meta", None),
    "NeuroSymbolicMemory": ("multimind.memory.neuro_symbolic", None),
    "NoveltyMemory": ("multimind.memory.novelty", "memory"),
    "PlanningMemory": ("multimind.memory.planning", "memory"),
    "ProceduralMemory": ("multimind.memory.procedural", None),
    "ProspectiveMemory": ("multimind.memory.prospective", None),
    "QAM": ("multimind.memory.quantum", "memory"),
    "QRAM": ("multimind.memory.quantum", "memory"),
    "QuantumClassicalHybridMemory": ("multimind.memory.quantum", "memory"),
    "ReadOnlyMemory": ("multimind.memory.readonly", None),
    "RedisMemory": ("multimind.memory.redis", "memory"),
    "ReinforcementMemory": ("multimind.memory.reinforcement", "finetune"),
    "SemanticMemory": ("multimind.memory.semantic", None),
    "SensoryMemory": ("multimind.memory.sensory", None),
    "SimpleMemory": ("multimind.memory.simple", None),
    "SketchMemory": ("multimind.memory.sketch", None),
    "SpatialMemory": ("multimind.memory.spatial", None),
    "SpikingMemory": ("multimind.memory.spiking", "memory"),
    "SQLAlchemyMemory": ("multimind.memory.sqlalchemy", None),
    "TemporalMemory": ("multimind.memory.temporal", None),
    "TimeWeightedMemory": ("multimind.memory.time_weighted", None),
    "TokenBufferMemory": ("multimind.memory.token_buffer", None),
    "TopologicalMemory": ("multimind.memory.quantum", "memory"),
    "VectorStoreMemory": ("multimind.memory.vector_store", "memory"),
    "VersionedMemory": ("multimind.memory.versioned", None),
    "WorkingMemory": ("multimind.memory.working", "memory"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute lookup, cached on the module after first access."""
    if name in _LAZY_ATTRS:
        module_path, extras_group = _LAZY_ATTRS[name]
        value = lazy_attr(name, module_path, extras_group)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'multimind.memory' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))


__all__ = [
    "BaseMemory",
    "BufferMemory",
    "SummaryMemory",
    "SummaryBufferMemory",
    "MemoryUtils",
    "TokenAwareMemory",
    *sorted(_LAZY_ATTRS),
]
