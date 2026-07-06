"""MultiMind SDK — the compliance-first AI agent framework.

This package exposes a wide public surface (models, RAG, agents, fine-tuning,
gateway, compliance, …) but the SDK is designed so that ``import multimind`` is
*cheap*. Heavy optional dependencies (``torch``, ``transformers``, ``chromadb``,
``fastapi``, ``faiss``, …) are **lazily loaded** the first time the user touches
an attribute that needs them.

What this means in practice:

* ``pip install multimind-sdk`` (core only) gives you a working install.
* ``from multimind import OpenAIModel`` or ``from multimind import ClaudeModel``
  works immediately.
* ``from multimind import RAG`` (or any other extras-only feature) only fails
  if you haven't installed the corresponding extra, and the error message
  tells you exactly what to install:

      ImportError: `RAG` requires additional dependencies.
      Install with: pip install 'multimind-sdk[rag]'

Implementation: PEP 562 ``__getattr__`` resolves attributes against
``_LAZY_ATTRS`` on demand, then caches them on the module so subsequent access
is free.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from multimind._lazy import lazy_attr

__version__ = "0.3.0"

# ─── Logging / warning configuration ──────────────────────────────────────────

OPTIONAL_DEPENDENCY_LOG_LEVEL = os.getenv("MULTIMIND_LOG_LEVEL", "WARNING")
logging.basicConfig(level=getattr(logging, OPTIONAL_DEPENDENCY_LOG_LEVEL, logging.WARNING))


def configure_warnings(show_backend_warnings: bool = False, log_level: str = "WARNING") -> None:
    """Tune MultiMind SDK runtime warning behaviour.

    Args:
        show_backend_warnings: emit warnings when optional vector-store
            backends are unavailable.
        log_level: standard ``logging`` level name (``"DEBUG"`` … ``"CRITICAL"``).
    """
    os.environ["MULTIMIND_SHOW_BACKEND_WARNINGS"] = str(show_backend_warnings).lower()
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.WARNING))


# ─── Eager (lightweight) imports ──────────────────────────────────────────────
#
# These two model classes are the most common entry points and their deps
# (``openai``, ``anthropic``) are in the core requirements, so we import them
# eagerly to keep ``from multimind import OpenAIModel`` fast.

from multimind.models.claude import ClaudeModel  # noqa: E402
from multimind.models.openai import OpenAIModel  # noqa: E402

# ─── Lazy attribute map ───────────────────────────────────────────────────────
#
# Each entry maps a public top-level name to:
#   (dotted_module_path, extras_group_or_None)
#
# * ``extras_group=None`` means the dependency lives in core; an ImportError
#   here is a real bug.
# * ``extras_group="rag"`` means the user must ``pip install
#   multimind-sdk[rag]``. The ImportError raised on access spells this out.

_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    # Core orchestration
    "BaseLLM": ("multimind.models.base", None),
    "Config": ("multimind.main_config", None),
    "ModelRouter": ("multimind.router", None),
    "MultiMind": ("multimind.core.multimind", None),
    "Router": ("multimind.core.router", None),
    "TaskType": ("multimind.core.router", None),
    "TaskConfig": ("multimind.core.router", None),
    "RoutingStrategy": ("multimind.core.router", None),
    # Memory
    "BaseMemory": ("multimind.memory", None),
    "BufferMemory": ("multimind.memory", None),
    "SummaryMemory": ("multimind.memory", None),
    "SummaryBufferMemory": ("multimind.memory", None),
    "MemoryUtils": ("multimind.memory", None),
    # Context transfer
    "ContextTransferManager": ("multimind.context_transfer", None),
    # Agents
    "Agent": ("multimind.agents", "agents"),
    "AgentMemory": ("multimind.agents", "agents"),
    "AgentLoader": ("multimind.agents", "agents"),
    "BaseTool": ("multimind.agents.tools", "agents"),
    "CalculatorTool": ("multimind.agents.tools", "agents"),
    # Orchestration
    "PromptChain": ("multimind.orchestration.prompt_chain", None),
    "TaskRunner": ("multimind.orchestration.task_runner", None),
    # Ensemble
    "AdvancedEnsemble": ("multimind.ensemble", None),
    "EnsembleMethod": ("multimind.ensemble.advanced", None),
    # MCP
    "MCPExecutor": ("multimind.mcp.executor", None),
    "MCPParser": ("multimind.mcp.parser", None),
    "AdvancedMCPExecutor": ("multimind.mcp.advanced_executor", None),
    # Integrations
    "IntegrationHandler": ("multimind.integrations.base", None),
    "GitHubIntegrationHandler": ("multimind.integrations.github", None),
    "SlackIntegrationHandler": ("multimind.integrations.slack", None),
    "DiscordIntegrationHandler": ("multimind.integrations.discord", None),
    "JiraIntegrationHandler": ("multimind.integrations.jira", None),
    # Logging / tracing
    "TraceLogger": ("multimind.multimind_logging.trace_logger", None),
    "UsageTracker": ("multimind.multimind_logging.usage_tracker", None),
    # Models (extras live in their own families)
    "OllamaModel": ("multimind.models.ollama", None),
    "MistralModel": ("multimind.models.ollama", None),
    "GroqModel": ("multimind.models.groq", None),
    "MistralAIModel": ("multimind.models.mistral", None),
    "GeminiModel": ("multimind.models.gemini", None),
    "DeepSeekModel": ("multimind.models.deepseek", None),
    "ModelFactory": ("multimind.models.factory", None),
    "MultiModelWrapper": ("multimind.models.multi_model", None),
    "HuggingFaceModel": ("multimind.models.huggingface", "finetune"),
    # LLM interface
    "LLMInterface": ("multimind.llm", None),
    "LLMConfig": ("multimind.llm", None),
    "ModelType": ("multimind.llm", None),
    # Non-transformer LLMs (torch-heavy)
    "NonTransformerLLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "SSM_LLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "MLPOnlyLLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "DiffusionTextLLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "MoELLMMixin": ("multimind.llm.non_transformer_llm", "finetune"),
    "PerceiverLLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "MegaS4LLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "LiquidS4LLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "S4DLLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "S4NDLLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "DSSLLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "GSSLLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "MambaLLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "MoEMambaLLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "H3LLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "RetNetLLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "RWKVLLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "SE3HyenaLLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "TopologicalNNLLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "CustomRNNLLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "QLoRALLM": ("multimind.llm.non_transformer_llm", "finetune"),
    "CompacterLLM": ("multimind.llm.non_transformer_llm", "finetune"),
    # Workflows
    "CodeReviewWorkflow": ("multimind.mcp.workflows.code_review", None),
    "CICDWorkflow": ("multimind.mcp.workflows.ci_cd", None),
    "DocumentationWorkflow": ("multimind.mcp.workflows.documentation", None),
    # API / server
    "multi_model_app": ("multimind.api", "gateway"),
    "unified_app": ("multimind.api", "gateway"),
    "MultiMindServer": ("multimind.server", "gateway"),
    # Splitter
    "TextSplitter": ("multimind.splitter", None),
    "DocumentSplitter": ("multimind.splitter", None),
    # Retrieval
    "Retriever": ("multimind.retrieval.retriever", "rag"),
    "RetrievalConfig": ("multimind.retrieval.retriever", "rag"),
    "EnhancedRetriever": ("multimind.retrieval.enhanced_retrieval", "rag"),
    # Pipeline
    "Pipeline": ("multimind.pipeline.pipeline", None),
    "PipelineBuilder": ("multimind.pipeline.pipeline", None),
    # RAG
    "RAG": ("multimind.rag", "rag"),
    "RAGConfig": ("multimind.rag", "rag"),
    "BaseRAG": ("multimind.rag", "rag"),
    "RAGError": ("multimind.rag", "rag"),
    "PostProcessor": ("multimind.rag", "rag"),
    "PostProcessingConfig": ("multimind.rag", "rag"),
    # Document loader
    "DataIngestion": ("multimind.document_loader", "documents"),
    # Embeddings
    "EmbeddingGenerator": ("multimind.embeddings", "rag"),
    "EmbeddingConfig": ("multimind.embeddings", "rag"),
    "Embedding": ("multimind.embeddings", "rag"),
    "EmbeddingType": ("multimind.embeddings", "rag"),
    "EmbeddingStandardizer": ("multimind.embeddings", "rag"),
    # Vector store (top-level types are core; backends load on demand
    # and prompt for the right extras when actually instantiated).
    "VectorStore": ("multimind.vector_store", None),
    "VectorStoreBackend": ("multimind.vector_store", None),
    "VectorStoreConfig": ("multimind.vector_store", None),
    "SearchResult": ("multimind.vector_store", None),
    "VectorStoreType": ("multimind.vector_store", None),
    "VectorStoreFactory": ("multimind.vector_store", None),
    # Compliance
    "ComplianceShard": ("multimind.compliance", "compliance"),
    "SelfHealingCompliance": ("multimind.compliance", "compliance"),
    "ExplainableDTO": ("multimind.compliance", "compliance"),
    "ModelWatermarking": ("multimind.compliance", "compliance"),
    "AdaptivePrivacy": ("multimind.compliance", "compliance"),
    "RegulatoryChangeDetector": ("multimind.compliance", "compliance"),
    "FederatedCompliance": ("multimind.compliance", "compliance"),
    "ComplianceLevel": ("multimind.compliance", "compliance"),
    "ComplianceMetrics": ("multimind.compliance", "compliance"),
    "ComplianceShardConfig": ("multimind.compliance", "compliance"),
    "SelfHealingConfig": ("multimind.compliance", "compliance"),
    "ExplainableDTOConfig": ("multimind.compliance", "compliance"),
    "ModelWatermarkingConfig": ("multimind.compliance", "compliance"),
    "AdaptivePrivacyConfig": ("multimind.compliance", "compliance"),
    "RegulatoryChangeConfig": ("multimind.compliance", "compliance"),
    "FederatedComplianceConfig": ("multimind.compliance", "compliance"),
    "load_advanced_config": ("multimind.compliance", "compliance"),
    "save_advanced_config": ("multimind.compliance", "compliance"),
    "GovernanceConfig": ("multimind.compliance", "compliance"),
    "Regulation": ("multimind.compliance", "compliance"),
    "ComplianceTrainer": ("multimind.compliance", "compliance"),
    # Fine-tuning (torch + transformers + peft)
    "AdapterDropTuner": ("multimind.fine_tuning", "finetune"),
    "AdapterFusionTuner": ("multimind.fine_tuning", "finetune"),
    "AdapterTuner": ("multimind.fine_tuning", "finetune"),
    "LoRATrainer": ("multimind.fine_tuning", "finetune"),
    "QLoraTuner": ("multimind.fine_tuning", "finetune"),
    "PromptTuner": ("multimind.fine_tuning", "finetune"),
    "PrefixTuner": ("multimind.fine_tuning", "finetune"),
    "PEFTTuner": ("multimind.fine_tuning", "finetune"),
    "UniPELTTuner": ("multimind.fine_tuning", "finetune"),
    "UniPELTPlusTuner": ("multimind.fine_tuning", "finetune"),
    "MoETrainer": ("multimind.fine_tuning", "finetune"),
    "RAGFineTuner": ("multimind.fine_tuning", "finetune"),
    "SSFTuner": ("multimind.fine_tuning", "finetune"),
    "IntrinsicSAIDTuner": ("multimind.fine_tuning", "finetune"),
    "IA3Tuner": ("multimind.fine_tuning", "finetune"),
    "BitFitTuner": ("multimind.fine_tuning", "finetune"),
    "PromptPoolingTuner": ("multimind.fine_tuning", "finetune"),
    "CompacterTuner": ("multimind.fine_tuning", "finetune"),
    "HyperLoRATuner": ("multimind.fine_tuning", "finetune"),
    "MAMAdapterTuner": ("multimind.fine_tuning", "finetune"),
    # Model conversion (heavy: torch, onnx, …)
    "BaseModelConverter": ("multimind.model_conversion", "finetune"),
    "HuggingFaceConverter": ("multimind.model_conversion", "finetune"),
    "OllamaConverter": ("multimind.model_conversion", "finetune"),
    "ONNXConverter": ("multimind.model_conversion", "finetune"),
    "TensorFlowConverter": ("multimind.model_conversion", "finetune"),
    "ONNXRuntimeConverter": ("multimind.model_conversion", "finetune"),
    "SafetensorsConverter": ("multimind.model_conversion", "finetune"),
    "GGMLConverter": ("multimind.model_conversion", "finetune"),
    "OptimizationConverter": ("multimind.model_conversion", "finetune"),
    "QuantizationConverter": ("multimind.model_conversion", "finetune"),
    "DistillationConverter": ("multimind.model_conversion", "finetune"),
    "HardwareOptimizedConverter": ("multimind.model_conversion", "finetune"),
    "ConversionPipeline": ("multimind.model_conversion", "finetune"),
    "PipelineConverter": ("multimind.model_conversion", "finetune"),
    "ModelConversionManager": ("multimind.model_conversion", "finetune"),
    # Context window
    "ContextManager": ("multimind.context_window", None),
    "ContextOptimizer": ("multimind.context_window", None),
    # Patterns (RAG-flavoured)
    "RetrievalStep": ("multimind.patterns", "rag"),
    "FusionResult": ("multimind.patterns", "rag"),
    "MultiHopRetriever": ("multimind.patterns", "rag"),
    "RAGFusion": ("multimind.patterns", "rag"),
    "GraphRAG": ("multimind.patterns", "rag"),
    "SelfImprovingRAG": ("multimind.patterns", "rag"),
    # Observability
    "MetricsCollector": ("multimind.observability", None),
    "Metric": ("multimind.observability", None),
    "LatencyMetric": ("multimind.observability", None),
    "CostMetric": ("multimind.observability", None),
    "TokenMetric": ("multimind.observability", None),
    "ErrorMetric": ("multimind.observability", None),
    # Gateway / API server
    "MultiMindAPI": ("multimind.gateway", "gateway"),
    "OpenAIHandler": ("multimind.gateway", "gateway"),
    "AnthropicHandler": ("multimind.gateway", "gateway"),
    "OllamaHandler": ("multimind.gateway", "gateway"),
    "HuggingFaceHandler": ("multimind.gateway", "gateway"),
    # Client
    "ModelClient": ("multimind.client", None),
    "FederatedRouter": ("multimind.client", None),
    "RAGClient": ("multimind.client", "rag"),
    # CLI
    "cli": ("multimind.cli", None),
    "main": ("multimind.cli", None),
    "compliance": ("multimind.cli", None),
    "chat": ("multimind.cli", None),
    "models": ("multimind.cli", None),
    "config": ("multimind.cli", None),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute lookup.

    Resolves ``name`` via ``_LAZY_ATTRS`` and caches the result on the module
    so subsequent accesses are free. Unknown names raise ``AttributeError`` as
    required by the data model.
    """
    if name in _LAZY_ATTRS:
        module_path, extras_group = _LAZY_ATTRS[name]
        value = lazy_attr(name, module_path, extras_group)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'multimind' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy names for ``dir()`` and IDE autocompletion."""
    return sorted(set(globals()) | set(_LAZY_ATTRS))


__all__ = [
    "__version__",
    "configure_warnings",
    *sorted(_LAZY_ATTRS),
    # Eagerly imported
    "OpenAIModel",
    "ClaudeModel",
]


# ─── Static type-checker support ──────────────────────────────────────────────
# Type checkers don't execute ``__getattr__``; they need explicit imports.
if TYPE_CHECKING:  # pragma: no cover
    from multimind.agents import Agent, AgentLoader, AgentMemory  # noqa: F401
    from multimind.agents.tools import BaseTool, CalculatorTool  # noqa: F401
    from multimind.client import FederatedRouter, ModelClient, RAGClient  # noqa: F401
    from multimind.compliance import (  # noqa: F401
        AdaptivePrivacy,
        AdaptivePrivacyConfig,
        ComplianceLevel,
        ComplianceMetrics,
        ComplianceShard,
        ComplianceShardConfig,
        ComplianceTrainer,
        ExplainableDTO,
        ExplainableDTOConfig,
        FederatedCompliance,
        FederatedComplianceConfig,
        GovernanceConfig,
        ModelWatermarking,
        ModelWatermarkingConfig,
        Regulation,
        RegulatoryChangeConfig,
        RegulatoryChangeDetector,
        SelfHealingCompliance,
        SelfHealingConfig,
        load_advanced_config,
        save_advanced_config,
    )
    from multimind.core.multimind import MultiMind  # noqa: F401
    from multimind.core.router import (  # noqa: F401
        Router,
        RoutingStrategy,
        TaskConfig,
        TaskType,
    )
    from multimind.gateway import (  # noqa: F401
        AnthropicHandler,
        HuggingFaceHandler,
        MultiMindAPI,
        OllamaHandler,
        OpenAIHandler,
    )
    from multimind.main_config import Config  # noqa: F401
    from multimind.memory import (  # noqa: F401
        BaseMemory,
        BufferMemory,
        MemoryUtils,
        SummaryBufferMemory,
        SummaryMemory,
    )
    from multimind.models.base import BaseLLM  # noqa: F401
    from multimind.models.deepseek import DeepSeekModel  # noqa: F401
    from multimind.models.factory import ModelFactory  # noqa: F401
    from multimind.models.gemini import GeminiModel  # noqa: F401
    from multimind.models.groq import GroqModel  # noqa: F401
    from multimind.models.mistral import MistralAIModel  # noqa: F401
    from multimind.models.multi_model import MultiModelWrapper  # noqa: F401
    from multimind.models.ollama import MistralModel, OllamaModel  # noqa: F401
    from multimind.rag import (  # noqa: F401
        RAG,
        BaseRAG,
        PostProcessingConfig,
        PostProcessor,
        RAGConfig,
        RAGError,
    )
    from multimind.router import ModelRouter  # noqa: F401
    from multimind.vector_store import (  # noqa: F401
        SearchResult,
        VectorStore,
        VectorStoreBackend,
        VectorStoreConfig,
        VectorStoreFactory,
        VectorStoreType,
    )
