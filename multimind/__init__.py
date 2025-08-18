"""
MultiMind SDK - A flexible and composable SDK for building AI applications.

This SDK provides a set of tools and abstractions for building AI applications,
including memory management, model integration, context transfer, and utility functions.

Core Components:
- Memory: Conversation and context management
- Models: LLM and embedding model integration
- Context Transfer: Advanced conversation context transfer between LLM providers
- Utils: Common utility functions

Each component is designed to be modular and composable, allowing for flexible
application design.
"""

__version__ = "0.2.1"

# Configuration for warnings and logging
import os
import logging

# Configure logging level for optional dependencies
OPTIONAL_DEPENDENCY_LOG_LEVEL = os.getenv('MULTIMIND_LOG_LEVEL', 'WARNING')
logging.basicConfig(level=getattr(logging, OPTIONAL_DEPENDENCY_LOG_LEVEL))

def configure_warnings(show_backend_warnings: bool = False, log_level: str = 'WARNING') -> None:
    """
    Configure warning behavior for MultiMind SDK.
    
    Args:
        show_backend_warnings: Whether to show warnings for missing vector database backends
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    os.environ['MULTIMIND_SHOW_BACKEND_WARNINGS'] = str(show_backend_warnings).lower()
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))

# Core components
from .main_config import Config
from .models.base import BaseLLM
from .router import ModelRouter
from .core.multimind import MultiMind
from .core.router import Router, TaskType, TaskConfig, RoutingStrategy

# Memory components
from .memory import (
    BaseMemory,
    BufferMemory,
    SummaryMemory,
    SummaryBufferMemory,
    MemoryUtils
)

# Context Transfer components
from .context_transfer import ContextTransferManager

# Agent components
from .agents import Agent, AgentMemory, AgentLoader
from .agents.tools import BaseTool, CalculatorTool

# Orchestration components
from .orchestration.prompt_chain import PromptChain
from .orchestration.task_runner import TaskRunner

# Ensemble components
from .ensemble import AdvancedEnsemble
from .ensemble.advanced import EnsembleMethod

# MCP components
from .mcp.executor import MCPExecutor
from .mcp.parser import MCPParser
from .mcp.advanced_executor import AdvancedMCPExecutor

# Integration handlers
from .integrations.base import IntegrationHandler
from .integrations.github import GitHubIntegrationHandler
from .integrations.slack import SlackIntegrationHandler
from .integrations.discord import DiscordIntegrationHandler
from .integrations.jira import JiraIntegrationHandler

# Logging components
from .multimind_logging.trace_logger import TraceLogger
from .multimind_logging.usage_tracker import UsageTracker

# Model implementations
from .models.claude import ClaudeModel
from .models.ollama import OllamaModel, MistralModel
from .models.openai import OpenAIModel
from .models.factory import ModelFactory
from .models.multi_model import MultiModelWrapper

# LLM Interface
from .llm import LLMInterface, LLMConfig, ModelType

# Non-transformer LLMs
from .llm.non_transformer_llm import (
    NonTransformerLLM,
    SSM_LLM,
    MLPOnlyLLM,
    DiffusionTextLLM,
    MoELLMMixin,
    PerceiverLLM,
    MegaS4LLM,
    LiquidS4LLM,
    S4DLLM,
    S4NDLLM,
    DSSLLM,
    GSSLLM,
    MambaLLM,
    MoEMambaLLM,
    H3LLM,
    RetNetLLM,
    RWKVLLM,
    SE3HyenaLLM,
    TopologicalNNLLM,
    CustomRNNLLM,
    QLoRALLM,
    CompacterLLM
)

# Pre-built workflows
from .mcp.workflows.code_review import CodeReviewWorkflow
from .mcp.workflows.ci_cd import CICDWorkflow
from .mcp.workflows.documentation import DocumentationWorkflow

# API components
from .api import multi_model_app, unified_app

# Server components
from .server import MultiMindServer

# Splitter components
from .splitter import TextSplitter, DocumentSplitter

# Retrieval components
from .retrieval.retriever import Retriever, RetrievalConfig
from .retrieval.enhanced_retrieval import EnhancedRetriever

# Pipeline components
from .pipeline.pipeline import Pipeline, PipelineBuilder

# RAG components
from .rag import RAG, RAGConfig, BaseRAG, RAGError, PostProcessor, PostProcessingConfig

# Document loader components
from .document_loader import DataIngestion

# Embeddings components
from .embeddings import EmbeddingGenerator, EmbeddingConfig, Embedding, EmbeddingType

# Vector store components
from .vector_store import VectorStore, VectorStoreBackend, VectorStoreConfig, SearchResult, VectorStoreType

# Compliance components
from .compliance import (
    ComplianceShard,
    SelfHealingCompliance,
    ExplainableDTO,
    ModelWatermarking,
    AdaptivePrivacy,
    RegulatoryChangeDetector,
    FederatedCompliance,
    ComplianceLevel,
    ComplianceMetrics,
    ComplianceShardConfig,
    SelfHealingConfig,
    ExplainableDTOConfig,
    ModelWatermarkingConfig,
    AdaptivePrivacyConfig,
    RegulatoryChangeConfig,
    FederatedComplianceConfig,
    load_advanced_config,
    save_advanced_config,
    GovernanceConfig,
    Regulation,
    ComplianceTrainer
)

# Fine-tuning components
from .fine_tuning import (
    AdapterDropTuner,
    AdapterFusionTuner,
    AdapterTuner,
    LoRATrainer,
    QLoraTuner,
    PromptTuner,
    PrefixTuner,
    PEFTTuner,
    UniPELTTuner,
    UniPELTPlusTuner,
    MoETrainer,
    RAGFineTuner,
    SSFTuner,
    IntrinsicSAIDTuner,
    IA3Tuner,
    BitFitTuner,
    PromptPoolingTuner,
    CompacterTuner,
    HyperLoRATuner,
    MAMAdapterTuner
)

# Model conversion components
from .model_conversion import (
    BaseModelConverter,
    HuggingFaceConverter,
    OllamaConverter,
    ONNXConverter,
    TensorFlowConverter,
    ONNXRuntimeConverter,
    SafetensorsConverter,
    GGMLConverter,
    OptimizationConverter,
    QuantizationConverter,
    DistillationConverter,
    HardwareOptimizedConverter,
    ConversionPipeline,
    PipelineConverter,
    ModelConversionManager
)

# Context window components
from .context_window import (
    ContextManager,
    ContextOptimizer
)

# Patterns components
from .patterns import (
    RetrievalStep,
    FusionResult,
    MultiHopRetriever,
    RAGFusion,
    GraphRAG,
    SelfImprovingRAG
)

# Observability components
from .observability import (
    MetricsCollector,
    Metric,
    LatencyMetric,
    CostMetric,
    TokenMetric,
    ErrorMetric
)

# Gateway components
from .gateway import (
    MultiMindAPI,
    OpenAIHandler,
    AnthropicHandler,
    OllamaHandler,
    HuggingFaceHandler
)

# Client components
from .client import (
    ModelClient,
    FederatedRouter,
    RAGClient
)

# CLI components
from .cli import (
    cli,
    main,
    compliance,
    chat,
    models,
    config
)

__all__ = [
    # Version
    "__version__",

    # Core
    "BaseLLM",
    "ModelRouter",
    "Router",
    "TaskType", 
    "TaskConfig",
    "RoutingStrategy",
    "Config",
    "MultiMind",

    # Memory
    "BaseMemory",
    "BufferMemory",
    "SummaryMemory",
    "SummaryBufferMemory",
    "MemoryUtils",

    # Context Transfer
    "ContextTransferManager",

    # Agents
    "Agent",
    "AgentMemory",
    "AgentLoader",
    "BaseTool",
    "CalculatorTool",

    # Orchestration
    "PromptChain",
    "TaskRunner",

    # Ensemble
    "AdvancedEnsemble",
    "EnsembleMethod",

    # MCP
    "MCPParser",
    "MCPExecutor",
    "AdvancedMCPExecutor",

    # Integrations
    "IntegrationHandler",
    "GitHubIntegrationHandler",
    "SlackIntegrationHandler",
    "DiscordIntegrationHandler",
    "JiraIntegrationHandler",

    # Logging
    "TraceLogger",
    "UsageTracker",

    # Models
    "OpenAIModel",
    "ClaudeModel",
    "OllamaModel",
    "MistralModel",
    "ModelFactory",
    "MultiModelWrapper",

    # LLM Interface
    "LLMInterface",
    "LLMConfig",
    "ModelType",

    # Non-transformer LLMs
    "NonTransformerLLM",
    "SSM_LLM",
    "MLPOnlyLLM",
    "DiffusionTextLLM",
    "MoELLMMixin",
    "PerceiverLLM",
    "MegaS4LLM",
    "LiquidS4LLM",
    "S4DLLM",
    "S4NDLLM",
    "DSSLLM",
    "GSSLLM",
    "MambaLLM",
    "MoEMambaLLM",
    "H3LLM",
    "RetNetLLM",
    "RWKVLLM",
    "SE3HyenaLLM",
    "TopologicalNNLLM",
    "CustomRNNLLM",
    "QLoRALLM",
    "CompacterLLM",

    # Workflows
    "CodeReviewWorkflow",
    "CICDWorkflow",
    "DocumentationWorkflow",

    # API
    "multi_model_app",
    "unified_app",

    # Server
    "MultiMindServer",

    # Splitter
    "TextSplitter",
    "DocumentSplitter",

    # Retrieval
    "Retriever",
    "RetrievalConfig",
    "EnhancedRetriever",

    # Pipeline
    "Pipeline",
    "PipelineBuilder",

    # RAG
    "RAG",
    "RAGConfig",
    "BaseRAG",
    "RAGError",
    "PostProcessor",
    "PostProcessingConfig",

    # Document Loader
    "DataIngestion",

    # Embeddings
    "EmbeddingGenerator",
    "EmbeddingConfig",
    "Embedding",
    "EmbeddingType",

    # Vector Store
    "VectorStore",
    "VectorStoreBackend",
    "VectorStoreConfig",
    "SearchResult",
    "VectorStoreType",

    # Compliance
    "ComplianceShard",
    "SelfHealingCompliance",
    "ExplainableDTO",
    "ModelWatermarking",
    "AdaptivePrivacy",
    "RegulatoryChangeDetector",
    "FederatedCompliance",
    "ComplianceLevel",
    "ComplianceMetrics",
    "ComplianceShardConfig",
    "SelfHealingConfig",
    "ExplainableDTOConfig",
    "ModelWatermarkingConfig",
    "AdaptivePrivacyConfig",
    "RegulatoryChangeConfig",
    "FederatedComplianceConfig",
    "load_advanced_config",
    "save_advanced_config",
    "GovernanceConfig",
    "Regulation",
    "ComplianceTrainer",

    # Fine-tuning
    "AdapterDropTuner",
    "AdapterFusionTuner",
    "AdapterTuner",
    "LoRATrainer",
    "QLoraTuner",
    "PromptTuner",
    "PrefixTuner",
    "PEFTTuner",
    "UniPELTTuner",
    "UniPELTPlusTuner",
    "MoETrainer",
    "RAGFineTuner",
    "SSFTuner",
    "IntrinsicSAIDTuner",
    "IA3Tuner",
    "BitFitTuner",
    "PromptPoolingTuner",
    "CompacterTuner",
    "HyperLoRATuner",
    "MAMAdapterTuner",

    # Model conversion
    "BaseModelConverter",
    "HuggingFaceConverter",
    "OllamaConverter",
    "ONNXConverter",
    "TensorFlowConverter",
    "ONNXRuntimeConverter",
    "SafetensorsConverter",
    "GGMLConverter",
    "OptimizationConverter",
    "QuantizationConverter",
    "DistillationConverter",
    "HardwareOptimizedConverter",
    "ConversionPipeline",
    "PipelineConverter",
    "ModelConversionManager",

    # Context window
    "ContextManager",
    "ContextOptimizer",

    # Patterns
    "RetrievalStep",
    "FusionResult",
    "MultiHopRetriever",
    "RAGFusion",
    "GraphRAG",
    "SelfImprovingRAG",

    # Observability
    "MetricsCollector",
    "Metric",
    "LatencyMetric",
    "CostMetric",
    "TokenMetric",
    "ErrorMetric",

    # Gateway
    "MultiMindAPI",
    "OpenAIHandler",
    "AnthropicHandler",
    "OllamaHandler",
    "HuggingFaceHandler",

    # Client
    "ModelClient",
    "FederatedRouter",
    "RAGClient",

    # CLI
    "cli",
    "main",
    "compliance",
    "chat",
    "models",
    "config",
]