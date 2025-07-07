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

from .memory import (
    BaseMemory,
    BufferMemory,
    SummaryMemory,
    SummaryBufferMemory,
    MemoryUtils
)

__version__ = "0.2.1"

# Core components
from .main_config import Config
from .models.base import BaseLLM
from .router.router import ModelRouter
from .core.multimind import MultiMind

# Context Transfer components
from .context_transfer import ContextTransferManager

# Agent components
from .agents.agent import Agent
from .agents.memory import AgentMemory

# Orchestration components
from .orchestration.prompt_chain import PromptChain
from .orchestration.task_runner import TaskRunner

# MCP components
from .mcp.executor import MCPExecutor
from .mcp.parser import MCPParser
from .mcp.advanced_executor import AdvancedMCPExecutor
from .mcp.api.base import MCPWorkflowAPI, DefaultMCPWorkflowAPI
from .mcp.api.registry import WorkflowRegistry

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
from .models.ollama import OllamaModel
from .models.openai import OpenAIModel

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

# New imports
from .document_loader.document_loader import DefaultFileLoader
from .llm.non_transformer_llm import QLoRALLM, CompacterLLM
from .vector_store.clarifai import ClarifaiBackend
from .vector_store.dashvector import DashVectorBackend
from .vector_store.dingo import DingoDBBackend
from .vector_store.epsilla import EpsillaBackend
from multimind.agents.tools.calculator import CalculatorTool

__all__ = [
    # Memory
    "BaseMemory",
    "BufferMemory",
    "SummaryMemory",
    "SummaryBufferMemory",
    "MemoryUtils",
    
    # Version
    "__version__",

    # Core
    "BaseLLM",
    "ModelRouter",
    "Config",
    "MultiMind",

    # Context Transfer
    "ContextTransferManager",

    # Agents
    "Agent",
    "AgentMemory",

    # Orchestration
    "PromptChain",
    "TaskRunner",

    # MCP
    "MCPParser",
    "MCPExecutor",
    "AdvancedMCPExecutor",
    "MCPWorkflowAPI",
    "WorkflowRegistry",

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

    # New components
    "DefaultMCPWorkflowAPI",
    "DefaultFileLoader",
    "QLoRALLM",
    "CompacterLLM",
    "ClarifaiBackend",
    "DashVectorBackend",
    "DingoDBBackend",
    "EpsillaBackend",
    "CalculatorTool",
]