"""
Multimind SDK - A unified interface for multiple LLM providers and local models.
"""

__version__ = "0.1.0"

# Core components
from multimind.config import Config
from multimind.models.base import BaseLLM
from multimind.router.router import ModelRouter
from multimind.core.multimind import MultiMind

# Agent components
from multimind.agents.agent import Agent
from multimind.agents.memory import AgentMemory
from multimind.agents.tools.calculator import CalculatorTool

# Orchestration components
from multimind.orchestration.prompt_chain import PromptChain
from multimind.orchestration.task_runner import TaskRunner

# MCP components
from multimind.mcp.executor import MCPExecutor
from multimind.mcp.parser import MCPParser
from multimind.mcp.advanced_executor import AdvancedMCPExecutor
from multimind.mcp.api.base import MCPWorkflowAPI
from multimind.mcp.api.registry import WorkflowRegistry

# Integration handlers
from multimind.integrations.base import IntegrationHandler
from multimind.integrations.github import GitHubIntegrationHandler
from multimind.integrations.slack import SlackIntegrationHandler
from multimind.integrations.discord import DiscordIntegrationHandler
from multimind.integrations.jira import JiraIntegrationHandler

# Logging components
from multimind.logging.trace_logger import TraceLogger
from multimind.logging.usage_tracker import UsageTracker

# Model implementations
from multimind.models.claude import ClaudeModel
from multimind.models.ollama import OllamaModel
from multimind.models.openai import OpenAIModel

# Pre-built workflows
from multimind.mcp.workflows.code_review import CodeReviewWorkflow
from multimind.mcp.workflows.ci_cd import CICDWorkflow
from multimind.mcp.workflows.documentation import DocumentationWorkflow
from multimind.mcp.workflows.multi_integration import MultiIntegrationWorkflow

__all__ = [
    # Core
    "BaseLLM",
    "ModelRouter",
    "Config",
    "MultiMind",

    # Agents
    "Agent",
    "AgentMemory",
    "CalculatorTool",

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
    "MultiIntegrationWorkflow",
]