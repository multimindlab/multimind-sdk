"""
Multimind SDK - A unified interface for multiple LLM providers and local models.
"""

__version__ = "0.1.0"

# Core components
from multimind.models.base import BaseLLM
from multimind.router.router import ModelRouter
from multimind.config import Config

# Agent components
from multimind.agents.agent import Agent
from multimind.agents.memory import AgentMemory
from multimind.agents.tools.calculator import CalculatorTool

# Orchestration components
from multimind.orchestration.prompt_chain import PromptChain
from multimind.orchestration.task_runner import TaskRunner

# MCP components
from multimind.mcp.parser import MCPParser
from multimind.mcp.executor import MCPExecutor

# Logging components
from multimind.logging.trace_logger import TraceLogger
from multimind.logging.usage_tracker import UsageTracker

# Model implementations
from multimind.models.openai import OpenAIModel
from multimind.models.claude import ClaudeModel
from multimind.models.mistral import MistralModel
from multimind.models.huggingface import HuggingFaceModel
from multimind.models.ollama import OllamaModel

__all__ = [
    # Core
    "BaseLLM",
    "ModelRouter",
    "Config",
    
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
    
    # Logging
    "TraceLogger",
    "UsageTracker",
    
    # Models
    "OpenAIModel",
    "ClaudeModel",
    "MistralModel",
    "HuggingFaceModel",
    "OllamaModel",
] 