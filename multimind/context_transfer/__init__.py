"""
Context Transfer Module for MultiMind SDK

This module provides functionality to transfer conversation context between different LLM providers.
It extracts conversation history, summarizes context, and formats it for target models.
"""

from multimind import __version__

from .adapters import AdapterFactory, ChatGPTAdapter, ClaudeAdapter, DeepSeekAdapter, ModelAdapter
from .manager import ContextTransferManager

__all__ = [
    "ContextTransferManager",
    "ModelAdapter",
    "DeepSeekAdapter",
    "ClaudeAdapter",
    "ChatGPTAdapter",
    "AdapterFactory",
]
