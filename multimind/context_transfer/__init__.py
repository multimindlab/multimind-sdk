"""
Context Transfer Module for MultiMind SDK

This module provides functionality to transfer conversation context between different LLM providers.
It extracts conversation history, summarizes context, and formats it for target models.
"""

from .manager import ContextTransferManager
from .adapters import ModelAdapter, DeepSeekAdapter, ClaudeAdapter, ChatGPTAdapter, AdapterFactory

__version__ = "1.0.0"
__all__ = [
    "ContextTransferManager",
    "ModelAdapter", 
    "DeepSeekAdapter",
    "ClaudeAdapter",
    "ChatGPTAdapter",
    "AdapterFactory"
] 