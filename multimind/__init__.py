"""
Multimind SDK - A unified interface for multiple LLM providers and local models.
"""

__version__ = "0.1.0"

from multimind.core.base import BaseLLM
from multimind.core.router import ModelRouter
from multimind.core.local_runner import LocalRunner

__all__ = [
    "BaseLLM",
    "ModelRouter",
    "LocalRunner",
] 