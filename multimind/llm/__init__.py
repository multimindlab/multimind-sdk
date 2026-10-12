"""
LLM module for language model interfaces.
"""

from .llm_interface import GenerationConfig as LLMConfig
from .llm_interface import LLMInterface, ModelType

__all__ = ["LLMInterface", "LLMConfig", "ModelType"]
