"""
LLM module for language model interfaces.
"""

from .llm_interface import LLMInterface, GenerationConfig as LLMConfig, ModelType

__all__ = [
    'LLMInterface',
    'LLMConfig',
    'ModelType'
] 