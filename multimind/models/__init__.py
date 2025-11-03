"""
Models package for MultiMind SDK.
"""

from .base import BaseLLM
from .factory import ModelFactory
from .openai import OpenAIModel
from .claude import ClaudeModel
from .ollama import OllamaModel, MistralModel
from .multi_model import MultiModelWrapper

# Try to import HuggingFace model
try:
    from .huggingface import HuggingFaceModel
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    HuggingFaceModel = None

__all__ = [
    'BaseLLM',
    'ModelFactory',
    'OpenAIModel',
    'ClaudeModel',
    'OllamaModel',
    'MistralModel',
    'MultiModelWrapper',
]

if HUGGINGFACE_AVAILABLE:
    __all__.append('HuggingFaceModel') 